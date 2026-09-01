"""The hard stop G0 asks for on the public training path.

``PLAN.md`` §9.3 item 1: public training must not start when the canonical
dataset audit or the Phase 5 positive gate fails.  Both checks already existed
as scripts, and both were reported as failing while the public facade trained on
the same corpus anyway -- that gap is the finding, not the checks.

The stop is scoped to *corpus* datasets: a root that ships a
``dataset_manifest.json`` claims provenance, so it is held to it.  A synthetic
fixture that claims nothing is not gated, because there is nothing to verify and
nothing it could mislabel.

There is deliberately no override flag and no environment variable.  A gate with
a documented bypass is a warning, and ``PLAN.md`` §9.2 says these invariants may
not be resolved by a warning.  A dataset that passes its own audit is not
blocked by this module at all; one that fails it is blocked for everyone.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from . import registry

CORRECTIVE_GATE_SCHEMA = "qdgrasp/corrective-gate/v1"

#: Name a dataset root must carry to be treated as a provenance-bearing corpus.
MANIFEST_FILE = "dataset_manifest.json"

#: Where the canonical audit and the positive gate still live.  ``COR-01``/G1
#: moves them into the package; until then the gate locates them by path and
#: refuses to run without them, because "the audit was not available" is not a
#: reason to train on an unaudited corpus.
_AUDIT_SCRIPT = "check_dataset_manifest.py"
_POSITIVE_GATE_SCRIPT = "check_phase5_inputs.py"
_PROTOCOL_DIRECTORY = Path("configs") / "phase5"


class CorrectiveGateError(RuntimeError):
    """Public training was refused because a corrective gate is still open."""


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """One gate check and what it found."""

    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == "fail"


@dataclasses.dataclass(frozen=True)
class GateReport:
    """What the gate looked at and what it decided."""

    purpose: str
    gated: bool
    dataset_root: str | None
    dataset_id: str | None
    checks: tuple[CheckResult, ...]

    @property
    def allowed(self) -> bool:
        return not any(check.failed for check in self.checks)

    @property
    def release_evidence_allowed(self) -> bool:
        """No run may produce release evidence while §9 findings are open."""

        return not registry.release_is_blocked()

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": CORRECTIVE_GATE_SCHEMA,
            "purpose": self.purpose,
            "gated": self.gated,
            "dataset_root": self.dataset_root,
            "dataset_id": self.dataset_id,
            "allowed": self.allowed,
            "release_evidence_allowed": self.release_evidence_allowed,
            "checks": [dataclasses.asdict(check) for check in self.checks],
        }

    def render(self) -> str:
        lines = [
            f"corrective gate refused {self.purpose} on {self.dataset_root}",
            f"dataset_id: {self.dataset_id}",
            "",
            "checks:",
        ]
        lines.extend(f"  {check.status:4s} {check.name}: {check.detail}" for check in self.checks)
        lines.extend(
            [
                "",
                (
                    f"This is the G0 hard stop from PLAN.md §9.3, added by {registry.REVISION_RECORD} after "
                    f"the cross-component audit in {registry.AUDIT_SESSION}."
                ),
                "Open findings:",
                registry.summary(),
                "",
                "Fix the dataset, or run the remediation gate that closes the finding. There is no override.",
            ]
        )
        return "\n".join(lines)


# -- locating the checks that have not moved into the package yet ----------


def repository_root() -> Path:
    """The checkout this package was imported from."""

    return Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType | None:
    """Import a repository script by path, or ``None`` when it is not shipped."""

    path = repository_root() / "scripts" / name
    if not path.is_file():
        return None
    root = str(repository_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(f"_qdgrasp_gate_{path.stem}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the individual checks -------------------------------------------------


def resolve_dataset_root(data_config: Any) -> Path | None:
    """The ``dataset_root`` a data configuration names, if it names one."""

    if isinstance(data_config, dict):
        raw = data_config.get("dataset_root")
    else:
        raw = getattr(data_config, "dataset_root", None)
    if raw in (None, ""):
        return None
    return Path(str(raw))


def is_corpus_dataset(root: Path | None) -> bool:
    """Does this root claim provenance a gate can hold it to?"""

    return root is not None and (root / MANIFEST_FILE).is_file()


def read_dataset_id(root: Path) -> str | None:
    try:
        document = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = document.get("dataset_id")
    return str(value) if value is not None else None


def canonical_audit(root: Path) -> CheckResult:
    """Run the canonical dataset audit, failing closed when it is missing."""

    module = _load_script(_AUDIT_SCRIPT)
    if module is None or not hasattr(module, "audit_dataset_manifest"):
        return CheckResult(
            name="canonical_dataset_audit",
            status="fail",
            detail=(
                f"the canonical audit ({_AUDIT_SCRIPT}) is not available from {repository_root()}, so this "
                "corpus cannot be verified; COR-01/G1 moves the audit into the package, and until then an "
                "unauditable corpus is not trainable"
            ),
        )
    try:
        summary = module.audit_dataset_manifest(root)
    except Exception as error:  # noqa: BLE001 - any refusal the audit raises is a refusal
        return CheckResult(name="canonical_dataset_audit", status="fail", detail=f"{type(error).__name__}: {error}")
    shards = summary.get("total_shards") if isinstance(summary, dict) else None
    return CheckResult(
        name="canonical_dataset_audit",
        status="pass",
        detail=f"audit passed{'' if shards is None else f' over {shards} shards'}",
    )


def _locked_protocol_for(dataset_id: str | None) -> Path | None:
    """The current locked protocol for this dataset, if one exists.

    A superseded protocol is still on disk -- that is the point of locking one
    by hash -- so candidates are parsed and the ones the current schema refuses
    are skipped.  Picking the first filename would measure the dataset against
    a protocol nobody is running.
    """

    if dataset_id is None:
        return None
    directory = repository_root() / _PROTOCOL_DIRECTORY
    if not directory.is_dir():
        return None

    from ..models.protocol import ProtocolError, load_protocol

    for path in sorted(directory.glob("*.yaml")):
        try:
            protocol = load_protocol(path)
        except (ProtocolError, OSError, ValueError):
            continue
        if protocol.dataset_id == dataset_id:
            return path
    return None


def positive_gate(root: Path, dataset_id: str | None, *, audited: bool = True) -> CheckResult:
    """Does the locked protocol's train split hold enough positives to train on?"""

    if not audited:
        return CheckResult(
            name="phase5_positive_gate",
            status="skip",
            detail=(
                "not measured: the canonical audit refused this corpus, and counting samples the artifact "
                "cannot vouch for would describe whatever bytes are on disk"
            ),
        )
    protocol = _locked_protocol_for(dataset_id)
    if protocol is None:
        return CheckResult(
            name="phase5_positive_gate",
            status="skip",
            detail=f"no locked protocol names dataset {dataset_id!r}; the positive floor is defined by a protocol",
        )
    module = _load_script(_POSITIVE_GATE_SCRIPT)
    if module is None or not hasattr(module, "measure"):
        return CheckResult(
            name="phase5_positive_gate",
            status="fail",
            detail=(
                f"{_POSITIVE_GATE_SCRIPT} is not available from {repository_root()}, so the positive floor "
                "cannot be measured for a dataset that has a locked protocol"
            ),
        )
    try:
        report = module.measure(root, protocol)
    except Exception as error:  # noqa: BLE001 - any refusal the gate raises is a refusal
        return CheckResult(name="phase5_positive_gate", status="fail", detail=f"{type(error).__name__}: {error}")
    if report.get("sufficient"):
        return CheckResult(
            name="phase5_positive_gate",
            status="pass",
            detail=f"every active hand clears {report.get('minimum_positives_per_hand')} positives under "
            f"{protocol.name}",
        )
    rows = [
        f"{row['split']}/{row['robot']}={row['positives']}"
        for row in report.get("rows", [])
        if row.get("split") == "train"
    ]
    return CheckResult(
        name="phase5_positive_gate",
        status="fail",
        detail=(
            f"under {protocol.name} the train split holds {', '.join(rows) or 'no positives'} against a floor "
            f"of {report.get('minimum_positives_per_hand')} per hand; regressing onto these labels teaches the "
            "proposal distribution, not grasping"
        ),
    )


# -- the gate --------------------------------------------------------------


def evaluate(data_config: Any, *, purpose: str = "training") -> GateReport:
    """Check a data configuration without raising."""

    root = resolve_dataset_root(data_config)
    if not is_corpus_dataset(root):
        return GateReport(
            purpose=purpose,
            gated=False,
            dataset_root=None if root is None else root.as_posix(),
            dataset_id=None,
            checks=(
                CheckResult(
                    name="corpus_dataset",
                    status="skip",
                    detail=(
                        "the configuration names no dataset root carrying a dataset_manifest.json, so there is "
                        "no provenance claim to verify"
                    ),
                ),
            ),
        )
    assert root is not None
    dataset_id = read_dataset_id(root)
    audit = canonical_audit(root)
    checks = (audit, positive_gate(root, dataset_id, audited=not audit.failed))
    return GateReport(
        purpose=purpose,
        gated=True,
        dataset_root=root.as_posix(),
        dataset_id=dataset_id,
        checks=checks,
    )


def assert_public_training_allowed(data_config: Any, *, purpose: str = "training") -> GateReport:
    """Refuse public training on a corpus that fails its own audit or gate."""

    report = evaluate(data_config, purpose=purpose)
    if not report.allowed:
        raise CorrectiveGateError(report.render())
    return report


__all__ = [
    "CORRECTIVE_GATE_SCHEMA",
    "CheckResult",
    "CorrectiveGateError",
    "GateReport",
    "assert_public_training_allowed",
    "canonical_audit",
    "evaluate",
    "is_corpus_dataset",
    "positive_gate",
    "read_dataset_id",
    "repository_root",
    "resolve_dataset_root",
]
