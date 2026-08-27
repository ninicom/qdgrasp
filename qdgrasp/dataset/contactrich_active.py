"""Public loader for QDGrasp-ContactRich-Active-Tiny (C06.8).

The loader is where a consumer meets the dataset, so it is where the release
contract has to hold rather than be assumed. By default it accepts only a v2
artifact that is clean and unblocked, and it verifies what it is told: the
schema, the declared counts against the shard records, every shard hash, and
that no shard path escapes the dataset root.

``allow_blocked=True`` exists for inspecting an artifact that is not
release-ready. It never promotes one: a blocked dataset stays blocked, and the
loader says so on the way past.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qdgrasp.dataset.dynamic_contracts import (
    CONTACTRICH_MANIFEST_SCHEMA_V2,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
)
from qdgrasp.dataset.dynamic_shards import read_trajectory_shard

#: Splits a release artifact has to carry, both non-empty.
REQUIRED_SPLITS: tuple[str, ...] = ("train", "val")


class DatasetRejected(ValueError):
    """The artifact does not satisfy the contract it claims to satisfy."""


@dataclasses.dataclass(frozen=True)
class ContactRichActiveDataset:
    """A verified artifact, and the disclosure that travels with it."""

    root: Path
    manifest: dict[str, Any]

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def release_blocked(self) -> bool:
        return bool(self.manifest.get("release_blocked", True))

    @property
    def active_hands(self) -> tuple[str, ...]:
        return tuple(self.manifest["scope"]["active_hands"])

    @property
    def paused_hands(self) -> tuple[str, ...]:
        return tuple(self.manifest["scope"]["paused_hands"])

    @property
    def three_hand_coverage(self) -> bool:
        # Never inferred from a count: a two-hand artifact must not be readable
        # as the three-hand contract under any arithmetic.
        return False

    def split(self, name: str) -> tuple[tuple[DynamicGraspTrajectory, DynamicSearchOutcome], ...]:
        """Every sample of one split, rebuilt through the typed contracts."""
        shards = [s for s in self.manifest["shards"] if s["split"] == name]
        if not shards:
            raise DatasetRejected(f"no shard for split {name!r}")
        samples: list[tuple[DynamicGraspTrajectory, DynamicSearchOutcome]] = []
        for shard in shards:
            samples.extend(read_trajectory_shard(self.root / shard["path"]))
        return tuple(samples)

    def iter_splits(self) -> Iterator[tuple[str, int]]:
        for name in REQUIRED_SPLITS:
            yield name, int(self.manifest["counts"][name])


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve a shard path, refusing anything that escapes the dataset root."""
    candidate = (root / relative).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise DatasetRejected(f"shard path {relative!r} escapes the dataset root")
    return candidate


def verify(root: str | Path, *, allow_blocked: bool = False) -> list[str]:
    """Every way this artifact fails its own contract, in one pass.

    Returns the list rather than raising on the first one, so a reviewer sees
    the whole picture instead of fixing problems one run at a time.
    """
    base = Path(root)
    problems: list[str] = []

    manifest_path = base / "dataset_manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path} does not exist"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"manifest is not valid JSON: {exc}"]

    schema = manifest.get("schema")
    if schema != CONTACTRICH_MANIFEST_SCHEMA_V2:
        problems.append(
            f"manifest schema {schema!r} is not the release schema "
            f"{CONTACTRICH_MANIFEST_SCHEMA_V2!r}; a v1 artifact is readable but never "
            "release-ready"
        )
        return problems

    scope = manifest.get("scope") or {}
    if scope.get("three_hand_coverage") is not False:
        problems.append("scope claims three_hand_coverage; ADR-0008 keeps it false")
    if scope.get("historical_p3_4_state") != "paused_by_ADR-0008":
        problems.append("scope must record the historical P3.4 contract as paused_by_ADR-0008")
    if not scope.get("active_hands"):
        problems.append("scope does not disclose which hands the artifact covers")

    if manifest.get("worktree_dirty"):
        problems.append(
            "generated from a dirty worktree; the artifact cannot be reproduced from a commit"
        )

    counts = manifest.get("counts") or {}
    total = 0
    for shard in manifest.get("shards") or []:
        try:
            path = _safe_path(base, shard["path"])
        except DatasetRejected as exc:
            problems.append(str(exc))
            continue
        if not path.is_file():
            problems.append(f"shard {shard['path']} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != shard.get("sha256"):
            problems.append(f"shard {shard['path']} hash {digest} != declared {shard.get('sha256')}")
        payload = json.loads(text)
        records = payload.get("records") or []
        if int(payload.get("count", -1)) != len(records):
            problems.append(f"shard {shard['path']} header count disagrees with its records")
        if len(records) != int(shard.get("count", -1)):
            problems.append(f"shard {shard['path']} record count disagrees with the manifest")
        total += len(records)

    if total != int(counts.get("samples", -1)):
        problems.append(
            f"manifest counts {counts.get('samples')} samples but the shards hold {total}"
        )
    for name in REQUIRED_SPLITS:
        if int(counts.get(name, 0)) <= 0:
            problems.append(f"split {name!r} is empty; a release artifact needs both")

    splits = manifest.get("splits") or {}
    groups = {name: set(entry.get("groups", ())) for name, entry in splits.items()}
    if groups.get("train") and groups.get("val"):
        leaked = groups["train"] & groups["val"]
        if leaked:
            problems.append(f"split leakage: groups on both sides {sorted(leaked)}")

    dispositions = counts.get("dispositions") or {}
    if dispositions.get("unexpected_control_outcome"):
        problems.append(
            f"{dispositions['unexpected_control_outcome']} negative control(s) passed; "
            "a control that does not control anything is not a negative"
        )

    coverage = manifest.get("coverage") or {}
    if coverage.get("observed_cells") != coverage.get("declared_cells"):
        problems.append(
            f"coverage grid incomplete: {coverage.get('observed_cells')} of "
            f"{coverage.get('declared_cells')} declared cells"
        )
    for hand in scope.get("active_hands", ()):
        if int((coverage.get("positives_by_hand") or {}).get(hand, 0)) < 1:
            problems.append(f"{hand} has no CPU-confirmed positive")

    if manifest.get("release_blocked", True) and not allow_blocked:
        reasons = manifest.get("blocked_reasons") or ["unspecified"]
        problems.append(f"release_blocked=true: {reasons}")

    return problems


def load(root: str | Path, *, allow_blocked: bool = False) -> ContactRichActiveDataset:
    """Load an artifact, or refuse it and say why."""
    problems = verify(root, allow_blocked=allow_blocked)
    if problems:
        raise DatasetRejected(
            f"{root} is not a loadable ContactRich-Active artifact:\n  - "
            + "\n  - ".join(problems)
        )
    base = Path(root)
    manifest = json.loads((base / "dataset_manifest.json").read_text(encoding="utf-8"))
    return ContactRichActiveDataset(root=base, manifest=manifest)


def dataset_card(manifest: dict[str, Any]) -> str:
    """The disclosure a consumer has to see before using this (C06.9)."""
    scope = manifest.get("scope") or {}
    counts = manifest.get("counts") or {}
    lines = [
        f"# {manifest.get('dataset_id')} v{manifest.get('version')}",
        "",
        "## Scope",
        "",
        f"- Active hands: {', '.join(scope.get('active_hands', ()))}",
        (
            f"- Paused hands: {', '.join(scope.get('paused_hands', ()))} "
            f"({scope.get('governing_decision')})"
        ),
        (
            "- Three-hand coverage: **no**. This artifact is two active hands and "
            "must not be read as the historical three-hand P3.4 contract, which "
            "stays `paused_by_ADR-0008`."
        ),
        "",
        "## Contents",
        "",
        (
            f"- Samples: {counts.get('samples')} "
            f"(train {counts.get('train')}, val {counts.get('val')})"
        ),
        f"- Dispositions: {counts.get('dispositions')}",
        (
            "- Every sample carries its trajectory, its contact stream and its "
            "outcome, including the failures: a critic or safety model trained "
            "later needs them."
        ),
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in manifest.get("limitations", ())]
    if manifest.get("release_blocked", True):
        lines += [
            "",
            "## Release status",
            "",
            "**release_blocked = true.** This artifact is not cleared for release:",
            "",
        ]
        lines += [f"- {reason}" for reason in manifest.get("blocked_reasons", ())]
    lines += [
        "",
        "## License",
        "",
        f"{manifest.get('license')}",
        "",
        (
            "Simulation-only contact measurements. Nothing here is a claim about "
            "what a physical hand survives; a hardware claim needs manufacturer "
            "limits, calibration, a safety factor and its own revision record."
        ),
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "REQUIRED_SPLITS",
    "ContactRichActiveDataset",
    "DatasetRejected",
    "dataset_card",
    "load",
    "verify",
]
