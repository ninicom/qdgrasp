"""Closed-world requirement ledger for Phase 3.4.3 (C08).

A plan written in prose can be read as passing by anyone who wants it to. This
module reads the same plan as a manifest of identified requirements, each of
which has to name the code that implements it, the tests that exercise it and
the evidence that was produced -- and then computes the closure verdict from
those facts alone.

The rules it enforces come from the plan, not from convenience:

* a requirement claiming ``passed`` must name tests **and** evidence that exist;
* a requirement mapped to an ID nobody declared is an untracked gate;
* a ``deferred_not_claimed`` item may not be cited as coverage by anything;
* a dirty worktree cannot produce a pass, because the artifacts would not be
  reproducible from the commit under review;
* the historical three-hand P3.4 verdict stays paused whatever this ledger says.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

#: The only statuses a requirement may carry. Anything else is a manifest bug,
#: not a new state: an unrecognised status must never be read as "fine".
ALLOWED_STATUS: frozenset[str] = frozenset({"pending", "passed", "failed", "blocked", "paused", "deferred_not_claimed"})

#: Statuses that leave the phase open. ``paused`` and ``deferred_not_claimed``
#: are open too: they are honest non-coverage, not silent coverage.
_OPEN_STATUS: frozenset[str] = frozenset({"pending", "failed", "blocked", "paused", "deferred_not_claimed"})

MANIFEST_SCHEMA = "qdgrasp/roadmap-requirements/v1"


class ManifestError(ValueError):
    """The manifest itself is malformed and cannot be audited."""


@dataclasses.dataclass(frozen=True)
class Requirement:
    """One identified obligation of the plan."""

    id: str
    category: str
    normative_source: str
    mapped_to: tuple[str, ...]
    required: bool
    owner: str
    implementation_refs: tuple[str, ...]
    test_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    status: str
    blocker_reason: str
    supersession_policy: str

    @property
    def is_closed(self) -> bool:
        return self.status == "passed"

    @property
    def is_open(self) -> bool:
        return self.status in _OPEN_STATUS


@dataclasses.dataclass(frozen=True)
class RequirementsManifest:
    """The parsed manifest plus the scope and closure rule it was written under."""

    schema: str
    plan_id: str
    plan_version: str
    scope: Mapping[str, Any]
    closure_rule: Mapping[str, Any]
    normative_sources: tuple[str, ...]
    requirements: tuple[Requirement, ...]
    source_path: Path
    source_sha256: str

    def by_id(self, requirement_id: str) -> Requirement:
        for requirement in self.requirements:
            if requirement.id == requirement_id:
                return requirement
        raise KeyError(requirement_id)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(r.id for r in self.requirements)

    def of_category(self, category: str) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.category == category)


@dataclasses.dataclass(frozen=True)
class ClosureVerdict:
    """What the ledger permits to be said out loud."""

    verdict: str
    release_blocked: bool
    exit_code: int
    closure_scope: str
    active_hands: tuple[str, ...]
    paused_hands: tuple[str, ...]
    three_hand_coverage: bool
    total_requirements: int
    mapped_requirements: int
    status_counts: Mapping[str, int]
    unmapped: tuple[str, ...]
    unknown: tuple[str, ...]
    open_required: tuple[str, ...]
    violations: tuple[str, ...]
    worktree_dirty: bool
    manifest_sha256: str

    def as_json_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["status_counts"] = dict(sorted(self.status_counts.items()))
        return payload


def _as_tuple(value: Any, *, field: str, requirement_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ManifestError(f"{requirement_id}.{field} must be a list, got {value!r}")
    return tuple(str(item) for item in value)


def load_manifest(path: str | Path) -> RequirementsManifest:
    """Parse and structurally validate a requirements manifest."""
    source = Path(path)
    raw_text = source.read_text(encoding="utf-8")
    document = yaml.safe_load(raw_text)
    if not isinstance(document, Mapping):
        raise ManifestError(f"{source} is not a YAML mapping")

    schema = str(document.get("schema", ""))
    if schema != MANIFEST_SCHEMA:
        raise ManifestError(f"unsupported manifest schema {schema!r}, expected {MANIFEST_SCHEMA!r}")

    entries = document.get("requirements")
    if not isinstance(entries, Sequence) or not entries:
        raise ManifestError("manifest declares no requirements")

    seen: set[str] = set()
    requirements: list[Requirement] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ManifestError(f"requirement entry is not a mapping: {entry!r}")
        requirement_id = str(entry.get("id", "")).strip()
        if not requirement_id:
            raise ManifestError(f"requirement entry without an id: {entry!r}")
        if requirement_id in seen:
            raise ManifestError(f"duplicate requirement id {requirement_id!r}")
        seen.add(requirement_id)

        status = str(entry.get("status", "pending"))
        if status not in ALLOWED_STATUS:
            raise ManifestError(
                f"{requirement_id} carries status {status!r}, which is not one of {sorted(ALLOWED_STATUS)}"
            )
        requirements.append(
            Requirement(
                id=requirement_id,
                category=str(entry.get("category", "unknown")),
                normative_source=str(entry.get("normative_source", "")),
                mapped_to=_as_tuple(entry.get("mapped_to"), field="mapped_to", requirement_id=requirement_id),
                required=bool(entry.get("required", True)),
                owner=str(entry.get("owner", "unassigned")),
                implementation_refs=_as_tuple(
                    entry.get("implementation_refs"), field="implementation_refs", requirement_id=requirement_id
                ),
                test_ids=_as_tuple(entry.get("test_ids"), field="test_ids", requirement_id=requirement_id),
                evidence_refs=_as_tuple(
                    entry.get("evidence_refs"), field="evidence_refs", requirement_id=requirement_id
                ),
                status=status,
                blocker_reason=str(entry.get("blocker_reason", "")),
                supersession_policy=str(entry.get("supersession_policy", "revision_required")),
            )
        )

    scope = document.get("scope") or {}
    closure_rule = document.get("closure_rule") or {}
    if not isinstance(scope, Mapping) or not isinstance(closure_rule, Mapping):
        raise ManifestError("scope and closure_rule must both be mappings")

    return RequirementsManifest(
        schema=schema,
        plan_id=str(document.get("plan_id", "")),
        plan_version=str(document.get("plan_version", "")),
        scope=dict(scope),
        closure_rule=dict(closure_rule),
        normative_sources=_as_tuple(
            document.get("normative_sources"), field="normative_sources", requirement_id="<manifest>"
        ),
        requirements=tuple(requirements),
        source_path=source,
        source_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    )


def worktree_is_dirty(repo_root: str | Path) -> bool:
    """True when the tree differs from HEAD, or when git cannot say.

    A tree whose state cannot be established is treated as dirty: a release
    claim needs a commit someone else can check out, and "git is missing" is not
    evidence that the tree is clean.
    """
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if completed.returncode != 0:
        return True
    return bool(completed.stdout.strip())


def _missing_paths(refs: Iterable[str], repo_root: Path) -> list[str]:
    missing: list[str] = []
    for ref in refs:
        # A ref may carry a ``path::anchor`` suffix; only the path part is checked.
        path_part = ref.split("::", 1)[0].strip()
        if not path_part:
            missing.append(ref)
            continue
        if not (repo_root / path_part).exists():
            missing.append(ref)
    return missing


def audit_closure(
    manifest: RequirementsManifest,
    *,
    repo_root: str | Path,
    worktree_dirty: bool | None = None,
    require_release: bool = False,
) -> ClosureVerdict:
    """Compute the closure verdict from the manifest and the tree it describes."""
    root = Path(repo_root)
    dirty = worktree_is_dirty(root) if worktree_dirty is None else bool(worktree_dirty)

    declared = manifest.ids
    violations: list[str] = []
    unmapped: list[str] = []
    unknown: set[str] = set()
    status_counts: dict[str, int] = {status: 0 for status in sorted(ALLOWED_STATUS)}

    deferred = {r.id for r in manifest.requirements if r.status == "deferred_not_claimed"}

    for requirement in manifest.requirements:
        status_counts[requirement.status] += 1

        if not requirement.mapped_to:
            unmapped.append(requirement.id)
        for target in requirement.mapped_to:
            if target not in declared:
                unknown.add(f"{requirement.id}->{target}")
        if not requirement.normative_source:
            violations.append(f"{requirement.id}: no normative_source")

        if requirement.status == "passed":
            if not requirement.implementation_refs:
                violations.append(f"{requirement.id}: passed without implementation_refs")
            if not requirement.test_ids:
                violations.append(f"{requirement.id}: passed without test_ids")
            if not requirement.evidence_refs:
                violations.append(f"{requirement.id}: passed without evidence_refs")
            # ``test_ids`` is checked alongside the other refs, not separately:
            # a passed claim naming a test file that is not in the tree is
            # exactly as empty as one naming missing evidence, and the only
            # thing that used to catch it was the dirty-worktree rule, which
            # says nothing about the test.
            missing = _missing_paths(
                (*requirement.implementation_refs, *requirement.evidence_refs, *requirement.test_ids), root
            )
            if missing:
                violations.append(f"{requirement.id}: passed but refs do not exist: {sorted(missing)}")
            if dirty:
                violations.append(f"{requirement.id}: passed claimed on a dirty worktree")

        if (
            requirement.status in {"failed", "blocked", "paused", "deferred_not_claimed"}
            and not requirement.blocker_reason
        ):
            violations.append(f"{requirement.id}: {requirement.status} without blocker_reason")

        if requirement.status == "deferred_not_claimed" and requirement.evidence_refs:
            violations.append(
                f"{requirement.id}: deferred_not_claimed may not carry evidence_refs (deferral is not coverage)"
            )

    # Deferral is an allowed disposition for an optional package -- the plan
    # says so for MPPI -- but not for a required one: dropping something the
    # contract requires needs a revision, not a status change. ``mapped_to`` is
    # a traceability link in both directions, so a gate that passes while
    # explicitly not claiming a deferred item is doing exactly what it should.
    for requirement_id in sorted(deferred):
        requirement = manifest.by_id(requirement_id)
        if requirement.required:
            violations.append(
                f"{requirement_id}: deferred_not_claimed while still required; "
                "deferring a required item needs a revision, not a status change"
            )

    scope = manifest.scope
    three_hand = bool(scope.get("three_hand_coverage", False))
    if three_hand:
        violations.append("scope claims three_hand_coverage=true; ADR-0008 keeps it false")
    historical = str(scope.get("historical_p3_4_state", ""))
    if historical != "paused_by_ADR-0008":
        violations.append(f"scope records historical P3.4 as {historical!r}; it stays 'paused_by_ADR-0008'")
    active_hands = tuple(str(h) for h in scope.get("active_hands", ()) or ())
    paused_hands = tuple(str(h) for h in scope.get("paused_hands", ()) or ())
    if set(active_hands) & set(paused_hands):
        violations.append("a hand is listed as both active and paused")

    rule = manifest.closure_rule
    if not rule.get("allow_unmapped", False) and unmapped:
        violations.append(f"unmapped requirements: {sorted(unmapped)}")
    if not rule.get("allow_unknown", False) and unknown:
        violations.append(f"requirements mapped to untracked ids: {sorted(unknown)}")
    if not rule.get("allow_dirty_candidate", True) and dirty and require_release:
        violations.append("release candidate audited on a dirty worktree")

    open_required = tuple(sorted(r.id for r in manifest.requirements if r.required and r.is_open))

    verdict, exit_code = _resolve_verdict(
        violations=violations,
        open_required=open_required,
        manifest=manifest,
    )
    release_blocked = verdict != "PASS"

    return ClosureVerdict(
        verdict=verdict,
        release_blocked=release_blocked,
        exit_code=exit_code,
        closure_scope=str(scope.get("verdict", "P3.4.3-ACTIVE")),
        active_hands=active_hands,
        paused_hands=paused_hands,
        three_hand_coverage=three_hand,
        total_requirements=len(manifest.requirements),
        mapped_requirements=len(manifest.requirements) - len(unmapped),
        status_counts=status_counts,
        unmapped=tuple(sorted(unmapped)),
        unknown=tuple(sorted(unknown)),
        open_required=open_required,
        violations=tuple(violations),
        worktree_dirty=dirty,
        manifest_sha256=manifest.source_sha256,
    )


def _resolve_verdict(
    *,
    violations: Sequence[str],
    open_required: Sequence[str],
    manifest: RequirementsManifest,
) -> tuple[str, int]:
    """Map ledger state onto the plan's exit-code convention.

    ``0`` is reserved for the exact requested scope passing. Partial, paused and
    blocked each get their own nonzero code so that CI cannot read one as the
    other -- the failure mode B-09 records.
    """
    if violations:
        return ("FAIL", 1)
    if not open_required:
        return ("PASS", 0)

    open_statuses = {manifest.by_id(requirement_id).status for requirement_id in open_required}
    if open_statuses == {"paused"}:
        return ("PAUSED", 2)
    if open_statuses <= {"paused", "deferred_not_claimed"}:
        return ("PAUSED", 2)
    if "failed" in open_statuses:
        return ("FAIL", 1)
    if "blocked" in open_statuses:
        return ("BLOCKED", 3)
    return ("INCOMPLETE", 3)
