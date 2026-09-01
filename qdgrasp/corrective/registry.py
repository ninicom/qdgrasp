"""The corrective findings of the 2026-09-01 cross-component audit, as data.

``PLAN.md`` §9.2 lists twelve failure chains that only appear when several
subsystems are put together, and §9.3 asks for a characterization test per
chain.  A prose table cannot be asserted against, so the same table lives here
in a form the test suite reads: every characterization test names the finding it
reproduces, and this module decides whether that test is still expected to fail.

The point of the indirection is the flip.  While a finding is ``open`` its
characterization test is an expected failure; when a remediation gate closes it,
the entry moves to ``closed`` and the very same test becomes a regression test.
Nothing about the test changes, so the test cannot quietly be weakened into
agreement with whatever the code happens to do.

This registry describes work, not results.  It is not evidence that anything has
been fixed, and closing an entry here without a passing test is a lie the suite
will catch on the next run.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

CORRECTIVE_REGISTRY_SCHEMA = "qdgrasp/corrective-registry/v1"

#: The revision that introduced the corrective track.
REVISION_RECORD = "REV-20260901-001"

#: The audit session that reproduced every chain below.
AUDIT_SESSION = "SESSION-20260901-001"

#: Plan revision this registry mirrors.  A later plan revision that changes §9.2
#: has to change this file too, and the registry test says so.
PLAN_REVISION = "PLAN-V2@4.6.0"

Status = Literal["open", "closed"]


class CorrectiveError(RuntimeError):
    """Something asked the corrective track a question it refuses to answer."""


@dataclasses.dataclass(frozen=True)
class Finding:
    """One reproduced failure chain and the state that would close it."""

    id: str
    severity: str
    gate: str
    pull_request: str
    chain: str
    target: str
    status: Status = "open"
    #: Why an open finding has no failing test left: the code landed, and the
    #: gate that closes it waits on something else -- usually the dataset
    #: regeneration in R8.  "Nothing fails any more" is not the same claim as
    #: "the gate is closed", and this field keeps them apart.
    blocked_on: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def describe(self) -> str:
        return f"{self.id} ({self.severity}, closes at {self.gate}/{self.pull_request}): {self.chain}"


#: §9.2, in plan order.  ``gate`` is the remediation gate that closes the
#: finding; ``pull_request`` is the PR in §9.11 that carries the fix.  ``R1``
#: itself carries no semantic fix, so no entry closes at ``R1``.
FINDINGS: tuple[Finding, ...] = (
    Finding(
        id="COR-00",
        severity="S0",
        gate="G1",
        pull_request="R2",
        chain=(
            "manifest-controlled paths escape the dataset root and shard/checkpoint loads reach "
            "torch.load(weights_only=False), so an untrusted artifact has an execution path"
        ),
        target=(
            "one safe artifact I/O path: weights_only=True, constrained relative paths, and a malicious "
            "reducer that never runs"
        ),
        status="closed",
    ),
    Finding(
        id="COR-01",
        severity="S1",
        gate="G1",
        pull_request="R2",
        chain=(
            "the canonical audit, the Phase 5 adapter and the public loader implement three different "
            "sample/manifest contracts, and the facade calls none of the gates"
        ),
        target="DatasetArtifact.open_verified() as the single entry point for audit, gate, facade and Runner",
        status="closed",
    ),
    Finding(
        id="COR-02",
        severity="S1",
        gate="G2",
        pull_request="R3",
        chain=(
            "the splitter stratifies inside a shape while claiming a family hold-out, and the protocol "
            "filters after the fact instead of reaching the trainer"
        ),
        target="a materialised ProtocolDatasetView keyed on (split, robot, object_id) that fails on leakage",
        blocked_on=(
            "positive yield: R8 regenerated the corpus, so the view reaches the public path and the "
            "canonical audit passes, but the G2 gate also asks that each declared train hand clear the "
            "positive floor and the recipe yields one positive per hand. That is a pipeline question, "
            "not a plumbing one, and no relabelling closes it"
        ),
    ),
    Finding(
        id="COR-03",
        severity="S1",
        gate="G3",
        pull_request="R4",
        chain=(
            "the mixed-robot guard lives in a side helper while the Runner collates with default_collate, "
            "so an Allegro sample reaches a LEAP-bound model because both have 16 joints"
        ),
        target="one canonical collator that carries robot identity, and a model that asserts or groups on it",
        status="closed",
    ),
    Finding(
        id="COR-04",
        severity="S1",
        gate="G2",
        pull_request="R3",
        chain=(
            "missing kinematics are serialised as zero/identity and then regressed onto as if they were "
            "measurements, because no sample carries target-validity flags"
        ),
        target="explicit validity masks, so a placeholder produces no geometric gradient",
        status="closed",
    ),
    Finding(
        id="COR-05",
        severity="S1",
        gate="G3",
        pull_request="R4",
        chain=(
            "encode_target writes physical joint angles while decode applies a tanh squash onto the joint "
            "limits, so the flow target and the decoded pose are two different parameterizations"
        ),
        target="an inverse parameterization whose round-trip is below 1e-5 rad on both active hands",
        status="closed",
    ),
    Finding(
        id="COR-06",
        severity="S1",
        gate="G3",
        pull_request="R4",
        chain=(
            "the quality head reads the observation only, so every candidate generated for one object "
            "scores identically and a ranking test passes on ties"
        ),
        target="a candidate-aware quality head, ranked against negatives that share the observation",
        status="closed",
    ),
    Finding(
        id="COR-07",
        severity="S1/S2",
        gate="G4",
        pull_request="R5",
        chain=(
            "validation draws from the training RNG and averages per batch, so the metric depends on the "
            "batch size and the loss curve depends on val_interval; EMA is updated but never used"
        ),
        target="separated RNG streams, sample-weighted deterministic validation and an explicit EMA contract",
        status="closed",
    ),
    Finding(
        id="COR-08",
        severity="S1",
        gate="G4",
        pull_request="R5",
        chain=(
            "resume carries no model/robot/data/protocol identity, accepts LEAP state into an Allegro run "
            "on matching shapes, and records no real AMP scaler"
        ),
        target="resume/v2: full identity validated before any state mutation, exact continuation only",
        status="closed",
    ),
    Finding(
        id="COR-09",
        severity="S1",
        gate="G5",
        pull_request="R6",
        chain=(
            "the bundle loader gates on tensor shape rather than semantics and from_bundle hard-codes "
            "robot/v1, while the exact-robot gate forbids the held-out inference the protocol requires"
        ),
        target="a versioned parser, exact semantic bundle checks and an explicit cross-embodiment binding",
        status="closed",
    ),
    Finding(
        id="COR-10",
        severity="S1",
        gate="G5",
        pull_request="R6",
        chain=(
            "the flow export traces a dataclass return, a stochastic draw and a Python-level token "
            "topology, and only the dummy model is exercised"
        ),
        target="a tensor-only deterministic export adapter with explicit noise and dynamic-shape parity",
        status="closed",
    ),
    Finding(
        id="COR-11",
        severity="S1/S2",
        gate="G6",
        pull_request="R7",
        chain=(
            "the MVP evaluation worker loads a checkpoint without its fingerprint and the report then "
            "stamps the current environment fingerprint onto it"
        ),
        target="the guard runs before the first episode, and the report records stored, effective and verdict",
        status="closed",
    ),
    Finding(
        id="COR-12",
        severity="S2",
        gate="G7",
        pull_request="R9",
        chain=(
            "zero point padding carries no input mask, several config keys are silent no-ops, and the "
            "packaged legacy namespace still holds exec/eval/unsafe loads"
        ),
        target="masked padding, config keys that either take effect or are refused, and a quarantined legacy surface",
        status="closed",
    ),
)

FINDINGS_BY_ID: dict[str, Finding] = {item.id: item for item in FINDINGS}


@dataclasses.dataclass(frozen=True)
class SchemaBump:
    """An artifact schema the corrective track expects to move, and why.

    G0 asks for the bump to be *declared* before the semantics change, so a
    reader of today's code can see which artifacts are about to stop being
    interchangeable.  The registry test pins the current value: bumping the
    constant without updating this table, or updating this table without a
    closed finding, both fail.
    """

    artifact: str
    module: str
    constant: str
    current: str
    planned: str
    finding: str
    reason: str


PLANNED_SCHEMA_BUMPS: tuple[SchemaBump, ...] = (
    SchemaBump(
        artifact="dataset manifest",
        module="qdgrasp.dataset.manifest",
        constant="DATASET_MANIFEST_SCHEMA",
        current="qdgrasp/dataset-manifest/v2",
        planned="qdgrasp/dataset-manifest/v3",
        finding="COR-04",
        reason="target-validity flags become required sample fields, so old shards stop being readable",
    ),
    SchemaBump(
        artifact="public bundle",
        module="qdgrasp.engine.checkpoint",
        constant="BUNDLE_SCHEMA",
        current="qdgrasp/bundle/v1",
        planned="qdgrasp/bundle/v2",
        finding="COR-09",
        reason="joint parameterization, weights source and the training/runtime robot split",
    ),
    SchemaBump(
        artifact="resume state",
        module="qdgrasp.engine.checkpoint",
        constant="RESUME_SCHEMA",
        current="qdgrasp/resume/v1",
        planned="qdgrasp/resume/v2",
        finding="COR-08",
        reason="model/robot/data/protocol identity, effective run config and the real AMP scaler",
    ),
    SchemaBump(
        artifact="MVP policy checkpoint",
        module="qdgrasp.mvp.policy",
        constant="POLICY_SCHEMA",
        current="qdgrasp/mvp-policy/v0",
        planned="qdgrasp/mvp-policy/v1",
        finding="COR-11",
        reason="typed payload, parent lineage and a settled bounded-action contract",
    ),
    SchemaBump(
        artifact="MVP evaluation report",
        module="qdgrasp.mvp.evaluate",
        constant="EVAL_REPORT_SCHEMA",
        current="qdgrasp/mvp-eval-report/v0",
        planned="qdgrasp/mvp-eval-report/v1",
        finding="COR-11",
        reason="stored fingerprint, effective fingerprint and the match verdict",
    ),
)


def get(finding_id: str) -> Finding:
    """The finding with this id, or a loud error naming the registry."""

    try:
        return FINDINGS_BY_ID[finding_id]
    except KeyError:
        raise CorrectiveError(
            f"{finding_id!r} is not a registered corrective finding; PLAN.md §9.2 lists {sorted(FINDINGS_BY_ID)}"
        ) from None


def open_findings() -> tuple[Finding, ...]:
    """Every finding still expected to reproduce."""

    return tuple(item for item in FINDINGS if item.is_open)


def findings_for_gate(gate: str) -> tuple[Finding, ...]:
    return tuple(item for item in FINDINGS if item.gate == gate)


def release_is_blocked() -> bool:
    """``PLAN.md`` §9.10: release reopens only when every finding is closed."""

    return bool(open_findings())


def summary() -> str:
    """One line per finding, for an error message a reader can act on."""

    return "\n".join(f"  {'open  ' if item.is_open else 'closed'} {item.describe()}" for item in FINDINGS)


__all__ = [
    "AUDIT_SESSION",
    "CORRECTIVE_REGISTRY_SCHEMA",
    "FINDINGS",
    "FINDINGS_BY_ID",
    "PLANNED_SCHEMA_BUMPS",
    "PLAN_REVISION",
    "REVISION_RECORD",
    "CorrectiveError",
    "Finding",
    "SchemaBump",
    "findings_for_gate",
    "get",
    "open_findings",
    "release_is_blocked",
    "summary",
]
