"""CPU replay and terminal grasp certification (P3.4-12).

A GPU search ranks candidates. Only a CPU replay admits one. This module runs
the finalist again on the oracle from the same initial state and command
sequence, and compares the outcome class -- not the trajectory, which contact
dynamics make divergent by nature.

Plan section 9 sets the standard: if the CPU replay changes the outcome or
breaks the budget, the sample is recorded with reason ``backend_divergence`` and
is not released as a positive.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    ContactClass,
    ContactPairKind,
    CpuReplayCertificate,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
)
from qdgrasp.dynamic.capsule import ReplayCapsule, certificate_matches


@dataclasses.dataclass(frozen=True)
class ParityTolerance:
    """Three-tier parity from plan section 9, pinned before any comparison."""

    #: Tier 1, no contact, short horizon: state must track closely.
    no_contact_state_atol: float = 1e-4
    #: Tier 2, one pinned contact: impulse and object delta within tolerance.
    single_contact_impulse_rtol: float = 0.2
    single_contact_object_delta_m: float = 2e-3
    #: Tier 3, full trajectory: only the outcome class has to agree.
    require_same_outcome_class: bool = True


@dataclasses.dataclass(frozen=True)
class CertificateResult:
    certified: bool
    tier: str
    reason: str
    metrics: dict[str, float]
    #: True when search and replay agreed, whatever they agreed on. Two matching
    #: failures are useful parity evidence -- they are just not a certification,
    #: which is what v1 folded them into (blocker B-05).
    parity_confirmed: bool = False
    outcome_class: str = ""


def _outcome_class(outcome: DynamicSearchOutcome) -> str:
    return "pass" if outcome.passed else f"fail:{outcome.failure_reason}"


def certify_replay(
    *,
    search_outcome: DynamicSearchOutcome,
    replay_outcome: DynamicSearchOutcome,
    search_trajectory: DynamicGraspTrajectory,
    replay_trajectory: DynamicGraspTrajectory,
    tolerance: ParityTolerance | None = None,
) -> CertificateResult:
    """Compare a search result against its CPU replay.

    Agreement is not admission. Two backends that both say "insufficient lift"
    have confirmed each other's *failure*, and folding that into ``certified``
    let a matching pair of failures reach the release path (blocker B-05).
    """
    tolerance = tolerance or ParityTolerance()
    search_class = _outcome_class(search_outcome)
    replay_class = _outcome_class(replay_outcome)

    if search_class != replay_class:
        return CertificateResult(
            certified=False,
            tier="full_trajectory",
            reason="backend_divergence",
            metrics={
                "search_passed": float(search_outcome.passed),
                "replay_passed": float(replay_outcome.passed),
            },
            parity_confirmed=False,
            outcome_class=replay_class,
        )

    if not (search_outcome.passed and replay_outcome.passed):
        # Kept as negative parity evidence, under the failure both agreed on.
        return CertificateResult(
            certified=False,
            tier="full_trajectory",
            reason=replay_outcome.failure_reason,
            metrics={
                "search_passed": float(search_outcome.passed),
                "replay_passed": float(replay_outcome.passed),
            },
            parity_confirmed=True,
            outcome_class=replay_class,
        )

    if replay_trajectory.hard_reject_events:
        return CertificateResult(
            certified=False,
            tier="full_trajectory",
            reason="replay_violates_safety_budget",
            metrics={"hard_reject_events": float(len(replay_trajectory.hard_reject_events))},
            parity_confirmed=True,
            outcome_class=replay_class,
        )

    steps = min(search_trajectory.num_steps, replay_trajectory.num_steps)
    objects = min(search_trajectory.num_objects, replay_trajectory.num_objects)
    delta = 0.0
    if steps and objects:
        delta = float(
            np.max(
                np.linalg.norm(
                    search_trajectory.object_pose[:steps, :objects, :3]
                    - replay_trajectory.object_pose[:steps, :objects, :3],
                    axis=-1,
                )
            )
        )

    return CertificateResult(
        certified=True,
        tier="full_trajectory",
        reason="none",
        metrics={"max_object_position_delta_m": delta, "compared_steps": float(steps)},
        parity_confirmed=True,
        outcome_class=replay_class,
    )


def certify_terminal_grasp(
    trajectory: DynamicGraspTrajectory,
    *,
    min_enclosure_links: int = 2,
    min_lift_m: float = 0.03,
    require_stage_progression: bool = False,
    target_slot: int = 0,
) -> CertificateResult:
    """Check the terminal conditions of plan section 4.3 on a finished rollout.

    Two things changed in v2. Support release is decided from *target*-support
    contacts alone: a robot link resting on the table is support-assisted too,
    and counting it made every lifted object read as still supported (blocker
    B-12). And the lift is measured on the declared target slot, so lifting the
    wrong object is a typed negative rather than a pass.
    """
    if trajectory.num_steps == 0:
        return CertificateResult(False, "terminal", "empty_trajectory", {})
    if trajectory.hard_reject_events:
        return CertificateResult(
            False, "terminal", "hard_reject_contact",
            {"events": float(len(trajectory.hard_reject_events))},
        )

    target_events = [
        e
        for e in trajectory.contact_graph
        if e.contact_class is ContactClass.TARGET_INTENTIONAL
    ]
    links = {e.body_a for e in target_events} | {e.body_b for e in target_events}
    enclosure = max(0, len(links) - 1)

    lift = float(
        trajectory.object_pose[-1, target_slot, 2] - trajectory.object_pose[0, target_slot, 2]
    ) if trajectory.num_objects > target_slot else 0.0

    final_events = [
        e for e in trajectory.contact_graph if e.time_index == trajectory.num_steps - 1
    ]
    still_supported = any(e.supports_target for e in final_events)
    # A pair whose roles were never resolved cannot answer the support question
    # either way, so it is not silently read as "released".
    unresolved = any(
        e.pair_kind is ContactPairKind.UNKNOWN for e in final_events
    )

    # The largest rise of any object that is not the declared target, so a
    # sequence that lifted a neighbour is not scored as a success.
    other_lift = 0.0
    for slot in range(trajectory.num_objects):
        if slot == target_slot:
            continue
        other_lift = max(
            other_lift,
            float(trajectory.object_pose[-1, slot, 2] - trajectory.object_pose[0, slot, 2]),
        )

    metrics = {
        "enclosure_links": float(enclosure),
        "lift_m": lift,
        "still_supported": float(still_supported),
        "max_non_target_lift_m": other_lift,
        "reached_required_stages": float(trajectory.has_required_terminal_stages),
    }
    if unresolved:
        return CertificateResult(False, "terminal", "hard_reject_contact", metrics)
    if enclosure < min_enclosure_links:
        return CertificateResult(False, "terminal", "insufficient_enclosure", metrics)
    if still_supported:
        return CertificateResult(False, "terminal", "support_not_released", metrics)
    if lift < min_lift_m:
        if other_lift >= min_lift_m:
            return CertificateResult(False, "terminal", "wrong_object_lift", metrics)
        return CertificateResult(False, "terminal", "insufficient_lift", metrics)
    if other_lift >= min_lift_m:
        return CertificateResult(False, "terminal", "wrong_object_lift", metrics)
    if require_stage_progression and not trajectory.terminal_stages_in_canonical_order:
        # Enclosing after the lift, or lifting before the support was released,
        # is not an acquisition -- it is a mislabelled one (C03.7).
        return CertificateResult(False, "terminal", "no_closure", metrics)
    return CertificateResult(True, "terminal", "none", metrics)


def release_decision(
    *,
    replay: CertificateResult,
    terminal: CertificateResult,
    certificate: CpuReplayCertificate,
    search_outcome: DynamicSearchOutcome | None = None,
    replay_outcome: DynamicSearchOutcome | None = None,
    capsule: ReplayCapsule | None = None,
    trajectory_ref: str = "",
    gpu_evidence: dict[str, Any] | None = None,
) -> DynamicSearchOutcome:
    """Fold the evidence into the outcome that may or may not be released.

    The typed ``certificate`` is what makes a positive releasable: it names the
    backend, the capsule, the commands and the compiled model, so the replay can
    be re-run by someone who was not there. v1 folded two booleans into a dict
    that said ``confirmed: True``, which is an assertion rather than evidence
    (blockers B-05, B-11).

    ``search_outcome`` and ``replay_outcome`` are the originals, not summaries of
    them. When they are supplied they are re-checked here, so that a caller
    cannot hand over two certificates whose provenance nobody can trace back
    (G04.2). When ``capsule`` is supplied the certificate has to be bound to it.
    """
    refused = _refusal(
        replay=replay,
        terminal=terminal,
        certificate=certificate,
        search_outcome=search_outcome,
        replay_outcome=replay_outcome,
        capsule=capsule,
    )
    if refused is not None:
        stage, reason = refused
        return DynamicSearchOutcome(
            trajectory_ref=trajectory_ref,
            passed=False,
            failure_stage=stage,
            failure_reason=reason,
            gpu_search_evidence=gpu_evidence,
        )
    return DynamicSearchOutcome(
        trajectory_ref=trajectory_ref,
        passed=True,
        failure_stage="none",
        failure_reason="none",
        objective_terms=dict(terminal.metrics),
        gpu_search_evidence=gpu_evidence,
        cpu_replay_evidence=certificate,
    )


def _refusal(
    *,
    replay: CertificateResult,
    terminal: CertificateResult,
    certificate: CpuReplayCertificate,
    search_outcome: DynamicSearchOutcome | None,
    replay_outcome: DynamicSearchOutcome | None,
    capsule: ReplayCapsule | None,
) -> tuple[str, str] | None:
    """Why this candidate may not be released, or ``None`` if it may."""
    if search_outcome is not None and not search_outcome.passed:
        return ("cpu_replay", search_outcome.failure_reason)
    if replay_outcome is not None and not replay_outcome.passed:
        return ("cpu_replay", replay_outcome.failure_reason)
    if not replay.certified:
        return ("cpu_replay", replay.reason)
    if not terminal.certified:
        return ("terminal", terminal.reason)
    if not certificate.is_positive:
        return ("cpu_replay", "evidence_hash_mismatch")
    if capsule is not None and not certificate_matches(capsule, certificate):
        return ("cpu_replay", "evidence_hash_mismatch")
    if "cuda" in certificate.backend_id:
        return ("cpu_replay", "gpu_only_evidence")
    return None


@dataclasses.dataclass
class ReleaseLedger:
    """Separate denominators for every stage a candidate passes through (G04.5).

    A single "yield" number cannot say whether a sample was never replayed or
    was replayed and refused, and folding those together makes a release rate
    look better than it is. Each stage is counted on its own, and a candidate
    that was never replayed is not in the released denominator.
    """

    searched: int = 0
    gpu_survived: int = 0
    replayed: int = 0
    cpu_confirmed: int = 0
    released: int = 0
    parity_confirmed_failures: int = 0
    dispositions: dict[str, int] = dataclasses.field(default_factory=dict)

    def record_candidate(
        self,
        *,
        search_outcome: DynamicSearchOutcome,
        gpu_survived: bool = False,
        replay: CertificateResult | None = None,
        released: DynamicSearchOutcome | None = None,
    ) -> None:
        """Record one candidate's whole journey, exactly once."""
        self.searched += 1
        if gpu_survived:
            self.gpu_survived += 1

        disposition: str
        if replay is None:
            disposition = "not_replayed"
        else:
            self.replayed += 1
            if replay.parity_confirmed and not replay.certified:
                self.parity_confirmed_failures += 1
            if replay.certified:
                self.cpu_confirmed += 1
            disposition = replay.reason if not replay.certified else "cpu_confirmed"

        if released is not None and released.passed:
            self.released += 1
            disposition = "released"
        elif released is not None and replay is not None and replay.certified:
            disposition = released.failure_reason

        if not search_outcome.passed and replay is None:
            disposition = search_outcome.failure_reason

        self.dispositions[disposition] = self.dispositions.get(disposition, 0) + 1

    def reconcile(self) -> bool:
        """Whether every candidate is accounted for exactly once."""
        return (
            sum(self.dispositions.values()) == self.searched
            and self.released <= self.cpu_confirmed <= self.replayed <= self.searched
            and self.gpu_survived <= self.searched
        )

    @property
    def release_rate(self) -> float:
        """Released as a fraction of what was actually replayed, not of what was
        sampled: a candidate nobody replayed cannot count against a replay rate."""
        return self.released / self.replayed if self.replayed else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "searched": self.searched,
            "gpu_survived": self.gpu_survived,
            "replayed": self.replayed,
            "cpu_confirmed": self.cpu_confirmed,
            "released": self.released,
            "parity_confirmed_failures": self.parity_confirmed_failures,
            "dispositions": dict(sorted(self.dispositions.items())),
            "release_rate_of_replayed": self.release_rate,
            "reconciled": self.reconcile(),
        }
