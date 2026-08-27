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
    """Compare a search result against its CPU replay."""
    tolerance = tolerance or ParityTolerance()

    if _outcome_class(search_outcome) != _outcome_class(replay_outcome):
        return CertificateResult(
            certified=False,
            tier="full_trajectory",
            reason="backend_divergence",
            metrics={
                "search_passed": float(search_outcome.passed),
                "replay_passed": float(replay_outcome.passed),
            },
        )

    if replay_trajectory.hard_reject_events:
        return CertificateResult(
            certified=False,
            tier="full_trajectory",
            reason="replay_violates_safety_budget",
            metrics={"hard_reject_events": float(len(replay_trajectory.hard_reject_events))},
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
    trajectory_ref: str = "",
    gpu_evidence: dict[str, Any] | None = None,
) -> DynamicSearchOutcome:
    """Fold both certificates into the outcome that may or may not be released.

    The typed ``certificate`` is what makes a positive releasable: it names the
    backend, the capsule, the commands and the compiled model, so the replay can
    be re-run by someone who was not there. v1 folded two booleans into a dict
    that said ``confirmed: True``, which is an assertion rather than evidence
    (blockers B-05, B-11).
    """
    if not replay.certified:
        return DynamicSearchOutcome(
            trajectory_ref=trajectory_ref,
            passed=False,
            failure_stage="cpu_replay",
            failure_reason=replay.reason,
            gpu_search_evidence=gpu_evidence,
        )
    if not terminal.certified:
        return DynamicSearchOutcome(
            trajectory_ref=trajectory_ref,
            passed=False,
            failure_stage="terminal",
            failure_reason=terminal.reason,
            gpu_search_evidence=gpu_evidence,
        )
    if not certificate.is_positive:
        return DynamicSearchOutcome(
            trajectory_ref=trajectory_ref,
            passed=False,
            failure_stage="cpu_replay",
            failure_reason="evidence_hash_mismatch",
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
