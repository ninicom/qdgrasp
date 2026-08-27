"""Refinement and CPU-replay certification tests (P3.4-11, P3.4-12)."""

from __future__ import annotations

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import ContactClass, DynamicSearchOutcome
from qdgrasp.dynamic.certify import (
    ParityTolerance,
    certify_replay,
    certify_terminal_grasp,
    release_decision,
)
from qdgrasp.dynamic.cem import ParameterSpace
from qdgrasp.dynamic.primitives import Primitive, PrimitiveKind
from qdgrasp.dynamic.refine import RefineConfig, refine_local

from .conftest import make_event, make_trajectory


def template() -> tuple[Primitive, ...]:
    return (
        Primitive(
            kind=PrimitiveKind.PUSH,
            direction=np.array([1.0, 0.0, 0.0]),
            speed=0.1,
            max_duration_s=0.2,
        ),
    )


def passing(lift=0.05, **extra) -> DynamicSearchOutcome:
    return DynamicSearchOutcome(
        trajectory_ref="t", passed=True, failure_stage="none", failure_reason="none",
        objective_terms={"lift_m": lift, **extra},
        peak_safety_metrics={"min_budget_margin": 0.5, "peak_normal_force_N": 1.0},
        cumulative_safety_metrics={"total_slip_m": 0.0},
        cpu_replay_evidence={"confirmed": True},
    )


def failing(reason="insufficient_lift") -> DynamicSearchOutcome:
    return DynamicSearchOutcome(
        trajectory_ref="t", passed=False, failure_stage="lift", failure_reason=reason
    )


# -- refinement -----------------------------------------------------------


def test_refinement_refuses_to_start_from_a_failure():
    # Otherwise a near-miss could be nudged into range and relabelled a success.
    with pytest.raises(ValueError, match="passing trajectory"):
        refine_local(
            template=template(), seed_sample=np.array([0.1, 0.5, 0.2]),
            seed_outcome=failing(), seed_trajectory=make_trajectory(),
            rollout=lambda p: (make_trajectory(), passing()),
        )


def test_refinement_never_accepts_a_failing_neighbour():
    def rollout(primitives):
        # Every neighbour fails; the seed must survive untouched.
        return make_trajectory(), failing()

    result = refine_local(
        template=template(), seed_sample=np.array([0.1, 0.5, 0.2]),
        seed_outcome=passing(), seed_trajectory=make_trajectory(),
        rollout=rollout, config=RefineConfig(iterations=2),
    )
    assert not result.improved
    assert result.outcome.passed
    assert result.rejected_regressions == result.evaluated > 0


def test_refinement_takes_a_genuine_improvement():
    space = ParameterSpace()

    def rollout(primitives):
        # Reward higher speed, so a step in that direction should be kept.
        return make_trajectory(), passing(lift=float(primitives[0].speed))

    result = refine_local(
        template=template(), seed_sample=space.lower().copy(),
        seed_outcome=passing(lift=0.0), seed_trajectory=make_trajectory(),
        rollout=rollout, space=space, config=RefineConfig(iterations=2),
    )
    assert result.improved
    assert result.sample[0] > space.lower()[0]


def test_refine_config_rejects_a_degenerate_neighbourhood():
    for kwargs, match in (
        ({"iterations": 0}, "iterations"),
        ({"initial_step_fraction": 0.9}, "initial_step_fraction"),
        ({"shrink": 1.0}, "shrink"),
    ):
        with pytest.raises(ValueError, match=match):
            RefineConfig(**kwargs)


# -- replay parity --------------------------------------------------------


def test_a_changed_outcome_class_is_backend_divergence():
    result = certify_replay(
        search_outcome=passing(), replay_outcome=failing(),
        search_trajectory=make_trajectory(), replay_trajectory=make_trajectory(),
    )
    assert not result.certified
    assert result.reason == "backend_divergence"


def test_matching_outcome_classes_certify():
    result = certify_replay(
        search_outcome=passing(), replay_outcome=passing(),
        search_trajectory=make_trajectory(), replay_trajectory=make_trajectory(),
    )
    assert result.certified
    assert result.reason == "none"
    assert "max_object_position_delta_m" in result.metrics


def test_a_replay_that_breaks_the_budget_is_refused_even_when_classes_match():
    unsafe = make_trajectory(
        contact_graph=(make_event(contact_class=ContactClass.DAMAGING),)
    )
    result = certify_replay(
        search_outcome=passing(), replay_outcome=passing(),
        search_trajectory=make_trajectory(), replay_trajectory=unsafe,
    )
    assert not result.certified
    assert result.reason == "replay_violates_safety_budget"


def test_two_failures_for_the_same_reason_are_not_divergence():
    result = certify_replay(
        search_outcome=failing("insufficient_lift"),
        replay_outcome=failing("insufficient_lift"),
        search_trajectory=make_trajectory(), replay_trajectory=make_trajectory(),
    )
    assert result.certified


def test_tolerances_are_pinned_before_comparison():
    tolerance = ParityTolerance()
    assert tolerance.require_same_outcome_class
    assert tolerance.no_contact_state_atol > 0.0


# -- terminal certificate -------------------------------------------------


def lifted_trajectory(links=2, lift=0.05, still_supported=False):
    events = [
        make_event(time_index=0, contact_class=ContactClass.TARGET_INTENTIONAL,
                   body_a=f"finger_{i}", body_b="target")
        for i in range(links)
    ]
    if still_supported:
        events.append(
            make_event(time_index=3, contact_class=ContactClass.SUPPORT_ASSISTED)
        )
    traj = make_trajectory(steps=4, contact_graph=tuple(events))
    pose = traj.object_pose.copy()
    pose[-1, 0, 2] = pose[0, 0, 2] + lift
    return type(traj)(
        time=traj.time, palm_pose=traj.palm_pose, joint_state=traj.joint_state,
        actuator_command=traj.actuator_command, object_pose=pose,
        object_velocity=traj.object_velocity, stage=traj.stage,
        contact_graph=traj.contact_graph,
    )


def test_a_lifted_enclosed_grasp_certifies():
    result = certify_terminal_grasp(lifted_trajectory())
    assert result.certified
    assert result.metrics["lift_m"] == pytest.approx(0.05)


def test_one_finger_is_not_enclosure():
    result = certify_terminal_grasp(lifted_trajectory(links=1))
    assert not result.certified
    assert result.reason == "insufficient_enclosure"


def test_a_grasp_still_touching_its_support_has_not_lifted_free():
    result = certify_terminal_grasp(lifted_trajectory(still_supported=True))
    assert not result.certified
    assert result.reason == "support_not_released"


def test_too_little_lift_is_refused():
    result = certify_terminal_grasp(lifted_trajectory(lift=0.001))
    assert not result.certified
    assert result.reason == "insufficient_lift"


def test_a_hard_reject_contact_blocks_the_terminal_certificate():
    traj = make_trajectory(contact_graph=(make_event(contact_class=ContactClass.FORBIDDEN),))
    result = certify_terminal_grasp(traj)
    assert not result.certified
    assert result.reason == "hard_reject_contact"


# -- release decision -----------------------------------------------------


def test_release_requires_both_certificates():
    good_replay = certify_replay(
        search_outcome=passing(), replay_outcome=passing(),
        search_trajectory=make_trajectory(), replay_trajectory=make_trajectory(),
    )
    good_terminal = certify_terminal_grasp(lifted_trajectory())
    released = release_decision(replay=good_replay, terminal=good_terminal)
    assert released.passed
    assert released.cpu_replay_evidence["confirmed"] is True

    bad_terminal = certify_terminal_grasp(lifted_trajectory(links=1))
    refused = release_decision(replay=good_replay, terminal=bad_terminal)
    assert not refused.passed
    assert refused.failure_reason == "insufficient_enclosure"


def test_gpu_evidence_alone_never_releases_a_positive():
    diverged = certify_replay(
        search_outcome=passing(), replay_outcome=failing(),
        search_trajectory=make_trajectory(), replay_trajectory=make_trajectory(),
    )
    refused = release_decision(
        replay=diverged,
        terminal=certify_terminal_grasp(lifted_trajectory()),
        gpu_evidence={"backend": "mjwarp_cuda", "score": 99.0},
    )
    assert not refused.passed
    assert refused.failure_reason == "backend_divergence"
    assert refused.gpu_search_evidence["backend"] == "mjwarp_cuda"
