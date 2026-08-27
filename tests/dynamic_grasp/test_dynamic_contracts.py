"""Contract tests for Phase 3.4 (P3.4-00, P3.4-01)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import (
    HARD_REJECT_CLASSES,
    ContactClass,
    DynamicGraspRequest,
    DynamicSearchOutcome,
    TrajectoryStage,
)

from .conftest import make_certificate, make_event, make_trajectory


def test_forbidden_and_damaging_are_the_only_hard_rejects():
    assert HARD_REJECT_CLASSES == {ContactClass.FORBIDDEN, ContactClass.DAMAGING}
    for allowed in (
        ContactClass.TARGET_INTENTIONAL,
        ContactClass.SUPPORT_ASSISTED,
        ContactClass.NEIGHBOR_INCIDENTAL,
        ContactClass.SELF_CONTACT_ALLOWED,
    ):
        assert not make_event(contact_class=allowed).is_hard_reject
    for rejected in (ContactClass.FORBIDDEN, ContactClass.DAMAGING):
        assert make_event(contact_class=rejected).is_hard_reject


def test_every_contact_class_is_a_stable_string():
    # The class travels into manifests and evidence JSON; renaming a value is a
    # dataset-breaking change, so pin the wire format.
    assert {c.value for c in ContactClass} == {
        "target_intentional",
        "support_assisted",
        "neighbor_incidental",
        "self_contact_allowed",
        "forbidden",
        "damaging",
    }
    # v2 adds support_release and retain: a positive has to show the target
    # leaving its support and still being held afterwards, and the original five
    # stages had no way to say either.
    assert {s.value for s in TrajectoryStage} == {
        "approach",
        "reposition",
        "enclose",
        "support_release",
        "lift",
        "perturb",
        "retain",
    }


@pytest.mark.parametrize(
    "field",
    ["peak_normal_force_N", "max_penetration_m", "contact_work_J", "max_wrist_torque_Nm"],
)
def test_budget_rejects_nonpositive_and_nonfinite_limits(budget, field):
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(budget, **{field: bad})


def test_trajectory_rejects_step_count_disagreement():
    traj = make_trajectory(steps=4)
    with pytest.raises(ValueError, match="joint_state has 3 steps but time has 4"):
        dataclasses.replace(traj, joint_state=np.zeros((3, 16)))


def test_trajectory_rejects_malformed_pose_shapes():
    traj = make_trajectory()
    with pytest.raises(ValueError, match=r"palm_pose must be \[T, 7\]"):
        dataclasses.replace(traj, palm_pose=np.zeros((traj.num_steps, 6)))
    with pytest.raises(ValueError, match=r"object_velocity must be \[T, O, 6\]"):
        dataclasses.replace(
            traj, object_velocity=np.zeros((traj.num_steps, traj.num_objects, 5))
        )


def test_trajectory_rejects_contact_event_outside_the_horizon():
    with pytest.raises(ValueError, match="outside"):
        make_trajectory(steps=4, contact_graph=(make_event(time_index=4),))
    # A negative index never gets as far as the trajectory: the event itself
    # refuses to exist, which is the earlier and better place to stop it.
    with pytest.raises(ValueError, match="time_index must be non-negative"):
        make_event(time_index=-1)


def test_trajectory_exposes_events_by_class():
    events = (
        make_event(0, ContactClass.TARGET_INTENTIONAL),
        make_event(1, ContactClass.SUPPORT_ASSISTED),
        make_event(2, ContactClass.DAMAGING),
    )
    traj = make_trajectory(steps=4, contact_graph=events)
    assert traj.events_of_class(ContactClass.SUPPORT_ASSISTED) == (events[1],)
    assert traj.hard_reject_events == (events[2],)


def test_request_rejects_degenerate_horizon_and_timestep():
    kwargs = {
        "scene_state_ref": "scene:table-leap-sparse#0",
        "observation_ref": "obs:table-leap-sparse/cam_top",
        "target_object_id": "obj_01",
        "robot_profile": "leap_hand",
        "strategy_id": "static_seeded_contact_rollout",
        "safety_budget_id": "micro-conservative-v1",
        "horizon": 10,
        "control_dt": 0.01,
        "seed": 42,
    }
    DynamicGraspRequest(**kwargs)
    with pytest.raises(ValueError, match="horizon"):
        DynamicGraspRequest(**{**kwargs, "horizon": 0})
    with pytest.raises(ValueError, match="control_dt"):
        DynamicGraspRequest(**{**kwargs, "control_dt": 0.0})


def test_a_passed_outcome_cannot_carry_a_failure_reason():
    with pytest.raises(ValueError, match="failure_reason"):
        DynamicSearchOutcome(
            trajectory_ref="t:0",
            passed=True,
            failure_stage="none",
            failure_reason="damaging_contact",
            cpu_replay_evidence=make_certificate(),
        )


def test_a_passed_outcome_requires_cpu_replay_evidence():
    # GPU search ranks candidates; it never admits a release positive on its own.
    with pytest.raises(ValueError, match="cpu_replay_evidence"):
        DynamicSearchOutcome(
            trajectory_ref="t:0",
            passed=True,
            failure_stage="none",
            failure_reason="none",
            gpu_search_evidence={"backend": "mjwarp_cuda"},
        )
    DynamicSearchOutcome(
        trajectory_ref="t:0",
        passed=True,
        failure_stage="none",
        failure_reason="none",
        gpu_search_evidence={"backend": "mjwarp_cuda"},
        cpu_replay_evidence=make_certificate(),
    )


def test_a_truthy_dict_is_not_a_cpu_certificate():
    # The v1 contract accepted any truthy object, so {"confirmed": True} admitted
    # a release positive with nothing behind it (blockers B-05, B-11).
    with pytest.raises(ValueError, match="typed"):
        DynamicSearchOutcome(
            trajectory_ref="t:0",
            passed=True,
            failure_stage="none",
            failure_reason="none",
            cpu_replay_evidence={"confirmed": True, "outcome_class": "pass"},
        )


def test_a_failed_outcome_needs_no_replay_evidence():
    outcome = DynamicSearchOutcome(
        trajectory_ref="t:1",
        passed=False,
        failure_stage="enclose",
        failure_reason="damaging_contact",
    )
    assert not outcome.passed
    assert outcome.gpu_search_evidence is None
