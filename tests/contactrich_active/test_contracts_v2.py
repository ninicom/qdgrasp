"""S2 — the typed contracts fail closed on an invalid corpus (C01, B-11).

Every case here is a payload that v1 accepted and should not have. The point is
not that these inputs are exotic: they are what a producer bug actually looks
like -- a NaN that survived a solver, a quaternion nobody normalised, a time
axis reconstructed from the wrong clock, a certificate that says ``confirmed``
and nothing else.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import (
    CONTACTRICH_MANIFEST_SCHEMA_V2,
    DYNAMIC_TRAJECTORY_SCHEMA_V1,
    DYNAMIC_TRAJECTORY_SCHEMA_V2,
    FAILURE_REASONS,
    LEGACY_TRAJECTORY_SCHEMAS,
    QUATERNION_ORDER,
    RELEASE_TRAJECTORY_SCHEMAS,
    REPLAY_CAPSULE_SCHEMA_V1,
    ContactClass,
    ContactEvent,
    ContractViolation,
    CpuReplayCertificate,
    DynamicGraspRequest,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    FailureStage,
    TrajectoryStage,
    TrajectoryTimebase,
    canonical_hash,
    certificate_from_dict,
    is_known_failure_reason,
    sequence_hash,
)

SAMPLE_PERIOD_S = 0.01


def timebase(**overrides) -> TrajectoryTimebase:
    defaults = {"simulator_dt": SAMPLE_PERIOD_S, "sample_every": 1}
    defaults.update(overrides)
    return TrajectoryTimebase(**defaults)


def event(**overrides) -> ContactEvent:
    defaults = {
        "time_index": 0,
        "contact_class": ContactClass.TARGET_INTENTIONAL,
        "geom_a": "leap_tip_0",
        "geom_b": "target_geom",
        "body_a": "leap_distal_0",
        "body_b": "target",
        "point": np.zeros(3),
        "frame": np.eye(3),
        "normal_force_N": 1.0,
        "tangential_force_N": 0.2,
        "normal_impulse_Ns": 0.05,
        "tangential_impulse_Ns": 0.01,
        "penetration_m": 2e-4,
        "relative_velocity_mps": 0.01,
        "slip_m": 5e-4,
        "work_J": 1e-3,
        "budget_margin": 0.5,
        "simulator_step": 7,
    }
    defaults.update(overrides)
    return ContactEvent(**defaults)


def trajectory(steps: int = 4, **overrides) -> DynamicGraspTrajectory:
    palm = np.zeros((steps, 7))
    palm[:, 3] = 1.0
    pose = np.zeros((steps, 1, 7))
    pose[:, :, 3] = 1.0
    defaults = {
        "time": np.arange(steps, dtype=float) * SAMPLE_PERIOD_S,
        "palm_pose": palm,
        "joint_state": np.zeros((steps, 16)),
        "actuator_command": np.zeros((steps, 16)),
        "object_pose": pose,
        "object_velocity": np.zeros((steps, 1, 6)),
        "stage": tuple([TrajectoryStage.APPROACH] * steps),
        "timebase": timebase(),
        "robot_profile": "leap_hand",
        "palm_body": "palm_lower",
    }
    defaults.update(overrides)
    return DynamicGraspTrajectory(**defaults)


def certificate(**overrides) -> CpuReplayCertificate:
    defaults = {
        "backend_id": "mujoco_cpu",
        "capsule_sha256": "a" * 64,
        "command_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "timestep_s": 0.002,
        "terminal_certified": True,
        "safety_certified": True,
        "outcome_class": "pass",
    }
    defaults.update(overrides)
    return CpuReplayCertificate(**defaults)


# -- schema identity ------------------------------------------------------


def test_only_v2_backs_a_release() -> None:
    assert RELEASE_TRAJECTORY_SCHEMAS == {DYNAMIC_TRAJECTORY_SCHEMA_V2}
    assert LEGACY_TRAJECTORY_SCHEMAS == {DYNAMIC_TRAJECTORY_SCHEMA_V1}
    assert trajectory().is_release_schema is True
    assert trajectory(schema=DYNAMIC_TRAJECTORY_SCHEMA_V1).is_release_schema is False


def test_reading_a_v1_payload_does_not_promote_it() -> None:
    legacy = trajectory(schema=DYNAMIC_TRAJECTORY_SCHEMA_V1)
    assert legacy.schema == DYNAMIC_TRAJECTORY_SCHEMA_V1
    assert not legacy.is_release_schema


def test_unknown_schema_is_refused() -> None:
    with pytest.raises(ContractViolation, match="unknown trajectory schema"):
        trajectory(schema="qdgrasp/dynamic-trajectory/v3")


def test_every_payload_has_a_versioned_schema() -> None:
    for schema in (
        DYNAMIC_TRAJECTORY_SCHEMA_V2,
        REPLAY_CAPSULE_SCHEMA_V1,
        CONTACTRICH_MANIFEST_SCHEMA_V2,
    ):
        assert schema.startswith("qdgrasp/")
        assert schema.rsplit("/", 1)[1].startswith("v")


# -- request --------------------------------------------------------------


def request(**overrides) -> DynamicGraspRequest:
    defaults = {
        "scene_state_ref": "scene:table-leap-sparse#0",
        "observation_ref": "obs:table-leap-sparse/cam_top",
        "target_object_id": "obj_01",
        "robot_profile": "leap_hand",
        "strategy_id": "primitive_sequence",
        "safety_budget_id": "contactrich-tiny-leap_hand-v1",
        "horizon": 40,
        "control_dt": 0.002,
        "seed": 7,
    }
    defaults.update(overrides)
    return DynamicGraspRequest(**defaults)


@pytest.mark.parametrize(
    "field",
    [
        "scene_state_ref",
        "observation_ref",
        "target_object_id",
        "robot_profile",
        "strategy_id",
        "safety_budget_id",
    ],
)
def test_request_rejects_an_empty_reference(field: str) -> None:
    with pytest.raises(ContractViolation, match=field):
        request(**{field: "   "})


def test_request_rejects_an_unknown_backend() -> None:
    with pytest.raises(ContractViolation, match="not one of"):
        request(backend_request="tpu")


def test_request_rejects_a_non_integer_seed() -> None:
    with pytest.raises(ContractViolation, match="seed must be an int"):
        request(seed=1.5)


def test_request_hash_changes_with_the_seed() -> None:
    assert request(seed=1).request_hash != request(seed=2).request_hash


# -- contact events -------------------------------------------------------


def test_event_rejects_a_non_orthonormal_frame() -> None:
    skewed = np.eye(3)
    skewed[0, 1] = 0.5
    with pytest.raises(ContractViolation, match="orthonormal"):
        event(frame=skewed)


def test_event_rejects_a_malformed_frame_or_point() -> None:
    with pytest.raises(ContractViolation, match=r"frame must be \[3, 3\]"):
        event(frame=np.eye(4))
    with pytest.raises(ContractViolation, match=r"point must be \[3\]"):
        event(point=np.zeros(2))


@pytest.mark.parametrize(
    "field",
    [
        "normal_force_N",
        "tangential_force_N",
        "normal_impulse_Ns",
        "tangential_impulse_Ns",
        "penetration_m",
        "work_J",
        "duration_s",
        "slip_m",
    ],
)
def test_event_rejects_negative_physical_quantities(field: str) -> None:
    with pytest.raises(ContractViolation, match=field):
        event(**{field: -1.0})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_event_rejects_non_finite_quantities(bad: float) -> None:
    with pytest.raises(ContractViolation, match="normal_force_N"):
        event(normal_force_N=bad)
    with pytest.raises(ContractViolation, match="budget_margin"):
        event(budget_margin=bad)


def test_event_rejects_an_empty_identity_reference() -> None:
    with pytest.raises(ContractViolation, match="geom_a"):
        event(geom_a="")


def test_event_carries_both_clocks() -> None:
    sample = event(time_index=3, simulator_step=41)
    assert sample.time_index == 3
    assert sample.simulator_step == 41


# -- trajectory -----------------------------------------------------------


def test_trajectory_rejects_non_finite_state() -> None:
    joints = np.zeros((4, 16))
    joints[2, 0] = np.nan
    with pytest.raises(ContractViolation, match="joint_state contains non-finite"):
        trajectory(joint_state=joints)


def test_trajectory_rejects_time_that_does_not_increase() -> None:
    with pytest.raises(ContractViolation, match="strictly increasing"):
        trajectory(time=np.array([0.0, 0.01, 0.01, 0.02]))


def test_trajectory_rejects_time_that_disagrees_with_the_sample_period() -> None:
    # The v1 recorder reconstructed time from the requested control period while
    # the integrator ran at its own timestep, so the recorded duration was not
    # the duration that was simulated (blocker B-06).
    with pytest.raises(ContractViolation, match="declared sample period"):
        trajectory(time=np.arange(4, dtype=float) * 0.02)


def test_trajectory_rejects_a_start_time_the_timebase_does_not_declare() -> None:
    with pytest.raises(ContractViolation, match="timebase declares"):
        trajectory(time=np.arange(4, dtype=float) * SAMPLE_PERIOD_S + 1.0)


def test_trajectory_accepts_a_declared_start_offset() -> None:
    offset = trajectory(
        time=np.arange(4, dtype=float) * SAMPLE_PERIOD_S + 0.05,
        timebase=timebase(start_time_s=0.05),
    )
    assert offset.duration_s == pytest.approx(3 * SAMPLE_PERIOD_S)


def test_trajectory_rejects_an_unnormalised_palm_quaternion() -> None:
    palm = np.zeros((4, 7))
    palm[:, 3] = 0.5  # never normalised
    with pytest.raises(ContractViolation, match="palm_pose quaternions"):
        trajectory(palm_pose=palm)


def test_trajectory_rejects_an_unnormalised_object_quaternion() -> None:
    pose = np.zeros((4, 1, 7))
    pose[:, :, 3] = 2.0
    with pytest.raises(ContractViolation, match="object_pose quaternions"):
        trajectory(object_pose=pose)


def test_quaternion_order_is_declared_and_pinned() -> None:
    assert QUATERNION_ORDER == "wxyz"
    assert timebase().quaternion_order == QUATERNION_ORDER
    with pytest.raises(ContractViolation, match="quaternion_order"):
        timebase(quaternion_order="xyzw")


def test_timebase_rejects_a_degenerate_rate() -> None:
    with pytest.raises(ContractViolation, match="simulator_dt"):
        timebase(simulator_dt=0.0)
    with pytest.raises(ContractViolation, match="sample_every"):
        timebase(sample_every=0)


def test_trajectory_rejects_an_untyped_stage() -> None:
    with pytest.raises(ContractViolation, match="must be a TrajectoryStage"):
        trajectory(stage=("approach", "approach", "approach", "approach"))


def test_required_terminal_stages_are_checked_by_order_not_presence() -> None:
    ordered = trajectory(
        steps=4,
        stage=(
            TrajectoryStage.ENCLOSE,
            TrajectoryStage.SUPPORT_RELEASE,
            TrajectoryStage.LIFT,
            TrajectoryStage.PERTURB,
        ),
    )
    assert ordered.has_required_terminal_stages
    assert ordered.terminal_stages_in_canonical_order

    scrambled = trajectory(
        steps=4,
        stage=(
            TrajectoryStage.LIFT,
            TrajectoryStage.ENCLOSE,
            TrajectoryStage.SUPPORT_RELEASE,
            TrajectoryStage.PERTURB,
        ),
    )
    assert scrambled.has_required_terminal_stages
    assert not scrambled.terminal_stages_in_canonical_order


def test_a_stopped_trajectory_is_still_constructible() -> None:
    # A negative that stopped at the enclose stage is evidence; refusing to
    # build it would destroy the failure record it exists to carry.
    stopped = trajectory(steps=2, stage=(TrajectoryStage.APPROACH, TrajectoryStage.ENCLOSE))
    assert not stopped.has_required_terminal_stages


def test_trajectory_rejects_a_contact_event_outside_the_rollout() -> None:
    with pytest.raises(ValueError, match="outside"):
        trajectory(steps=4, contact_graph=(event(time_index=4),))


def test_an_empty_trajectory_admits_no_contact_events() -> None:
    empty = trajectory(
        steps=0,
        time=np.zeros(0),
        palm_pose=np.zeros((0, 7)),
        joint_state=np.zeros((0, 1)),
        actuator_command=np.zeros((0, 1)),
        object_pose=np.zeros((0, 1, 7)),
        object_velocity=np.zeros((0, 1, 6)),
        stage=(),
    )
    assert empty.num_steps == 0
    with pytest.raises(ValueError, match="outside"):
        dataclasses.replace(empty, contact_graph=(event(time_index=0),))


# -- certificate ----------------------------------------------------------


def test_certificate_refuses_a_cuda_backend() -> None:
    with pytest.raises(ContractViolation, match="cannot name a CUDA backend"):
        certificate(backend_id="mjwarp_cuda")


@pytest.mark.parametrize("field", ["capsule_sha256", "command_sha256", "model_sha256"])
def test_certificate_requires_real_digests(field: str) -> None:
    with pytest.raises(ContractViolation, match=field):
        certificate(**{field: "deadbeef"})


def test_certificate_requires_a_positive_timestep() -> None:
    with pytest.raises(ContractViolation, match="timestep_s"):
        certificate(timestep_s=0.0)


def test_certificate_from_dict_rejects_unknown_keys() -> None:
    payload = certificate().as_dict() | {"confirmed": True}
    with pytest.raises(ContractViolation, match="unknown keys"):
        certificate_from_dict(payload)


def test_certificate_round_trips_through_a_dict() -> None:
    original = certificate()
    assert certificate_from_dict(original.as_dict()) == original


# -- outcome --------------------------------------------------------------


def outcome(**overrides) -> DynamicSearchOutcome:
    defaults = {
        "trajectory_ref": "t:0",
        "passed": False,
        "failure_stage": "lift",
        "failure_reason": "insufficient_lift",
    }
    defaults.update(overrides)
    return DynamicSearchOutcome(**defaults)


def test_outcome_rejects_an_unknown_failure_stage() -> None:
    with pytest.raises(ContractViolation, match="failure_stage"):
        outcome(failure_stage="somewhere")


def test_outcome_rejects_an_invented_failure_reason() -> None:
    with pytest.raises(ContractViolation, match="not a known reason"):
        outcome(failure_reason="it_just_did_not_work")


def test_namespaced_reasons_carry_their_qualifier() -> None:
    assert is_known_failure_reason("transition_timeout:support_released")
    assert is_known_failure_reason("validated_rollout:lift")
    # A bare namespace says nothing about which condition never arrived.
    assert not is_known_failure_reason("transition_timeout")
    assert not is_known_failure_reason("transition_timeout:")
    outcome(failure_stage="enclose", failure_reason="transition_timeout:enclosure_closed")


def test_outcome_rejects_non_finite_objective_terms() -> None:
    with pytest.raises(ContractViolation, match="objective_terms"):
        outcome(objective_terms={"lift_m": float("nan")})
    with pytest.raises(ContractViolation, match="peak_safety_metrics"):
        outcome(peak_safety_metrics={"peak_normal_force_N": float("inf")})


def test_a_failed_outcome_has_to_name_a_reason_and_a_stage() -> None:
    with pytest.raises(ContractViolation, match="must name a failure_reason"):
        outcome(failure_reason="none")
    with pytest.raises(ContractViolation, match="must name a failure_stage"):
        outcome(failure_stage="none")


def test_a_positive_needs_a_typed_certificate() -> None:
    with pytest.raises(ValueError, match="typed"):
        outcome(passed=True, failure_stage="none", failure_reason="none")
    with pytest.raises(ValueError, match="typed"):
        outcome(
            passed=True,
            failure_stage="none",
            failure_reason="none",
            cpu_replay_evidence={"confirmed": True},
        )
    released = outcome(
        passed=True,
        failure_stage="none",
        failure_reason="none",
        cpu_replay_evidence=certificate(),
    )
    assert released.is_release_positive
    assert released.outcome_class == "pass"


def test_a_certificate_that_did_not_certify_cannot_carry_a_positive() -> None:
    with pytest.raises(ContractViolation, match="terminal and safety"):
        outcome(
            passed=True,
            failure_stage="none",
            failure_reason="none",
            cpu_replay_evidence=certificate(terminal_certified=False),
        )


def test_gpu_evidence_alone_is_never_a_positive() -> None:
    with pytest.raises(ValueError, match="typed"):
        outcome(
            passed=True,
            failure_stage="none",
            failure_reason="none",
            gpu_search_evidence={"backend": "mjwarp_cuda", "score": 99.0},
        )


def test_failure_stage_vocabulary_is_closed_and_covers_the_ledger() -> None:
    from qdgrasp.dynamic.objective import _REASON_STAGE

    assert set(_REASON_STAGE) <= FAILURE_REASONS
    assert "none" in {stage.value for stage in FailureStage}


# -- hashing --------------------------------------------------------------


def test_sequence_hash_separates_shape_and_dtype() -> None:
    flat = np.zeros(6, dtype=np.float64)
    shaped = np.zeros((2, 3), dtype=np.float64)
    assert sequence_hash(flat) != sequence_hash(shaped)
    assert sequence_hash(flat) == sequence_hash(np.zeros(6))


def test_sequence_hash_changes_with_one_value() -> None:
    commands = np.zeros((3, 4))
    changed = commands.copy()
    changed[1, 2] = 1e-9
    assert sequence_hash(commands) != sequence_hash(changed)


def test_canonical_hash_is_order_independent() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
