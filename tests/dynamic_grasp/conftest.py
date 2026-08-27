from __future__ import annotations

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import (
    ContactClass,
    ContactEvent,
    ContactSafetyBudget,
    CpuReplayCertificate,
    DynamicGraspTrajectory,
    TrajectoryStage,
    TrajectoryTimebase,
)


@pytest.fixture
def budget() -> ContactSafetyBudget:
    return ContactSafetyBudget(
        budget_id="micro-conservative-v1",
        robot_profile="leap_hand",
        peak_normal_force_N=20.0,
        peak_tangential_force_N=12.0,
        normal_impulse_Ns=2.0,
        tangential_impulse_Ns=1.2,
        contact_duration_s=5.0,
        contact_work_J=0.5,
        max_penetration_m=0.002,
        max_wrist_force_N=40.0,
        max_wrist_torque_Nm=6.0,
        max_joint_or_tendon_load=15.0,
        max_non_target_translation_m=0.01,
        max_non_target_rotation_rad=0.15,
        max_non_target_velocity_mps=0.05,
    )


def make_event(
    time_index: int = 0,
    contact_class: ContactClass = ContactClass.TARGET_INTENTIONAL,
    **overrides,
) -> ContactEvent:
    defaults = {
        "time_index": time_index,
        "contact_class": contact_class,
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
        "penetration_m": 0.0002,
        "relative_velocity_mps": 0.01,
        "slip_m": 0.0005,
        "work_J": 0.001,
        "budget_margin": 0.5,
    }
    defaults.update(overrides)
    return ContactEvent(**defaults)


SAMPLE_PERIOD_S = 0.01


def make_timebase(sample_period_s: float = SAMPLE_PERIOD_S) -> TrajectoryTimebase:
    return TrajectoryTimebase(simulator_dt=sample_period_s, sample_every=1)


def make_trajectory(
    steps: int = 4,
    joints: int = 16,
    actuators: int = 16,
    objects: int = 2,
    contact_graph: tuple[ContactEvent, ...] = (),
    stage: tuple[TrajectoryStage, ...] | None = None,
) -> DynamicGraspTrajectory:
    palm = np.zeros((steps, 7))
    palm[:, 3] = 1.0
    pose = np.zeros((steps, objects, 7))
    pose[:, :, 3] = 1.0
    return DynamicGraspTrajectory(
        time=np.arange(steps, dtype=float) * SAMPLE_PERIOD_S,
        palm_pose=palm,
        joint_state=np.zeros((steps, joints)),
        actuator_command=np.zeros((steps, actuators)),
        object_pose=pose,
        object_velocity=np.zeros((steps, objects, 6)),
        stage=stage if stage is not None else tuple([TrajectoryStage.APPROACH] * steps),
        timebase=make_timebase(),
        contact_graph=contact_graph,
        robot_profile="leap_hand",
        palm_body="palm_lower",
    )


def make_certificate(**overrides) -> CpuReplayCertificate:
    """A certificate whose hashes are real digests, not placeholder strings."""
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
