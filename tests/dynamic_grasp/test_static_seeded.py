"""Static-seeded contact rollout tests (P3.4-08)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import ContactClass, ContactSafetyBudget
from qdgrasp.dynamic.primitives import Primitive, PrimitiveKind, TransitionCondition
from qdgrasp.dynamic.safety import SceneRoles
from qdgrasp.dynamic.static_seeded import (
    RolloutLimits,
    SeedPose,
    run_static_seeded_rollout,
)

MICRO_SCENE = (Path(__file__).parent / "micro_scene.xml").read_text(encoding="utf-8")


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MICRO_SCENE)


def gid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


@pytest.fixture
def roles(model) -> SceneRoles:
    return SceneRoles(
        target_geoms=frozenset({gid(model, "target_geom")}),
        support_geoms=frozenset({gid(model, "table")}),
        non_target_geoms=frozenset(),
        robot_geoms=frozenset({gid(model, "pusher_geom")}),
    )


@pytest.fixture
def seed(model) -> SeedPose:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SeedPose(
        qpos=np.array(data.qpos), ctrl=np.zeros(model.nu), source_candidate_id="seed:0"
    )


def push_sequence() -> tuple[Primitive, ...]:
    return (
        Primitive(
            kind=PrimitiveKind.PUSH,
            direction=np.array([1.0, 0.0, 0.0]),
            speed=0.2,
            max_duration_s=0.4,
            until=TransitionCondition.TARGET_CONTACT_MADE,
        ),
        Primitive(
            kind=PrimitiveKind.SQUEEZE,
            direction=np.array([1.0, 0.0, 0.0]),
            speed=0.1,
            max_duration_s=0.4,
            grip=1.0,
        ),
    )


def test_rollout_produces_a_valid_trajectory(model, roles, budget, seed):
    trajectory, outcome = run_static_seeded_rollout(
        model, roles=roles, budget=budget, seed=seed,
        primitives=push_sequence(), horizon=40, control_dt=0.01,
    )
    assert trajectory.num_steps > 0
    assert trajectory.joint_state.shape[0] == trajectory.num_steps
    assert trajectory.object_pose.shape[2] == 7
    assert len(trajectory.stage) == trajectory.num_steps
    assert outcome.trajectory_ref == "seed:0"


def test_the_target_moves_because_contact_moved_it(model, roles, budget, seed):
    trajectory, _ = run_static_seeded_rollout(
        model, roles=roles, budget=budget, seed=seed,
        primitives=push_sequence(), horizon=60, control_dt=0.01,
    )
    x = trajectory.object_pose[:, 0, 0]
    assert x[-1] > x[0] + 1e-4, "the pushed target should have travelled"
    assert any(
        e.contact_class is ContactClass.TARGET_INTENTIONAL for e in trajectory.contact_graph
    ), "and there must be a recorded robot-target contact explaining it"


def test_a_rollout_that_never_lifts_is_a_negative_with_a_reason(model, roles, budget, seed):
    # The micro pusher cannot lift anything: the outcome must say so rather than
    # pass, and must not be silently dropped.
    _, outcome = run_static_seeded_rollout(
        model, roles=roles, budget=budget, seed=seed,
        primitives=push_sequence(), horizon=40, control_dt=0.01,
    )
    assert not outcome.passed
    assert outcome.failure_reason != "none"
    assert outcome.failure_stage != "none"
    assert not outcome.cpu_replay_evidence, "a failed outcome carries no replay claim"


def test_a_blown_budget_is_reported_as_damaging_contact(model, roles, seed):
    tight = ContactSafetyBudget(
        budget_id="tight", robot_profile="micro_pusher",
        peak_normal_force_N=1e-6, peak_tangential_force_N=1e-6,
        normal_impulse_Ns=1e-9, tangential_impulse_Ns=1e-9,
        contact_duration_s=1e-6, contact_work_J=1e-9, max_penetration_m=1e-9,
        max_wrist_force_N=1.0, max_wrist_torque_Nm=1.0, max_joint_or_tendon_load=1.0,
        max_non_target_translation_m=1e-6, max_non_target_rotation_rad=1e-6,
        max_non_target_velocity_mps=1e-6,
    )
    _, outcome = run_static_seeded_rollout(
        model, roles=roles, budget=tight, seed=seed,
        primitives=push_sequence(), horizon=20, control_dt=0.01,
    )
    assert not outcome.passed
    assert outcome.failure_reason == "damaging_contact"
    assert outcome.failure_stage == "contact"


def test_an_unknown_geom_role_is_a_forbidden_contact(model, budget, seed):
    # Declare nothing as support: the table becomes an unknown geom, and an
    # unknown role must reject rather than pass unnoticed.
    blind = SceneRoles(
        target_geoms=frozenset({gid(model, "target_geom")}),
        support_geoms=frozenset(),
        non_target_geoms=frozenset(),
        robot_geoms=frozenset({gid(model, "pusher_geom")}),
    )
    _, outcome = run_static_seeded_rollout(
        model, roles=blind, budget=budget, seed=seed,
        primitives=push_sequence(), horizon=20, control_dt=0.01,
    )
    assert outcome.failure_reason == "forbidden_contact"


def test_limits_are_a_frozen_object_the_manifest_can_hash():
    limits = RolloutLimits()
    assert dataclasses.is_dataclass(limits)
    with pytest.raises(dataclasses.FrozenInstanceError):
        limits.min_lift_m = 0.0  # type: ignore[misc]


def test_rollout_rejects_a_degenerate_horizon_or_seed(model, roles, budget, seed):
    with pytest.raises(ValueError, match="horizon"):
        run_static_seeded_rollout(
            model, roles=roles, budget=budget, seed=seed,
            primitives=push_sequence(), horizon=0, control_dt=0.01,
        )
    bad = dataclasses.replace(seed, qpos=np.zeros(3))
    with pytest.raises(ValueError, match="seed qpos"):
        run_static_seeded_rollout(
            model, roles=roles, budget=budget, seed=bad,
            primitives=push_sequence(), horizon=10, control_dt=0.01,
        )


def test_rollout_is_deterministic(model, roles, budget, seed):
    def run():
        traj, out = run_static_seeded_rollout(
            model, roles=roles, budget=budget, seed=seed,
            primitives=push_sequence(), horizon=30, control_dt=0.01,
        )
        return traj.object_pose.copy(), out.failure_reason

    first, second = run(), run()
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_a_velocity_command_does_not_compound_into_a_runaway(model, roles, budget, seed):
    """Regression: wrist velocity must be integrated over the control step.

    Adding a velocity straight onto a position target compounds every step. The
    first version of this rollout drove the 5 cm box 9.4 metres across the scene,
    and the velocity-consistency check accepted it because the box really was
    moving that fast -- the setpoint, not the physics, was wrong.
    """
    trajectory, _ = run_static_seeded_rollout(
        model, roles=roles, budget=budget, seed=seed,
        primitives=push_sequence(), horizon=60, control_dt=0.01,
    )
    travel = abs(float(trajectory.object_pose[-1, 0, 0] - trajectory.object_pose[0, 0, 0]))
    assert travel < 0.5, f"target travelled {travel:.3f} m from a 0.2 m/s push"

    commands = trajectory.actuator_command
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    for index in range(int(model.nu)):
        if limited[index]:
            low, high = ctrlrange[index]
            assert np.all(commands[:, index] >= low - 1e-9)
            assert np.all(commands[:, index] <= high + 1e-9)
