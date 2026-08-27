"""S8/S9 — the two backends describe the same world the same way (C02, G07).

**B-13**: ``reset`` put every world into the compiled model's default state, so
a batch of requests that differed in initial state rolled out identical physics
and the seed did nothing. The scene signature hashed seven fields, so two models
with different actuator counts or contact capacities could share a bucket.

**B-14**: the CPU oracle returned empty objective and safety dicts, so the
reference the GPU was supposed to be checked against said nothing about contact.

**B-03**: ``hard_reject`` covered NaN and nothing else, so a world that
overflowed its contact buffer -- and therefore observed an unknown number of
contacts -- survived to be ranked and could become a finalist.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import mujoco
import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import ContactSafetyBudget, DynamicGraspRequest
from qdgrasp.dynamic.capsule import InitialState, ReplayCapsule
from qdgrasp.dynamic.safety import SceneRoles
from qdgrasp.sim.batched.contracts import (
    ROLLOUT_SUMMARY_SCHEMA_V2,
    BackendCapabilityError,
    BackendStateError,
    ContactTelemetry,
    RolloutSummary,
    SceneSignature,
    WorldRejected,
)
from qdgrasp.sim.batched.mjwarp_cuda import MjWarpCudaBackend
from qdgrasp.sim.batched.mujoco_cpu import MuJoCoCpuBackend

MICRO_SCENE = (
    Path(__file__).resolve().parents[1] / "dynamic_grasp" / "micro_scene.xml"
).read_text(encoding="utf-8")


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MICRO_SCENE)


def signature(model: mujoco.MjModel) -> SceneSignature:
    return SceneSignature.from_model(
        model,
        robot_profile="micro_pusher",
        environment="table",
        support_count=1,
        robot_asset_sha256="a" * 64,
    )


def request(seed: int = 0, horizon: int = 20) -> DynamicGraspRequest:
    return DynamicGraspRequest(
        scene_state_ref="scene:micro#0",
        observation_ref="obs:micro/cam_top",
        target_object_id="target",
        robot_profile="micro_pusher",
        strategy_id="primitive_sequence",
        safety_budget_id="micro-conservative-v1",
        horizon=horizon,
        control_dt=0.002,
        seed=seed,
    )


def budget() -> ContactSafetyBudget:
    return ContactSafetyBudget(
        budget_id="micro-conservative-v1",
        robot_profile="micro_pusher",
        peak_normal_force_N=200.0,
        peak_tangential_force_N=120.0,
        normal_impulse_Ns=20.0,
        tangential_impulse_Ns=12.0,
        contact_duration_s=50.0,
        contact_work_J=5.0,
        max_penetration_m=0.02,
        max_wrist_force_N=4000.0,
        max_wrist_torque_Nm=600.0,
        max_joint_or_tendon_load=1500.0,
        max_non_target_translation_m=1.0,
        max_non_target_rotation_rad=6.0,
        max_non_target_velocity_mps=10.0,
    )


def roles(model: mujoco.MjModel) -> SceneRoles:
    def gid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def bid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    return SceneRoles(
        target_geoms=frozenset({gid("target_geom")}),
        support_geoms=frozenset({gid("table")}),
        non_target_geoms=frozenset(),
        robot_geoms=frozenset({gid("pusher_geom")}),
        wrist_body=bid("pusher"),
        palm_body=bid("pusher"),
    )


@pytest.fixture
def backend(model: mujoco.MjModel) -> MuJoCoCpuBackend:
    cpu = MuJoCoCpuBackend(MICRO_SCENE)
    cpu.compile(signature(model), "micro_pusher", batch_capacity=4)
    return cpu


# -- scene signature ------------------------------------------------------


def test_the_signature_covers_every_topology_field(model) -> None:
    sig = signature(model)
    assert sig.actuator_count == int(model.nu)
    assert sig.dof_count == int(model.nv)
    assert sig.body_count == int(model.nbody)
    assert sig.integrator == int(model.opt.integrator)
    assert sig.solver == int(model.opt.solver)


@pytest.mark.parametrize(
    "field",
    [
        "dof_count",
        "actuator_count",
        "tendon_count",
        "equality_count",
        "site_count",
        "mocap_count",
        "body_count",
        "collision_geom_count",
        "contact_capacity",
        "constraint_capacity",
        "integrator",
        "solver",
        "cone",
        "solver_iterations",
        "non_target_count",
    ],
)
def test_changing_any_topology_field_moves_the_bucket(model, field: str) -> None:
    # v1 hashed seven fields, so two models that differed in any of these could
    # share a compiled model (blocker B-13).
    base = signature(model)
    changed = dataclasses.replace(base, **{field: int(getattr(base, field)) + 1})
    assert changed.bucket_key != base.bucket_key


def test_the_signature_does_not_hash_per_world_data(model) -> None:
    # Mass and friction are batched data, not topology; a backend that cannot
    # vary them per world says so at preflight instead.
    fields = {f.name for f in dataclasses.fields(signature(model))}
    assert "body_mass" not in fields
    assert "geom_friction" not in fields


# -- request hydration ----------------------------------------------------


def initial_state(model: mujoco.MjModel, *, target_x: float) -> InitialState:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = InitialState.from_data(model, data)
    qpos = state.qpos.copy()
    qpos[1] = target_x  # the free target's x coordinate
    return dataclasses.replace(state, qpos=qpos)


def test_reset_hydrates_each_world_from_its_own_state(model, backend) -> None:
    states = [initial_state(model, target_x=0.0), initial_state(model, target_x=0.2)]
    seated = backend.reset([request(0), request(1)], states)
    assert seated.qpos[0, 1] == pytest.approx(0.0)
    assert seated.qpos[1, 1] == pytest.approx(0.2)


def test_without_states_reset_is_still_deterministic(model, backend) -> None:
    first = backend.reset([request(0), request(1)]).qpos.copy()
    second = backend.reset([request(0), request(1)]).qpos
    assert np.array_equal(first, second)


def test_a_batch_that_disagrees_on_mass_is_bucketed_not_collapsed(model, backend) -> None:
    light = initial_state(model, target_x=0.0)
    heavy = dataclasses.replace(light, body_mass=light.body_mass * 2.0)
    with pytest.raises(BackendCapabilityError, match="bucket those requests separately"):
        backend.reset([request(0), request(1)], [light, heavy])


def test_a_request_naming_an_absent_target_is_refused(model, backend) -> None:
    bad = dataclasses.replace(request(0), target_object_id="not_in_this_scene")
    with pytest.raises(BackendCapabilityError, match="not a body"):
        backend.reset([bad])


def test_reset_after_a_rollout_clears_the_previous_run(model, backend) -> None:
    backend.reset([request(0)])
    backend.rollout(np.full((1, 10, 1), 0.2))
    assert backend.timing.steps_executed > 0

    backend.reset([request(0)])
    assert backend.timing.steps_executed == 0
    with pytest.raises(BackendStateError, match="before rollout"):
        backend.export_finalists([0])


# -- live worlds vs capacity ----------------------------------------------


def test_num_worlds_is_the_live_count_not_the_pool(model, backend) -> None:
    # v1 returned the pool size on CPU and the live count on GPU, so the same
    # number meant two different things (C02.5).
    assert backend.batch_capacity == 4
    backend.reset([request(0), request(1)])
    assert backend.num_worlds == 2
    assert backend.batch_capacity == 4


def test_both_backends_expose_the_same_world_accounting() -> None:
    for name in ("num_worlds", "batch_capacity"):
        assert isinstance(getattr(MuJoCoCpuBackend, name), property)
        assert isinstance(getattr(MjWarpCudaBackend, name), property)


def test_the_two_backends_still_agree_on_signatures() -> None:
    for name in ("compile", "reset", "step", "observe", "rollout", "export_finalists"):
        cpu = inspect.signature(getattr(MuJoCoCpuBackend, name))
        gpu = inspect.signature(getattr(MjWarpCudaBackend, name))
        assert list(cpu.parameters) == list(gpu.parameters), name


# -- world isolation ------------------------------------------------------


def test_driving_one_world_does_not_move_another(model, backend) -> None:
    backend.reset([request(0), request(1)])
    commands = np.zeros((2, 40, 1))
    commands[0, :, 0] = 0.2
    backend.rollout(commands)
    state = backend.observe()
    driven = float(state.object_pose[0, 0, 0])
    idle = float(state.object_pose[1, 0, 0])
    assert driven > idle + 1e-4


def test_a_rejected_world_does_not_reject_its_neighbours(model, backend) -> None:
    backend.reset([request(0), request(1)])
    with pytest.raises(WorldRejected):
        backend.step(np.array([[np.nan], [0.1]]))
    # The batch is refused as a whole before any world is stepped, so world 1 is
    # untouched rather than half-integrated.
    assert backend.observe().invalid_worlds == ()


# -- summary v2 -----------------------------------------------------------


def test_the_cpu_oracle_summarises_contact(model, backend) -> None:
    backend.attach_safety(roles(model), budget())
    backend.reset([request(0)])
    (summary,) = backend.rollout(np.full((1, 60, 1), 0.2))
    assert summary.schema == ROLLOUT_SUMMARY_SCHEMA_V2
    assert summary.backend_id == "mujoco_cpu"
    # v1 returned empty dicts here, so the oracle said nothing about contact.
    assert summary.objective_terms
    assert summary.peak_safety_metrics
    assert "min_budget_margin" in summary.peak_safety_metrics
    assert summary.contact.unavailable_fields == ()


def test_without_scene_roles_the_oracle_says_it_cannot_classify(model, backend) -> None:
    backend.reset([request(0)])
    (summary,) = backend.rollout(np.full((1, 10, 1), 0.2))
    assert "contact_classification" in summary.contact.unavailable_fields
    assert not summary.contact.observed


def test_a_summary_cannot_disagree_with_itself() -> None:
    with pytest.raises(ValueError, match="must name a failure reason"):
        RolloutSummary(
            world_index=0, steps_executed=0, objective_terms={},
            peak_safety_metrics={}, cumulative_safety_metrics={},
            hard_reject=True, failure_stage="rollout", failure_reason="none",
        )
    with pytest.raises(ValueError, match="must carry failure_reason 'none'"):
        RolloutSummary(
            world_index=0, steps_executed=5, objective_terms={},
            peak_safety_metrics={}, cumulative_safety_metrics={},
            hard_reject=False, failure_stage="none", failure_reason="insufficient_lift",
        )


def test_a_non_finite_metric_rejects_the_world() -> None:
    with pytest.raises(WorldRejected, match="not finite"):
        RolloutSummary(
            world_index=0, steps_executed=5,
            objective_terms={"lift_m": float("nan")},
            peak_safety_metrics={}, cumulative_safety_metrics={},
            hard_reject=False, failure_stage="none", failure_reason="none",
        )


def test_an_unobserved_contact_stream_is_not_treated_as_no_contact() -> None:
    telemetry = ContactTelemetry(contact_count=0, buffer_overflow=True)
    assert not telemetry.observed
    truncated = ContactTelemetry(contact_count=0, unavailable_fields=("frame",))
    assert not truncated.observed
    clean = ContactTelemetry(contact_count=3)
    assert clean.observed


# -- finalist export ------------------------------------------------------


def test_a_finalist_carries_the_exact_commands_that_were_applied(model, backend) -> None:
    backend.reset([request(0), request(1)])
    commands = np.zeros((2, 15, 1))
    commands[1, :, 0] = 0.17
    backend.rollout(commands)
    (finalist,) = backend.export_finalists([1])
    assert isinstance(finalist, ReplayCapsule)
    assert np.array_equal(finalist.control_sequence, commands[1])
    assert finalist.model.nu == int(model.nu)


def test_a_rejected_world_cannot_be_exported(model, backend) -> None:
    backend.reset([request(0)])
    backend.rollout(np.full((1, 10, 1), 0.1))
    backend._reject(0, "non_finite_state")
    with pytest.raises(WorldRejected):
        backend.export_finalists([0])


def test_the_gpu_hard_reject_covers_more_than_nan() -> None:
    # The device path cannot run here, so the reasoning is checked at the source
    # level; the Kaggle harness exercises it against a real T4 (blocker B-03).
    source = inspect.getsource(MjWarpCudaBackend._summarise)
    assert "contact_buffer_overflow" in source
    assert "truncated_contact_stream" in source


def test_the_gpu_requires_every_contact_field_the_budget_needs() -> None:
    from qdgrasp.sim.batched.mjwarp_cuda import REQUIRED_CONTACT_FIELDS

    # ``pos`` alone is not enough: the budget needs the frame and the identity
    # too, or the forces cannot be resolved or attributed (G08.1).
    assert set(REQUIRED_CONTACT_FIELDS) >= {"dist", "pos", "frame", "geom"}
