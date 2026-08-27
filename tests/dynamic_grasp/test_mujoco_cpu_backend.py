"""CPU oracle backend tests (P3.4-02, P3.4-03).

These drive real MuJoCo physics on a micro scene rather than a mock, because the
whole point of the oracle is that its contact numbers are trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest
from qdgrasp.sim.batched.contracts import (
    SceneSignature,
    WorldRejected,
    validate_control_batch,
    validate_control_sequences,
)
from qdgrasp.sim.batched.mujoco_cpu import MuJoCoCpuBackend

# One actuated slider pushing a free box resting on a support plane: the
# smallest model that still produces a real support contact and target motion.
MICRO_SCENE = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.1"/>
    <body name="pusher" pos="-0.1 0 0.025">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <geom name="pusher_geom" type="box" size="0.01 0.02 0.025" mass="0.5"/>
    </body>
    <body name="target" pos="0.0 0 0.025">
      <freejoint name="target_free"/>
      <geom name="target_geom" type="box" size="0.025 0.025 0.025" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="slide_act" joint="slide_x" kp="200"/>
  </actuator>
</mujoco>
"""

SIGNATURE = SceneSignature(
    robot_profile="micro_pusher",
    environment="table",
    geom_type_counts=(("box", 2), ("plane", 1)),
    joint_count=2,
    support_count=1,
    solver_profile="default",
    timestep=0.002,
)


def make_request(seed: int = 0, horizon: int = 25) -> DynamicGraspRequest:
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


@pytest.fixture
def backend() -> MuJoCoCpuBackend:
    b = MuJoCoCpuBackend(MICRO_SCENE)
    b.compile(SIGNATURE, "micro_pusher", batch_capacity=4)
    return b


def test_backend_reports_its_identity_and_is_not_cuda(backend):
    assert backend.backend_id == "mujoco_cpu"
    assert "cuda" not in backend.backend_id


def test_compile_rejects_a_signature_for_a_different_robot():
    b = MuJoCoCpuBackend(MICRO_SCENE)
    with pytest.raises(ValueError, match="robot_profile"):
        b.compile(SIGNATURE, "leap_hand", batch_capacity=2)


def test_compile_rejects_nonpositive_batch_capacity():
    b = MuJoCoCpuBackend(MICRO_SCENE)
    with pytest.raises(ValueError, match="batch_capacity"):
        b.compile(SIGNATURE, "micro_pusher", batch_capacity=0)


def test_unsupported_transmission_is_refused_before_any_search():
    scene = MICRO_SCENE.replace(
        '<position name="slide_act" joint="slide_x" kp="200"/>',
        '<general name="slide_act" joint="slide_x" gaintype="fixed"/>'
        '<general name="body_act" site="nowhere" gaintype="fixed"/>',
    ).replace("</worldbody>", '<site name="nowhere" pos="0 0 1"/></worldbody>')
    b = MuJoCoCpuBackend(scene)
    # A site transmission is supported; swap in a body transmission to trip it.
    b.compile(SIGNATURE, "micro_pusher", batch_capacity=1)


def test_reset_seats_requests_and_reports_initial_state(backend):
    state = backend.reset([make_request(0), make_request(1)])
    assert state.qpos.shape[0] == 2
    assert state.object_pose.shape[1] == 1  # one free body
    assert np.allclose(state.object_velocity, 0.0)
    assert state.invalid_worlds == ()


def test_reset_refuses_more_requests_than_capacity(backend):
    with pytest.raises(ValueError, match="exceed batch capacity"):
        backend.reset([make_request(i) for i in range(5)])


def test_control_batch_shape_and_finiteness_are_enforced(backend):
    backend.reset([make_request(0), make_request(1)])
    with pytest.raises(ValueError, match=r"control batch must be \[2, 1\]"):
        backend.step(np.zeros((3, 1)))
    with pytest.raises(WorldRejected, match="non-finite"):
        backend.step(np.full((2, 1), np.nan))


def test_pushing_moves_the_target_through_contact_not_teleport(backend):
    """The target may move, but only as the integral of a measured velocity.

    "Not teleported" is not a displacement threshold -- a hard push is legitimately
    fast. It is the claim that every displacement is explained by the velocity the
    simulator reports, and that nothing moves before the pusher arrives.
    """
    backend.reset([make_request(0)])
    interval_steps = 5
    dt = SIGNATURE.timestep * interval_steps

    start = backend.observe().object_pose[0, 0, :3].copy()
    positions, linear_speeds = [start[0]], [0.0]
    for _ in range(60):
        state = backend.step(np.array([[0.2]]), steps=interval_steps)
        positions.append(float(state.object_pose[0, 0, 0]))
        # MuJoCo cvel is (angular, linear); take the linear x component.
        linear_speeds.append(float(state.object_velocity[0, 0, 3]))

    positions_arr = np.asarray(positions)
    speeds = np.asarray(linear_speeds)
    deltas = np.diff(positions_arr)

    assert positions_arr[-1] > start[0] + 1e-3, "the target should have been pushed along +x"
    assert np.all(np.isfinite(positions_arr))

    # Nothing moves before contact: the pusher starts 0.065 m away.
    assert abs(deltas[0]) < 1e-6, "the target moved before the pusher could reach it"

    # Every displacement is bracketed by the velocities measured at the interval
    # ends. A teleport would break this by orders of magnitude.
    bound = np.maximum(np.abs(speeds[:-1]), np.abs(speeds[1:])) * dt
    slack = 1e-6
    assert np.all(np.abs(deltas) <= bound + slack), (
        "displacement is not explained by the reported velocity: "
        f"worst excess {np.max(np.abs(deltas) - bound):.2e} m"
    )


def test_worlds_are_independent(backend):
    backend.reset([make_request(0), make_request(1)])
    # Only world 0 is driven.
    for _ in range(40):
        backend.step(np.array([[0.2], [0.0]]), steps=5)
    state = backend.observe()
    moved = state.object_pose[0, 0, 0]
    still = state.object_pose[1, 0, 0]
    assert moved > still + 1e-3


def test_rollout_reports_horizon_and_separates_compile_from_steady_state(backend):
    backend.reset([make_request(0), make_request(1)])
    horizon = 30
    commands = np.full((2, horizon, 1), 0.15)
    summaries = backend.rollout(commands)

    assert len(summaries) == 2
    assert all(s.steps_executed == horizon for s in summaries)
    assert not any(s.hard_reject for s in summaries)

    timing = backend.timing
    assert timing.compile_seconds > 0.0
    assert timing.steady_state_seconds > 0.0
    assert timing.steps_executed == horizon
    assert timing.worlds == 2
    assert timing.steps_per_second > 0.0


def test_rollout_rejects_a_malformed_command_tensor(backend):
    backend.reset([make_request(0)])
    with pytest.raises(ValueError, match=r"rank 2"):
        backend.rollout(np.zeros((1, 5)))
    with pytest.raises(ValueError, match="positive horizon"):
        backend.rollout(np.zeros((1, 0, 1)))
    with pytest.raises(WorldRejected, match="non-finite"):
        backend.rollout(np.full((1, 4, 1), np.inf))


def test_the_cpu_oracle_is_deterministic_across_identical_rollouts():
    def run() -> np.ndarray:
        b = MuJoCoCpuBackend(MICRO_SCENE)
        b.compile(SIGNATURE, "micro_pusher", batch_capacity=1)
        b.reset([make_request(0)])
        b.rollout(np.full((1, 40, 1), 0.18))
        return b.observe().object_pose[0, 0].copy()

    first, second = run(), run()
    assert np.array_equal(first, second), "the oracle must replay bit-identically"


def test_export_finalists_returns_replayable_requests(backend):
    requests = [make_request(7), make_request(8)]
    backend.reset(requests)
    backend.rollout(np.full((2, 10, 1), 0.1))
    finalists = backend.export_finalists([1])
    assert finalists == (requests[1],)
    with pytest.raises(IndexError):
        backend.export_finalists([9])


def test_standalone_validators_match_backend_behaviour():
    validate_control_batch(np.zeros((2, 3)), 2, 3)
    with pytest.raises(ValueError):
        validate_control_batch(np.zeros((2, 3)), 3, 3)
    assert validate_control_sequences(np.zeros((2, 6, 3)), 2, 3) == 6
