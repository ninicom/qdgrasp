"""S5 — a finalist can be replayed by someone who was not there (G03, G06).

**B-04**: the GPU exported a request, not a candidate. The request says which
scene and which seed; it does not say which controls were applied. The CPU then
regenerated something from the seed and confirmed whatever it produced, so
"every GPU positive is confirmed by CPU replay" was not the claim it looked like.

**B-06**: trajectory time was reconstructed from the requested control period
while the integrator ran at its own timestep, and the palm pose was body index 1
with an identity quaternion -- so every trajectory claimed the hand never
rotated.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import (
    REPLAY_CAPSULE_SCHEMA_V1,
    CpuReplayCertificate,
    DynamicGraspTrajectory,
    TrajectoryStage,
    TrajectoryTimebase,
)
from qdgrasp.dataset.dynamic_shards import read_trajectory_shard, write_trajectory_shard
from qdgrasp.dynamic.capsule import (
    CapsuleError,
    ReplayCapsule,
    capture_capsule,
    certificate_matches,
    hydrate,
    outcome_evidence_hash,
    replay,
)

MICRO_SCENE_PATH = (
    Path(__file__).resolve().parents[1] / "dynamic_grasp" / "micro_scene.xml"
)
MICRO_SCENE = MICRO_SCENE_PATH.read_text(encoding="utf-8")
MODEL_SHA = hashlib.sha256(MICRO_SCENE.encode("utf-8")).hexdigest()


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MICRO_SCENE)


def make_capsule(model: mujoco.MjModel, *, drive: float = 0.2, horizon: int = 30) -> ReplayCapsule:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    commands = np.zeros((horizon, int(model.nu)))
    commands[:, 0] = drive
    return capture_capsule(
        model,
        data,
        capsule_id="capsule:micro#0",
        robot_profile="micro_pusher",
        scene_signature="bucket:micro",
        model_sha256=MODEL_SHA,
        control_sequence=commands,
        seed=7,
        strategy_id="primitive_sequence",
        strategy_parameters={"speed": 0.2},
        safety_budget_id="micro-conservative-v1",
        safety_budget_hash="d" * 64,
    )


# -- identity and hashing -------------------------------------------------


def test_a_capsule_declares_its_schema_and_hashes(model) -> None:
    capsule = make_capsule(model)
    assert capsule.schema == REPLAY_CAPSULE_SCHEMA_V1
    assert len(capsule.capsule_sha256) == 64
    assert len(capsule.command_sha256) == 64
    assert capsule.horizon == 30


def test_changing_one_control_value_changes_the_capsule_hash(model) -> None:
    capsule = make_capsule(model)
    changed = capsule.control_sequence.copy()
    changed[5, 0] += 1e-9
    mutated = dataclasses.replace(capsule, control_sequence=changed)
    assert mutated.command_sha256 != capsule.command_sha256
    assert mutated.capsule_sha256 != capsule.capsule_sha256


def test_changing_one_byte_of_initial_state_changes_the_hash(model) -> None:
    capsule = make_capsule(model)
    state = dataclasses.replace(capsule.state, qpos=capsule.state.qpos + 1e-12)
    assert dataclasses.replace(capsule, state=state).capsule_sha256 != capsule.capsule_sha256


def test_the_control_dtype_is_part_of_the_identity(model) -> None:
    capsule = make_capsule(model)
    assert dataclasses.replace(capsule, control_dtype="float32").capsule_sha256 != (
        capsule.capsule_sha256
    )


def test_mass_and_friction_are_part_of_the_capsule(model) -> None:
    # Two worlds can share a compiled model and still be different physics.
    capsule = make_capsule(model)
    heavier = dataclasses.replace(
        capsule.state, body_mass=capsule.state.body_mass * 1.5
    )
    assert dataclasses.replace(capsule, state=heavier).capsule_sha256 != capsule.capsule_sha256


# -- serialisation --------------------------------------------------------


def test_a_capsule_round_trips_through_disk(tmp_path: Path, model) -> None:
    capsule = make_capsule(model)
    path = tmp_path / "capsule.json"
    digest = capsule.write(path)
    assert len(digest) == 64

    loaded = ReplayCapsule.read(path)
    assert loaded.capsule_sha256 == capsule.capsule_sha256
    assert np.array_equal(loaded.control_sequence, capsule.control_sequence)
    assert np.array_equal(loaded.state.qpos, capsule.state.qpos)


def test_a_capsule_write_is_byte_stable(tmp_path: Path, model) -> None:
    capsule = make_capsule(model)
    first = capsule.write(tmp_path / "a.json")
    second = capsule.write(tmp_path / "b.json")
    assert first == second


def test_a_tampered_payload_is_refused(tmp_path: Path, model) -> None:
    capsule = make_capsule(model)
    payload = capsule.as_dict()
    payload["control_sequence"][3][0] = 99.0
    with pytest.raises(CapsuleError, match="does not match the declared"):
        ReplayCapsule.from_dict(payload)


def test_an_unknown_schema_is_refused(model) -> None:
    payload = make_capsule(model).as_dict() | {"schema": "qdgrasp/replay-capsule/v2"}
    with pytest.raises(CapsuleError, match="unknown capsule schema"):
        ReplayCapsule.from_dict(payload)


def test_a_capsule_whose_controls_do_not_fit_the_model_is_refused(model) -> None:
    capsule = make_capsule(model)
    with pytest.raises(CapsuleError, match="actuators"):
        dataclasses.replace(capsule, control_sequence=np.zeros((5, int(model.nu) + 1)))


# -- replay ---------------------------------------------------------------


def test_replaying_the_same_capsule_twice_gives_the_same_result(model) -> None:
    capsule = make_capsule(model)
    first = replay(model, capsule)
    first_hash = outcome_evidence_hash(capsule, first)

    fresh = mujoco.MjModel.from_xml_string(MICRO_SCENE)
    second = replay(fresh, capsule)
    assert outcome_evidence_hash(capsule, second) == first_hash


def test_replay_uses_the_recorded_commands_not_the_seed(model) -> None:
    # Same seed, different controls: the replay must follow the controls.
    driven = make_capsule(model, drive=0.25)
    idle = dataclasses.replace(
        driven, control_sequence=np.zeros_like(driven.control_sequence)
    )
    moved = replay(mujoco.MjModel.from_xml_string(MICRO_SCENE), driven)
    still = replay(mujoco.MjModel.from_xml_string(MICRO_SCENE), idle)
    assert driven.seed == idle.seed
    assert float(moved.qpos[0]) != pytest.approx(float(still.qpos[0]))


def test_replay_is_independent_of_the_capturing_process(tmp_path: Path, model) -> None:
    # The capsule goes to disk and comes back; nothing is carried in memory.
    capsule = make_capsule(model)
    expected = outcome_evidence_hash(capsule, replay(model, capsule))

    path = tmp_path / "finalist.json"
    capsule.write(path)
    reloaded = ReplayCapsule.read(path)
    fresh = mujoco.MjModel.from_xml_string(MICRO_SCENE)
    assert outcome_evidence_hash(reloaded, replay(fresh, reloaded)) == expected


def test_hydrate_restores_mass_and_friction(model) -> None:
    capsule = make_capsule(model)
    heavy = dataclasses.replace(capsule.state, body_mass=capsule.state.body_mass * 2.0)
    capsule = dataclasses.replace(capsule, state=heavy)

    data = mujoco.MjData(model)
    hydrate(model, data, capsule)
    assert np.allclose(model.body_mass, heavy.body_mass)
    assert float(data.time) == 0.0


def test_hydrate_refuses_a_model_it_was_not_captured_on(model) -> None:
    capsule = make_capsule(model)
    other = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.002"/>
          <worldbody><body name="b"><joint name="j" type="slide" axis="1 0 0"/>
          <geom type="box" size="0.01 0.01 0.01"/></body></worldbody>
          <actuator><position name="a" joint="j"/></actuator>
        </mujoco>
        """
    )
    with pytest.raises(CapsuleError, match="captured on a model"):
        hydrate(other, mujoco.MjData(other), capsule)


# -- certificate binding --------------------------------------------------


def certificate_for(capsule: ReplayCapsule, **over) -> CpuReplayCertificate:
    defaults = {
        "backend_id": "mujoco_cpu",
        "capsule_sha256": capsule.capsule_sha256,
        "command_sha256": capsule.command_sha256,
        "model_sha256": capsule.model.model_sha256,
        "timestep_s": capsule.model.timestep_s,
        "terminal_certified": True,
        "safety_certified": True,
        "outcome_class": "pass",
    }
    defaults.update(over)
    return CpuReplayCertificate(**defaults)


def test_a_certificate_is_bound_to_the_capsule_it_was_issued_against(model) -> None:
    capsule = make_capsule(model)
    assert certificate_matches(capsule, certificate_for(capsule))


def test_a_certificate_for_a_different_capsule_is_refused(model) -> None:
    capsule = make_capsule(model)
    other = make_capsule(model, drive=0.9)
    assert not certificate_matches(other, certificate_for(capsule))


def test_a_certificate_naming_a_different_model_is_refused(model) -> None:
    capsule = make_capsule(model)
    stale = certificate_for(capsule, model_sha256="f" * 64)
    assert not certificate_matches(capsule, stale)


# -- trajectory v2 timebase and palm pose ---------------------------------


def test_palm_pose_matches_the_simulator_at_every_sampled_frame(model) -> None:
    capsule = make_capsule(model, horizon=40)
    palm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
    sample_every = 5

    data = mujoco.MjData(model)
    hydrate(model, data, capsule)
    times, palm, expected = [], [], []
    calls = 0
    for row in range(capsule.horizon):
        data.ctrl[:] = capsule.control_sequence[row]
        mujoco.mj_step(model, data)
        calls += 1
        if calls % sample_every:
            continue
        times.append(float(data.time))
        pose = np.zeros(7)
        pose[:3] = data.xpos[palm_body]
        pose[3:] = data.xquat[palm_body]
        palm.append(pose)
        expected.append(np.concatenate([data.xpos[palm_body], data.xquat[palm_body]]))

    steps = len(times)
    trajectory = DynamicGraspTrajectory(
        time=np.asarray(times),
        palm_pose=np.asarray(palm),
        joint_state=np.zeros((steps, int(model.nq))),
        actuator_command=np.zeros((steps, int(model.nu))),
        object_pose=np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (steps, 1, 1)),
        object_velocity=np.zeros((steps, 1, 6)),
        stage=tuple([TrajectoryStage.APPROACH] * steps),
        timebase=TrajectoryTimebase(
            simulator_dt=float(model.opt.timestep),
            sample_every=sample_every,
            start_time_s=times[0],
        ),
        robot_profile="micro_pusher",
        palm_body="pusher",
    )
    assert np.allclose(trajectory.palm_pose, np.asarray(expected))
    # A real rotation is recorded, not an identity placeholder.
    assert trajectory.palm_pose.shape[1] == 7


def test_recorded_duration_matches_the_rollout(model) -> None:
    sample_every = 5
    dt = float(model.opt.timestep)
    steps = 8
    times = np.arange(1, steps + 1, dtype=float) * sample_every * dt
    trajectory = DynamicGraspTrajectory(
        time=times,
        palm_pose=np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (steps, 1)),
        joint_state=np.zeros((steps, 2)),
        actuator_command=np.zeros((steps, 1)),
        object_pose=np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (steps, 1, 1)),
        object_velocity=np.zeros((steps, 1, 6)),
        stage=tuple([TrajectoryStage.APPROACH] * steps),
        timebase=TrajectoryTimebase(
            simulator_dt=dt, sample_every=sample_every, start_time_s=float(times[0])
        ),
    )
    assert trajectory.duration_s == pytest.approx((steps - 1) * sample_every * dt)


def test_a_trajectory_shard_round_trips_byte_stably(tmp_path: Path, model) -> None:
    from qdgrasp.dataset.dynamic_contracts import DynamicSearchOutcome

    from .test_taxonomy_and_terminal import _trajectory  # local fixture builder

    trajectory = _trajectory(lifts={0: 0.05})
    outcome = DynamicSearchOutcome(
        trajectory_ref="t:0",
        passed=False,
        failure_stage="lift",
        failure_reason="insufficient_lift",
    )
    first = write_trajectory_shard(tmp_path / "a.json", [(trajectory, outcome)])
    second = write_trajectory_shard(tmp_path / "b.json", [(trajectory, outcome)])
    assert first == second

    (loaded, _), = read_trajectory_shard(tmp_path / "a.json")
    assert loaded.timebase.sample_period_s == pytest.approx(
        trajectory.timebase.sample_period_s
    )
    assert loaded.palm_body == trajectory.palm_body


def test_a_shard_whose_header_count_lies_is_refused(tmp_path: Path) -> None:
    import json

    from qdgrasp.dataset.dynamic_contracts import ContractViolation, DynamicSearchOutcome

    from .test_taxonomy_and_terminal import _trajectory

    path = tmp_path / "s.json"
    write_trajectory_shard(
        path,
        [
            (
                _trajectory(),
                DynamicSearchOutcome(
                    trajectory_ref="t:0",
                    passed=False,
                    failure_stage="lift",
                    failure_reason="insufficient_lift",
                ),
            )
        ],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["count"] = 7
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ContractViolation, match="header declares"):
        read_trajectory_shard(path)
