"""Trajectory storage tests (P3.4-13)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import ContactClass, DynamicSearchOutcome
from qdgrasp.dataset.dynamic_shards import (
    SCHEMA,
    read_trajectory_shard,
    record_to_trajectory,
    storage_cost,
    trajectory_to_record,
    write_trajectory_shard,
)

from .conftest import make_certificate, make_event, make_trajectory


def sample(passed=False):
    events = (
        make_event(0, ContactClass.TARGET_INTENTIONAL),
        make_event(2, ContactClass.SUPPORT_ASSISTED),
    )
    trajectory = make_trajectory(steps=4, contact_graph=events)
    outcome = DynamicSearchOutcome(
        trajectory_ref="t:1",
        passed=passed,
        failure_stage="none" if passed else "lift",
        failure_reason="none" if passed else "insufficient_lift",
        objective_terms={"lift_m": 0.05},
        peak_safety_metrics={"peak_normal_force_N": 1.5},
        cumulative_safety_metrics={"contact_work_J": 0.01},
        cpu_replay_evidence=make_certificate() if passed else None,
    )
    return trajectory, outcome


def test_a_record_round_trips_exactly():
    trajectory, outcome = sample()
    rebuilt_traj, rebuilt_outcome = record_to_trajectory(
        trajectory_to_record(trajectory, outcome)
    )
    assert np.array_equal(rebuilt_traj.object_pose, trajectory.object_pose)
    assert np.array_equal(rebuilt_traj.joint_state, trajectory.joint_state)
    assert rebuilt_traj.stage == trajectory.stage
    assert len(rebuilt_traj.contact_graph) == len(trajectory.contact_graph)
    assert rebuilt_traj.contact_graph[0].contact_class is ContactClass.TARGET_INTENTIONAL
    assert rebuilt_outcome.failure_reason == outcome.failure_reason
    assert rebuilt_outcome.objective_terms == outcome.objective_terms


def test_contact_frame_survives_the_round_trip_as_a_matrix():
    trajectory, outcome = sample()
    rebuilt, _ = record_to_trajectory(trajectory_to_record(trajectory, outcome))
    assert rebuilt.contact_graph[0].frame.shape == (3, 3)
    assert np.array_equal(rebuilt.contact_graph[0].frame, trajectory.contact_graph[0].frame)


def test_failure_trajectories_are_stored_not_dropped(tmp_path):
    # A critic trained later needs the failures as much as the successes.
    path = tmp_path / "shard.json"
    write_trajectory_shard(path, [sample(passed=False), sample(passed=True)])
    loaded = read_trajectory_shard(path)
    assert len(loaded) == 2
    assert [o.passed for _, o in loaded] == [False, True]


def test_a_shard_write_is_deterministic(tmp_path):
    first = write_trajectory_shard(tmp_path / "a.json", [sample(), sample(passed=True)])
    second = write_trajectory_shard(tmp_path / "b.json", [sample(), sample(passed=True)])
    assert first == second
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_an_unknown_schema_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something/else", "records": []}))
    with pytest.raises(ValueError, match="unsupported shard schema"):
        read_trajectory_shard(path)
    with pytest.raises(ValueError, match="unsupported trajectory schema"):
        record_to_trajectory({"schema": "nope"})


def test_storage_is_sparse_in_contacts_not_one_array_per_step():
    # The release must not grow with the integrator timestep: a finer timestep is
    # a simulation choice, not more data.
    trajectory = make_trajectory(steps=200, contact_graph=(make_event(5),))
    cost = storage_cost(trajectory)
    assert cost["state_samples"] == 200
    assert cost["contact_events"] == 1


def test_record_declares_its_schema():
    trajectory, outcome = sample()
    assert trajectory_to_record(trajectory, outcome)["schema"] == SCHEMA


def test_gpu_evidence_is_preserved_alongside_cpu_replay(tmp_path):
    trajectory, outcome = sample(passed=True)
    outcome = DynamicSearchOutcome(
        trajectory_ref=outcome.trajectory_ref, passed=True,
        failure_stage="none", failure_reason="none",
        cpu_replay_evidence=make_certificate(),
        gpu_search_evidence={"backend": "mjwarp_cuda", "worlds": 64},
    )
    path = tmp_path / "s.json"
    write_trajectory_shard(path, [(trajectory, outcome)])
    _, loaded = read_trajectory_shard(path)[0]
    assert loaded.gpu_search_evidence["worlds"] == 64
    assert loaded.cpu_replay_evidence.backend_id == "mujoco_cpu"
    assert loaded.cpu_replay_evidence == outcome.cpu_replay_evidence
