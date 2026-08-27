"""Trajectory storage for contact-rich samples (P3.4-13).

Storage is keyframes plus fixed-rate state samples plus a **sparse** contact
stream. Writing one array per simulator step would make the release dataset grow
with the integrator timestep, which the plan rules out: a finer timestep is a
simulation choice, not more data.

Round-trips are byte-stable for a given trajectory, so a manifest can hash a
shard and a regeneration can be compared to it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    DYNAMIC_TRAJECTORY_SCHEMA_V1,
    DYNAMIC_TRAJECTORY_SCHEMA_V2,
    ContactClass,
    ContactEvent,
    ContractViolation,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    TrajectoryStage,
    TrajectoryTimebase,
    certificate_from_dict,
    outcome_evidence_dict,
)

#: Payloads are written at v2 and only v2 backs a release. v1 stays readable so
#: old evidence can still be inspected, but reading it never promotes it to
#: release-ready (C01.5).
SCHEMA = DYNAMIC_TRAJECTORY_SCHEMA_V2
LEGACY_SCHEMA = DYNAMIC_TRAJECTORY_SCHEMA_V1
READABLE_SCHEMAS = (SCHEMA, LEGACY_SCHEMA)


def _event_to_dict(event: ContactEvent) -> dict[str, Any]:
    return {
        "time_index": int(event.time_index),
        "contact_class": event.contact_class.value,
        "geom_a": event.geom_a,
        "geom_b": event.geom_b,
        "body_a": event.body_a,
        "body_b": event.body_b,
        "point": [float(v) for v in event.point],
        "frame": [float(v) for v in np.asarray(event.frame).ravel()],
        "normal_force_N": float(event.normal_force_N),
        "tangential_force_N": float(event.tangential_force_N),
        "normal_impulse_Ns": float(event.normal_impulse_Ns),
        "tangential_impulse_Ns": float(event.tangential_impulse_Ns),
        "penetration_m": float(event.penetration_m),
        "relative_velocity_mps": float(event.relative_velocity_mps),
        "slip_m": float(event.slip_m),
        "work_J": float(event.work_J),
        "budget_margin": float(event.budget_margin),
        "duration_s": float(event.duration_s),
        "link_class": event.link_class,
        "simulator_step": int(event.simulator_step),
        "episode_index": int(event.episode_index),
    }


def _event_from_dict(payload: dict[str, Any]) -> ContactEvent:
    return ContactEvent(
        time_index=int(payload["time_index"]),
        contact_class=ContactClass(payload["contact_class"]),
        geom_a=payload["geom_a"],
        geom_b=payload["geom_b"],
        body_a=payload["body_a"],
        body_b=payload["body_b"],
        point=np.asarray(payload["point"], dtype=float),
        frame=np.asarray(payload["frame"], dtype=float).reshape(3, 3),
        normal_force_N=float(payload["normal_force_N"]),
        tangential_force_N=float(payload["tangential_force_N"]),
        normal_impulse_Ns=float(payload["normal_impulse_Ns"]),
        tangential_impulse_Ns=float(payload["tangential_impulse_Ns"]),
        penetration_m=float(payload["penetration_m"]),
        relative_velocity_mps=float(payload["relative_velocity_mps"]),
        slip_m=float(payload["slip_m"]),
        work_J=float(payload["work_J"]),
        budget_margin=float(payload["budget_margin"]),
        duration_s=float(payload["duration_s"]),
        link_class=payload["link_class"],
        simulator_step=int(payload.get("simulator_step", -1)),
        episode_index=int(payload.get("episode_index", 0)),
    )


def trajectory_to_record(
    trajectory: DynamicGraspTrajectory, outcome: DynamicSearchOutcome
) -> dict[str, Any]:
    """Serialise one sample, positive or negative.

    Failure trajectories are stored deliberately: a critic or safety model
    trained later needs them as much as the successes.
    """
    return {
        "schema": SCHEMA,
        "trajectory": {
            "time": [float(v) for v in trajectory.time],
            "palm_pose": trajectory.palm_pose.tolist(),
            "joint_state": trajectory.joint_state.tolist(),
            "actuator_command": trajectory.actuator_command.tolist(),
            "object_pose": trajectory.object_pose.tolist(),
            "object_velocity": trajectory.object_velocity.tolist(),
            "stage": [s.value for s in trajectory.stage],
            "timebase": trajectory.timebase.as_dict(),
            "robot_profile": trajectory.robot_profile,
            "palm_body": trajectory.palm_body,
            "contact_graph": [_event_to_dict(e) for e in trajectory.contact_graph],
            "terminal_grasp": trajectory.terminal_grasp,
        },
        "outcome": {
            "trajectory_ref": outcome.trajectory_ref,
            "passed": bool(outcome.passed),
            "failure_stage": outcome.failure_stage,
            "failure_reason": outcome.failure_reason,
            "objective_terms": {k: float(v) for k, v in outcome.objective_terms.items()},
            "peak_safety_metrics": {
                k: float(v) for k, v in outcome.peak_safety_metrics.items()
            },
            "cumulative_safety_metrics": {
                k: float(v) for k, v in outcome.cumulative_safety_metrics.items()
            },
            "cpu_replay_evidence": outcome_evidence_dict(outcome),
            "gpu_search_evidence": outcome.gpu_search_evidence,
        },
    }


def record_to_trajectory(
    record: dict[str, Any],
) -> tuple[DynamicGraspTrajectory, DynamicSearchOutcome]:
    """Rebuild a sample from its record."""
    schema = record.get("schema")
    if schema not in READABLE_SCHEMAS:
        raise ValueError(f"unsupported trajectory schema: {schema!r}")
    payload = record["trajectory"]
    raw_timebase = payload.get("timebase")
    if raw_timebase is None:
        if schema != LEGACY_SCHEMA:
            raise ContractViolation("a v2 record must declare its timebase")
        # A v1 record has no declared timebase, so one is reconstructed from the
        # samples purely to make the payload inspectable. It is stamped with the
        # legacy schema, which the release path refuses.
        times = [float(v) for v in payload["time"]]
        period = (times[1] - times[0]) if len(times) > 1 else 1.0
        raw_timebase = {"simulator_dt": period or 1.0, "sample_every": 1, "start_time_s": times[0] if times else 0.0}
    timebase = TrajectoryTimebase(
        simulator_dt=float(raw_timebase["simulator_dt"]),
        sample_every=int(raw_timebase["sample_every"]),
        start_time_s=float(raw_timebase.get("start_time_s", 0.0)),
    )
    trajectory = DynamicGraspTrajectory(
        time=np.asarray(payload["time"], dtype=float),
        palm_pose=np.asarray(payload["palm_pose"], dtype=float),
        joint_state=np.asarray(payload["joint_state"], dtype=float),
        actuator_command=np.asarray(payload["actuator_command"], dtype=float),
        object_pose=np.asarray(payload["object_pose"], dtype=float),
        object_velocity=np.asarray(payload["object_velocity"], dtype=float),
        stage=tuple(TrajectoryStage(s) for s in payload["stage"]),
        timebase=timebase,
        contact_graph=tuple(_event_from_dict(e) for e in payload["contact_graph"]),
        terminal_grasp=payload.get("terminal_grasp"),
        schema=schema,
        robot_profile=payload.get("robot_profile", ""),
        palm_body=payload.get("palm_body", ""),
    )
    raw = record["outcome"]
    outcome = DynamicSearchOutcome(
        trajectory_ref=raw["trajectory_ref"],
        passed=bool(raw["passed"]),
        failure_stage=raw["failure_stage"],
        failure_reason=raw["failure_reason"],
        objective_terms=dict(raw["objective_terms"]),
        peak_safety_metrics=dict(raw["peak_safety_metrics"]),
        cumulative_safety_metrics=dict(raw["cumulative_safety_metrics"]),
        cpu_replay_evidence=certificate_from_dict(raw.get("cpu_replay_evidence")),
        gpu_search_evidence=raw.get("gpu_search_evidence"),
    )
    return trajectory, outcome


def write_trajectory_shard(
    path: Path,
    samples: Sequence[tuple[DynamicGraspTrajectory, DynamicSearchOutcome]],
) -> str:
    """Write a shard deterministically and return its sha256."""
    payload = {
        "schema": SCHEMA,
        "count": len(samples),
        "records": [trajectory_to_record(t, o) for t, o in samples],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_trajectory_shard(
    path: Path,
) -> tuple[tuple[DynamicGraspTrajectory, DynamicSearchOutcome], ...]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if payload.get("schema") not in READABLE_SCHEMAS:
        raise ValueError(f"unsupported shard schema: {payload.get('schema')!r}")
    records = payload["records"]
    # The header count is checked on read as well as on write: a shard whose
    # header disagrees with its records is how a manifest ends up counting
    # samples that are not there (C01.6).
    declared = int(payload.get("count", -1))
    if declared != len(records):
        raise ContractViolation(
            f"shard header declares {declared} records but carries {len(records)}"
        )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    samples = tuple(record_to_trajectory(r) for r in records)
    if payload.get("sha256") not in (None, digest):
        raise ContractViolation(
            f"shard content hash {digest} does not match the declared {payload['sha256']}"
        )
    return samples


def storage_cost(trajectory: DynamicGraspTrajectory) -> dict[str, int]:
    """Report what a trajectory costs, so sparsity can be checked not assumed."""
    return {
        "state_samples": trajectory.num_steps,
        "contact_events": len(trajectory.contact_graph),
        "objects": trajectory.num_objects,
    }
