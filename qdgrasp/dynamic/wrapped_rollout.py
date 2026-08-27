"""Contact-rich observation layered on the validated rollout (P3.4-16 support).

Phase 3.4 needs the target and the scene to react physically. That protocol
already exists and is validated: ``validate_grasp_rollout`` carries weld
settling, approach/squeeze/lift/perturb phasing at pinned step counts, actuator
gains read from the compiled MJCF, a contact noise floor and tracking
tolerances.

So this module does not re-run physics. It attaches the Phase 3.4 contact
observer to that rollout through its own ``step_observer`` hook, and reads the
result against the multi-quantity safety budget. What makes Phase 3.4 different
-- permitted support and neighbour contact under a measured budget -- sits above
the protocol rather than inside it.

An earlier draft reimplemented a fraction of the protocol instead. It produced
zero positives across nine hand-iterations and flung a hand six kilometres; the
diagnosis is in `evidence/phase3_4/p16-dataset-blocked/`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    ContactClass,
    ContactEvent,
    ContactSafetyBudget,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    TrajectoryStage,
)
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout
from qdgrasp.dynamic.safety import ContactObserver, SceneRoles, summarise_safety
from qdgrasp.objects.schema import SubGeomSpec

#: Rollout stage names used by the validated protocol, mapped onto the Phase 3.4
#: trajectory stages. An unrecognised stage is an error rather than a guess.
_STAGE_MAP: dict[str, TrajectoryStage] = {
    "approach": TrajectoryStage.APPROACH,
    "pregrasp": TrajectoryStage.APPROACH,
    "squeeze": TrajectoryStage.ENCLOSE,
    "close": TrajectoryStage.ENCLOSE,
    "lift": TrajectoryStage.LIFT,
    "perturbation": TrajectoryStage.PERTURB,
    "perturb": TrajectoryStage.PERTURB,
    "settle": TrajectoryStage.APPROACH,
}


def _stage_of(name: str) -> TrajectoryStage:
    key = name.strip().lower()
    for prefix, stage in _STAGE_MAP.items():
        if key.startswith(prefix):
            return stage
    return TrajectoryStage.APPROACH


@dataclasses.dataclass
class _Recorder:
    """Accumulates a trajectory while the validated rollout runs."""

    observer: ContactObserver
    control_dt: float
    free_bodies: Sequence[int]
    sample_every: int = 5

    def __post_init__(self) -> None:
        self.index = 0
        self.calls = 0
        self.times: list[float] = []
        self.joints: list[np.ndarray] = []
        self.commands: list[np.ndarray] = []
        self.palm: list[np.ndarray] = []
        self.poses: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.stages: list[TrajectoryStage] = []
        self.events: list[ContactEvent] = []

    def __call__(self, stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.calls += 1
        # Contacts are read every step so impulse and work accumulate honestly;
        # state is sampled sparsely so the record does not grow with the
        # integrator timestep.
        self.events.extend(
            self.observer.observe(data, time_index=self.index, dt=self.control_dt)
        )
        if self.calls % self.sample_every:
            return

        self.times.append(self.index * self.control_dt)
        self.joints.append(np.array(data.qpos))
        self.commands.append(np.array(data.ctrl))
        pose = np.zeros(7)
        pose[3] = 1.0
        pose[:3] = data.xpos[1] if int(model.nbody) > 1 else 0.0
        self.palm.append(pose)

        count = max(1, len(self.free_bodies))
        poses = np.zeros((count, 7))
        velocities = np.zeros((count, 6))
        for slot, body in enumerate(self.free_bodies):
            poses[slot, :3] = data.xpos[body]
            poses[slot, 3:] = data.xquat[body]
            velocities[slot] = data.cvel[body]
        if not self.free_bodies:
            poses[0, 3] = 1.0
        self.poses.append(poses)
        self.velocities.append(velocities)
        self.stages.append(_stage_of(stage))
        self.index += 1


def run_wrapped_contact_rollout(
    *,
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    fingertip_body_names: Sequence[str],
    roles_from_model: Callable[[mujoco.MjModel], SceneRoles],
    budget: ContactSafetyBudget,
    rollout_kwargs: Mapping[str, object],
    control_dt: float = 0.002,
    trajectory_ref: str = "",
) -> tuple[DynamicGraspTrajectory, DynamicSearchOutcome, object]:
    """Run the validated rollout and observe it as a Phase 3.4 trajectory.

    ``roles_from_model`` receives the compiled model and returns the
    :class:`SceneRoles` for it, because geom ids only exist after compilation.
    """
    recorder: dict[str, _Recorder] = {}

    def install(stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if "r" not in recorder:
            roles = roles_from_model(model)
            free_bodies = [
                int(model.jnt_bodyid[j])
                for j in range(int(model.njnt))
                if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
                and int(model.jnt_bodyid[j])
                in {int(model.geom_bodyid[g]) for g in roles.target_geoms}
            ]
            recorder["r"] = _Recorder(
                observer=ContactObserver(model, roles, budget),
                control_dt=control_dt,
                free_bodies=free_bodies,
            )
        recorder["r"](stage, model, data)

    validation = validate_grasp_rollout(
        hand_xml_path,
        collision_geoms,
        fingertip_body_names,
        step_observer=install,
        **rollout_kwargs,
    )

    record = recorder.get("r")
    if record is None or not record.times:
        empty = DynamicGraspTrajectory(
            time=np.zeros(0),
            palm_pose=np.zeros((0, 7)),
            joint_state=np.zeros((0, 1)),
            actuator_command=np.zeros((0, 1)),
            object_pose=np.zeros((0, 1, 7)),
            object_velocity=np.zeros((0, 1, 6)),
            stage=(),
        )
        return (
            empty,
            DynamicSearchOutcome(
                trajectory_ref=trajectory_ref,
                passed=False,
                failure_stage="rollout",
                failure_reason="empty_trajectory",
            ),
            validation,
        )

    steps = len(record.times)
    trajectory = DynamicGraspTrajectory(
        time=np.asarray(record.times),
        palm_pose=np.asarray(record.palm),
        joint_state=np.asarray(record.joints),
        actuator_command=np.asarray(record.commands),
        object_pose=np.asarray(record.poses),
        object_velocity=np.asarray(record.velocities),
        stage=tuple(record.stages),
        contact_graph=tuple(
            dataclasses.replace(e, time_index=min(e.time_index, steps - 1))
            for e in record.events
        ),
    )

    peak, cumulative = summarise_safety(trajectory.contact_graph)
    hard = trajectory.hard_reject_events
    if hard:
        classes = {e.contact_class for e in hard}
        stage, reason = (
            ("contact", "forbidden_contact")
            if ContactClass.FORBIDDEN in classes
            else ("contact", "damaging_contact")
        )
    elif not validation.passed:
        # The validated protocol has the final say on whether this is a grasp.
        stage, reason = (validation.failure_stage or "dynamic", "validated_rollout_failed")
    else:
        stage, reason = ("none", "none")

    passed = reason == "none"
    lift = float(trajectory.object_pose[-1, 0, 2] - trajectory.object_pose[0, 0, 2])
    outcome = DynamicSearchOutcome(
        trajectory_ref=trajectory_ref or "wrapped",
        passed=passed,
        failure_stage=stage,
        failure_reason=reason,
        objective_terms={
            "lift_m": lift,
            "steps": float(steps),
            "contact_events": float(len(trajectory.contact_graph)),
        },
        peak_safety_metrics=peak,
        cumulative_safety_metrics=cumulative,
        cpu_replay_evidence=(
            {
                "backend": "mujoco_cpu",
                "confirmed": True,
                "protocol": "mocap-weld-v3",
                "validated_rollout_passed": True,
            }
            if passed
            else {}
        ),
    )
    return trajectory, outcome, validation
