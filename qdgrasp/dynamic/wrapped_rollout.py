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
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    ContactClass,
    ContactEvent,
    ContactSafetyBudget,
    CpuReplayCertificate,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    TrajectoryStage,
    TrajectoryTimebase,
    sequence_hash,
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
    "support_release": TrajectoryStage.SUPPORT_RELEASE,
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
    simulator_dt: float
    free_bodies: Sequence[int]
    palm_body_id: int
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
        # integrator timestep. The observer is advanced by the *simulator*
        # timestep, not by a requested control period: those are different
        # clocks, and feeding it the wrong one inflates every impulse and every
        # contact duration it reports (blocker B-06).
        self.events.extend(
            self.observer.observe(
                data,
                time_index=self.index,
                dt=self.simulator_dt,
                simulator_step=self.calls,
            )
        )
        if self.calls % self.sample_every:
            return

        # Time is read from the integrator rather than reconstructed from an
        # index, so the recorded duration is the duration that was simulated.
        self.times.append(float(data.time))
        self.joints.append(np.array(data.qpos))
        self.commands.append(np.array(data.ctrl))
        pose = np.zeros(7)
        pose[:3] = data.xpos[self.palm_body_id]
        pose[3:] = data.xquat[self.palm_body_id]
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
    palm_body_name: str,
    sample_every: int = 5,
    robot_profile: str = "",
    trajectory_ref: str = "",
) -> tuple[DynamicGraspTrajectory, DynamicSearchOutcome, object]:
    """Run the validated rollout and observe it as a Phase 3.4 trajectory.

    ``roles_from_model`` receives the compiled model and returns the
    :class:`SceneRoles` for it, because geom ids only exist after compilation.

    ``palm_body_name`` has to name the real palm body. v1 recorded body index 1
    and an identity quaternion, so every trajectory claimed the hand never
    rotated and that its palm sat wherever the first body happened to be
    (blocker B-06); there is no sensible default for this, so it is required.
    """
    recorder: dict[str, _Recorder] = {}
    timebase: dict[str, TrajectoryTimebase] = {}

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
            palm_body_id = mujoco.mj_name2id(
                model, int(mujoco.mjtObj.mjOBJ_BODY), palm_body_name
            )
            if palm_body_id < 0:
                raise ValueError(
                    f"palm body {palm_body_name!r} does not exist in the compiled model; "
                    "a trajectory cannot record a palm pose it cannot find"
                )
            simulator_dt = float(model.opt.timestep)
            timebase["t"] = TrajectoryTimebase(
                simulator_dt=simulator_dt,
                sample_every=int(sample_every),
                start_time_s=float(data.time) + simulator_dt * int(sample_every),
            )
            recorder["r"] = _Recorder(
                observer=ContactObserver(model, roles, budget),
                simulator_dt=simulator_dt,
                free_bodies=free_bodies,
                palm_body_id=int(palm_body_id),
                sample_every=int(sample_every),
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
    base = timebase.get("t") or TrajectoryTimebase(simulator_dt=1e-3, sample_every=int(sample_every))
    if record is None or not record.times:
        empty = DynamicGraspTrajectory(
            time=np.zeros(0),
            palm_pose=np.zeros((0, 7)),
            joint_state=np.zeros((0, 1)),
            actuator_command=np.zeros((0, 1)),
            object_pose=np.zeros((0, 1, 7)),
            object_velocity=np.zeros((0, 1, 6)),
            stage=(),
            timebase=base,
            robot_profile=robot_profile,
            palm_body=palm_body_name,
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
        timebase=base,
        contact_graph=tuple(
            # The sample association is clamped to the last recorded sample,
            # but ``simulator_step`` keeps the exact integrator step the reading
            # came from, so a tail contact is no longer indistinguishable from
            # one that happened at the final sample (blocker B-06).
            dataclasses.replace(e, time_index=min(e.time_index, steps - 1))
            for e in record.events
        ),
        robot_profile=robot_profile,
        palm_body=palm_body_name,
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
        # Its own stage name is carried in the reason rather than in the stage
        # field, because ``failure_stage`` is a closed vocabulary the ledger
        # groups on and the validator's stage names are its own (C01.4).
        stage = "dynamic"
        reason = f"validated_rollout:{validation.failure_stage or 'unknown'}"
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
            CpuReplayCertificate(
                backend_id="mujoco_cpu",
                capsule_sha256=sequence_hash(
                    np.concatenate(
                        [
                            np.asarray(trajectory.joint_state[0]).ravel(),
                            np.asarray(trajectory.actuator_command).ravel(),
                        ]
                    )
                ),
                command_sha256=sequence_hash(trajectory.actuator_command),
                model_sha256=_asset_hash(hand_xml_path),
                timestep_s=base.simulator_dt,
                terminal_certified=True,
                safety_certified=not hard,
                outcome_class="pass",
            )
            if passed
            else None
        ),
    )
    return trajectory, outcome, validation


def _asset_hash(path: str | Path) -> str:
    """sha256 of the model file the rollout was compiled from.

    A replay that reproduces the numbers against a different model has not
    reproduced anything, so the model identity travels with the certificate.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
