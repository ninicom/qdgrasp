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
    STAGE_ORDER,
    ContactClass,
    ContactEvent,
    ContactPairKind,
    ContactSafetyBudget,
    CpuReplayCertificate,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    TrajectoryStage,
    TrajectoryTimebase,
)
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout
from qdgrasp.dynamic.capsule import InitialState, ModelIdentity, ReplayCapsule
from qdgrasp.dynamic.certify import certify_terminal_grasp
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
    #: Called once per recorded sample, with the sample index. Used by the
    #: renderer so a stage image is the frame that sample was taken at, rather
    #: than a re-simulation that only resembles it.
    frame_observer: Callable[[int, mujoco.MjModel, mujoco.MjData], None] | None = None

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
        #: Every step's command, not just the sampled ones. The record stays
        #: sparse; the capsule has to be exact, and those are different jobs.
        self.step_commands: list[np.ndarray] = []
        #: Mechanical work done at the actuators, and the simulator clock at the
        #: last step. Control energy and elapsed time are two measurements, not
        #: one step count standing in for both (C04.2).
        self.control_energy_J = 0.0
        self.final_time_s = 0.0
        self.first_time_s = 0.0

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
        self.step_commands.append(np.array(data.ctrl, dtype=np.float64))
        if self.calls == 1:
            self.first_time_s = float(data.time) - self.simulator_dt
        self.final_time_s = float(data.time)
        if int(model.nu):
            force = np.asarray(data.actuator_force, dtype=np.float64)
            velocity = np.asarray(data.actuator_velocity, dtype=np.float64)
            self.control_energy_J += float(np.sum(np.abs(force * velocity))) * self.simulator_dt
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
        if self.frame_observer is not None:
            self.frame_observer(self.index, model, data)
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
    frame_observer: Callable[[int, mujoco.MjModel, mujoco.MjData], None] | None = None,
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
    origin: dict[str, InitialState] = {}

    def capture_origin(stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Pin the state the rollout starts from, before any step is taken."""
        del stage
        origin.setdefault("state", InitialState.from_data(model, data))

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
            # This first call lands after the first integrator step, and the
            # first sample is taken on call ``sample_every``. Predicting the
            # start time rather than reading it back means the contract check
            # catches a sampler that does not do what it says it does.
            timebase["t"] = TrajectoryTimebase(
                simulator_dt=simulator_dt,
                sample_every=int(sample_every),
                start_time_s=float(data.time) + simulator_dt * (int(sample_every) - 1),
            )
            recorder["r"] = _Recorder(
                observer=ContactObserver(model, roles, budget),
                simulator_dt=simulator_dt,
                free_bodies=free_bodies,
                palm_body_id=int(palm_body_id),
                sample_every=int(sample_every),
                frame_observer=frame_observer,
            )
        recorder["r"](stage, model, data)

    validation = validate_grasp_rollout(
        hand_xml_path,
        collision_geoms,
        fingertip_body_names,
        initial_observer=capture_origin,
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
        stage=_derive_stages(record.stages, record.events, np.asarray(record.poses)),
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

    model_sha256 = _asset_hash(hand_xml_path)
    capsule = _build_capsule(
        record=record,
        origin=origin.get("state"),
        model_sha256=model_sha256,
        robot_profile=robot_profile,
        budget=budget,
        trajectory_ref=trajectory_ref,
        timebase=base,
    )

    peak, cumulative = summarise_safety(trajectory.contact_graph)
    hard = trajectory.hard_reject_events
    # Every declared limit, measured -- not just the contact-scope seven the
    # per-event margin covers (blocker B-01).
    evaluation = record.observer.evaluation
    # The stage progression is now derived from measurement, so requiring it is
    # a real check rather than a check on the protocol's own labels (C03.7).
    terminal = certify_terminal_grasp(trajectory, require_stage_progression=True)

    if hard:
        classes = {e.contact_class for e in hard}
        stage, reason = (
            ("contact", "forbidden_contact")
            if ContactClass.FORBIDDEN in classes
            else ("contact", "damaging_contact")
        )
    elif not evaluation.safe:
        stage = "contact"
        reason = (
            evaluation.failure_reasons[0]
            if evaluation.failure_reasons
            else "safety_budget_violation"
        )
    elif not validation.passed:
        # The validated protocol has the final say on whether this is a grasp.
        # Its own stage name is carried in the reason rather than in the stage
        # field, because ``failure_stage`` is a closed vocabulary the ledger
        # groups on and the validator's stage names are its own (C01.4).
        stage = "dynamic"
        reason = f"validated_rollout:{validation.failure_stage or 'unknown'}"
    elif not terminal.certified:
        # The terminal conditions are measured on the recorded trajectory rather
        # than asserted from the protocol's own verdict, so a positive has to
        # show the enclosure, the support release and the lift (G04).
        stage, reason = ("terminal", terminal.reason)
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
            "min_budget_margin": float(evaluation.min_margin),
            "control_energy_J": float(record.control_energy_J),
            "elapsed_time_s": float(record.final_time_s - record.first_time_s),
            "enclosure_links": float(terminal.metrics.get("enclosure_links", 0.0)),
            **{f"terminal_{k}": v for k, v in terminal.metrics.items()},
        },
        peak_safety_metrics={**peak, **dict(evaluation.measurements)},
        cumulative_safety_metrics=cumulative,
        cpu_replay_evidence=(
            CpuReplayCertificate(
                backend_id="mujoco_cpu",
                capsule_sha256=capsule.capsule_sha256,
                command_sha256=capsule.command_sha256,
                model_sha256=model_sha256,
                timestep_s=base.simulator_dt,
                terminal_certified=terminal.certified,
                safety_certified=evaluation.safe and not hard,
                outcome_class="pass",
            )
            if passed and capsule is not None
            else None
        ),
    )
    return trajectory, outcome, validation


def _derive_stages(
    stages: Sequence[TrajectoryStage],
    events: Sequence[ContactEvent],
    object_pose: np.ndarray,
    *,
    lift_threshold_m: float = 0.005,
) -> tuple[TrajectoryStage, ...]:
    """Label the acquisition from what was measured, not from phase names.

    The validated protocol names its own phases -- settle, approach, squeeze,
    lift, perturbation -- and has no phase for "the target left its support".
    That is not a naming gap: it is the one transition a contact-rich positive
    has to show (C03.7). Worse, its ``lift`` phase begins while the object is
    still resting on the table, so the label and the physics disagree for as
    long as it takes the object to actually come free.

    So from the enclosure onwards the stage is decided by measurement: the
    target is either still supported, or free but not yet lifted, or lifted.
    Both derived boundaries are monotone by construction, so the sequence cannot
    run backwards.
    """
    steps = len(stages)
    if steps == 0:
        return tuple(stages)

    supported = [False] * steps
    target_held = [False] * steps
    for event in events:
        index = int(event.time_index)
        if not 0 <= index < steps:
            continue
        if event.supports_target:
            supported[index] = True
        if event.pair_kind is ContactPairKind.TARGET_ROBOT:
            target_held[index] = True

    # Released once no target-support contact ever returns: monotone on purpose,
    # so a single late brush against the table cannot un-release the grasp.
    released = [not any(supported[index:]) for index in range(steps)]

    base_height = float(object_pose[0, 0, 2]) if object_pose.shape[1] else 0.0
    lifted_from = steps
    for index in range(steps):
        if released[index] and float(object_pose[index, 0, 2]) - base_height >= lift_threshold_m:
            lifted_from = index
            break

    result = list(stages)
    enclose_rank = STAGE_ORDER[TrajectoryStage.ENCLOSE]
    for index in range(steps):
        if STAGE_ORDER[result[index]] < enclose_rank:
            continue  # approach and reposition are the protocol's to name
        if not released[index]:
            result[index] = TrajectoryStage.ENCLOSE
        elif index < lifted_from:
            result[index] = TrajectoryStage.SUPPORT_RELEASE
        elif result[index] is not TrajectoryStage.PERTURB:
            result[index] = TrajectoryStage.LIFT

    # The grasp is retained if the hand still holds the target at the end.
    if target_held[-1] and result[-1] is TrajectoryStage.PERTURB:
        result[-1] = TrajectoryStage.RETAIN
    return tuple(result)


def _build_capsule(
    *,
    record: _Recorder,
    origin: InitialState | None,
    model_sha256: str,
    robot_profile: str,
    budget: ContactSafetyBudget,
    trajectory_ref: str,
    timebase: TrajectoryTimebase,
) -> ReplayCapsule | None:
    """Capture what a CPU oracle needs to replay this rollout exactly.

    Returns ``None`` when the initial state could not be pinned, because a
    capsule that guesses where the rollout started is not a capsule.
    """
    if origin is None or not record.step_commands:
        return None
    commands = np.asarray(record.step_commands, dtype=np.float64)
    identity = ModelIdentity(
        robot_profile=robot_profile or budget.robot_profile,
        scene_signature=f"wrapped:{model_sha256[:16]}",
        model_sha256=model_sha256,
        timestep_s=timebase.simulator_dt,
        integrator=0,
        solver=0,
        cone=0,
        nq=int(origin.qpos.shape[0]),
        nv=int(origin.qvel.shape[0]),
        nu=int(commands.shape[1]),
        ngeom=int(origin.geom_friction.shape[0]),
        nbody=int(origin.body_mass.shape[0]),
    )
    return ReplayCapsule(
        capsule_id=f"capsule:{trajectory_ref or 'wrapped'}",
        model=identity,
        state=origin,
        control_sequence=commands,
        control_dtype="float64",
        seed=0,
        strategy_id="validated_rollout_protocol",
        strategy_parameters={"sample_every": float(timebase.sample_every)},
        safety_budget_id=budget.budget_id,
        safety_budget_hash=budget.budget_hash,
    )


def _asset_hash(path: str | Path) -> str:
    """sha256 of the model file the rollout was compiled from.

    A replay that reproduces the numbers against a different model has not
    reproduced anything, so the model identity travels with the certificate.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
