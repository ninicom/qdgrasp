"""Static-seeded contact rollout (P3.4-08).

Takes a static pose that Phase 3.3 blocked or nearly admitted, then lets the
target and the scene react physically through approach and squeeze instead of
freezing them. It is the cheapest way to measure what dropping the frozen-object
assumption actually buys, which is why the plan makes it the first strategy.

The rollout records; it does not repair. A trajectory that violates the safety
budget is returned as a negative with its reason, because a critic trained later
needs the failures as much as the successes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    ContactClass,
    ContactEvent,
    ContactSafetyBudget,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
)
from qdgrasp.dynamic.primitives import Primitive, PrimitiveSequenceController
from qdgrasp.dynamic.safety import ContactObserver, SceneRoles, summarise_safety


@dataclasses.dataclass(frozen=True)
class RolloutLimits:
    """Terminal thresholds, pinned before a search rather than after.

    Changing any of these after looking at results is exactly the move the plan
    forbids, so they travel as one frozen object that the manifest can hash.
    """

    min_lift_m: float = 0.03
    max_non_target_translation_m: float = 0.01
    #: Distinct robot links that must touch the target for enclosure to count.
    min_enclosure_links: int = 2
    #: Ceiling on per-step target displacement, as a multiple of the distance
    #: its own reported velocity could produce. Catches teleporting.
    max_velocity_consistency_ratio: float = 3.0


@dataclasses.dataclass(frozen=True)
class SeedPose:
    """The static candidate a rollout starts from."""

    qpos: np.ndarray
    ctrl: np.ndarray
    source_candidate_id: str = ""
    static_verdict: str = "blocked"


def _actuator_command(
    model: mujoco.MjModel,
    grip: float,
    wrist_velocity: np.ndarray,
    base_ctrl: np.ndarray,
    control_dt: float,
) -> np.ndarray:
    """Map a primitive's grip and wrist velocity onto actuator commands.

    Deliberately simple: the grip interpolates every actuator between its
    control-range bounds, and the wrist velocity advances the first three
    actuators. A richer mapping is a strategy concern, not a rollout concern.

    The velocity is **integrated over the control step** before it is added to a
    position target. Adding a velocity directly to a position command compounds
    every step into a runaway setpoint, which launches the target across the
    scene at a speed the velocity-consistency check still considers legitimate.
    Every command is then clamped to the actuator's own range.
    """
    command = np.array(base_ctrl, dtype=np.float64, copy=True)
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    for index in range(int(model.nu)):
        if limited[index]:
            low, high = ctrlrange[index]
            command[index] = low + float(np.clip(grip, 0.0, 1.0)) * (high - low)
    for axis in range(min(3, int(model.nu))):
        command[axis] = command[axis] + float(wrist_velocity[axis]) * control_dt
    for index in range(int(model.nu)):
        if limited[index]:
            command[index] = float(
                np.clip(command[index], ctrlrange[index][0], ctrlrange[index][1])
            )
    return command


def run_static_seeded_rollout(
    model: mujoco.MjModel,
    *,
    roles: SceneRoles,
    budget: ContactSafetyBudget,
    seed: SeedPose,
    primitives: Sequence[Primitive],
    horizon: int,
    control_dt: float,
    limits: RolloutLimits | None = None,
    trajectory_ref: str = "",
) -> tuple[DynamicGraspTrajectory, DynamicSearchOutcome]:
    """Roll one seeded candidate forward and judge it against the budget."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    limits = limits or RolloutLimits()
    observer = ContactObserver(model, roles, budget)
    controller = PrimitiveSequenceController(primitives, control_dt)

    data = mujoco.MjData(model)
    if seed.qpos.shape[0] != int(model.nq):
        raise ValueError(
            f"seed qpos has {seed.qpos.shape[0]} entries, model expects {model.nq}"
        )
    data.qpos[:] = seed.qpos
    mujoco.mj_forward(model, data)

    target_bodies = sorted({int(model.geom_bodyid[g]) for g in roles.target_geoms})
    non_target_bodies = sorted({int(model.geom_bodyid[g]) for g in roles.non_target_geoms})
    free_bodies = target_bodies + non_target_bodies

    steps_per_control = max(1, round(control_dt / float(model.opt.timestep)))

    times, palm_poses, joints, commands = [], [], [], []
    object_poses, object_velocities, stages = [], [], []
    contact_graph: list[ContactEvent] = []
    start_target_z = float(data.xpos[target_bodies[0]][2]) if target_bodies else 0.0
    non_target_start = {b: np.array(data.xpos[b]) for b in non_target_bodies}
    teleported = False
    enclosure_links: set[str] = set()
    saw_support_assist = False

    for index in range(horizon):
        if controller.finished:
            break
        events = observer.observe(data, time_index=index, dt=control_dt)
        step = controller.step(events)
        command = _actuator_command(
            model, step.grip, step.wrist_velocity, data.ctrl, control_dt
        )

        previous = np.array([data.xpos[b] for b in free_bodies]) if free_bodies else None
        data.ctrl[:] = command
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)

        if previous is not None:
            current = np.array([data.xpos[b] for b in free_bodies])
            speed = np.zeros(len(free_bodies))
            for slot, body in enumerate(free_bodies):
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    model, data, int(mujoco.mjtObj.mjOBJ_BODY), body, velocity, 0
                )
                speed[slot] = float(np.linalg.norm(velocity[3:]))
            travelled = np.linalg.norm(current - previous, axis=1)
            allowed = speed * control_dt * limits.max_velocity_consistency_ratio + 1e-6
            if np.any(travelled > allowed):
                teleported = True

        post_events = observer.observe(data, time_index=index, dt=control_dt)
        contact_graph.extend(post_events)
        for event in post_events:
            if event.contact_class is ContactClass.SUPPORT_ASSISTED:
                saw_support_assist = True
            if event.contact_class in (
                ContactClass.TARGET_INTENTIONAL,
                ContactClass.DAMAGING,
            ):
                enclosure_links.update({event.body_a, event.body_b})

        times.append(index * control_dt)
        palm = np.zeros(7)
        palm[3] = 1.0
        palm[:3] = data.xpos[1] if int(model.nbody) > 1 else 0.0
        palm_poses.append(palm)
        joints.append(np.array(data.qpos[: int(model.nq)]))
        commands.append(command)
        poses = np.zeros((max(1, len(free_bodies)), 7))
        velocities = np.zeros((max(1, len(free_bodies)), 6))
        for slot, body in enumerate(free_bodies):
            poses[slot, :3] = data.xpos[body]
            poses[slot, 3:] = data.xquat[body]
            velocities[slot] = data.cvel[body]
        if not free_bodies:
            poses[0, 3] = 1.0
        object_poses.append(poses)
        object_velocities.append(velocities)
        stages.append(step.stage)

    trajectory = DynamicGraspTrajectory(
        time=np.asarray(times, dtype=float),
        palm_pose=np.asarray(palm_poses, dtype=float),
        joint_state=np.asarray(joints, dtype=float),
        actuator_command=np.asarray(commands, dtype=float),
        object_pose=np.asarray(object_poses, dtype=float),
        object_velocity=np.asarray(object_velocities, dtype=float),
        stage=tuple(stages),
        contact_graph=tuple(contact_graph),
    )

    peak, cumulative = summarise_safety(contact_graph)
    lift = (
        float(data.xpos[target_bodies[0]][2]) - start_target_z if target_bodies else 0.0
    )
    disturbance = max(
        (float(np.linalg.norm(np.array(data.xpos[b]) - non_target_start[b]))
         for b in non_target_bodies),
        default=0.0,
    )
    enclosure_count = max(0, len(enclosure_links) - 1)

    failure_stage, failure_reason = _judge(
        trajectory=trajectory,
        teleported=teleported,
        lift=lift,
        disturbance=disturbance,
        enclosure_count=enclosure_count,
        saw_support_assist=saw_support_assist,
        limits=limits,
    )
    passed = failure_reason == "none"

    outcome = DynamicSearchOutcome(
        trajectory_ref=trajectory_ref or seed.source_candidate_id or "static_seeded",
        passed=passed,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        objective_terms={
            "lift_m": lift,
            "non_target_disturbance_m": disturbance,
            "enclosure_links": float(enclosure_count),
            "steps": float(trajectory.num_steps),
        },
        peak_safety_metrics=peak,
        cumulative_safety_metrics=cumulative,
        # A CPU rollout is its own oracle, so the replay evidence is this run.
        cpu_replay_evidence=(
            {"backend": "mujoco_cpu", "confirmed": True, "steps": trajectory.num_steps}
            if passed
            else {}
        ),
    )
    return trajectory, outcome


def _judge(
    *,
    trajectory: DynamicGraspTrajectory,
    teleported: bool,
    lift: float,
    disturbance: float,
    enclosure_count: int,
    saw_support_assist: bool,
    limits: RolloutLimits,
) -> tuple[str, str]:
    """Apply the terminal conditions of plan section 4.3, in order."""
    if trajectory.num_steps == 0:
        return ("rollout", "empty_trajectory")
    if teleported:
        return ("acquisition", "target_teleported")

    hard = trajectory.hard_reject_events
    if hard:
        classes = {e.contact_class.value for e in hard}
        if ContactClass.FORBIDDEN.value in classes:
            return ("contact", "forbidden_contact")
        return ("contact", "damaging_contact")

    if disturbance > limits.max_non_target_translation_m:
        return ("scene", "non_target_disturbance")
    if not saw_support_assist:
        return ("acquisition", "no_environmental_assistance")
    if enclosure_count < limits.min_enclosure_links:
        return ("enclose", "insufficient_enclosure")
    if lift < limits.min_lift_m:
        return ("lift", "insufficient_lift")
    return ("none", "none")
