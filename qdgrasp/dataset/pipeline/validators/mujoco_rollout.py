from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.pipeline.contracts import DynamicValidation
from qdgrasp.dataset.pipeline.observers.contact_load import extract_contact_loads
from qdgrasp.dataset.pipeline.validators.dynamic_predicate import (
    DynamicPredicateEvidence,
    RolloutProtocol,
    evaluate_dynamic_success,
)
from qdgrasp.objects.schema import SubGeomSpec


@dataclass(frozen=True)
class RolloutSceneObject:
    """A physical non-target object compiled into the grasp rollout."""

    object_id: str
    collision_geoms: Sequence[SubGeomSpec]
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 0.1


def _add_rollout_object(
    spec: mujoco.MjSpec,
    *,
    body_name: str,
    geom_prefix: str,
    collision_geoms: Sequence[SubGeomSpec],
    pos: Sequence[float],
    quat: Sequence[float],
    mass: float,
) -> None:
    if not collision_geoms:
        raise ConfigError(f"rollout object {body_name} has no collision geoms")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ConfigError(f"rollout object {body_name} mass must be finite and positive")
    position = np.asarray(pos, dtype=np.float64)
    quaternion = np.asarray(quat, dtype=np.float64)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(np.concatenate([position, quaternion])))
    ):
        raise ConfigError(f"rollout object {body_name} pose is invalid")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= np.finfo(np.float64).eps:
        raise ConfigError(f"rollout object {body_name} quaternion has zero norm")
    object_body = spec.worldbody.add_body(
        name=body_name,
        pos=position.tolist(),
        quat=(quaternion / quaternion_norm).tolist(),
    )
    object_body.add_freejoint(name=f"{body_name}::freejoint")
    for index, geom in enumerate(collision_geoms):
        geom_type = getattr(mujoco.mjtGeom, f"mjGEOM_{geom.type.upper()}")
        object_body.add_geom(
            name=f"{geom_prefix}{index}",
            type=geom_type,
            size=[float(value) for value in geom.size],
            pos=[float(value) for value in geom.pos],
            quat=[float(value) for value in geom.quat],
            mass=float(mass / len(collision_geoms)),
            condim=4,
            friction=[1.0, 0.005, 0.0001],
            rgba=[0.8, 0.3, 0.3, 1.0],
        )


def build_rollout_scene_model(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    *,
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
    non_target_objects: Sequence[RolloutSceneObject] = (),
) -> mujoco.MjModel:
    """Build a compiled MuJoCo model containing the hand and the procedural object geoms."""
    hand_p = Path(hand_xml_path).resolve()
    if not hand_p.is_file():
        raise ConfigError(f"hand XML file not found: {hand_p}")

    spec = mujoco.MjSpec.from_file(str(hand_p))
    if len(spec.worldbody.bodies) == 0:
        raise ConfigError(f"No bodies found in hand model: {hand_p}")

    hand_root = spec.worldbody.bodies[0]
    has_free = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in hand_root.joints)
    if not has_free:
        hand_root.add_freejoint(name="hand_freejoint")

    spec.worldbody.add_body(name="hand_mocap", mocap=True)
    mocap_weld = spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name="mocap_weld",
        name1="hand_mocap",
        name2=hand_root.name,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        # Keep the weld softer than the simulation timestep scale.  A very
        # stiff weld amplifies even a small initialization mismatch into an
        # unstable free-joint acceleration.
        solref=[0.01, 1.0],
        solimp=[0.99, 0.999, 0.001, 0.5, 2],
    )
    # A body-body weld otherwise inherits the bodies' reference-pose offset
    # during compilation.  At rollout initialization both bodies are placed
    # at the same world pose, so that inherited offset immediately violates
    # the constraint.  Declare the intended identity relative pose explicitly:
    # anchor1, anchor2, relative quaternion (wxyz), torque scale.
    mocap_weld.data = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    _add_rollout_object(
        spec,
        body_name="target_object",
        geom_prefix="object_subgeom_",
        collision_geoms=collision_geoms,
        pos=object_pos,
        quat=(1.0, 0.0, 0.0, 0.0),
        mass=object_mass,
    )
    reserved_names = {"target_object", "hand_mocap", "floor"}
    object_ids = [item.object_id for item in non_target_objects]
    if len(object_ids) != len(set(object_ids)) or any(
        not object_id or object_id in reserved_names for object_id in object_ids
    ):
        raise ConfigError("non-target rollout object IDs must be unique and non-reserved")
    for item in non_target_objects:
        _add_rollout_object(
            spec,
            body_name=item.object_id,
            geom_prefix=f"scene_object::{item.object_id}::geom::",
            collision_geoms=item.collision_geoms,
            pos=item.pos,
            quat=item.quat,
            mass=item.mass,
        )

    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.1],
        pos=[0.0, 0.0, 0.0],
        rgba=[0.9, 0.9, 0.9, 1.0],
    )

    try:
        return spec.compile()
    except Exception as exc:
        raise ConfigError(f"failed to compile labeled scene for {hand_p}: {exc}") from exc


def smoothstep(t: float) -> float:
    """Standard smoothstep polynomial: 3*t^2 - 2*t^3 for t in [0, 1]."""
    t_c = np.clip(t, 0.0, 1.0)
    return float(3.0 * t_c**2 - 2.0 * t_c**3)


def validate_grasp_rollout(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    fingertip_body_names: Sequence[str],
    *,
    palm_pos: tuple[float, float, float] = (0.0, 0.0, 0.1),
    palm_rot: np.ndarray | None = None,  # 3x3 rotation matrix or quaternion
    joint_targets: Mapping[str, float] | None = None,
    initial_joint_targets: Mapping[str, float] | None = None,
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
    squeeze_steps: int = 150,
    lift_steps: int = 150,
    lift_height: float = 0.05,
    perturbation_steps: int = 50,
    perturbation_wrench: np.ndarray | None = None,  # [6] force & torque
    max_allowed_penetration: float = 0.002,
    min_active_fingers: int = 2,
    pregrasp_distance: float = 0.03,
    expected_fingertip_positions: np.ndarray | None = None,
    fingertip_local_offsets: np.ndarray | None = None,
    active_fingers: np.ndarray | None = None,
    desired_fingertip_displacement: np.ndarray | None = None,
    contact_joint_targets: Mapping[str, float] | None = None,
    rollout_protocol: RolloutProtocol | None = None,
    non_target_objects: Sequence[RolloutSceneObject] = (),
    initial_observer: Callable[[str, mujoco.MjModel, mujoco.MjData], None] | None = None,
    stage_observer: Callable[[str, mujoco.MjModel, mujoco.MjData], None] | None = None,
    step_observer: Callable[[str, mujoco.MjModel, mujoco.MjData], None] | None = None,
) -> DynamicValidation:
    """
    Executes a multi-stage physical rollout:
    1. Squeeze
    2. Smoothstep Lift
    3. Wrench Perturbation via xfrc_applied

    Returns a DynamicValidation contract instance.
    """
    model = build_rollout_scene_model(
        hand_xml_path=hand_xml_path,
        collision_geoms=collision_geoms,
        object_pos=object_pos,
        object_mass=object_mass,
        non_target_objects=non_target_objects,
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    protocol = rollout_protocol or RolloutProtocol()
    protocol_error = protocol.validation_error()
    if protocol_error is not None:
        return DynamicValidation(
            trajectory_metrics={"protocol_error": protocol_error},
            per_finger_loads=np.zeros((len(fingertip_body_names), 6), dtype=np.float64),
            failure_stage="controller_protocol",
            passed=False,
        )

    fingertip_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in fingertip_body_names]
    missing_tips = [name for name, body_id in zip(fingertip_body_names, fingertip_body_ids) if body_id < 0]
    if missing_tips:
        raise ConfigError(f"fingertip bodies are absent from rollout model: {missing_tips}")

    expected_tips = None
    if expected_fingertip_positions is not None:
        expected_tips = np.asarray(expected_fingertip_positions, dtype=np.float64)
        if expected_tips.shape != (len(fingertip_body_names), 3):
            raise ConfigError(
                "expected_fingertip_positions must have shape "
                f"({len(fingertip_body_names)}, 3), got {expected_tips.shape}"
            )
    local_tip_offsets = np.zeros((len(fingertip_body_names), 3), dtype=np.float64)
    if fingertip_local_offsets is not None:
        local_tip_offsets = np.asarray(fingertip_local_offsets, dtype=np.float64)
        if local_tip_offsets.shape != (len(fingertip_body_names), 3):
            raise ConfigError(
                "fingertip_local_offsets must have shape "
                f"({len(fingertip_body_names)}, 3), got {local_tip_offsets.shape}"
            )
    task_active_fingers = (
        np.ones(len(fingertip_body_names), dtype=bool)
        if active_fingers is None
        else np.asarray(active_fingers, dtype=bool)
    )
    if task_active_fingers.shape != (len(fingertip_body_names),):
        raise ConfigError(f"active_fingers must have shape ({len(fingertip_body_names)},)")

    palm_candidates = [
        b_id
        for b_id in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or "").lower().endswith("palm")
    ]
    palm_id = palm_candidates[0] if palm_candidates else -1
    if palm_id < 0:
        raise ConfigError("could not identify palm body in model")

    root_id = palm_id
    while int(model.body_parentid[root_id]) != 0:
        root_id = int(model.body_parentid[root_id])

    # Establish the complete articulated initial shape before measuring the
    # root->palm transform.  Shadow has two wrist joints above the palm; using
    # the q=0 transform and applying wrist targets afterwards rotates the palm
    # away from the requested pose even while the free root tracks mocap.
    if initial_joint_targets:
        for joint_name, value in initial_joint_targets.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0 or int(model.jnt_type[joint_id]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ConfigError(f"initial joint target refers to unsupported joint: {joint_name}")
            data.qpos[model.jnt_qposadr[joint_id]] = float(value)

    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mocap")
    if mocap_body_id < 0:
        raise ConfigError("could not identify mocap body in model")
    mocap_idx = model.body_mocapid[mocap_body_id]

    # Determine the complete root->palm transform.  Translation-only alignment
    # is wrong whenever the requested palm rotation is not identity.
    mujoco.mj_forward(model, data)
    root_rot_initial = np.array(data.xmat[root_id]).reshape(3, 3)
    palm_rot_initial = np.array(data.xmat[palm_id]).reshape(3, 3)
    root_to_palm_rot = root_rot_initial.T @ palm_rot_initial
    root_to_palm_pos = root_rot_initial.T @ (np.array(data.xpos[palm_id]) - np.array(data.xpos[root_id]))

    requested_palm_rot = np.eye(3, dtype=np.float64)
    if palm_rot is not None:
        if palm_rot.shape == (3, 3):
            requested_palm_rot = np.asarray(palm_rot, dtype=np.float64)
        elif palm_rot.shape == (4,):
            quat = np.asarray(palm_rot, dtype=np.float64)
            requested_palm_rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        else:
            raise ConfigError("palm_rot must have shape (3, 3) or (4,)")

    root_start_rot = requested_palm_rot @ root_to_palm_rot.T
    root_target_pos = np.asarray(palm_pos, dtype=np.float64) - root_start_rot @ root_to_palm_pos
    outward = np.asarray(palm_pos, dtype=np.float64) - np.asarray(object_pos, dtype=np.float64)
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm < 1e-8:
        raise ConfigError("target palm position coincides with object center")
    root_pregrasp_pos = root_target_pos + pregrasp_distance * outward / outward_norm

    jnt_id = model.body_jntadr[root_id]
    if jnt_id < 0 or int(model.jnt_type[jnt_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ConfigError(f"Root body {root_id} must have a freejoint for mocap-weld control.")

    qpos_adr = model.jnt_qposadr[jnt_id]
    data.qpos[qpos_adr : qpos_adr + 3] = root_pregrasp_pos
    quat_xyzw = Rotation.from_matrix(root_start_rot).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
    data.mocap_quat[mocap_idx] = quat_wxyz

    data.mocap_pos[mocap_idx][:3] = root_pregrasp_pos

    # Identify articulated hand joints and actuators in compiled scene model
    hand_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        for j_id in range(model.njnt)
        if int(model.jnt_type[j_id]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        and not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or "").startswith("object_")
    ]
    hand_actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_id) for a_id in range(model.nu)]

    actuated_damping = []
    for joint_name in hand_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dof_address = int(model.jnt_dofadr[joint_id])
        actuated_damping.append(float(model.dof_damping[dof_address]))
    if not actuated_damping or not np.all(np.isfinite(actuated_damping)) or np.any(np.asarray(actuated_damping) <= 0.0):
        return DynamicValidation(
            trajectory_metrics={
                "protocol_error": "nonpositive_actuated_damping",
                "minimum_actuated_damping": min(actuated_damping, default=float("nan")),
            },
            per_finger_loads=np.zeros((len(fingertip_body_names), 6), dtype=np.float64),
            failure_stage="controller_protocol",
            passed=False,
        )

    has_tendon = any(int(model.actuator_trntype[a_id]) == int(mujoco.mjtTrn.mjTRN_TENDON) for a_id in range(model.nu))

    from qdgrasp.robot.transmission.direct import DirectJointTransmission
    from qdgrasp.robot.transmission.fixed_tendon import FixedTendonTransmission

    if has_tendon:
        tm = FixedTendonTransmission(hand_joint_names, hand_actuator_names, model)
    else:
        tm = DirectJointTransmission(hand_joint_names, hand_actuator_names, model)

    mujoco.mj_forward(model, data)
    initial_trans_state = tm.extract_state(model, data)
    q_init = initial_trans_state.joint_position
    q_target = q_init.copy()

    command_plan = None
    task_command_requested = desired_fingertip_displacement is not None
    if task_command_requested:
        from qdgrasp.robot.transmission.command import plan_controllable_task_command

        task_active = task_active_fingers
        desired_tip_delta = np.asarray(desired_fingertip_displacement, dtype=np.float64)
        if task_active.shape != (len(fingertip_body_names),) or desired_tip_delta.shape != (
            len(fingertip_body_names),
            3,
        ):
            raise ConfigError("task command active_fingers/displacement must have shapes [K] and [K,3]")
        active_count = int(np.sum(task_active))
        if active_count < min_active_fingers:
            return DynamicValidation(
                trajectory_metrics={
                    "active_finger_count": float(active_count),
                    "minimum_active_fingers": float(min_active_fingers),
                },
                per_finger_loads=np.zeros((len(fingertip_body_names), 6), dtype=np.float64),
                failure_stage="insufficient_active_fingers",
                passed=False,
            )

        task_rows = []
        desired_rows = []
        for tip_index in np.where(task_active)[0]:
            body_id = fingertip_body_ids[int(tip_index)]
            body_rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
            contact_point = np.asarray(data.xpos[body_id]) + (body_rotation @ local_tip_offsets[int(tip_index)])
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                contact_point,
                body_id,
            )
            task_rows.append(
                np.stack(
                    [
                        jacobian_position[
                            :, int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)])
                        ]
                        for joint_name in hand_joint_names
                    ],
                    axis=1,
                )
            )
            desired_rows.append(desired_tip_delta[int(tip_index)])

        task_jacobian = np.concatenate(task_rows, axis=0)
        task_delta = np.concatenate(desired_rows, axis=0)
        joint_limits = np.empty((len(hand_joint_names), 2), dtype=np.float64)
        for joint_index, joint_name in enumerate(hand_joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_limits[joint_index] = (
                np.asarray(model.jnt_range[joint_id], dtype=np.float64)
                if bool(model.jnt_limited[joint_id])
                else np.array([-np.inf, np.inf])
            )
        q_contact = q_init.copy()
        if contact_joint_targets:
            for index, joint_name in enumerate(hand_joint_names):
                if joint_name in contact_joint_targets:
                    q_contact[index] = float(contact_joint_targets[joint_name])

        command_plan = plan_controllable_task_command(
            current_state=initial_trans_state,
            task_jacobian=task_jacobian,
            desired_task_delta=task_delta,
            joint_limits=joint_limits,
            actuator_ctrlrange=tm.actuator_ctrlrange,
            active_fingers=task_active,
            q_contact=q_contact,
        )
        if command_plan.rejection_reason != "converged":
            return DynamicValidation(
                trajectory_metrics={
                    "transmission_rank": float(tm.rank),
                    "joint_state_dimensions": float(tm.num_joints),
                    "control_dimensions": float(tm.num_actuators),
                    "task_residual": float(command_plan.task_residual),
                    "nullspace_residual": float(command_plan.nullspace_residual),
                    "actuator_saturation_count": float(np.sum(command_plan.saturated)),
                },
                per_finger_loads=np.zeros((len(fingertip_body_names), 6), dtype=np.float64),
                failure_stage=command_plan.rejection_reason,
                passed=False,
            )
        q_target = command_plan.q_preload.copy()

    if joint_targets and not task_command_requested:
        for idx, j_name in enumerate(hand_joint_names):
            if j_name in joint_targets:
                q_target[idx] = float(joint_targets[j_name])

    dq_desired = q_target - q_init
    cmd = None if command_plan is not None else tm.project_joint_delta(dq_desired, initial_trans_state)

    if cmd is not None and cmd.reason in ("nullspace_rejection", "actuator_saturation"):
        return DynamicValidation(
            trajectory_metrics={
                "transmission_rank": float(tm.rank),
                "joint_state_dimensions": float(tm.num_joints),
                "control_dimensions": float(tm.num_actuators),
                "controllable_residual": float(cmd.controllable_residual),
                "nullspace_residual": float(cmd.nullspace_residual),
                "actuator_saturation_count": float(np.sum(cmd.saturated)),
            },
            per_finger_loads=np.zeros((len(fingertip_body_names), 6), dtype=np.float64),
            failure_stage=("underactuated_targets" if cmd.reason == "nullspace_rejection" else "actuator_saturation"),
            passed=False,
        )

    start_controls = initial_trans_state.actuator_coordinate.copy()
    target_controls = command_plan.control_target.copy() if command_plan is not None else cmd.control_target.copy()

    ctrl_mins = tm.actuator_ctrlrange[:, 0]
    ctrl_maxs = tm.actuator_ctrlrange[:, 1]
    data.ctrl[: model.nu] = np.clip(start_controls, ctrl_mins, ctrl_maxs)

    # Object & floor geom IDs
    object_geom_ids = {
        g_id
        for g_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g_id) or "").startswith("object_subgeom_")
    }
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")

    # Calibrate the contact observer on the same compiled scene with the hand
    # translated far from the object.  This measures numerical force noise
    # without changing timestep, gains, solver, or object/floor contacts.
    noise_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, noise_data)
    noise_data.qpos[:] = data.qpos
    noise_data.qpos[qpos_adr : qpos_adr + 3] = np.array([10.0, 10.0, 10.0])
    noise_data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
    noise_data.mocap_pos[mocap_idx] = np.array([10.0, 10.0, 10.0])
    noise_data.mocap_quat[mocap_idx] = quat_wxyz
    noise_data.ctrl[: model.nu] = np.clip(start_controls, ctrl_mins, ctrl_maxs)
    contact_noise_floor = 0.0
    for _ in range(4):
        mujoco.mj_step(model, noise_data)
        noise_loads = extract_contact_loads(
            model,
            noise_data,
            object_geom_ids,
            fingertip_body_names,
        )
        contact_noise_floor = max(
            contact_noise_floor,
            float(np.max(noise_loads["per_finger_f_normal"], initial=0.0)),
        )
    contact_force_threshold = max(1e-6, 10.0 * contact_noise_floor)
    dt = float(model.opt.timestep)

    def summarize_contact_window(samples: list[dict[str, object]]) -> dict[str, object]:
        if not samples:
            return {
                "sustained_count": 0,
                "duty_cycle": np.zeros(len(fingertip_body_names)),
                "normal_impulse": np.zeros(len(fingertip_body_names)),
                "palm_support": False,
                "max_cone_violation": float("inf"),
            }
        normal_forces = np.stack([np.asarray(sample["per_finger_f_normal"], dtype=np.float64) for sample in samples])
        duty_cycle = np.mean(normal_forces > contact_force_threshold, axis=0)
        normal_impulse = np.sum(normal_forces, axis=0) * dt
        minimum_impulse = contact_force_threshold * dt * len(samples) * protocol.minimum_contact_impulse_ratio
        sustained = (
            (duty_cycle >= protocol.minimum_contact_duty_cycle)
            & (normal_impulse >= minimum_impulse)
            & task_active_fingers
        )
        return {
            "sustained_count": int(np.sum(sustained)),
            "duty_cycle": duty_cycle,
            "normal_impulse": normal_impulse,
            "palm_support": any(bool(sample["has_palm_contact"]) for sample in samples),
            "max_cone_violation": max(float(np.max(sample["cone_violations"])) for sample in samples),
        }

    def get_max_penetration() -> float:
        max_pen = 0.0
        for idx in range(int(data.ncon)):
            c = data.contact[idx]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in object_geom_ids or g2 in object_geom_ids) and floor_geom_id not in (g1, g2) and c.dist < 0:
                max_pen = max(max_pen, abs(float(c.dist)))
        return max_pen

    def object_has_floor_support() -> bool:
        return any(
            floor_geom_id in (int(data.contact[idx].geom1), int(data.contact[idx].geom2))
            and (int(data.contact[idx].geom1) in object_geom_ids or int(data.contact[idx].geom2) in object_geom_ids)
            for idx in range(int(data.ncon))
        )

    def tracking_metrics() -> dict[str, float]:
        current_state = tm.extract_state(model, data)
        current_q = current_state.joint_position
        current_coords = current_state.actuator_coordinate

        joint_errors = np.abs(current_q - q_target)
        actuator_errors = np.abs(current_coords - target_controls)
        joint_ranges = np.empty(len(hand_joint_names), dtype=np.float64)
        for index, joint_name in enumerate(hand_joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_ranges[index] = (
                float(np.diff(model.jnt_range[joint_id])[0])
                if bool(model.jnt_limited[joint_id])
                else max(abs(float(q_target[index] - q_init[index])), 1.0)
            )
        control_ranges = ctrl_maxs - ctrl_mins
        control_scales = np.where(
            np.isfinite(control_ranges) & (control_ranges > 1e-12),
            control_ranges,
            np.maximum(np.abs(target_controls - start_controls), 1.0),
        )

        mocap_quat_wxyz = np.asarray(data.mocap_quat[mocap_idx], dtype=np.float64)
        mocap_rot = Rotation.from_quat(
            [
                mocap_quat_wxyz[1],
                mocap_quat_wxyz[2],
                mocap_quat_wxyz[3],
                mocap_quat_wxyz[0],
            ]
        ).as_matrix()
        commanded_palm_pos = data.mocap_pos[mocap_idx] + mocap_rot @ root_to_palm_pos
        commanded_palm_rot = mocap_rot @ root_to_palm_rot
        command_controllable_residual = (
            float(command_plan.task_residual) if command_plan is not None else float(cmd.controllable_residual)
        )
        command_nullspace_residual = (
            float(command_plan.nullspace_residual) if command_plan is not None else float(cmd.nullspace_residual)
        )
        command_saturation_count = (
            float(np.sum(command_plan.saturated)) if command_plan is not None else float(np.sum(cmd.saturated))
        )
        metrics = {
            "transmission_rank": float(tm.rank),
            "joint_state_dimensions": float(tm.num_joints),
            "control_dimensions": float(tm.num_actuators),
            "controllable_residual": command_controllable_residual,
            "task_residual": command_controllable_residual,
            "nullspace_residual": command_nullspace_residual,
            "actuator_saturation_count": command_saturation_count,
            "max_joint_tracking_error": float(np.max(joint_errors) if len(joint_errors) > 0 else 0.0),
            "max_actuator_coordinate_error": float(np.max(actuator_errors) if len(actuator_errors) > 0 else 0.0),
            "max_normalized_joint_tracking_error": float(
                np.max(joint_errors / np.maximum(joint_ranges, 1e-12), initial=0.0)
            ),
            "max_normalized_actuator_tracking_error": float(np.max(actuator_errors / control_scales, initial=0.0)),
            "palm_position_tracking_error": float(np.linalg.norm(np.asarray(data.xpos[palm_id]) - commanded_palm_pos)),
            "root_mocap_position_error": float(
                np.linalg.norm(np.asarray(data.xpos[root_id]) - data.mocap_pos[mocap_idx])
            ),
        }
        actual_palm_rot = np.asarray(data.xmat[palm_id]).reshape(3, 3)
        rotation_cosine = np.clip(
            (np.trace(actual_palm_rot.T @ commanded_palm_rot) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
        metrics["palm_rotation_tracking_error"] = float(np.arccos(rotation_cosine))
        if expected_tips is not None:
            actual_tips = np.asarray(
                [
                    data.xpos[body_id] + np.asarray(data.xmat[body_id]).reshape(3, 3) @ local_offset
                    for body_id, local_offset in zip(fingertip_body_ids, local_tip_offsets)
                ],
                dtype=np.float64,
            )
            commanded_tips = (
                data.mocap_pos[mocap_idx] + (mocap_rot @ root_start_rot.T @ (expected_tips - root_target_pos).T).T
            )
            tip_errors = np.linalg.norm(actual_tips - commanded_tips, axis=1)
            metrics["mean_fingertip_tracking_error"] = float(np.mean(tip_errors))
            metrics["max_fingertip_tracking_error"] = float(np.max(tip_errors))
        return metrics

    def simulation_is_stable() -> bool:
        state_is_finite = all(np.all(np.isfinite(values)) for values in (data.qpos, data.qvel, data.qacc))
        warning_count = sum(int(warning.number) for warning in data.warning)
        return bool(state_is_finite and warning_count == 0)

    def instability_result(stage: str) -> DynamicValidation:
        return DynamicValidation(
            trajectory_metrics={
                "max_penetration": float(overall_max_penetration),
                "instability_stage": stage,
                **tracking_metrics(),
            },
            per_finger_loads=np.zeros((len(fingertip_body_names), 6), dtype=np.float64),
            failure_stage="simulation_instability",
            passed=False,
        )

    overall_max_penetration = 0.0
    tracking_history: list[dict[str, float]] = []
    if initial_observer is not None:
        initial_observer("initial", model, data)

    # Stage 1: Squeeze (apply closing torque via smoothstep trajectory)
    squeeze_contact_samples: list[dict[str, object]] = []
    squeeze_window_start = max(
        0,
        squeeze_steps - max(1, int(np.ceil(squeeze_steps * protocol.contact_window_fraction))),
    )
    for squeeze_index in range(squeeze_steps):
        squeeze_progress = smoothstep((squeeze_index + 1) / max(1, squeeze_steps))
        data.mocap_pos[mocap_idx][:3] = root_pregrasp_pos + squeeze_progress * (root_target_pos - root_pregrasp_pos)
        u_val = start_controls + squeeze_progress * (target_controls - start_controls)
        data.ctrl[: model.nu] = np.clip(u_val, ctrl_mins, ctrl_maxs)

        mujoco.mj_step(model, data)
        if step_observer is not None:
            step_observer("squeeze", model, data)
        overall_max_penetration = max(overall_max_penetration, get_max_penetration())
        if not simulation_is_stable():
            return instability_result("squeeze")
        if squeeze_index >= squeeze_window_start:
            squeeze_contact_samples.append(
                extract_contact_loads(
                    model,
                    data,
                    object_geom_ids,
                    fingertip_body_names,
                    palm_body_names=(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, palm_id) or "palm",),
                )
            )

    palm_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, palm_id) or "palm"
    squeeze_loads = extract_contact_loads(
        model,
        data,
        object_geom_ids,
        fingertip_body_names,
        palm_body_names=(palm_name,),
    )
    squeeze_window = summarize_contact_window(squeeze_contact_samples)
    squeeze_tracking = tracking_metrics()
    tracking_history.append(squeeze_tracking)

    if (
        int(squeeze_window["sustained_count"]) < min_active_fingers
        or bool(squeeze_window["palm_support"])
        or overall_max_penetration > max_allowed_penetration
    ):
        squeeze_stage = (
            "palm_support"
            if bool(squeeze_window["palm_support"])
            else ("penetration" if overall_max_penetration > max_allowed_penetration else "active_contact")
        )
        return DynamicValidation(
            trajectory_metrics={
                "max_penetration": overall_max_penetration,
                "squeeze_active_fingers": int(squeeze_window["sustained_count"]),
                "contact_noise_floor": contact_noise_floor,
                "contact_force_threshold": contact_force_threshold,
                "squeeze_contact_duty_cycle": np.asarray(squeeze_window["duty_cycle"]).tolist(),
                "squeeze_normal_impulse": np.asarray(squeeze_window["normal_impulse"]).tolist(),
                "contacting_links": len(squeeze_loads["contacting_links"]),
                "has_palm_contact": float(bool(squeeze_window["palm_support"])),
                "max_cone_violation": float(np.max(squeeze_loads["cone_violations"])),
                "measured_fingertip_wrench_norm": float(np.linalg.norm(squeeze_loads["net_fingertip_wrench"])),
                **squeeze_tracking,
            },
            per_finger_loads=squeeze_loads["per_finger_loads"],
            failure_stage=squeeze_stage,
            passed=False,
        )
    if stage_observer is not None:
        stage_observer("squeeze", model, data)

    # Stage 2: Smooth Lift
    init_obj_z = float(data.xpos[obj_body_id][2])
    start_mocap_z = float(data.mocap_pos[mocap_idx][2])
    for s in range(1, lift_steps + 1):
        progress = float(s) / max(1, lift_steps)
        s_val = smoothstep(progress)
        data.mocap_pos[mocap_idx][2] = start_mocap_z + lift_height * s_val
        data.ctrl[: model.nu] = np.clip(target_controls, ctrl_mins, ctrl_maxs)

        mujoco.mj_step(model, data)
        if step_observer is not None:
            step_observer("lift", model, data)
        overall_max_penetration = max(overall_max_penetration, get_max_penetration())
        if not simulation_is_stable():
            return instability_result("lift")

    post_lift_z = float(data.xpos[obj_body_id][2])
    actual_lift = post_lift_z - init_obj_z
    floor_support_after_lift = object_has_floor_support()
    tracking_history.append(tracking_metrics())

    if actual_lift < protocol.lift_success_fraction * lift_height or floor_support_after_lift:
        return DynamicValidation(
            trajectory_metrics={
                "max_penetration": overall_max_penetration,
                "lift_achieved": actual_lift,
                **tracking_metrics(),
            },
            per_finger_loads=squeeze_loads["per_finger_loads"],
            failure_stage=("floor_support" if floor_support_after_lift else "lift"),
            passed=False,
        )
    if stage_observer is not None:
        stage_observer("lift", model, data)

    # Stage 3: Perturbation Wrench
    if perturbation_wrench is None:
        gravity_magnitude = float(np.linalg.norm(model.opt.gravity))
        object_weight = float(object_mass) * gravity_magnitude
        characteristic_length = max(
            (2.0 * float(np.max(np.asarray(geom.size, dtype=np.float64))) for geom in collision_geoms),
            default=0.05,
        )
        force_amplitude = 0.5 * object_weight
        torque_amplitude = 0.25 * object_weight * characteristic_length
        applied_wrench = np.array(
            [
                force_amplitude,
                force_amplitude,
                0.0,
                torque_amplitude,
                torque_amplitude,
                torque_amplitude,
            ],
            dtype=np.float64,
        )
    else:
        applied_wrench = np.asarray(perturbation_wrench, dtype=np.float64)

    total_impulse = np.zeros(6, dtype=np.float64)
    perturbation_contact_samples: list[dict[str, object]] = []

    for step_p in range(perturbation_steps):
        # Alternate perturbation direction periodically
        scale = np.sin(2.0 * np.pi * step_p / max(1, perturbation_steps))
        current_wrench = applied_wrench * scale
        data.xfrc_applied[obj_body_id] = current_wrench
        total_impulse += np.abs(current_wrench) * dt
        mujoco.mj_step(model, data)
        if step_observer is not None:
            step_observer("perturbation", model, data)
        overall_max_penetration = max(overall_max_penetration, get_max_penetration())
        if not simulation_is_stable():
            return instability_result("perturbation")
        floor_support_after_lift |= object_has_floor_support()
        perturbation_contact_samples.append(
            extract_contact_loads(
                model,
                data,
                object_geom_ids,
                fingertip_body_names,
                palm_body_names=(palm_name,),
            )
        )

    # Clear applied force
    data.xfrc_applied[obj_body_id] = np.zeros(6)

    final_obj_z = float(data.xpos[obj_body_id][2])
    final_lift = final_obj_z - init_obj_z

    final_loads = extract_contact_loads(
        model,
        data,
        object_geom_ids,
        fingertip_body_names,
        palm_body_names=(palm_name,),
    )
    perturbation_window = summarize_contact_window(perturbation_contact_samples)
    final_tracking = tracking_metrics()
    tracking_history.append(final_tracking)
    if stage_observer is not None:
        stage_observer("perturbation", model, data)

    passed_penetration = overall_max_penetration <= max_allowed_penetration
    passed_lift = final_lift >= protocol.lift_success_fraction * lift_height
    passed_fingers = int(perturbation_window["sustained_count"]) >= min_active_fingers
    passed_floor = not floor_support_after_lift
    passed_cone = bool(float(perturbation_window["max_cone_violation"]) <= protocol.cone_tolerance)
    max_normalized_actuator_error = max(
        snapshot["max_normalized_actuator_tracking_error"] for snapshot in tracking_history
    )
    max_normalized_joint_error = max(snapshot["max_normalized_joint_tracking_error"] for snapshot in tracking_history)
    max_palm_position_error = max(snapshot["palm_position_tracking_error"] for snapshot in tracking_history)
    max_palm_rotation_error = max(snapshot["palm_rotation_tracking_error"] for snapshot in tracking_history)
    max_root_mocap_error = max(snapshot["root_mocap_position_error"] for snapshot in tracking_history)
    actuator_tracking_pass = (
        max_normalized_actuator_error <= protocol.actuator_tracking_range_fraction
        and max_normalized_joint_error <= protocol.joint_tracking_range_fraction
    )
    palm_tracking_pass = (
        max_palm_position_error <= protocol.palm_position_tolerance
        and max_palm_rotation_error <= protocol.palm_rotation_tolerance
        and max_root_mocap_error <= protocol.root_mocap_position_tolerance
    )
    predicate = DynamicPredicateEvidence(
        stable=simulation_is_stable(),
        actuator_tracking_pass=actuator_tracking_pass,
        palm_tracking_pass=palm_tracking_pass,
        active_contact_sustained=passed_fingers,
        palm_support=bool(perturbation_window["palm_support"]),
        floor_support_after_lift=not passed_floor,
        penetration_pass=passed_penetration,
        lift_pass=passed_lift,
        disturbance_survival_pass=passed_lift and passed_fingers,
        friction_cone_pass=passed_cone,
    )
    passed, failure_stage = evaluate_dynamic_success(predicate)

    metrics = {
        "max_penetration": float(overall_max_penetration),
        "lift_achieved": float(final_lift),
        "final_active_fingers": float(perturbation_window["sustained_count"]),
        "contact_noise_floor": contact_noise_floor,
        "contact_force_threshold": contact_force_threshold,
        "contact_duty_cycle": np.asarray(perturbation_window["duty_cycle"]).tolist(),
        "normal_impulse": np.asarray(perturbation_window["normal_impulse"]).tolist(),
        "impulse_applied": float(np.sum(total_impulse)),
        "has_palm_contact": float(bool(perturbation_window["palm_support"])),
        "floor_support": float(not passed_floor),
        "max_cone_violation": float(perturbation_window["max_cone_violation"]),
        "actuator_tracking_pass": float(actuator_tracking_pass),
        "palm_tracking_pass": float(palm_tracking_pass),
        "active_contact_sustained": float(passed_fingers),
        "max_window_normalized_actuator_tracking_error": max_normalized_actuator_error,
        "max_window_normalized_joint_tracking_error": max_normalized_joint_error,
        "max_window_palm_position_tracking_error": max_palm_position_error,
        "max_window_palm_rotation_tracking_error": max_palm_rotation_error,
        "max_window_root_mocap_position_error": max_root_mocap_error,
        "protocol_gains_source": protocol.gains_source,
        "protocol_timestep_source": protocol.timestep_source,
        "measured_hand_wrench": final_loads["net_wrench"].tolist(),
        "measured_fingertip_wrench": final_loads["net_fingertip_wrench"].tolist(),
        **tracking_metrics(),
    }

    return DynamicValidation(
        trajectory_metrics=metrics,
        per_finger_loads=final_loads["per_finger_loads"],
        failure_stage=failure_stage,
        passed=passed,
    )
