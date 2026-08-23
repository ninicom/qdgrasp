from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from qdgrasp.config.schema import ConfigError
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.dataset.pipeline.contracts import DynamicValidation
from qdgrasp.dataset.pipeline.observers.contact_load import extract_contact_loads


def build_rollout_scene_model(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    *,
    object_pos: Tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
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

    obj_body = spec.worldbody.add_body(
        name="target_object",
        pos=[float(object_pos[0]), float(object_pos[1]), float(object_pos[2])],
    )
    obj_body.add_freejoint(name="object_freejoint")

    for i, g in enumerate(collision_geoms):
        geom_type = getattr(mujoco.mjtGeom, f"mjGEOM_{g.type.upper()}")
        size_list = [float(s) for s in g.size]
        pos_list = [float(p) for p in g.pos]
        quat_list = [float(q) for q in g.quat]

        obj_body.add_geom(
            name=f"object_subgeom_{i}",
            type=geom_type,
            size=size_list,
            pos=pos_list,
            quat=quat_list,
            mass=float(object_mass / len(collision_geoms)),
            condim=4,
            friction=[1.0, 0.005, 0.0001],
            rgba=[0.8, 0.3, 0.3, 1.0],
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
    palm_pos: Tuple[float, float, float] = (0.0, 0.0, 0.1),
    palm_rot: Optional[np.ndarray] = None, # 3x3 rotation matrix or quaternion
    joint_targets: Optional[Mapping[str, float]] = None,
    initial_joint_targets: Optional[Mapping[str, float]] = None,
    object_pos: Tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
    squeeze_steps: int = 150,
    lift_steps: int = 150,
    lift_height: float = 0.05,
    perturbation_steps: int = 50,
    perturbation_wrench: Optional[np.ndarray] = None, # [6] force & torque
    max_allowed_penetration: float = 0.002,
    min_active_fingers: int = 2,
    pregrasp_distance: float = 0.03,
    expected_fingertip_positions: Optional[np.ndarray] = None,
    fingertip_local_offsets: Optional[np.ndarray] = None,
    stage_observer: Optional[
        Callable[[str, mujoco.MjModel, mujoco.MjData], None]
    ] = None,
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
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    fingertip_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in fingertip_body_names
    ]
    missing_tips = [
        name for name, body_id in zip(fingertip_body_names, fingertip_body_ids)
        if body_id < 0
    ]
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
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0 or int(model.jnt_type[joint_id]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ConfigError(
                    f"initial joint target refers to unsupported joint: {joint_name}"
                )
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
    root_to_palm_pos = root_rot_initial.T @ (
        np.array(data.xpos[palm_id]) - np.array(data.xpos[root_id])
    )

    requested_palm_rot = np.eye(3, dtype=np.float64)
    if palm_rot is not None:
        if palm_rot.shape == (3, 3):
            requested_palm_rot = np.asarray(palm_rot, dtype=np.float64)
        elif palm_rot.shape == (4,):
            quat = np.asarray(palm_rot, dtype=np.float64)
            requested_palm_rot = Rotation.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()
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
    if jnt_id < 0 or int(model.jnt_type[jnt_id]) != int(
        mujoco.mjtJoint.mjJNT_FREE
    ):
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
    hand_actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_id)
        for a_id in range(model.nu)
    ]

    has_tendon = any(
        int(model.actuator_trntype[a_id]) == int(mujoco.mjtTrn.mjTRN_TENDON)
        for a_id in range(model.nu)
    )

    from qdgrasp.robot.transmission.direct import DirectJointTransmission
    from qdgrasp.robot.transmission.fixed_tendon import FixedTendonTransmission

    if has_tendon:
        tm = FixedTendonTransmission(hand_joint_names, hand_actuator_names, model)
    else:
        tm = DirectJointTransmission(hand_joint_names, hand_actuator_names, model)

    initial_trans_state = tm.extract_state(model, data)
    q_init = initial_trans_state.joint_position
    q_target = q_init.copy()

    if joint_targets:
        for idx, j_name in enumerate(hand_joint_names):
            if j_name in joint_targets:
                q_target[idx] = float(joint_targets[j_name])

    dq_desired = q_target - q_init
    cmd = tm.project_joint_delta(dq_desired, initial_trans_state)

    if cmd.reason == "nullspace_rejection":
        return DynamicValidation(
            trajectory_metrics={
                "transmission_rank": float(tm.rank),
                "joint_state_dimensions": float(tm.num_joints),
                "control_dimensions": float(tm.num_actuators),
                "controllable_residual": float(cmd.controllable_residual),
                "nullspace_residual": float(cmd.nullspace_residual),
                "actuator_saturation_count": float(np.sum(cmd.saturated)),
            },
            per_finger_loads=np.zeros(
                (len(fingertip_body_names), 6), dtype=np.float64
            ),
            failure_stage="underactuated_targets",
            passed=False,
        )

    start_controls = initial_trans_state.actuator_coordinate.copy()
    target_controls = cmd.control_target.copy()

    ctrl_mins = tm.actuator_ctrlrange[:, 0]
    ctrl_maxs = tm.actuator_ctrlrange[:, 1]
    data.ctrl[:model.nu] = np.clip(start_controls, ctrl_mins, ctrl_maxs)

    # Object & floor geom IDs
    object_geom_ids = {
        g_id for g_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g_id) or "").startswith("object_subgeom_")
    }
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")

    def get_max_penetration() -> float:
        max_pen = 0.0
        for idx in range(int(data.ncon)):
            c = data.contact[idx]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in object_geom_ids or g2 in object_geom_ids) and floor_geom_id not in (g1, g2):
                if c.dist < 0:
                    max_pen = max(max_pen, abs(float(c.dist)))
        return max_pen

    def object_has_floor_support() -> bool:
        return any(
            floor_geom_id in (int(data.contact[idx].geom1), int(data.contact[idx].geom2))
            and (
                int(data.contact[idx].geom1) in object_geom_ids
                or int(data.contact[idx].geom2) in object_geom_ids
            )
            for idx in range(int(data.ncon))
        )

    def tracking_metrics() -> Dict[str, float]:
        current_state = tm.extract_state(model, data)
        current_q = current_state.joint_position
        current_coords = current_state.actuator_coordinate

        joint_errors = np.abs(current_q - q_target)
        actuator_errors = np.abs(current_coords - target_controls)

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
        metrics = {
            "transmission_rank": float(tm.rank),
            "joint_state_dimensions": float(tm.num_joints),
            "control_dimensions": float(tm.num_actuators),
            "controllable_residual": float(cmd.controllable_residual),
            "nullspace_residual": float(cmd.nullspace_residual),
            "actuator_saturation_count": float(np.sum(cmd.saturated)),
            "max_joint_tracking_error": float(np.max(joint_errors) if len(joint_errors) > 0 else 0.0),
            "max_actuator_coordinate_error": float(np.max(actuator_errors) if len(actuator_errors) > 0 else 0.0),
            "palm_position_tracking_error": float(
                np.linalg.norm(np.asarray(data.xpos[palm_id]) - commanded_palm_pos)
            ),
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
                    data.xpos[body_id]
                    + np.asarray(data.xmat[body_id]).reshape(3, 3) @ local_offset
                    for body_id, local_offset in zip(
                        fingertip_body_ids, local_tip_offsets
                    )
                ],
                dtype=np.float64,
            )
            commanded_tips = data.mocap_pos[mocap_idx] + (
                mocap_rot
                @ root_start_rot.T
                @ (expected_tips - root_target_pos).T
            ).T
            tip_errors = np.linalg.norm(actual_tips - commanded_tips, axis=1)
            metrics["mean_fingertip_tracking_error"] = float(np.mean(tip_errors))
            metrics["max_fingertip_tracking_error"] = float(np.max(tip_errors))
        return metrics

    def simulation_is_stable() -> bool:
        state_is_finite = all(
            np.all(np.isfinite(values))
            for values in (data.qpos, data.qvel, data.qacc)
        )
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

    # Stage 1: Squeeze (apply closing torque via smoothstep trajectory)
    for squeeze_index in range(squeeze_steps):
        squeeze_progress = smoothstep((squeeze_index + 1) / max(1, squeeze_steps))
        data.mocap_pos[mocap_idx][:3] = root_pregrasp_pos + squeeze_progress * (
            root_target_pos - root_pregrasp_pos
        )
        u_val = start_controls + squeeze_progress * (target_controls - start_controls)
        data.ctrl[:model.nu] = np.clip(u_val, ctrl_mins, ctrl_maxs)

        mujoco.mj_step(model, data)
        overall_max_penetration = max(overall_max_penetration, get_max_penetration())
        if not simulation_is_stable():
            return instability_result("squeeze")

    palm_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, palm_id) or "palm"
    squeeze_loads = extract_contact_loads(
        model,
        data,
        object_geom_ids,
        fingertip_body_names,
        palm_body_names=(palm_name,),
    )

    if (
        squeeze_loads["active_fingers_count"] < min_active_fingers
        or overall_max_penetration > max_allowed_penetration
    ):
        return DynamicValidation(
            trajectory_metrics={
                "max_penetration": overall_max_penetration,
                "squeeze_active_fingers": squeeze_loads["active_fingers_count"],
                "contacting_links": len(squeeze_loads["contacting_links"]),
                "has_palm_contact": float(squeeze_loads["has_palm_contact"]),
                "max_cone_violation": float(np.max(squeeze_loads["cone_violations"])),
                "measured_fingertip_wrench_norm": float(
                    np.linalg.norm(squeeze_loads["net_fingertip_wrench"])
                ),
                **tracking_metrics(),
            },
            per_finger_loads=squeeze_loads["per_finger_loads"],
            failure_stage="squeeze",
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
        data.ctrl[:model.nu] = np.clip(target_controls, ctrl_mins, ctrl_maxs)

        mujoco.mj_step(model, data)
        overall_max_penetration = max(overall_max_penetration, get_max_penetration())
        if not simulation_is_stable():
            return instability_result("lift")

    post_lift_z = float(data.xpos[obj_body_id][2])
    actual_lift = post_lift_z - init_obj_z

    if actual_lift < 0.5 * lift_height or object_has_floor_support():
        return DynamicValidation(
            trajectory_metrics={
                "max_penetration": overall_max_penetration,
                "lift_achieved": actual_lift,
                **tracking_metrics(),
            },
            per_finger_loads=squeeze_loads["per_finger_loads"],
            failure_stage="lift",
            passed=False
        )
    if stage_observer is not None:
        stage_observer("lift", model, data)

    # Stage 3: Perturbation Wrench
    if perturbation_wrench is None:
        # Default perturbation wrench (shaking force + torque)
        applied_wrench = np.array([0.5, 0.5, 0.0, 0.05, 0.05, 0.05], dtype=np.float64)
    else:
        applied_wrench = np.asarray(perturbation_wrench, dtype=np.float64)

    total_impulse = np.zeros(6, dtype=np.float64)
    dt = float(model.opt.timestep)

    for step_p in range(perturbation_steps):
        # Alternate perturbation direction periodically
        scale = np.sin(2.0 * np.pi * step_p / max(1, perturbation_steps))
        current_wrench = applied_wrench * scale
        data.xfrc_applied[obj_body_id] = current_wrench
        total_impulse += np.abs(current_wrench) * dt
        mujoco.mj_step(model, data)
        overall_max_penetration = max(overall_max_penetration, get_max_penetration())
        if not simulation_is_stable():
            return instability_result("perturbation")

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
    if stage_observer is not None:
        stage_observer("perturbation", model, data)

    passed_penetration = (overall_max_penetration <= max_allowed_penetration)
    passed_lift = (final_lift >= 0.5 * lift_height)
    passed_fingers = (final_loads["active_fingers_count"] >= min_active_fingers)
    passed_floor = not object_has_floor_support()
    passed_cone = bool(np.max(final_loads["cone_violations"]) <= 1e-6)

    passed = bool(
        passed_penetration
        and passed_lift
        and passed_fingers
        and passed_floor
        and passed_cone
    )
    failure_stage = "none" if passed else ("penetration" if not passed_penetration else "perturbation")

    metrics = {
        "max_penetration": float(overall_max_penetration),
        "lift_achieved": float(final_lift),
        "final_active_fingers": float(final_loads["active_fingers_count"]),
        "impulse_applied": float(np.sum(total_impulse)),
        "has_palm_contact": float(final_loads["has_palm_contact"]),
        "floor_support": float(not passed_floor),
        "max_cone_violation": float(np.max(final_loads["cone_violations"])),
        "measured_hand_wrench": final_loads["net_wrench"].tolist(),
        "measured_fingertip_wrench": final_loads["net_fingertip_wrench"].tolist(),
        **tracking_metrics(),
    }

    return DynamicValidation(
        trajectory_metrics=metrics,
        per_finger_loads=final_loads["per_finger_loads"],
        failure_stage=failure_stage,
        passed=passed
    )
