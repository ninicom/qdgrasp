from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple
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

    mocap_body = spec.worldbody.add_body(name="hand_mocap", mocap=True)
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name="mocap_weld",
        name1="hand_mocap",
        name2=hand_root.name,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        solref=[0.002, 1.0],
        solimp=[0.99, 0.999, 0.001, 0.5, 2],
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
        pos=[0.0, 0.0, -0.1],
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
    object_pos: Tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
    squeeze_steps: int = 150,
    lift_steps: int = 150,
    lift_height: float = 0.05,
    perturbation_steps: int = 50,
    perturbation_wrench: Optional[np.ndarray] = None, # [6] force & torque
    max_allowed_penetration: float = 0.02,
    min_active_fingers: int = 1,
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

    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mocap")
    if mocap_body_id < 0:
        raise ConfigError("could not identify mocap body in model")
    mocap_idx = model.body_mocapid[mocap_body_id]

    # Determine palm vs root offset
    mujoco.mj_forward(model, data)
    palm_delta = np.asarray(palm_pos, dtype=np.float64) - data.xpos[palm_id]
    root_start_pos = np.array(data.xpos[root_id]) + palm_delta

    jnt_id = model.body_jntadr[root_id]
    if jnt_id < 0 or model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ConfigError(f"Root body {root_id} must have a freejoint for mocap-weld control.")

    qpos_adr = model.jnt_qposadr[jnt_id]
    data.qpos[qpos_adr : qpos_adr + 3] = root_start_pos
    if palm_rot is not None:
        if palm_rot.shape == (3, 3):
            # Convert rotation matrix to quat [w, x, y, z]
            r = Rotation.from_matrix(palm_rot)
            quat_xyzw = r.as_quat()
            quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        elif palm_rot.shape == (4,):
            quat_wxyz = list(palm_rot)
        else:
            quat_wxyz = [1.0, 0.0, 0.0, 0.0]
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
        data.mocap_quat[mocap_idx] = quat_wxyz

    data.mocap_pos[mocap_idx][:3] = root_start_pos

    # Set initial joint angles and map to actuators
    target_by_joint: Dict[int, float] = {}
    target_by_actuator: Dict[int, float] = {}

    if joint_targets:
        for j_name, val in joint_targets.items():
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id >= 0:
                data.qpos[model.jnt_qposadr[j_id]] = float(val)
                target_by_joint[j_id] = float(val)

    for a_id in range(model.nu):
        a_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_id) or ""
        if model.actuator_trntype[a_id] == mujoco.mjtTrn.mjTRN_JOINT:
            j_id = int(model.actuator_trnid[a_id, 0])
            if j_id in target_by_joint:
                target_by_actuator[a_id] = target_by_joint[j_id]
        else:
            # Handle tendon actuators (e.g. rh_A_FFJ0 -> rh_FFJ2)
            if joint_targets:
                for j_name, val in joint_targets.items():
                    if j_name in a_name or (j_name.replace("J2", "J0") in a_name) or (j_name.replace("rh_", "rh_A_").replace("J2", "J0") == a_name):
                        target_by_actuator[a_id] = float(val)
                        break

    mujoco.mj_forward(model, data)

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

    overall_max_penetration = 0.0

    # Stage 1: Squeeze (apply closing torque via squeeze bias)
    for _ in range(squeeze_steps):
        for a_id in range(model.nu):
            if a_id in target_by_actuator:
                target_val = target_by_actuator[a_id]
                if model.actuator_ctrllimited[a_id]:
                    lo, hi = model.actuator_ctrlrange[a_id]
                    # Apply closing squeeze pressure
                    data.ctrl[a_id] = min(hi, target_val + 0.5)
                else:
                    data.ctrl[a_id] = target_val + 0.5
            elif model.actuator_ctrllimited[a_id]:
                lo, hi = model.actuator_ctrlrange[a_id]
                data.ctrl[a_id] = (lo + hi) * 0.5
        mujoco.mj_step(model, data)
        overall_max_penetration = max(overall_max_penetration, get_max_penetration())

    squeeze_loads = extract_contact_loads(model, data, object_geom_ids, fingertip_body_names)

    # Check squeeze criteria: at least min_active_fingers must have positive normal force
    if squeeze_loads["active_fingers_count"] < min_active_fingers:
        # Also check if any hand contact exists
        if len(squeeze_loads["contacting_links"]) > 0:
            active_count = max(squeeze_loads["active_fingers_count"], len(squeeze_loads["contacting_links"]))
        else:
            active_count = squeeze_loads["active_fingers_count"]

        if active_count < min_active_fingers:
            return DynamicValidation(
                trajectory_metrics={"max_penetration": overall_max_penetration, "squeeze_active_fingers": squeeze_loads["active_fingers_count"], "contacting_links": len(squeeze_loads["contacting_links"])},
                per_finger_loads=squeeze_loads["per_finger_loads"],
                failure_stage="squeeze",
                passed=False
            )

    # Stage 2: Smooth Lift
    init_obj_z = float(data.xpos[obj_body_id][2])
    start_mocap_z = float(data.mocap_pos[mocap_idx][2])
    for s in range(1, lift_steps + 1):
        progress = float(s) / max(1, lift_steps)
        s_val = smoothstep(progress)
        data.mocap_pos[mocap_idx][2] = start_mocap_z + lift_height * s_val
        data.qpos[qpos_adr + 2] = root_start_pos[2] + lift_height * s_val

        for a_id in range(model.nu):
            if a_id in target_by_actuator:
                target_val = target_by_actuator[a_id]
                if model.actuator_ctrllimited[a_id]:
                    lo, hi = model.actuator_ctrlrange[a_id]
                    data.ctrl[a_id] = min(hi, target_val + 0.5)
                else:
                    data.ctrl[a_id] = target_val + 0.5
            elif model.actuator_ctrllimited[a_id]:
                lo, hi = model.actuator_ctrlrange[a_id]
                data.ctrl[a_id] = (lo + hi) * 0.5

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        overall_max_penetration = max(overall_max_penetration, get_max_penetration())

    post_lift_z = float(data.xpos[obj_body_id][2])
    actual_lift = post_lift_z - init_obj_z

    if actual_lift < 0.5 * lift_height:
        return DynamicValidation(
            trajectory_metrics={"max_penetration": overall_max_penetration, "lift_achieved": actual_lift},
            per_finger_loads=squeeze_loads["per_finger_loads"],
            failure_stage="lift",
            passed=False
        )

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

    # Clear applied force
    data.xfrc_applied[obj_body_id] = np.zeros(6)

    final_obj_z = float(data.xpos[obj_body_id][2])
    final_lift = final_obj_z - init_obj_z

    final_loads = extract_contact_loads(model, data, object_geom_ids, fingertip_body_names)

    passed_penetration = (overall_max_penetration <= max_allowed_penetration)
    passed_lift = (final_lift >= 0.5 * lift_height)
    passed_fingers = (final_loads["active_fingers_count"] >= min_active_fingers)

    passed = bool(passed_penetration and passed_lift and passed_fingers)
    failure_stage = "none" if passed else ("penetration" if not passed_penetration else "perturbation")

    metrics = {
        "max_penetration": float(overall_max_penetration),
        "lift_achieved": float(final_lift),
        "final_active_fingers": float(final_loads["active_fingers_count"]),
        "impulse_applied": float(np.sum(total_impulse)),
        "has_palm_contact": float(final_loads["has_palm_contact"]),
    }

    return DynamicValidation(
        trajectory_metrics=metrics,
        per_finger_loads=final_loads["per_finger_loads"],
        failure_stage=failure_stage,
        passed=passed
    )
