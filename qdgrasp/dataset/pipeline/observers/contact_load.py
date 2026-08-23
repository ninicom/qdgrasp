from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import mujoco

def extract_contact_loads(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom_ids: Set[int],
    fingertip_body_names: Sequence[str],
    palm_body_names: Sequence[str] = ("palm", "base_link"),
    mu: float = 0.5,
) -> Dict[str, Any]:
    """
    Extracts physically accurate contact forces from MuJoCo using mj_contactForce.
    Separates normal and tangential forces, tracks per-finger loads, and computes net wrench on the object.

    Returns a dictionary containing:
    - 'per_finger_forces': np.ndarray of shape [K, 3] in world frame
    - 'per_finger_torques': np.ndarray of shape [K, 3] in world frame
    - 'per_finger_normals': np.ndarray of shape [K, 3] (unit normal directions)
    - 'per_finger_f_normal': np.ndarray of shape [K] (magnitude of normal force in N)
    - 'per_finger_f_tangential': np.ndarray of shape [K] (magnitude of friction force in N)
    - 'cone_violations': np.ndarray of shape [K] (max(0, f_t - mu * f_n))
    - 'net_wrench': np.ndarray of shape [6] (total force and torque on object COM in N, N*m)
    - 'active_fingers_count': int
    - 'has_palm_contact': bool
    - 'contacting_links': List[str]
    """
    num_fingers = len(fingertip_body_names)
    palm_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in palm_body_names
    }

    palm_id_set = {
        b_id for b_id in range(model.nbody)
        if any(p in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or "").lower() for p in ("palm", "forearm", "wrist", "base_link", "root", "world"))
    }

    body_to_finger_idx = {}
    for idx, name in enumerate(fingertip_body_names):
        b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        curr = b_id
        while curr > 0 and curr not in palm_id_set:
            body_to_finger_idx[curr] = idx
            curr = int(model.body_parentid[curr])

    per_finger_forces = np.zeros((num_fingers, 3), dtype=np.float64)
    per_finger_torques = np.zeros((num_fingers, 3), dtype=np.float64)
    per_finger_normals = np.zeros((num_fingers, 3), dtype=np.float64)
    per_finger_f_normal = np.zeros(num_fingers, dtype=np.float64)
    per_finger_f_tangential = np.zeros(num_fingers, dtype=np.float64)
    cone_violations = np.zeros(num_fingers, dtype=np.float64)

    # Get object center of mass / root position
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    obj_com = np.array(data.xpos[obj_id]) if obj_id >= 0 else np.zeros(3)

    net_force = np.zeros(3, dtype=np.float64)
    net_torque = np.zeros(3, dtype=np.float64)

    has_palm_contact = False
    contacting_links = set()

    force_buf = np.zeros(6, dtype=np.float64)

    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)

        # Check if one geom is the object and the other is a hand geom
        is_g1_obj = g1 in object_geom_ids
        is_g2_obj = g2 in object_geom_ids

        if not (is_g1_obj ^ is_g2_obj):
            continue  # Not a hand-object contact

        hand_geom = g2 if is_g1_obj else g1
        b_id = int(model.geom_bodyid[hand_geom])
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or f"body_{b_id}"
        contacting_links.add(b_name)

        if b_id in palm_body_ids:
            has_palm_contact = True

        # Extract 6D contact force in contact frame using standard MuJoCo routine
        mujoco.mj_contactForce(model, data, i, force_buf)

        # Contact frame orientation (3x3 row-major)
        c_rot = c.frame.reshape(3, 3)

        # In contact frame:
        # force_buf[0] is normal force (along c_rot[0])
        # force_buf[1], force_buf[2] are tangential forces (along c_rot[1], c_rot[2])
        # Note: force is applied to geom1 by geom2.
        # If geom1 is object, force on object is +force_buf in contact frame.
        # If geom2 is object, force on object is -force_buf in contact frame.
        sign = 1.0 if is_g1_obj else -1.0

        f_normal_mag = force_buf[0]
        f_tan_mag = np.linalg.norm(force_buf[1:3])

        # World frame force vector on the object
        f_world = sign * (c_rot.T @ force_buf[0:3])
        p_world = np.array(c.pos)
        r = p_world - obj_com
        tau_world = np.cross(r, f_world) + sign * (c_rot.T @ force_buf[3:6])

        net_force += f_world
        net_torque += tau_world

        if b_id in body_to_finger_idx:
            f_idx = body_to_finger_idx[b_id]
            per_finger_forces[f_idx] += f_world
            per_finger_torques[f_idx] += tau_world
            per_finger_f_normal[f_idx] += f_normal_mag
            per_finger_f_tangential[f_idx] += f_tan_mag

            # Normal vector in world coords pointing inward into the object
            n_world = -c_rot[0] if is_g1_obj else c_rot[0]
            per_finger_normals[f_idx] = n_world

            violation = max(0.0, f_tan_mag - mu * f_normal_mag)
            cone_violations[f_idx] = max(cone_violations[f_idx], violation)

    active_fingers_count = int(np.sum(per_finger_f_normal > 1e-3))

    per_finger_loads = np.concatenate([per_finger_forces, per_finger_torques], axis=-1)

    return {
        "per_finger_loads": per_finger_loads, # [K, 6]
        "per_finger_forces": per_finger_forces, # [K, 3]
        "per_finger_torques": per_finger_torques, # [K, 3]
        "per_finger_normals": per_finger_normals, # [K, 3]
        "per_finger_f_normal": per_finger_f_normal, # [K]
        "per_finger_f_tangential": per_finger_f_tangential, # [K]
        "cone_violations": cone_violations, # [K]
        "net_wrench": np.concatenate([net_force, net_torque]), # [6]
        "active_fingers_count": active_fingers_count,
        "has_palm_contact": has_palm_contact,
        "contacting_links": list(sorted(contacting_links)),
    }
