"""Exact MuJoCo collision admission for a generated contact pose.

The trimesh filter remains a cheap prefilter.  This module is the admission
oracle: it compiles the same hand asset and procedural object geoms as the
dynamic rollout, places the articulated hand at the requested palm pose, and
allows object contact only on declared active fingertip bodies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.pipeline.contracts import CollisionAdmission
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import build_rollout_scene_model
from qdgrasp.objects.schema import SubGeomSpec


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(index)) or f"unnamed_{index}"


def _set_articulated_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    palm_body_name: str,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    joint_targets: Mapping[str, float],
) -> None:
    for joint_name, value in joint_targets.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0 or int(model.jnt_type[joint_id]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ConfigError(f"collision pose refers to unsupported joint: {joint_name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)

    palm_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, palm_body_name
    )
    if palm_id < 0:
        raise ConfigError(f"palm body is absent from collision model: {palm_body_name}")

    root_id = palm_id
    while int(model.body_parentid[root_id]) != 0:
        root_id = int(model.body_parentid[root_id])

    mujoco.mj_forward(model, data)
    root_rot = np.asarray(data.xmat[root_id]).reshape(3, 3).copy()
    palm_rot_initial = np.asarray(data.xmat[palm_id]).reshape(3, 3).copy()
    root_to_palm_rot = root_rot.T @ palm_rot_initial
    root_to_palm_pos = root_rot.T @ (
        np.asarray(data.xpos[palm_id]) - np.asarray(data.xpos[root_id])
    )

    requested_rotation = np.asarray(palm_rot, dtype=np.float64)
    if requested_rotation.shape != (3, 3):
        raise ConfigError("collision palm_rot must have shape (3, 3)")
    root_world_rot = requested_rotation @ root_to_palm_rot.T
    root_world_pos = np.asarray(palm_pos, dtype=np.float64) - (
        root_world_rot @ root_to_palm_pos
    )

    root_joint_id = int(model.body_jntadr[root_id])
    if root_joint_id < 0 or int(model.jnt_type[root_joint_id]) != int(
        mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ConfigError("collision model hand root must have a free joint")
    qpos_address = int(model.jnt_qposadr[root_joint_id])
    data.qpos[qpos_address : qpos_address + 3] = root_world_pos
    quat_xyzw = Rotation.from_matrix(root_world_rot).as_quat()
    data.qpos[qpos_address + 3 : qpos_address + 7] = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    mujoco.mj_forward(model, data)


def admit_mujoco_collision_pose(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    *,
    palm_body_name: str,
    fingertip_body_names: Sequence[str],
    active_fingers: np.ndarray,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    joint_targets: Mapping[str, float],
    object_pos: tuple[float, float, float],
    object_mass: float,
    max_allowed_tip_penetration: float = 0.002,
    contact_tolerance: float = 1e-6,
) -> CollisionAdmission:
    """Admit only active-tip/object contacts in the compiled rollout scene."""
    active = np.asarray(active_fingers, dtype=bool)
    if active.shape != (len(fingertip_body_names),):
        raise ConfigError(
            "active_fingers must match fingertip_body_names in collision admission"
        )

    model = build_rollout_scene_model(
        hand_xml_path,
        collision_geoms,
        object_pos=object_pos,
        object_mass=object_mass,
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    _set_articulated_pose(
        model,
        data,
        palm_body_name=palm_body_name,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        joint_targets=joint_targets,
    )

    object_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).startswith(
            "object_subgeom_"
        )
    }
    floor_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    active_tip_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name, enabled in zip(fingertip_body_names, active)
        if enabled
    }
    if -1 in active_tip_body_ids:
        raise ConfigError("an active fingertip body is absent from collision model")

    enabled_hand_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id not in object_geom_ids
        and geom_id != floor_geom_id
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    }

    pair_records: list[dict[str, object]] = []
    forbidden_object: list[dict[str, object]] = []
    excessive_tip: list[dict[str, object]] = []
    floor_contacts: list[dict[str, object]] = []
    self_contacts: list[dict[str, object]] = []
    max_penetration = 0.0

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        distance = float(contact.dist)
        if distance > contact_tolerance:
            continue
        penetration = max(0.0, -distance)
        max_penetration = max(max_penetration, penetration)
        record: dict[str, object] = {
            "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "body1": _name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1])
            ),
            "body2": _name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2])
            ),
            "distance": distance,
            "penetration": penetration,
        }
        pair_records.append(record)

        if floor_geom_id in (geom1, geom2):
            other = geom2 if geom1 == floor_geom_id else geom1
            if other in enabled_hand_geom_ids:
                floor_contacts.append(record)
            continue

        object_contact = (geom1 in object_geom_ids) ^ (geom2 in object_geom_ids)
        if object_contact:
            hand_geom = geom2 if geom1 in object_geom_ids else geom1
            hand_body = int(model.geom_bodyid[hand_geom])
            if hand_body not in active_tip_body_ids:
                forbidden_object.append(record)
            elif penetration > max_allowed_tip_penetration:
                excessive_tip.append(record)
            continue

        if geom1 in enabled_hand_geom_ids and geom2 in enabled_hand_geom_ids:
            self_contacts.append(record)

    floor_distances = []
    if floor_geom_id >= 0:
        for geom_id in sorted(enabled_hand_geom_ids):
            floor_distances.append(
                float(
                    mujoco.mj_geomDistance(
                        model, data, geom_id, floor_geom_id, 10.0, None
                    )
                )
            )
    min_floor_clearance = min(floor_distances, default=float("inf"))

    if floor_contacts or min_floor_clearance < -contact_tolerance:
        reason = "hand_floor_contact"
    elif forbidden_object:
        first = forbidden_object[0]
        reason = f"forbidden_object_contact:{first['body1']}:{first['body2']}"
    elif self_contacts:
        reason = "hand_self_collision"
    elif excessive_tip:
        reason = "active_tip_excessive_penetration"
    else:
        reason = "passed"

    return CollisionAdmission(
        passed=reason == "passed",
        reason=reason,
        contact_pairs=tuple(pair_records),
        max_penetration=max_penetration,
        min_hand_floor_clearance=min_floor_clearance,
    )
