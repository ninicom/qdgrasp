"""Swept whole-scene collision admission for a hand approach path."""

from __future__ import annotations

from collections.abc import Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class ClearanceError(Exception):
    """A fail-closed scene-clearance rejection with deterministic telemetry."""

    def __init__(self, reason: str, details: str, *, telemetry: dict[str, object] | None = None):
        self.reason = reason
        self.details = details
        self.telemetry = telemetry or {}
        super().__init__(f"{reason}: {details}")


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, int(index)) or f"unnamed_{index}"


def _descendant_body_ids(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    result: set[int] = set()
    for body_id in range(model.nbody):
        cursor = body_id
        while cursor > 0:
            if cursor == root_body_id:
                result.add(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    return result


def _top_level_body(model: mujoco.MjModel, body_id: int) -> int:
    cursor = int(body_id)
    while int(model.body_parentid[cursor]) != 0:
        cursor = int(model.body_parentid[cursor])
    return cursor


def _resolve_hand_root(model: mujoco.MjModel, hand_geom_ids: set[int]) -> tuple[int, int]:
    roots = {_top_level_body(model, int(model.geom_bodyid[g])) for g in hand_geom_ids}
    if len(roots) != 1:
        raise ClearanceError(
            "source_frame_invalid",
            "hand geoms must belong to exactly one top-level body",
            telemetry={"hand_root_body_ids": sorted(roots)},
        )
    root_body_id = roots.pop()
    joint_id = int(model.body_jntadr[root_body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ClearanceError(
            "source_frame_invalid",
            "hand root must have a free joint for swept-pose evaluation",
            telemetry={"hand_root_body": _name(model, mujoco.mjtObj.mjOBJ_BODY, root_body_id)},
        )
    return root_body_id, joint_id


def _resolve_target_geoms(
    model: mujoco.MjModel,
    target_object_id: str,
    explicit_geom_ids: Iterable[int] | None,
) -> set[int]:
    if explicit_geom_ids is not None:
        target_geoms = {int(geom_id) for geom_id in explicit_geom_ids}
    else:
        target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_object_id)
        if target_body_id < 0:
            raise ClearanceError(
                "source_frame_invalid", f"target body is absent from scene: {target_object_id}"
            )
        target_bodies = _descendant_body_ids(model, target_body_id)
        target_geoms = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in target_bodies
        }
    invalid = sorted(g for g in target_geoms if not 0 <= g < model.ngeom)
    if invalid or not target_geoms:
        raise ClearanceError(
            "source_frame_invalid",
            "target must resolve to at least one valid scene geom",
            telemetry={"invalid_target_geom_ids": invalid},
        )
    return target_geoms


def _validate_transform(transform: np.ndarray, index: int) -> None:
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ClearanceError(
            "source_frame_invalid", f"approach transform {index} must be finite and 4x4"
        )
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ClearanceError(
            "source_frame_invalid", f"approach transform {index} has invalid homogeneous row"
        )
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ClearanceError(
            "source_frame_invalid", f"approach transform {index} has invalid rotation"
        )


def _swept_samples(
    path: np.ndarray, max_translation_step: float, max_rotation_step: float
) -> list[tuple[np.ndarray, float]]:
    samples: list[tuple[np.ndarray, float]] = [(path[0].copy(), 0.0)]
    segment_count = max(1, len(path) - 1)
    for segment in range(len(path) - 1):
        start, end = path[segment], path[segment + 1]
        translation_distance = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
        relative_rotation = Rotation.from_matrix(start[:3, :3].T @ end[:3, :3])
        rotation_distance = float(relative_rotation.magnitude())
        subdivisions = max(
            1,
            int(np.ceil(translation_distance / max_translation_step)),
            int(np.ceil(rotation_distance / max_rotation_step)),
        )
        rotation_vector = relative_rotation.as_rotvec()
        for substep in range(1, subdivisions + 1):
            alpha = substep / subdivisions
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * end[:3, 3]
            transform[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(
                alpha * rotation_vector
            ).as_matrix()
            samples.append((transform, (segment + alpha) / segment_count))
    return samples


def check_approach_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_object_id: str,
    approach_path: np.ndarray,
    hand_geom_ids: list[int],
    *,
    target_geom_ids: Iterable[int] | None = None,
    max_translation_step: float = 0.005,
    max_rotation_step: float = np.deg2rad(5.0),
    contact_tolerance: float = 1e-8,
    allow_target_contact_at_goal: bool = True,
) -> bool:
    """Reject collision anywhere along a densified kinematic hand-root path.

    ``approach_path`` contains world transforms of the hand's top-level free
    body. Target contact is intentional only at the final sampled pose when
    ``allow_target_contact_at_goal`` is true. The supplied ``MjData`` is always
    restored, including when validation rejects or raises.
    """
    path = np.asarray(approach_path, dtype=np.float64)
    if path.size == 0:
        raise ClearanceError("approach_blocked", "approach path is empty")
    if path.ndim != 3 or path.shape[1:] != (4, 4):
        raise ClearanceError("source_frame_invalid", "approach path must have shape [N, 4, 4]")
    if max_translation_step <= 0.0 or max_rotation_step <= 0.0:
        raise ClearanceError("source_frame_invalid", "sweep step limits must be positive")
    for index, transform in enumerate(path):
        _validate_transform(transform, index)

    hand_geoms = {int(geom_id) for geom_id in hand_geom_ids}
    invalid_hand_geoms = sorted(g for g in hand_geoms if not 0 <= g < model.ngeom)
    if invalid_hand_geoms or not hand_geoms:
        raise ClearanceError(
            "source_frame_invalid",
            "hand_geom_ids must contain valid scene geoms",
            telemetry={"invalid_hand_geom_ids": invalid_hand_geoms},
        )
    target_geoms = _resolve_target_geoms(model, target_object_id, target_geom_ids)
    overlap = hand_geoms & target_geoms
    if overlap:
        raise ClearanceError(
            "source_frame_invalid",
            "hand and target geom sets overlap",
            telemetry={"overlapping_geom_ids": sorted(overlap)},
        )
    _, hand_root_joint = _resolve_hand_root(model, hand_geoms)
    qpos_address = int(model.jnt_qposadr[hand_root_joint])
    samples = _swept_samples(path, max_translation_step, max_rotation_step)

    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)
    try:
        for sample_index, (transform, progress) in enumerate(samples):
            data.qpos[qpos_address : qpos_address + 3] = transform[:3, 3]
            quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
            data.qpos[qpos_address + 3 : qpos_address + 7] = [
                quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
            ]
            mujoco.mj_forward(model, data)
            for contact_index in range(int(data.ncon)):
                contact = data.contact[contact_index]
                if float(contact.dist) > contact_tolerance:
                    continue
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if geom1 not in hand_geoms and geom2 not in hand_geoms:
                    continue
                other_geom = geom2 if geom1 in hand_geoms else geom1
                is_target = other_geom in target_geoms
                at_goal = sample_index == len(samples) - 1
                if is_target and allow_target_contact_at_goal and at_goal:
                    continue
                reason = "approach_blocked" if is_target else "hand_scene_collision"
                other_kind = "target" if is_target else "non_target_or_support"
                hand_geom = geom1 if geom1 in hand_geoms else geom2
                telemetry = {
                    "sample_index": sample_index,
                    "sample_count": len(samples),
                    "path_progress": progress,
                    "hand_geom": _name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom),
                    "other_geom": _name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom),
                    "other_kind": other_kind,
                    "distance": float(contact.dist),
                }
                raise ClearanceError(
                    reason,
                    f"hand contact with {other_kind} at swept sample {sample_index}",
                    telemetry=telemetry,
                )
        return True
    finally:
        mujoco.mj_setState(model, data, state, state_spec)
        mujoco.mj_forward(model, data)
