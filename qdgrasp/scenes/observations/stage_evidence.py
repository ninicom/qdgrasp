"""Rendered grasp-stage QA captured from the exact labeling ``MjData``."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from qdgrasp.config.schema import ConfigError

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _project_points(
    points: np.ndarray,
    camera: mujoco.MjvGLCamera,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Project world points through MuJoCo's realized render camera."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ConfigError("overlay points must have shape [N, 3]")
    forward = np.asarray(camera.forward, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    relative = values - np.asarray(camera.pos, dtype=np.float64)
    depth = relative @ forward
    near = float(camera.frustum_near)
    half_width = float(camera.frustum_width) / 2.0
    half_height = float(camera.frustum_top - camera.frustum_center)
    if half_height <= np.finfo(np.float64).eps:
        raise ConfigError("render camera has invalid vertical frustum")
    if half_width <= np.finfo(np.float64).eps:
        half_width = half_height * width / height
    projected = np.full((len(values), 2), np.nan, dtype=np.float64)
    visible = depth > near
    projected[visible, 0] = (0.5 + 0.5 * (relative[visible] @ right) * near / depth[visible] / half_width) * width
    projected[visible, 1] = (0.5 - 0.5 * (relative[visible] @ up) * near / depth[visible] / half_height) * height
    return projected


def _target_contact_points(model: mujoco.MjModel, data: mujoco.MjData, target_body_name: str) -> np.ndarray:
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    if target_body < 0:
        raise ConfigError(f"stage evidence target body missing: {target_body_name}")
    target_geoms: set[int] = set()
    for geom_id in range(model.ngeom):
        ancestor = int(model.geom_bodyid[geom_id])
        while ancestor > 0 and ancestor != target_body:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == target_body:
            target_geoms.add(geom_id)
    points = [
        np.asarray(data.contact[index].pos, dtype=np.float64).copy()
        for index in range(int(data.ncon))
        if target_geoms.intersection((int(data.contact[index].geom1), int(data.contact[index].geom2)))
    ]
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _body_geom_ids(model: mujoco.MjModel, body_name: str) -> set[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ConfigError(f"stage evidence body missing: {body_name}")
    result: set[int] = set()
    for geom_id in range(model.ngeom):
        ancestor = int(model.geom_bodyid[geom_id])
        while ancestor > 0 and ancestor != body_id:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == body_id:
            result.add(geom_id)
    return result


def capture_stage_evidence(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_root: str | Path,
    *,
    scene_id: str,
    robot_profile: str,
    stage: str,
    target_object_id: str,
    fingertip_body_names: Sequence[str],
    active_fingers: Sequence[bool],
    approach_path: np.ndarray | None,
    failure_reason: str,
    width: int = 320,
    height: int = 240,
) -> dict[str, Any]:
    """Render and annotate one exact rollout stage, returning immutable metadata."""
    for label, value in (
        ("scene_id", scene_id),
        ("robot_profile", robot_profile),
        ("stage", stage),
    ):
        if value in {".", ".."} or not _SAFE_KEY.fullmatch(value):
            raise ConfigError(f"unsafe {label} for stage evidence path: {value}")
    if width <= 0 or height <= 0:
        raise ConfigError("stage evidence dimensions must be positive")
    active = np.asarray(active_fingers, dtype=bool)
    if active.shape != (len(fingertip_body_names),):
        raise ConfigError("stage evidence active_fingers shape mismatch")

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.08]
    camera.distance = 0.55
    camera.azimuth = 135.0
    camera.elevation = -30.0
    target_geoms = _body_geom_ids(model, "target_object")
    hand_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id not in target_geoms
        and not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("scene_object::")
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "") != "floor"
    }
    original_rgba = np.asarray(model.geom_rgba, dtype=np.float32).copy()
    for geom_id in hand_geoms:
        model.geom_rgba[geom_id] = [0.55, 0.68, 0.86, 1.0]
    for geom_id in target_geoms:
        model.geom_rgba[geom_id] = [0.95, 0.28, 0.12, 1.0]
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        rgb = np.asarray(renderer.render()).copy()
        realized_camera = renderer.scene.camera[0]
        contact_points = _target_contact_points(model, data, "target_object")
        tip_points: list[np.ndarray] = []
        active_names: list[str] = []
        for is_active, body_name in zip(active, fingertip_body_names):
            if not is_active:
                continue
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ConfigError(f"stage evidence fingertip body missing: {body_name}")
            tip_points.append(np.asarray(data.xpos[body_id], dtype=np.float64).copy())
            active_names.append(body_name)
        tip_pixels = _project_points(
            np.asarray(tip_points, dtype=np.float64).reshape(-1, 3),
            realized_camera,
            width=width,
            height=height,
        )
        contact_pixels = _project_points(contact_points, realized_camera, width=width, height=height)
        path_pixels = np.empty((0, 2), dtype=np.float64)
        if approach_path is not None:
            path = np.asarray(approach_path, dtype=np.float64)
            if path.ndim != 3 or path.shape[1:] != (4, 4):
                raise ConfigError("stage evidence approach_path must have shape [N, 4, 4]")
            path_pixels = _project_points(path[:, :3, 3], realized_camera, width=width, height=height)
    finally:
        renderer.close()
        model.geom_rgba[:] = original_rgba

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 48), fill=(0, 0, 0))
    draw.text(
        (6, 4),
        f"{scene_id} | {robot_profile} | {stage}",
        fill=(255, 255, 255),
    )
    draw.text(
        (6, 18),
        f"target={target_object_id}",
        fill=(255, 220, 80),
    )
    draw.text(
        (6, 32),
        f"active={len(active_names)} | contacts={len(contact_points)} | failure={failure_reason}",
        fill=(255, 220, 80),
    )
    finite_path = path_pixels[np.all(np.isfinite(path_pixels), axis=1)]
    if len(finite_path) >= 2:
        draw.line([tuple(point) for point in finite_path], fill=(80, 180, 255), width=3)
    for point in tip_pixels[np.all(np.isfinite(tip_pixels), axis=1)]:
        x, y = point
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(80, 255, 80), width=2)
    for point in contact_pixels[np.all(np.isfinite(contact_pixels), axis=1)]:
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 60, 60), width=3)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    payload = buffer.getvalue()
    root = Path(output_root).resolve()
    path = root / "qa" / scene_id / robot_profile / f"{stage}.png"
    _atomic_write(path, payload)
    return {
        "scene_id": scene_id,
        "robot_profile": robot_profile,
        "stage": stage,
        "target_object_id": target_object_id,
        "failure_reason": failure_reason,
        "active_fingers": active_names,
        "contact_count": len(contact_points),
        "approach_waypoint_count": int(0 if approach_path is None else len(approach_path)),
        "image_ref": path.relative_to(root).as_posix(),
        "image_sha256": hashlib.sha256(payload).hexdigest(),
        "mjdata_time": float(data.time),
    }
