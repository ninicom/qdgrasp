"""MuJoCo scene rendering with canonical object masks and packed evidence."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.contracts import CameraSpec, SceneObservation, SceneSpec

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


def _camera_id(model: mujoco.MjModel, camera_name: str) -> int:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ConfigError(f"scene camera not found in MuJoCo model: {camera_name}")
    return camera_id


def canonical_instance_mask(
    model: mujoco.MjModel,
    segmentation: np.ndarray,
    object_ids: Sequence[str],
) -> tuple[np.ndarray, dict[str, float]]:
    """Map MuJoCo geom segmentation to stable, one-based canonical object IDs."""
    raw = np.asarray(segmentation)
    if raw.ndim != 3 or raw.shape[2] != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ConfigError("MuJoCo segmentation must have shape [height, width, 2]")
    if len(object_ids) != len(set(object_ids)):
        raise ConfigError("canonical object IDs must be unique")
    instance_mask = np.zeros(raw.shape[:2], dtype=np.uint16)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_ids = raw[:, :, 0]
    object_types = raw[:, :, 1]

    for label, object_id in enumerate(object_ids, start=1):
        if label > np.iinfo(np.uint16).max:
            raise ConfigError("scene has too many objects for uint16 instance masks")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        if body_id < 0:
            raise ConfigError(f"canonical object body not found in MuJoCo model: {object_id}")
        object_geom_ids: list[int] = []
        for geom_id in range(model.ngeom):
            ancestor = int(model.geom_bodyid[geom_id])
            while ancestor > 0 and ancestor != body_id:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == body_id:
                object_geom_ids.append(geom_id)
        if not object_geom_ids:
            raise ConfigError(f"canonical object body has no renderable geoms: {object_id}")
        pixels = (object_types == geom_type) & np.isin(geom_ids, object_geom_ids)
        if np.any(instance_mask[pixels] != 0):
            raise ConfigError(f"segmentation geom ownership overlaps at object {object_id}")
        instance_mask[pixels] = label

    total_pixels = float(instance_mask.size)
    visibility = {
        object_id: float(np.count_nonzero(instance_mask == label) / total_pixels)
        for label, object_id in enumerate(object_ids, start=1)
    }
    return instance_mask, visibility


def render_camera_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    *,
    object_ids: Sequence[str] = (),
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    """Render RGB/depth/raw segmentation and a canonical instance mask."""
    if width <= 0 or height <= 0:
        raise ConfigError("render width and height must be positive")
    _camera_id(model, camera_name)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb = np.asarray(renderer.render()).copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        depth = np.asarray(renderer.render()).copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_name)
        segmentation = np.asarray(renderer.render()).copy()
    finally:
        renderer.close()
    if rgb.shape != (height, width, 3) or depth.shape != (height, width):
        raise ConfigError("MuJoCo renderer returned unexpected RGB/depth shapes")
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
        raise ConfigError("MuJoCo renderer returned invalid depth")
    instance_mask, visibility = canonical_instance_mask(model, segmentation, object_ids)
    return {
        "rgb": rgb,
        "depth": depth,
        "segmentation": segmentation,
        "instance_mask": instance_mask,
        "visibility": visibility,
    }


def _configure_camera_intrinsics(model: mujoco.MjModel, camera: CameraSpec, *, width: int, height: int) -> None:
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ConfigError(f"camera {camera.camera_id} intrinsics must be a finite 3x3 matrix")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    expected_cx, expected_cy = width / 2.0, height / 2.0
    if (
        fx <= 0.0
        or fy <= 0.0
        or not math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-6)
        or not math.isclose(float(intrinsics[0, 1]), 0.0, abs_tol=1e-8)
        or not math.isclose(float(intrinsics[0, 2]), expected_cx, abs_tol=1e-6)
        or not math.isclose(float(intrinsics[1, 2]), expected_cy, abs_tol=1e-6)
        or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ConfigError(
            f"camera {camera.camera_id} intrinsics are not representable by MuJoCo's centered fovy camera"
        )
    camera_id = _camera_id(model, camera.camera_id)
    model.cam_fovy[camera_id] = math.degrees(2.0 * math.atan(height / (2.0 * fy)))


def _calibration_hash(camera: CameraSpec) -> str:
    payload = {
        "T_world_camera": np.asarray(camera.T_world_camera, dtype=np.float64).tolist(),
        "intrinsics": np.asarray(camera.intrinsics, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _png_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _depth_to_camera_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth_array = np.asarray(depth, dtype=np.float64)
    rows, columns = np.indices(depth_array.shape)
    valid = np.isfinite(depth_array) & (depth_array > 0.0)
    z = depth_array[valid]
    return np.stack(
        [
            (columns[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (rows[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ],
        axis=1,
    ).astype(np.float32)


def build_scene_observation(
    spec: SceneSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: str,
    frame_id: str,
    output_root: str | Path,
    *,
    width: int = 640,
    height: int = 480,
    timestamp: float | None = None,
) -> SceneObservation:
    """Render and atomically pack one observation for a native scene shard."""
    for key, value in (("scene_id", spec.scene_id), ("camera_id", camera_id), ("frame_id", frame_id)):
        if value in {".", ".."} or not _SAFE_KEY.fullmatch(value):
            raise ConfigError(f"unsafe {key} for observation path: {value}")
    resolved_timestamp = float(data.time) if timestamp is None else float(timestamp)
    if not math.isfinite(resolved_timestamp):
        raise ConfigError("observation timestamp must be finite")
    camera = next((item for item in spec.cameras if item.camera_id == camera_id), None)
    if camera is None:
        raise ConfigError(f"camera {camera_id} is absent from SceneSpec {spec.scene_id}")
    _configure_camera_intrinsics(model, camera, width=width, height=height)
    rendered = render_camera_view(
        model,
        data,
        camera_id,
        object_ids=[item.object_id for item in spec.objects],
        width=width,
        height=height,
    )
    dataset_root = Path(output_root).resolve()
    output_dir = dataset_root / "observations" / spec.scene_id / camera_id / frame_id
    rgb_path = output_dir / "rgb.png"
    depth_path = output_dir / "depth.npy"
    mask_path = output_dir / "instance_mask.png"
    point_cloud_path = output_dir / "point_cloud.npy"
    depth_buffer = io.BytesIO()
    np.save(depth_buffer, rendered["depth"], allow_pickle=False)
    point_cloud_buffer = io.BytesIO()
    np.save(
        point_cloud_buffer,
        _depth_to_camera_points(rendered["depth"], np.asarray(camera.intrinsics, dtype=np.float64)),
        allow_pickle=False,
    )
    _atomic_write(rgb_path, _png_bytes(rendered["rgb"]))
    _atomic_write(depth_path, depth_buffer.getvalue())
    _atomic_write(mask_path, _png_bytes(rendered["instance_mask"]))
    _atomic_write(point_cloud_path, point_cloud_buffer.getvalue())
    return SceneObservation(
        scene_id=spec.scene_id,
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=resolved_timestamp,
        T_world_camera=np.asarray(camera.T_world_camera, dtype=np.float64).copy(),
        calibration_hash=_calibration_hash(camera),
        rgb_ref=rgb_path.relative_to(dataset_root).as_posix(),
        depth_ref=depth_path.relative_to(dataset_root).as_posix(),
        point_cloud_ref=point_cloud_path.relative_to(dataset_root).as_posix(),
        point_cloud_frame="camera",
        instance_mask_ref=mask_path.relative_to(dataset_root).as_posix(),
        visibility_by_object=rendered["visibility"],
    )


def scene_observation_record(observation: SceneObservation) -> dict[str, Any]:
    """Convert a packed observation to the canonical scene-shard record shape."""
    return {
        "record_type": "observation",
        "scene_id": observation.scene_id,
        "camera_id": observation.camera_id,
        "frame_id": observation.frame_id,
        "timestamp": observation.timestamp,
        "T_world_camera": observation.T_world_camera,
        "calibration_hash": observation.calibration_hash,
        "rgb_ref": observation.rgb_ref,
        "depth_ref": observation.depth_ref,
        "point_cloud_ref": observation.point_cloud_ref,
        "point_cloud_frame": observation.point_cloud_frame,
        "instance_mask_ref": observation.instance_mask_ref,
        "normal_ref": observation.normal_ref,
        "visibility_by_object": observation.visibility_by_object,
    }
