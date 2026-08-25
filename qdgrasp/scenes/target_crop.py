"""Deterministic target-local point-cloud crops from canonical scene observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qdgrasp.config.schema import ConfigError


@dataclass(frozen=True)
class TargetSceneCrop:
    points_object_frame: np.ndarray
    target_point_mask: np.ndarray
    source_pixel_indices: np.ndarray


def _require_transform(value: np.ndarray, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ConfigError(f"{label} must be a finite 4x4 transform")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ConfigError(f"{label} is not a rigid transform")
    return transform


def build_target_scene_crop(
    depth: np.ndarray,
    instance_mask: np.ndarray,
    *,
    target_label: int,
    intrinsics: np.ndarray,
    T_world_camera: np.ndarray,
    T_world_object: np.ndarray,
    context_radius: float,
    max_points: int | None = None,
) -> TargetSceneCrop:
    """Unproject depth and retain target/context points in the target object frame."""
    depth_array = np.asarray(depth, dtype=np.float64)
    mask_array = np.asarray(instance_mask)
    calibration = np.asarray(intrinsics, dtype=np.float64)
    if depth_array.ndim != 2 or depth_array.shape != mask_array.shape or depth_array.size == 0:
        raise ConfigError("depth and instance mask must be non-empty arrays with the same [H, W] shape")
    if calibration.shape != (3, 3) or not np.all(np.isfinite(calibration)):
        raise ConfigError("target crop intrinsics must be a finite 3x3 matrix")
    if target_label <= 0 or not np.any(mask_array == target_label):
        raise ConfigError(f"target label is absent from instance mask: {target_label}")
    if not math.isfinite(context_radius) or context_radius <= 0.0:
        raise ConfigError("target crop context radius must be finite and positive")
    if max_points is not None and max_points <= 0:
        raise ConfigError("target crop max_points must be positive")
    fx, fy = float(calibration[0, 0]), float(calibration[1, 1])
    if fx <= 0.0 or fy <= 0.0 or not np.allclose(calibration[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ConfigError("target crop intrinsics have invalid focal length or homogeneous row")
    world_camera = _require_transform(T_world_camera, "target crop T_world_camera")
    world_object = _require_transform(T_world_object, "target crop T_world_object")

    valid = np.isfinite(depth_array) & (depth_array > 0.0)
    rows, columns = np.nonzero(valid)
    if len(rows) == 0:
        raise ConfigError("target crop contains no valid depth pixels")
    z = depth_array[rows, columns]
    camera_points = np.stack(
        [
            (columns - calibration[0, 2]) * z / fx,
            (rows - calibration[1, 2]) * z / fy,
            z,
            np.ones_like(z),
        ],
        axis=1,
    )
    object_camera = np.linalg.inv(world_object) @ world_camera
    object_points = (object_camera @ camera_points.T).T[:, :3]
    in_context = np.linalg.norm(object_points, axis=1) <= context_radius
    if not np.any(in_context):
        raise ConfigError("target crop context radius contains no valid points")
    selected_points = object_points[in_context]
    selected_pixels = np.stack([rows[in_context], columns[in_context]], axis=1).astype(np.int32)
    selected_target = (mask_array[rows[in_context], columns[in_context]] == target_label).astype(bool)
    if not np.any(selected_target):
        raise ConfigError("target depth points are absent after context cropping")

    if max_points is not None and len(selected_points) > max_points:
        indices = np.linspace(0, len(selected_points) - 1, num=max_points, dtype=np.int64)
        selected_points = selected_points[indices]
        selected_pixels = selected_pixels[indices]
        selected_target = selected_target[indices]
        if not np.any(selected_target):
            target_index = int(np.flatnonzero(in_context & (mask_array[rows, columns] == target_label))[0])
            selected_points[-1] = object_points[target_index]
            selected_pixels[-1] = [rows[target_index], columns[target_index]]
            selected_target[-1] = True

    return TargetSceneCrop(
        points_object_frame=np.asarray(selected_points, dtype=np.float32),
        target_point_mask=selected_target,
        source_pixel_indices=selected_pixels,
    )
