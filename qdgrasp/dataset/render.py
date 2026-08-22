"""Analytic sensor simulation: deterministic camera model and point cloud sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from .rng import sample_sphere_surface


@dataclass(frozen=True)
class CameraModel:
    """Standard pinhole camera intrinsic parameters."""

    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    @property
    def intrinsics_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def sample_analytic_point_cloud(
    mesh: trimesh.Trimesh,
    camera_pos: np.ndarray,
    camera_rot: np.ndarray,
    *,
    num_points: int = 1024,
    camera: CameraModel | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample object point cloud analytically in camera coordinate frame.

    Uses direct geometric surface interpolation on mesh faces without depending
    on GPU rasterizer drivers or non-deterministic trimesh sampling (§6.1).
    """
    if camera is None:
        camera = CameraModel()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if len(faces) == 0:
        # Fallback to vertices directly
        pts_world = vertices[:num_points]
    else:
        # Compute face areas for area-weighted sampling
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross_prod, axis=-1)
        total_area = np.sum(areas)
        if total_area > 0:
            probs = areas / total_area
        else:
            probs = np.ones(len(faces)) / len(faces)

        if rng is not None:
            face_indices = rng.choice(len(faces), size=num_points, p=probs)
            # Uniform barycentric coordinates
            r1 = rng.uniform(0.0, 1.0, size=num_points)
            r2 = rng.uniform(0.0, 1.0, size=num_points)
        else:
            # Deterministic linear spacing
            face_indices = np.linspace(0, len(faces) - 1, num_points, dtype=np.int64)
            r1 = np.linspace(0.1, 0.9, num_points)
            r2 = np.linspace(0.1, 0.9, num_points)

        sqrt_r1 = np.sqrt(r1)
        u = 1.0 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1.0 - u - v

        f_v0 = v0[face_indices]
        f_v1 = v1[face_indices]
        f_v2 = v2[face_indices]

        pts_world = (
            u[:, None] * f_v0 + v[:, None] * f_v1 + w[:, None] * f_v2
        )

    # Transform to camera coordinate frame: P_cam = R_cam^T (P_world - t_cam)
    R_cam = np.asarray(camera_rot, dtype=np.float64).reshape(3, 3)
    t_cam = np.asarray(camera_pos, dtype=np.float64).reshape(3)

    pts_cam = (R_cam.T @ (pts_world - t_cam).T).T

    metadata: dict[str, Any] = {
        "camera_intrinsics": camera.intrinsics_matrix.tolist(),
        "camera_pos": t_cam.tolist(),
        "camera_rot": R_cam.tolist(),
        "num_points": len(pts_cam),
        "frame": "camera",
    }

    return pts_cam.astype(np.float32), metadata
