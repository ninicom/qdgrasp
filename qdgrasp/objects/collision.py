"""Collision representation guard ensuring physical geoms match visual meshes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import trimesh

from ..config.schema import ConfigError
from .schema import SubGeomSpec

SUPPORTED_CONVEX_TYPES = {"box", "sphere", "cylinder", "capsule", "ellipsoid"}


def _geom_bounding_box(geom: SubGeomSpec) -> tuple[np.ndarray, np.ndarray]:
    """Compute local axis-aligned bounding box [min_pt, max_pt] for a convex subgeom."""
    pos = np.array(geom.pos, dtype=np.float64)
    if geom.type == "box":
        half = np.array(geom.size[:3], dtype=np.float64)
        return pos - half, pos + half
    if geom.type == "sphere":
        r = float(geom.size[0])
        half = np.array([r, r, r], dtype=np.float64)
        return pos - half, pos + half
    if geom.type in ("cylinder", "capsule"):
        r = float(geom.size[0])
        half_h = float(geom.size[1])
        # Cylinders/capsules aligned with Z axis in MuJoCo
        half = np.array([r, r, half_h + (r if geom.type == "capsule" else 0.0)], dtype=np.float64)
        return pos - half, pos + half
    if geom.type == "ellipsoid":
        half = np.array(geom.size[:3], dtype=np.float64)
        return pos - half, pos + half
    raise ConfigError(f"unsupported collision geom type: {geom.type}")


def validate_collision_representation(
    mesh: trimesh.Trimesh,
    collision_geoms: Sequence[SubGeomSpec],
    tolerance: float = 0.015,  # 15mm bounding envelope tolerance
) -> None:
    """Validate that physical collision geoms match visual geometry within tolerance.

    Guards against the MuJoCo convex hull mismatch defect:
    MuJoCo collapses concave visual meshes into their outer convex hull, causing
    physical contacts to occur on phantom geometry.  To be valid, every collision
    element must be an explicit elementary convex primitive, and their union must
    closely envelope the visual mesh bounds.
    """
    if not collision_geoms:
        raise ConfigError("object has no collision geometries declared")

    for i, geom in enumerate(collision_geoms):
        if geom.type not in SUPPORTED_CONVEX_TYPES:
            raise ConfigError(
                f"collision geom #{i} has unsupported non-convex type '{geom.type}'; "
                f"supported convex primitives: {sorted(SUPPORTED_CONVEX_TYPES)}"
            )

    # Compute bounding box of visual mesh
    v_bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    v_min, v_max = v_bounds[0], v_bounds[1]

    # Compute composite bounding box of collision geoms
    c_mins = []
    c_maxs = []
    for geom in collision_geoms:
        g_min, g_max = _geom_bounding_box(geom)
        c_mins.append(g_min)
        c_maxs.append(g_max)

    c_min = np.min(c_mins, axis=0)
    c_max = np.max(c_maxs, axis=0)

    # Check bounding box error
    diff_min = np.abs(v_min - c_min)
    diff_max = np.abs(v_max - c_max)
    max_error = float(max(np.max(diff_min), np.max(diff_max)))

    if max_error > tolerance:
        raise ConfigError(
            f"collision representation bounding mismatch: max discrepancy {max_error:.4f} m "
            f"exceeds tolerance {tolerance:.4f} m (visual: {v_min.round(3)}..{v_max.round(3)}, "
            f"collision: {c_min.round(3)}..{c_max.round(3)})"
        )
