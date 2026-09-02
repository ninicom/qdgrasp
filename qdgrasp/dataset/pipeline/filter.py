"""Collision and physical plausibility filtering for candidate grasps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh

from ...robot.spec import RobotSpec


@dataclass(frozen=True)
class CollisionFilterResult:
    """Outcome of collision and kinematic sanity filtering."""

    valid: bool
    reason: str
    min_tip_dist: float
    estimated_penetration: float


def _point_inside_watertight_mesh(point: np.ndarray, mesh: trimesh.Trimesh) -> bool:
    """Brute-force odd/even ray test without the optional rtree dependency."""
    if not mesh.is_watertight:
        return False
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    origin = np.asarray(point, dtype=np.float64)
    direction = np.array([1.0, 0.37139068, 0.127831], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(a) > 1e-10
    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]
    s = origin - triangles[:, 0]
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = f * (q @ direction)
    t = f * np.einsum("ij,ij->i", edge2, q)
    hits = valid & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > 1e-9)
    return bool(np.count_nonzero(hits) % 2 == 1)


def filter_grasp_candidate(
    spec: RobotSpec,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    q: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    max_penetration: float = 0.002,
    min_fingertip_distance: float = 0.008,  # 8mm self-collision clearance
    max_reach_distance: float = 0.25,
) -> CollisionFilterResult:
    """Filter out physically infeasible grasp candidates before simulation.

    1. Joint limit verification.
    2. Self-collision clearance between non-adjacent fingertips.
    3. Hand-object penetration depth check.
    4. Proximity reachability check.
    """
    # 1. Check joint limits
    for j_idx, j_name in enumerate(spec.actuated_joint_names):
        val = float(q[j_idx])
        lo, hi = spec.joint_limits[j_name]
        if val < lo - 1e-4 or val > hi + 1e-4:
            return CollisionFilterResult(
                valid=False,
                reason=f"joint_limit_violation on {j_name} ({val:.4f} not in [{lo:.4f}, {hi:.4f}])",
                min_tip_dist=0.0,
                estimated_penetration=0.0,
            )

    # 2. Compute fingertip positions
    t_palm_pos = torch.from_numpy(np.array(palm_pos, copy=True, dtype=np.float32)).view(1, 3)
    t_palm_rot = torch.from_numpy(np.array(palm_rot, copy=True, dtype=np.float32)).view(1, 3, 3)
    t_q = torch.from_numpy(np.array(q, copy=True, dtype=np.float32)).view(1, -1)

    transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, t_q)
    tips = spec.fingertip_positions(t_palm_pos, t_palm_rot, t_q)[0].numpy()
    num_tips = len(tips)

    # Pairwise fingertip distance check (self-collision)
    min_tip_dist = float("inf")
    for i in range(num_tips):
        for j in range(i + 1, num_tips):
            dist = float(np.linalg.norm(tips[i] - tips[j]))
            min_tip_dist = min(min_tip_dist, dist)

    if num_tips > 1 and min_tip_dist < min_fingertip_distance:
        return CollisionFilterResult(
            valid=False,
            reason=f"self_collision: min fingertip distance {min_tip_dist:.4f} < {min_fingertip_distance:.4f}",
            min_tip_dist=min_tip_dist,
            estimated_penetration=0.0,
        )

    # 3. Object proximity and penetration check
    obj_center = mesh.centroid
    palm_dist = float(np.linalg.norm(palm_pos - obj_center))
    if palm_dist > max_reach_distance:
        return CollisionFilterResult(
            valid=False,
            reason=f"out_of_reach: palm distance {palm_dist:.4f} > {max_reach_distance:.4f}",
            min_tip_dist=min_tip_dist,
            estimated_penetration=0.0,
        )

    # Probe the complete declared contact chain against the actual watertight mesh.  An AABB is
    # not a collision representation for cylinders, superquadrics or concave
    # compounds.
    probe_parts = [palm_pos.reshape(1, 3), tips]
    for link_name in spec.contact_links:
        if link_name not in transforms:
            continue
        child = transforms[link_name][0, :3, 3].numpy()
        probe_parts.append(child.reshape(1, 3))
        parent_name = spec.links[link_name].parent_link
        if parent_name in transforms:
            parent = transforms[parent_name][0, :3, 3].numpy()
            fractions = np.linspace(0.2, 0.8, 4)[:, None]
            probe_parts.append(parent + fractions * (child - parent))
    probe_points = np.vstack(probe_parts)
    _, distances, _ = trimesh.proximity.closest_point_naive(mesh, probe_points)

    max_penetration_found = 0.0
    for pt, distance in zip(probe_points, distances):
        if _point_inside_watertight_mesh(pt, mesh):
            max_penetration_found = max(max_penetration_found, float(distance))

    if max_penetration_found > max_penetration:
        return CollisionFilterResult(
            valid=False,
            reason=f"excessive_penetration: {max_penetration_found:.4f} > {max_penetration:.4f}",
            min_tip_dist=min_tip_dist,
            estimated_penetration=max_penetration_found,
        )

    return CollisionFilterResult(
        valid=True,
        reason="passed",
        min_tip_dist=min_tip_dist,
        estimated_penetration=max_penetration_found,
    )
