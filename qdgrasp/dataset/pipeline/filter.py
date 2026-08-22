"""Collision and physical plausibility filtering for candidate grasps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def filter_grasp_candidate(
    spec: RobotSpec,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    q: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    max_penetration: float = 0.02,  # 20mm penetration limit
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

    tips = spec.fingertip_positions(t_palm_pos, t_palm_rot, t_q)[0].numpy()  # [K, 3]
    num_tips = len(tips)

    # Pairwise fingertip distance check (self-collision)
    min_tip_dist = float("inf")
    for i in range(num_tips):
        for j in range(i + 1, num_tips):
            dist = float(np.linalg.norm(tips[i] - tips[j]))
            if dist < min_tip_dist:
                min_tip_dist = dist

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

    # Approximate penetration using signed distance / closest point
    # Sample points on palm and fingertips
    probe_points = np.vstack([palm_pos.reshape(1, 3), tips])
    # Fast bounding box signed distance approximation
    bounds = mesh.bounds
    b_min, b_max = bounds[0], bounds[1]

    max_penetration_found = 0.0
    for pt in probe_points:
        # Check if inside bounding box
        if np.all(pt >= b_min) and np.all(pt <= b_max):
            # Point is inside bounding box: compute distance to surface
            d_to_face = np.min(np.abs(np.concatenate([pt - b_min, b_max - pt])))
            max_penetration_found = max(max_penetration_found, float(d_to_face))

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
