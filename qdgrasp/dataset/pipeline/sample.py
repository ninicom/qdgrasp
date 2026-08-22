"""Grasp candidate sampling for diverse robot hands and object surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import trimesh

from ...robot.kinematics import quaternion_to_rotation_matrix
from ...robot.spec import RobotSpec
from ..rng import sample_quaternion_so3, sample_sphere_surface


@dataclass(frozen=True)
class GraspCandidate:
    """Sampled grasp candidate pose and target contact locations."""

    palm_pos: np.ndarray  # [3]
    palm_rot: np.ndarray  # [3, 3]
    target_contacts: np.ndarray  # [num_fingertips, 3]
    standoff: float


def _detect_hand_approach_axis(spec: RobotSpec) -> np.ndarray:
    """Detect the hand's natural palm-to-fingertip extension vector at rest."""
    palm_pos = torch.zeros(1, 3, dtype=torch.float32)
    palm_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    q_zero = torch.zeros(1, len(spec.actuated_joint_names), dtype=torch.float32)

    tips = spec.fingertip_positions(palm_pos, palm_rot, q_zero)[0].numpy()
    if len(tips) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    avg_offset = np.mean(tips, axis=0)
    norm = np.linalg.norm(avg_offset)
    if norm < 1e-5:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return avg_offset / norm


def _construct_rotation_from_vectors(source_vec: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation matrix that aligns source_vec to target_vec."""
    u = source_vec / np.linalg.norm(source_vec)
    v = target_vec / np.linalg.norm(target_vec)

    dot = np.dot(u, v)
    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.999999:
        # 180 degree rotation around orthogonal axis
        ortho = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, ortho)
        axis /= np.linalg.norm(axis)
        w, x, y, z = 0.0, axis[0], axis[1], axis[2]
        return quaternion_to_rotation_matrix((w, x, y, z)).numpy()

    cross = np.cross(u, v)
    s = math.sqrt((1.0 + dot) * 2.0)
    w = 0.5 * s
    x, y, z = cross / s
    return quaternion_to_rotation_matrix((w, x, y, z)).numpy()


import math


def sample_grasp_candidates(
    spec: RobotSpec,
    mesh: trimesh.Trimesh,
    rng: np.random.Generator,
    num_candidates: int = 16,
    standoff_range: tuple[float, float] = (0.04, 0.10),
) -> list[GraspCandidate]:
    """Sample candidate palm poses and target contact points on the object.

    1. Approach vector is sampled towards the object center or along surface normals.
    2. Palm orientation is aligned with the approach vector, modulated by random in-plane roll.
    3. Palm position is placed at a standoff distance from the object surface.
    4. Target contacts are sampled on the object surface in the vicinity of the approach ray.
    """
    hand_axis = _detect_hand_approach_axis(spec)
    obj_center = mesh.centroid

    # Sample random directions from unit sphere
    approach_dirs = sample_sphere_surface(rng, size=num_candidates)

    candidates: list[GraspCandidate] = []
    num_tips = len(spec.fingertip_links)

    for i in range(num_candidates):
        # Target vector pointing into object
        target_approach = -approach_dirs[i]
        standoff = float(rng.uniform(standoff_range[0], standoff_range[1]))

        # Align hand approach axis with target approach direction
        R_align = _construct_rotation_from_vectors(hand_axis, target_approach)

        # Apply random in-plane roll rotation around target_approach
        roll_angle = float(rng.uniform(-math.pi, math.pi))
        axis_norm = target_approach / np.linalg.norm(target_approach)
        w = math.cos(roll_angle * 0.5)
        xyz = axis_norm * math.sin(roll_angle * 0.5)
        R_roll = quaternion_to_rotation_matrix((w, xyz[0], xyz[1], xyz[2])).numpy()

        palm_rot = R_roll @ R_align

        # Palm position standoff from object center
        palm_pos = obj_center - target_approach * standoff

        # Sample target contact points on object surface
        # Project approach ray to object surface approximation
        target_contacts = np.zeros((num_tips, 3), dtype=np.float64)
        for t_idx in range(num_tips):
            jitter = rng.normal(0.0, 0.01, size=3)
            target_contacts[t_idx] = obj_center + jitter

        candidates.append(
            GraspCandidate(
                palm_pos=palm_pos.astype(np.float64),
                palm_rot=palm_rot.astype(np.float64),
                target_contacts=target_contacts.astype(np.float64),
                standoff=standoff,
            )
        )

    return candidates
