"""Differentiable Forward Kinematics and Batch Kinematics in pure PyTorch."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


def rpy_to_rotation_matrix(rpy: tuple[float, float, float] | torch.Tensor, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu") -> torch.Tensor:
    """Compute 3x3 rotation matrix from roll-pitch-yaw (XYZ extrinsic / ZYX intrinsic)."""
    if isinstance(rpy, tuple):
        r, p, y = rpy
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)

        # R = Rz(y) * Ry(p) * Rx(r)
        R = torch.tensor(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=dtype,
            device=device,
        )
        return R

    # Tensor input [B, 3]
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quaternion_to_rotation_matrix(
    quaternion: tuple[float, float, float, float],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a 3x3 rotation matrix from a ``(w, x, y, z)`` quaternion.

    Used instead of a quaternion-to-RPY-to-matrix detour: the Euler round trip
    is degenerate at ``|pitch| == 90 deg``, where roll and yaw are no longer
    separable and the recovered triple describes a different rotation.  Several
    Menagerie bodies sit exactly on that singularity.
    """

    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=dtype,
        device=device,
    )


def rodrigues_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Compute batched 3x3 rotation matrix for rotation around unit axis by angle.

    Args:
        axis: Tensor of shape [3] or [B, 3] (unit vector).
        angle: Tensor of shape [B] or [B, 1].

    Returns:
        Tensor of shape [B, 3, 3].
    """
    if angle.ndim == 1:
        angle = angle.unsqueeze(-1)  # [B, 1]
    B = angle.shape[0]

    if axis.ndim == 1:
        axis = axis.unsqueeze(0).expand(B, 3)  # [B, 3]
    axis = F.normalize(axis, dim=-1)

    ax, ay, az = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    zero = torch.zeros_like(ax)

    # Cross-product skew-symmetric matrix K
    K_row0 = torch.cat([zero, -az, ay], dim=-1)
    K_row1 = torch.cat([az, zero, -ax], dim=-1)
    K_row2 = torch.cat([-ay, ax, zero], dim=-1)
    K = torch.stack([K_row0, K_row1, K_row2], dim=-2)  # [B, 3, 3]

    K2 = torch.bmm(K, K)  # [B, 3, 3]
    I = torch.eye(3, dtype=angle.dtype, device=angle.device).unsqueeze(0).expand(B, 3, 3)

    sin_a = torch.sin(angle).unsqueeze(-1)  # [B, 1, 1]
    cos_a = torch.cos(angle).unsqueeze(-1)  # [B, 1, 1]

    R = I + sin_a * K + (1.0 - cos_a) * K2
    return R


def compute_joint_transform(
    joint_type: str,
    axis: tuple[float, float, float] | torch.Tensor,
    origin_xyz: tuple[float, float, float],
    origin_rotation: Sequence[Sequence[float]] | torch.Tensor,
    q: torch.Tensor,  # [B]
) -> torch.Tensor:
    """Compute [B, 4, 4] homogeneous transform for a joint given its q value.

    ``origin_rotation`` is the parent-to-child rotation as a 3x3 matrix.  Taking
    the matrix rather than an RPY triple keeps the transform exact for bodies
    that sit on the Euler singularity.
    """

    B = q.shape[0]
    device = q.device
    dtype = q.dtype

    # Base origin transform
    if isinstance(origin_rotation, torch.Tensor):
        R_orig = origin_rotation.to(device=device, dtype=dtype)
    else:
        R_orig = torch.tensor(origin_rotation, dtype=dtype, device=device)
    R_orig = R_orig.reshape(3, 3).unsqueeze(0).expand(B, 3, 3)
    t_orig = torch.tensor(origin_xyz, dtype=dtype, device=device).unsqueeze(0).expand(B, 3)

    if joint_type == "fixed":
        R_local = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, 3, 3)
        t_local = torch.zeros((B, 3), dtype=dtype, device=device)
    elif joint_type in ("revolute", "continuous"):
        if isinstance(axis, tuple):
            ax = torch.tensor(axis, dtype=dtype, device=device)
        else:
            ax = axis.to(device=device, dtype=dtype)
        R_local = rodrigues_rotation(ax, q)
        t_local = torch.zeros((B, 3), dtype=dtype, device=device)
    elif joint_type == "prismatic":
        R_local = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, 3, 3)
        if isinstance(axis, tuple):
            ax = torch.tensor(axis, dtype=dtype, device=device)
        else:
            ax = axis.to(device=device, dtype=dtype)
        ax_norm = F.normalize(ax.unsqueeze(0), dim=-1)
        t_local = q.unsqueeze(-1) * ax_norm
    else:
        R_local = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, 3, 3)
        t_local = torch.zeros((B, 3), dtype=dtype, device=device)

    # T_joint = T_origin * T_motion
    # R_total = R_orig * R_local
    # t_total = R_orig * t_local + t_orig
    R_total = torch.bmm(R_orig, R_local)
    t_total = torch.bmm(R_orig, t_local.unsqueeze(-1)).squeeze(-1) + t_orig

    T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    T[:, :3, :3] = R_total
    T[:, :3, 3] = t_total
    return T


def invert_rigid_transform(T: torch.Tensor) -> torch.Tensor:
    """Invert a batch of rigid ``[B, 4, 4]`` transforms exactly.

    Uses the closed form ``(R', -R'@t)`` rather than a general matrix inverse:
    it is exact in FP32, cheaper, and keeps the round-trip bit-exact so a link
    expressed relative to itself comes back as the identity.
    """

    if T.shape[-2:] != (4, 4):
        raise ValueError(f"expected [..., 4, 4] transforms, got {tuple(T.shape)}")
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inverse = R.transpose(-1, -2)
    inverted = torch.zeros_like(T)
    inverted[..., :3, :3] = R_inverse
    inverted[..., :3, 3:4] = -torch.matmul(R_inverse, t)
    inverted[..., 3, 3] = 1.0
    return inverted


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform points by homogeneous matrix T.

    Args:
        T: Tensor of shape [B, 4, 4] or [4, 4]
        points: Tensor of shape [P, 3] or [B, P, 3]

    Returns:
        Tensor of shape [B, P, 3]
    """
    if T.ndim == 2:
        T = T.unsqueeze(0)
    B = T.shape[0]

    if points.ndim == 2:
        P = points.shape[0]
        points = points.unsqueeze(0).expand(B, P, 3)
    elif points.shape[0] != B:
        points = points.expand(B, -1, 3)

    R = T[:, :3, :3]  # [B, 3, 3]
    t = T[:, :3, 3:4]  # [B, 3, 1]

    # [B, P, 3] = [B, P, 3] x [B, 3, 3]^T + [B, 1, 3]
    transformed = torch.baddbmm(t.transpose(1, 2), points, R.transpose(1, 2))
    return transformed
