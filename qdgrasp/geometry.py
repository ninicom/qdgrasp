"""Rotation helpers shared by heads, results and export paths."""

from __future__ import annotations

import torch


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Map a ``[..., 6]`` continuous representation onto ``SO(3)``.

    The two input vectors are orthonormalised with Gram-Schmidt and the third
    column is their cross product, so the result is a proper rotation with
    ``det == +1`` for any finite input.
    """

    if not torch.jit.is_tracing() and rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d expects a trailing dimension of 6, got {tuple(rot6d.shape)}")
    first, second = rot6d[..., :3], rot6d[..., 3:]
    column0 = torch.nn.functional.normalize(first, dim=-1, eps=1e-8)
    projection = (column0 * second).sum(dim=-1, keepdim=True) * column0
    column1 = torch.nn.functional.normalize(second - projection, dim=-1, eps=1e-8)
    column2 = torch.cross(column0, column1, dim=-1)
    return torch.stack((column0, column1, column2), dim=-1)


def is_rotation_matrix(matrix: torch.Tensor, *, atol: float = 1e-4) -> bool:
    """Return whether every ``[..., 3, 3]`` entry is orthonormal with positive determinant."""

    if matrix.shape[-2:] != (3, 3):
        return False
    matrix = matrix.detach().to(torch.float64)
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
    orthonormal = torch.allclose(matrix @ matrix.transpose(-1, -2), identity, atol=atol)
    return bool(orthonormal and torch.all(torch.det(matrix) > 0))
