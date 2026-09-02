"""Deterministic hierarchical random number generation for datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def derive_seed(base_seed: int, *path: str | int) -> int:
    """Derive a deterministic 64-bit integer seed from a base seed and hierarchy path.

    Uses cryptographic SHA-256 domain separation to ensure complete statistical
    independence between different splits, object IDs, candidate IDs, and runs.
    """
    token = f"{base_seed}:{':'.join(str(p) for p in path)}".encode()
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def get_generator(base_seed: int, *path: str | int) -> np.random.Generator:
    """Create a standalone NumPy Generator backed by the PCG64 bit generator."""
    seed = derive_seed(base_seed, *path)
    return np.random.Generator(np.random.PCG64(seed))


def sample_quaternion_so3(rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """Sample uniform random quaternions on SO(3) using Shoemake's subgroup algorithm.

    Returns quaternions in ``(w, x, y, z)`` format with unit norm.
    If ``size`` is None, returns shape (4,), else (size, 4).
    """
    if size is None:
        u1, u2, u3 = rng.uniform(0.0, 1.0, size=3)
        w = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
        x = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
        y = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
        z = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
        return np.array([w, x, y, z], dtype=np.float64)

    u = rng.uniform(0.0, 1.0, size=(size, 3))
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    w = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    x = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.stack([w, x, y, z], axis=-1).astype(np.float64)


def sample_sphere_surface(rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Sample uniform points on the unit sphere S^2.

    Returns array of shape (size, 3).
    """
    # Marsaglia / inverse transform sampling
    z = rng.uniform(-1.0, 1.0, size=size)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=size)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


def sample_box_uniform(
    rng: np.random.Generator,
    bounds: Sequence[tuple[float, float]],
    size: int = 1,
) -> np.ndarray:
    """Sample uniform points inside axis-aligned bounding intervals."""
    mins = np.array([b[0] for b in bounds], dtype=np.float64)
    maxs = np.array([b[1] for b in bounds], dtype=np.float64)
    u = rng.uniform(0.0, 1.0, size=(size, len(bounds)))
    return mins + u * (maxs - mins)
