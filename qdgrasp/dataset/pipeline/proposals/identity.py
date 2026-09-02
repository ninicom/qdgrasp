"""Stable proposal identity and active-set validation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np


def normalize_active_fingers(
    active_fingers: np.ndarray | None,
    num_fingers: int,
    *,
    min_active_fingers: int = 2,
) -> np.ndarray:
    """Return a validated boolean active set, defaulting legacy proposals to all."""
    if active_fingers is None:
        active = np.ones(num_fingers, dtype=bool)
    else:
        active = np.asarray(active_fingers, dtype=bool)
        if active.shape != (num_fingers,):
            raise ValueError(
                f"active_fingers must have shape ({num_fingers},), got {active.shape}"
            )
        active = active.copy()
    if int(active.sum()) < min_active_fingers:
        raise ValueError(
            f"proposal requires at least {min_active_fingers} active fingers, "
            f"got {int(active.sum())}"
        )
    return active


def stable_candidate_id(
    provenance: str,
    *,
    target_points: np.ndarray,
    inward_normals: np.ndarray,
    face_ids: np.ndarray,
    finger_ids: np.ndarray,
    active_fingers: np.ndarray,
    opposition_pairs: np.ndarray | None = None,
) -> str:
    """Hash canonical proposal content without depending on object identity/CWD."""
    digest = hashlib.sha256(provenance.encode("utf-8"))
    values: Iterable[tuple[np.ndarray, np.dtype]] = (
        (np.asarray(target_points), np.dtype("<f8")),
        (np.asarray(inward_normals), np.dtype("<f8")),
        (np.asarray(face_ids), np.dtype("<i8")),
        (np.asarray(finger_ids), np.dtype("<i8")),
        (np.asarray(active_fingers), np.dtype("u1")),
    )
    for value, dtype in values:
        canonical = np.ascontiguousarray(value, dtype=dtype)
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    if opposition_pairs is not None:
        pairs = np.ascontiguousarray(opposition_pairs, dtype="<i8")
        digest.update(np.asarray(pairs.shape, dtype="<i8").tobytes())
        digest.update(pairs.tobytes())
    return f"{provenance}:{digest.hexdigest()[:20]}"
