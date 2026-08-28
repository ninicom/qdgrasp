"""The disturbance a rollout applies, resolved in one place.

WRK-R2. The ablation used to re-derive this formula beside the validator's copy.
Two implementations of one policy drift, and the first drift was silent: reading
only ``rollout_kwargs`` scored a hand at zero disturbance while the validator was
disturbing it all along, and the frozen test was reported as passing on that
basis. There is one function now, and both callers use it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

#: Fractions of the object's weight used when a recipe declares no wrench.
FORCE_FRACTION: float = 0.5
TORQUE_FRACTION: float = 0.25

#: Fallback characteristic length when a scene declares no geometry, in metres.
DEFAULT_CHARACTERISTIC_LENGTH_M: float = 0.05


def characteristic_length(collision_geoms: Sequence[Any]) -> float:
    """Twice the largest half-extent among the target's geoms."""
    sizes = [
        2.0 * float(np.max(np.asarray(geom.size, dtype=np.float64)))
        for geom in collision_geoms
    ]
    return max(sizes) if sizes else DEFAULT_CHARACTERISTIC_LENGTH_M


def resolve_perturbation_wrench(
    declared: Any | None,
    *,
    object_mass: float,
    collision_geoms: Sequence[Any],
    gravity_magnitude: float = 9.81,
) -> np.ndarray:
    """The wrench the protocol will apply, declared or derived.

    A recipe that names no wrench does not go undisturbed: the disturbance is
    derived from the object's own weight and size, so every hand faces one.
    """
    if declared is not None:
        return np.asarray(declared, dtype=np.float64).reshape(6)

    weight = float(object_mass) * float(gravity_magnitude)
    length = characteristic_length(collision_geoms)
    force = FORCE_FRACTION * weight
    torque = TORQUE_FRACTION * weight * length
    return np.array([force, force, 0.0, torque, torque, torque], dtype=np.float64)


def wrench_hash(wrench: np.ndarray) -> str:
    """Stable digest of an applied wrench, for pinning it to a record."""
    values = [round(float(v), 12) for v in np.asarray(wrench, dtype=np.float64).reshape(6)]
    blob = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
