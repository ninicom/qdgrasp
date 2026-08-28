"""One frozen description of a grasp candidate, forked by both arms.

WRK-R2. The paired evidence in section 16.3 only means something if the two arms
describe the same grasp. Before this, the static arm certified the contacts a
recipe planned while the dynamic arm ran whatever the hand did, so a disagreement
between them could always have been a disagreement about which grasp was under
discussion rather than about frozen versus reactive physics.

A snapshot fixes everything both arms need and hashes it. Each arm takes the same
snapshot and changes exactly one thing: ``physics_mode``. Anything measured after
the dynamic rollout -- the contacts the fingers actually landed on, for instance
-- is diagnostic, and never the primary evidence, because it exists on only one
side of the comparison.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import numpy as np

#: The only thing the two arms are allowed to differ in.
PHYSICS_MODES: tuple[str, str] = ("frozen", "reactive")


def _plain(value: Any) -> Any:
    """JSON-able form of a value, with arrays rounded so hashes are stable."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 12)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclasses.dataclass(frozen=True)
class CandidateSnapshot:
    """Everything both arms read, and nothing either arm may change."""

    hand: str
    scene: str
    seed: int
    object_mass_kg: float
    friction_mu: float
    torsional_friction: float
    characteristic_length_m: float
    horizon_steps: int
    applied_wrench: tuple[float, ...]
    applied_wrench_hash: str
    contact_points: tuple[tuple[float, float, float], ...]
    contact_normals: tuple[tuple[float, float, float], ...]
    centroid: tuple[float, float, float]
    force_limit_N: float
    safety_budget_id: str
    recipe_id: str

    def digest(self) -> str:
        """Stable hash of the snapshot, identical for both arms."""
        payload = _plain(dataclasses.asdict(self))
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def fork(self, physics_mode: str) -> dict[str, Any]:
        """The arm's view: the snapshot, plus the one factor that differs."""
        if physics_mode not in PHYSICS_MODES:
            raise ValueError(
                f"physics_mode must be one of {PHYSICS_MODES}, not {physics_mode!r}"
            )
        return {
            "snapshot_hash": self.digest(),
            "physics_mode": physics_mode,
            "hand": self.hand,
            "scene": self.scene,
            "seed": self.seed,
            "object_mass_kg": self.object_mass_kg,
            "applied_wrench_hash": self.applied_wrench_hash,
        }


def one_factor_diff(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, ...]:
    """Keys where two forks differ. A controlled comparison differs in one."""
    keys = set(left) | set(right)
    return tuple(sorted(key for key in keys if left.get(key) != right.get(key)))
