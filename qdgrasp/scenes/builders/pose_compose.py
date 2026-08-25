"""Pose-template scene composition with bounded physical reconciliation."""

from __future__ import annotations

import mujoco

from qdgrasp.scenes.builders.drop_and_settle import drop_and_settle_scene
from qdgrasp.scenes.contracts import SceneSpec


def compose_scene(
    spec: SceneSpec, *, repair_steps: int = 1000
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile a pose template and settle it without silently changing thresholds."""
    return drop_and_settle_scene(
        spec,
        max_steps=repair_steps,
        consecutive_stable_steps=10,
    )
