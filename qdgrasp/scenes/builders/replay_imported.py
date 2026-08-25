"""Exact kinematic replay of an imported canonical scene."""

from __future__ import annotations

import mujoco

from qdgrasp.scenes.builders.base import build_scene_mujoco_model
from qdgrasp.scenes.contracts import SceneSpec


def build_replay_scene(spec: SceneSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile source poses without integrating or reconciling physics."""
    model = build_scene_mujoco_model(spec, include_objects=True, dynamic_objects=False)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
