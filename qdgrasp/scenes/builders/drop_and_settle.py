"""Bounded physical settling for native scene composition."""

from __future__ import annotations

import mujoco
import numpy as np

from qdgrasp.scenes.builders.base import build_scene_mujoco_model
from qdgrasp.scenes.contracts import SceneSpec


class SettlingError(RuntimeError):
    def __init__(self, reason: str, telemetry: dict[str, float]):
        self.reason = reason
        self.telemetry = telemetry
        super().__init__(f"{reason}: {telemetry}")


def _object_velocity_norms(
    model: mujoco.MjModel, data: mujoco.MjData, object_ids: list[str]
) -> tuple[float, float]:
    max_linear = 0.0
    max_angular = 0.0
    for object_id in object_ids:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        joint_id = int(model.body_jntadr[body_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        max_linear = max(max_linear, float(np.linalg.norm(data.qvel[dof_address : dof_address + 3])))
        max_angular = max(
            max_angular, float(np.linalg.norm(data.qvel[dof_address + 3 : dof_address + 6]))
        )
    return max_linear, max_angular


def drop_and_settle_scene(
    spec: SceneSpec,
    max_steps: int = 5000,
    *,
    linear_velocity_threshold: float = 0.002,
    angular_velocity_threshold: float = 0.05,
    consecutive_stable_steps: int = 25,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Drop all canonical objects and reject scenes that do not settle in time."""
    if max_steps <= 0 or consecutive_stable_steps <= 0:
        raise ValueError("settling step limits must be positive")
    model = build_scene_mujoco_model(spec, include_objects=True, dynamic_objects=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    object_ids = [scene_object.object_id for scene_object in spec.objects]
    if not object_ids:
        return model, data
    stable_steps = 0
    max_linear = float("inf")
    max_angular = float("inf")
    for _ in range(max_steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise SettlingError("scene_unstable", {"time": float(data.time)})
        max_linear, max_angular = _object_velocity_norms(model, data, object_ids)
        if (
            max_linear <= linear_velocity_threshold
            and max_angular <= angular_velocity_threshold
        ):
            stable_steps += 1
            if stable_steps >= consecutive_stable_steps:
                return model, data
        else:
            stable_steps = 0
    raise SettlingError(
        "settle_timeout",
        {
            "max_steps": float(max_steps),
            "max_linear_velocity": max_linear,
            "max_angular_velocity": max_angular,
            "stable_steps": float(stable_steps),
        },
    )
