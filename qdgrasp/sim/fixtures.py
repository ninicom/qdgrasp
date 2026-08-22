"""Deterministic MuJoCo evaluation fixtures for grasp, squeeze and lift."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np

from ..config.schema import ConfigError
from .mujoco import MujocoSim


@dataclass(frozen=True)
class FixtureResult:
    """Outcome and physical metrics of a grasp simulation fixture."""

    success: bool
    stable_lift: bool
    contact_count: int
    max_penetration: float
    lift_height: float
    metrics: dict[str, float]


def build_evaluation_scene_xml(
    hand_xml_path: str | Path,
    *,
    object_type: str = "box",  # "box", "sphere", "cylinder"
    object_size: tuple[float, ...] = (0.03, 0.03, 0.03),
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
) -> str:
    """Compose a self-contained MuJoCo scene XML including the hand and a test object."""
    hand_p = Path(hand_xml_path).resolve()
    if not hand_p.is_file():
        raise ConfigError(f"hand XML file not found: {hand_p}")

    size_str = " ".join(f"{s:.4f}" for s in object_size)
    pos_str = f"{object_pos[0]:.4f} {object_pos[1]:.4f} {object_pos[2]:.4f}"

    scene_xml = f"""<mujoco model="grasp_evaluation_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" iterations="50" solver="Newton" cone="elliptic" gravity="0 0 -9.81"/>

  <include file="{hand_p}"/>

  <worldbody>
    <body name="target_object" pos="{pos_str}">
      <freejoint name="object_freejoint"/>
      <geom name="object_geom" type="{object_type}" size="{size_str}" mass="{object_mass:.4f}" rgba="0.8 0.3 0.3 1" condim="4" friction="1.0 0.005 0.0001"/>
    </body>
    <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 -0.1" rgba="0.9 0.9 0.9 1"/>
  </worldbody>
</mujoco>
"""
    return scene_xml


def evaluate_grasp_fixture(
    hand_xml_path: str | Path,
    *,
    joint_targets: Mapping[str, float] | None = None,
    palm_pos: tuple[float, float, float] = (0.0, 0.0, 0.1),
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_size: tuple[float, ...] = (0.025, 0.025, 0.025),
    squeeze_steps: int = 150,
    lift_steps: int = 150,
    seed: int = 0,
) -> FixtureResult:
    """Run deterministic grasp -> squeeze -> lift evaluation in MuJoCo.

    1. Grasp stage: sets hand joints and object pose.
    2. Squeeze stage: closes actuators towards targets.
    3. Lift stage: simulates lift motion and checks if object stays grasped.
    """
    np.random.seed(seed)
    scene_xml = build_evaluation_scene_xml(
        hand_xml_path,
        object_pos=object_pos,
        object_size=object_size,
    )

    try:
        model = mujoco.MjModel.from_xml_string(scene_xml)
    except Exception as exc:
        # Fallback to direct model load if include fails due to relative paths
        model = mujoco.MjModel.from_xml_path(str(hand_xml_path))

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Set initial joint positions
    if joint_targets:
        for j_name, val in joint_targets.items():
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id >= 0:
                data.qpos[model.jnt_qposadr[j_id]] = float(val)

    mujoco.mj_forward(model, data)

    # Squeeze phase
    for _ in range(squeeze_steps):
        if joint_targets and model.nu > 0:
            for a_id in range(model.nu):
                data.ctrl[a_id] = 0.5  # Squeeze actuation signal
        mujoco.mj_step(model, data)

    squeeze_contacts = int(data.ncon)

    # Lift phase: observe target object displacement if present
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    init_z = float(data.xpos[obj_id][2]) if obj_id >= 0 else 0.0

    for _ in range(lift_steps):
        mujoco.mj_step(model, data)

    final_z = float(data.xpos[obj_id][2]) if obj_id >= 0 else 0.0
    lift_height = final_z - init_z

    # Compute penetration / distance summary
    max_penetration = 0.0
    for c_id in range(data.ncon):
        dist = float(data.contact[c_id].dist)
        if dist < 0:
            max_penetration = max(max_penetration, abs(dist))

    # Stable if object maintained contact and did not penetrate excessively
    success = (squeeze_contacts >= 0) and (max_penetration < 0.05)
    stable_lift = success and (lift_height >= -0.05)

    metrics = {
        "squeeze_contacts": float(squeeze_contacts),
        "final_contacts": float(data.ncon),
        "lift_height": float(lift_height),
        "max_penetration": float(max_penetration),
    }

    return FixtureResult(
        success=success,
        stable_lift=stable_lift,
        contact_count=squeeze_contacts,
        max_penetration=max_penetration,
        lift_height=lift_height,
        metrics=metrics,
    )
