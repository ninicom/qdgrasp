"""Compatibility facade for the staged MuJoCo grasp validator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..config.schema import ConfigError
from ..dataset.pipeline.validators.mujoco_rollout import (
    build_rollout_scene_model,
    validate_grasp_rollout,
)
from ..objects.schema import SubGeomSpec
from ..robot.spec import RobotSpec, resolve_robot_asset


@dataclass(frozen=True)
class PhysicsLabelResult:
    """Legacy result shape backed by the fail-closed staged validator."""

    success: bool
    stable_lift: bool
    contact_count: int
    contacting_links: tuple[str, ...]
    lift_height: float
    max_penetration: float
    metrics: dict[str, float]


def build_labeled_scene_model(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    *,
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
) -> mujoco.MjModel:
    return build_rollout_scene_model(
        hand_xml_path,
        collision_geoms,
        object_pos=object_pos,
        object_mass=object_mass,
    )


def evaluate_grasp_physics(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    *,
    palm_pos: tuple[float, float, float] = (0.0, 0.0, 0.1),
    joint_targets: Mapping[str, float] | None = None,
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
    squeeze_steps: int = 150,
    lift_steps: int = 150,
    lift_height: float = 0.05,
    perturbation_steps: int = 50,
    seed: int = 0,
) -> PhysicsLabelResult:
    """Run the canonical validator; ``seed`` is retained for API compatibility."""
    del seed
    requested_asset = Path(hand_xml_path).resolve()
    robot_spec = None
    for profile in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
        candidate = RobotSpec.from_config(profile, sample_anchors=False)
        if resolve_robot_asset(candidate.config.source_asset).resolve() == requested_asset:
            robot_spec = candidate
            break
    if robot_spec is None:
        raise ConfigError(
            "legacy physics facade requires one of the three pinned robot assets; "
            f"no profile owns {requested_asset}"
        )
    fingertip_names = robot_spec.fingertip_links
    expanded_joint_targets = (
        None
        if joint_targets is None
        else robot_spec.expand_mimic_joint_targets(joint_targets)
    )
    validation = validate_grasp_rollout(
        hand_xml_path,
        collision_geoms,
        fingertip_names,
        palm_pos=palm_pos,
        palm_rot=np.eye(3),
        joint_targets=expanded_joint_targets,
        object_pos=object_pos,
        object_mass=object_mass,
        squeeze_steps=squeeze_steps,
        lift_steps=lift_steps,
        lift_height=lift_height,
        perturbation_steps=perturbation_steps,
        min_active_fingers=2,
        fingertip_local_offsets=np.stack(
            [
                robot_spec.fingertip_contact_offsets[name]
                for name in robot_spec.fingertip_links
            ]
        ),
    )
    raw_metrics = validation.trajectory_metrics
    metrics = {
        key: float(value)
        for key, value in raw_metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    observed_lift = float(metrics.get("lift_achieved", 0.0))
    active_contacts = int(
        max(
            metrics.get("squeeze_active_fingers", 0.0),
            metrics.get("final_active_fingers", 0.0),
        )
    )
    return PhysicsLabelResult(
        success=validation.passed,
        stable_lift=observed_lift >= 0.5 * lift_height,
        contact_count=active_contacts,
        contacting_links=(),
        lift_height=observed_lift,
        max_penetration=float(metrics.get("max_penetration", 0.0)),
        metrics=metrics,
    )
