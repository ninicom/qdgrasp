"""Canonical measured-positive grasp recipes used by the scene tiny release job."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.dataset.pipeline.validators.dynamic_predicate import RolloutProtocol
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset


@dataclass(frozen=True)
class SceneGraspRecipe:
    recipe_id: str
    robot_profile: str
    robot_spec: RobotSpec
    target_geoms: tuple[SubGeomSpec, ...]
    target_object_pos: tuple[float, float, float]
    target_object_mass: float
    rollout_kwargs: dict[str, Any]
    protocol_hash: str
    recipe_hash: str

    @property
    def hand_xml_path(self) -> str:
        return str(resolve_robot_asset(self.robot_spec.config.source_asset))


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finish_recipe(
    recipe_id: str,
    robot_profile: str,
    spec: RobotSpec,
    half_width: float,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    initial_targets: dict[str, float],
    joint_targets: dict[str, float],
    contact_points: np.ndarray,
    object_half_height: float = 0.02,
    **extra_rollout: Any,
) -> SceneGraspRecipe:
    object_pos = (0.0, 0.0, float(object_half_height))
    target_geoms = (
        SubGeomSpec(
            type="box",
            size=(float(half_width), 0.015, float(object_half_height)),
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    protocol = RolloutProtocol()
    rollout_kwargs: dict[str, Any] = {
        "palm_pos": tuple(float(value) for value in palm_pos),
        "palm_rot": np.asarray(palm_rot, dtype=np.float64),
        "initial_joint_targets": initial_targets,
        "joint_targets": joint_targets,
        "object_pos": object_pos,
        "object_mass": 0.02,
        "expected_fingertip_positions": np.asarray(contact_points, dtype=np.float64),
        "fingertip_local_offsets": np.stack([spec.fingertip_contact_offsets[name] for name in spec.fingertip_links]),
        "pregrasp_distance": 0.03,
        "rollout_protocol": protocol,
        **extra_rollout,
    }
    payload = {
        "recipe_id": recipe_id,
        "robot_profile": robot_profile,
        "target_geoms": [geom.model_dump(mode="json") for geom in target_geoms],
        "rollout_kwargs": rollout_kwargs,
    }
    return SceneGraspRecipe(
        recipe_id=recipe_id,
        robot_profile=robot_profile,
        robot_spec=spec,
        target_geoms=target_geoms,
        target_object_pos=object_pos,
        target_object_mass=0.02,
        rollout_kwargs=rollout_kwargs,
        protocol_hash=_hash(dataclasses.asdict(protocol)),
        recipe_hash=_hash(payload),
    )


def _leap_recipe() -> SceneGraspRecipe:
    profile = "leap_hand.yaml"
    spec = RobotSpec.from_config(profile, sample_anchors=False)
    q_contact = np.array(
        [
            0.5927356227,
            -0.3791691612,
            0.6132688578,
            1.692338131,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.228141244,
            0.1354573565,
            -0.1336592733,
            1.666422321,
        ],
        dtype=np.float32,
    )
    local_contacts = spec.fingertip_positions(torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None]))[
        0
    ].numpy()
    pinch_axis = local_contacts[3] - local_contacts[0]
    pinch_axis /= np.linalg.norm(pinch_axis)
    palm_rot = Rotation.align_vectors(np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None])[0].as_matrix()
    palm_pos = np.array([0.0, 0.0, 0.02]) - palm_rot @ (0.5 * (local_contacts[0] + local_contacts[3]))
    palm_pos_batch = palm_pos.astype(np.float32)[None]
    palm_rot_batch = palm_rot.astype(np.float32)[None]
    contact_points = spec.fingertip_positions(
        torch.from_numpy(palm_pos_batch),
        torch.from_numpy(palm_rot_batch),
        torch.from_numpy(q_contact[None]),
    )[0].numpy()
    contact_axes = spec.fingertip_contact_directions(
        torch.from_numpy(palm_pos_batch),
        torch.from_numpy(palm_rot_batch),
        torch.from_numpy(q_contact[None]),
    )[0].numpy()
    open_contacts = contact_points.copy()
    squeeze_contacts = contact_points.copy()
    open_contacts[[0, 3]] -= 0.004 * contact_axes[[0, 3]]
    squeeze_contacts[[0, 3]] += 0.003 * contact_axes[[0, 3]]
    commands = solve_dls_ik_batch(
        spec,
        np.repeat(palm_pos_batch, 2, axis=0),
        np.repeat(palm_rot_batch, 2, axis=0),
        np.stack([open_contacts, squeeze_contacts]),
        np.repeat(contact_axes[None], 2, axis=0),
        init_q=np.repeat(q_contact[None], 2, axis=0),
        max_iter=35,
        pos_tolerance=0.0007,
        normal_tolerance_dot=0.8,
        require_normal_alignment=False,
    )
    if not np.all(commands.converged):
        raise RuntimeError("canonical LEAP release recipe IK did not converge")
    return _finish_recipe(
        "scene_pinch_leap_v1",
        profile,
        spec,
        0.5 * float(np.linalg.norm(local_contacts[3] - local_contacts[0])),
        palm_pos,
        palm_rot,
        dict(zip(spec.actuated_joint_names, commands.q[0])),
        dict(zip(spec.actuated_joint_names, commands.q[1])),
        contact_points,
        approach_steps=100,
        pregrasp_distance=0.08,
        pregrasp_direction=np.array([0.0, 0.0, 1.0]),
        squeeze_steps=300,
    )


def _allegro_recipe() -> SceneGraspRecipe:
    profile = "wonik_allegro.yaml"
    spec = RobotSpec.from_config(profile, sample_anchors=False)
    q_contact = np.array(
        [
            -0.1410063654,
            0.7589393854,
            0.2905291915,
            1.610496521,
            -0.1829498112,
            0.7104878426,
            0.4637212753,
            0.6895720363,
            -0.3722456992,
            0.4500102401,
            1.241124988,
            1.336122274,
            1.066359162,
            0.5970826745,
            0.1071554348,
            1.677100062,
        ],
        dtype=np.float32,
    )
    local_contacts = spec.fingertip_positions(torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None]))[
        0
    ].numpy()
    pinch_vector = local_contacts[3] - local_contacts[0]
    half_width = 0.5 * float(np.linalg.norm(pinch_vector))
    pinch_vector /= np.linalg.norm(pinch_vector)
    palm_rot = Rotation.align_vectors(np.array([[-1.0, 0.0, 0.0]]), pinch_vector[None])[0].as_matrix()
    palm_pos = np.array([0.0, 0.0, 0.02]) - palm_rot @ (0.5 * (local_contacts[0] + local_contacts[3]))
    palm_pos_batch = palm_pos.astype(np.float32)[None]
    palm_rot_batch = palm_rot.astype(np.float32)[None]
    contact_points = spec.fingertip_positions(
        torch.from_numpy(palm_pos_batch),
        torch.from_numpy(palm_rot_batch),
        torch.from_numpy(q_contact[None]),
    )[0].numpy()
    contact_axes = spec.fingertip_contact_directions(
        torch.from_numpy(palm_pos_batch),
        torch.from_numpy(palm_rot_batch),
        torch.from_numpy(q_contact[None]),
    )[0].numpy()
    open_contacts = contact_points.copy()
    squeeze_contacts = contact_points.copy()
    open_contacts[[0, 3]] -= 0.008 * contact_axes[[0, 3]]
    squeeze_contacts[[0, 3]] += 0.002 * contact_axes[[0, 3]]
    commands = solve_dls_ik_batch(
        spec,
        np.repeat(palm_pos_batch, 2, axis=0),
        np.repeat(palm_rot_batch, 2, axis=0),
        np.stack([open_contacts, squeeze_contacts]),
        np.repeat(contact_axes[None], 2, axis=0),
        init_q=np.repeat(q_contact[None], 2, axis=0),
        active_fingers=np.array([True, False, False, True]),
        max_iter=100,
        pos_tolerance=0.003,
        normal_tolerance_dot=0.8,
        require_normal_alignment=False,
    )
    if not np.all(commands.converged):
        raise RuntimeError("canonical Allegro release recipe IK did not converge")
    return _finish_recipe(
        "scene_pinch_allegro_v1",
        profile,
        spec,
        half_width,
        palm_pos,
        palm_rot,
        spec.expand_mimic_joint_targets(dict(zip(spec.actuated_joint_names, commands.q[0]))),
        spec.expand_mimic_joint_targets(dict(zip(spec.actuated_joint_names, commands.q[1]))),
        contact_points,
        approach_steps=100,
        pregrasp_distance=0.08,
        pregrasp_direction=np.array([0.0, 0.0, 1.0]),
        squeeze_steps=500,
        perturbation_wrench=np.array([0.15, 0.15, 0.0, 0.01, 0.01, 0.01]),
    )


def _shadow_recipe() -> SceneGraspRecipe:
    profile = "shadow_hand.yaml"
    spec = RobotSpec.from_config(profile, sample_anchors=False)
    q_contact = np.zeros(len(spec.actuated_joint_names), dtype=np.float32)
    joint_names = list(spec.actuated_joint_names)
    for name, value in {
        "rh_MFJ3": 1.4,
        "rh_MFJ2": 1.2,
        "rh_MFJ1": 1.2,
        "rh_RFJ3": 1.4,
        "rh_RFJ2": 1.2,
        "rh_RFJ1": 1.2,
        "rh_LFJ3": 1.4,
        "rh_LFJ2": 1.2,
        "rh_LFJ1": 1.2,
        "rh_FFJ3": 0.6,
        "rh_FFJ2": 0.5,
        "rh_FFJ1": 0.5,
        "rh_THJ5": 0.0,
        "rh_THJ4": 1.0,
        "rh_THJ2": 0.5,
        "rh_THJ1": 0.5,
    }.items():
        q_contact[joint_names.index(name)] = value
    local_contacts = spec.fingertip_positions(torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None]))[
        0
    ].numpy()
    pinch_vector = local_contacts[4] - local_contacts[0]
    distance = float(np.linalg.norm(pinch_vector))
    pinch_vector /= distance
    palm_rot = Rotation.align_vectors(np.array([[-1.0, 0.0, 0.0]]), pinch_vector[None])[0].as_matrix()
    palm_pos = np.array([0.0, 0.0, 0.02]) - palm_rot @ (0.5 * (local_contacts[0] + local_contacts[4]))
    q_open = q_contact.copy()
    q_squeeze = q_contact.copy()
    for name, delta_open, delta_squeeze in (
        ("rh_FFJ3", -0.05, 0.12),
        ("rh_FFJ2", -0.05, 0.10),
        ("rh_FFJ1", -0.05, 0.10),
        ("rh_THJ4", -0.05, 0.10),
        ("rh_THJ2", -0.05, 0.10),
        ("rh_THJ1", -0.05, 0.10),
    ):
        q_open[joint_names.index(name)] += delta_open
        q_squeeze[joint_names.index(name)] += delta_squeeze
    contact_points = spec.fingertip_positions(
        torch.from_numpy(palm_pos.astype(np.float32)[None]),
        torch.from_numpy(palm_rot.astype(np.float32)[None]),
        torch.from_numpy(q_contact[None]),
    )[0].numpy()
    vertical_shift = np.array([0.0, 0.0, 0.08])
    return _finish_recipe(
        "scene_pinch_shadow_v1",
        profile,
        spec,
        0.5 * distance - 0.0075,
        palm_pos + vertical_shift,
        palm_rot,
        dict(zip(spec.actuated_joint_names, q_open)),
        dict(zip(spec.actuated_joint_names, q_squeeze)),
        contact_points + vertical_shift,
        object_half_height=0.1,
        approach_steps=150,
        pregrasp_distance=0.12,
        pregrasp_direction=np.array([0.0, 0.0, 1.0]),
        squeeze_steps=250,
        lift_steps=150,
        lift_height=0.05,
        perturbation_steps=40,
        perturbation_wrench=np.array([0.02, 0.02, 0.0, 0.002, 0.002, 0.002]),
    )


def build_release_grasp_recipe(robot_profile: str) -> SceneGraspRecipe:
    builders = {
        "leap_hand.yaml": _leap_recipe,
        "wonik_allegro.yaml": _allegro_recipe,
        "shadow_hand.yaml": _shadow_recipe,
    }
    if robot_profile not in builders:
        raise ValueError(f"unsupported scene release robot profile: {robot_profile}")
    return builders[robot_profile]()
