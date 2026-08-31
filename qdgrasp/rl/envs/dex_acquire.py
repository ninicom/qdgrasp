"""``QDGrasp-DexAcquire-v0`` and its clutter variant (P3.5-10/11).

The hand approaches, closes, lifts and holds a target on a scene that was
resolved and settled first.  What makes this an RL *readiness* artifact rather
than a grasping result is the discipline around the edges:

*The policy cannot write the world.*  An action is a bounded palm delta and a
bounded set of named joint deltas, clamped by workspace, joint limits and the
safety budget before it reaches ``ctrl``.  There is no path from an action to an
object's pose, to the solver, or to the terminal flag.

*A barrier terminates; it is never a negative number.*  Penetration and impulse
budgets end the episode with their own terminal reason, so no positive reward
term can pay for crossing one.

*Non-target objects are accounted for separately.*  In the clutter variant,
disturbing a neighbour is its own terminal reason and its own reward term, not a
rounding error inside the target's.

P3.5's gate asks these environments to run, not to be solved.  A scripted
fixture reaching its expected outcome class and a random policy staying finite
is what "ready" means here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from qdgrasp.rl.contracts import (
    BoxSpace,
    ObservationField,
    ObservationSchema,
    RewardBreakdown,
    RlActionSpec,
    StepResult,
    TerminalReason,
)
from qdgrasp.rl.envs.hand_scene import (
    build_hand_scene_model,
    geom_owner_map,
    resolve_hand_scene_indices,
)
from qdgrasp.rl.randomization import DomainRandomization, SeedStreams, apply_randomization, scene_signature
from qdgrasp.scenes.resolver import resolve_scene
from qdgrasp.scenes.virtual_drop import DropObjectRequest, VirtualDropSceneSpec


@dataclasses.dataclass(frozen=True)
class AcquireSuccessSpec:
    """The measured task predicate.  Reward never enters it."""

    lift_height_m: float = 0.05
    retain_steps: int = 20
    min_contact_links: int = 2
    support_clearance_m: float = 0.001

    def validate(self) -> None:
        if self.lift_height_m <= 0.0 or self.retain_steps < 1 or self.min_contact_links < 1:
            raise ValueError("success spec requires a positive lift, retain window and contact requirement")


@dataclasses.dataclass(frozen=True)
class AcquireSafetySpec:
    """Hard barriers.  Crossing one ends the episode; it is never priced in."""

    #: Interpenetration allowed while contact is sustained.
    max_penetration_m: float = 0.005
    #: Control steps of grace before sustained penetration ends the episode.
    #: First touch compresses the contact -- measured at 5.75 mm on the first
    #: control step a finger meets the target at 0.08 m/s -- and that transient
    #: is how the solver represents an impulse, not a defect.  What is a defect
    #: is penetration that persists.
    penetration_grace_steps: int = 5
    #: Instantaneous ceiling.  Tunnelling and solver blow-up live here, an order
    #: of magnitude above any legitimate contact transient.
    max_transient_penetration_m: float = 0.02
    max_step_impulse_ns: float = 1.0
    max_non_target_displacement_m: float = 0.05

    def validate(self) -> None:
        for name in (
            "max_penetration_m",
            "max_transient_penetration_m",
            "max_step_impulse_ns",
            "max_non_target_displacement_m",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.penetration_grace_steps < 1:
            raise ValueError("penetration_grace_steps must be at least 1")
        if self.max_transient_penetration_m < self.max_penetration_m:
            raise ValueError("max_transient_penetration_m must be at least max_penetration_m")


@dataclasses.dataclass(frozen=True)
class AcquireRewardWeights:
    """Weights for the §6.4 terms.  Penalties are stated as positive magnitudes."""

    reach: float = 1.0
    contact_progress: float = 0.5
    enclosure: float = 0.5
    lift: float = 4.0
    retention: float = 1.0
    penetration: float = 2.0
    unsafe_impulse: float = 1.0
    non_target_disturbance: float = 2.0
    action_rate: float = 0.02
    joint_limit: float = 0.1
    drop: float = 2.0


@dataclasses.dataclass(frozen=True)
class DexAcquireConfig:
    """Everything the acquisition environment needs before its first reset."""

    robot_profile: str
    objects: tuple[DropObjectRequest, ...]
    target_object_id: str
    virtual_scene: VirtualDropSceneSpec = dataclasses.field(default_factory=VirtualDropSceneSpec)
    randomization: DomainRandomization = dataclasses.field(default_factory=DomainRandomization)
    success: AcquireSuccessSpec = dataclasses.field(default_factory=AcquireSuccessSpec)
    safety: AcquireSafetySpec = dataclasses.field(default_factory=AcquireSafetySpec)
    reward: AcquireRewardWeights = dataclasses.field(default_factory=AcquireRewardWeights)
    max_steps: int = 120
    substeps: int = 10
    #: Physics steps run before the hand acts, so the scene starts at rest.
    settle_steps: int = 600
    #: Palm height above the target at the start of an episode.  Measured, not
    #: guessed: at 0.14 m both active hands already interpenetrate the target at
    #: reset, and 0.20 m is the first height where neither does.
    approach_height_m: float = 0.20
    palm_translation_limit_m: float = 0.008
    palm_rotation_limit_rad: float = 0.08
    joint_delta_limit_rad: float = 0.15
    workspace_radius_m: float = 0.30

    def validate(self) -> None:
        if not self.objects:
            raise ValueError("the acquisition environment needs at least one object")
        ids = [item.object_id for item in self.objects]
        if self.target_object_id not in ids:
            raise ValueError(f"target {self.target_object_id!r} is not among the scene objects {ids}")
        if self.max_steps < 1 or self.substeps < 1 or self.settle_steps < 1:
            raise ValueError("max_steps, substeps and settle_steps must be positive")
        self.virtual_scene.validate()
        self.randomization.validate()
        self.success.validate()
        self.safety.validate()

    @property
    def non_target_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.objects if item.object_id != self.target_object_id)


def initial_palm_rotation(robot: Any) -> np.ndarray:
    """Rotate the hand so its fingers reach toward the support, per profile.

    Identity is not a neutral choice: it happens to point LEAP's fingers
    usefully and points Allegro's the other way, so a fixture tuned on one hand
    drives the back of the other into the table.  The direction that has to face
    down is the one from the palm origin to the fingertips -- the hand's reach --
    which the profile's forward kinematics already give at the zero posture.

    The fingertip *approach* axes are the wrong quantity here and were tried
    first: they point inward at the object being pinched, so on an opposed hand
    like LEAP they largely cancel, and their mean says nothing about which way
    the hand faces.
    """

    import torch

    joint_count = len(robot.actuated_joint_names)
    tips = robot.fingertip_positions(torch.zeros(1, 3), torch.eye(3)[None], torch.zeros(1, joint_count))[0].numpy()
    reach = np.mean(np.asarray(tips, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(reach))
    if norm < 1e-6:
        return np.eye(3)
    return Rotation.align_vectors(np.array([[0.0, 0.0, -1.0]]), (reach / norm)[None])[0].as_matrix()


def build_acquire_observation_schema(joint_count: int, tip_count: int, non_target_count: int) -> ObservationSchema:
    fields = [
        ObservationField("joint_position", joint_count, "rad", "hand", "named joint order from the profile"),
        ObservationField("joint_velocity", joint_count, "rad/s", "hand"),
        ObservationField("joint_mask", joint_count, "unitless", "hand", "1 where the joint is actuated"),
        ObservationField("palm_position", 3, "m", "world"),
        ObservationField("palm_rotation_6d", 6, "unitless", "world"),
        ObservationField("palm_linear_velocity", 3, "m/s", "world"),
        ObservationField("palm_angular_velocity", 3, "rad/s", "world"),
        ObservationField("target_position", 3, "m", "palm"),
        ObservationField("target_rotation_6d", 6, "unitless", "palm"),
        ObservationField("target_linear_velocity", 3, "m/s", "palm"),
        ObservationField("target_angular_velocity", 3, "rad/s", "palm"),
        ObservationField("target_lift", 1, "m", "world", "rise above the settled height"),
        ObservationField("fingertip_position", 3 * tip_count, "m", "palm"),
        ObservationField("fingertip_contact", tip_count, "unitless", "hand", "1 where the finger touches the target"),
        ObservationField("contact_force", tip_count, "N", "hand", "log1p of the summed normal force"),
        ObservationField("support_contact", 1, "unitless", "scene", "1 while the target still rests on support"),
        ObservationField("previous_action", joint_count + 6, "unitless", "action"),
        ObservationField("time_remaining", 1, "unitless", "episode"),
    ]
    if non_target_count:
        fields.append(ObservationField("non_target_position", 3 * non_target_count, "m", "palm"))
        fields.append(ObservationField("non_target_displacement", non_target_count, "m", "world"))
    schema = ObservationSchema(fields=tuple(fields))
    schema.validate()
    return schema


class DexAcquireEnv:
    """Approach, close, lift and hold a target on a settled scene."""

    environment_id = "QDGrasp-DexAcquire-v0"

    def __init__(self, config: DexAcquireConfig, *, scene_ref: str | None = None) -> None:
        config.validate()
        self.config = config
        self.scene_ref = scene_ref
        self._done = True
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None

    # -- spaces -----------------------------------------------------------

    @property
    def action_spec(self) -> RlActionSpec:
        if getattr(self, "_action_spec", None) is None:
            raise RuntimeError("the action spec is known only after the first reset")
        return self._action_spec  # type: ignore[return-value]

    def observation_space(self) -> BoxSpace:
        return self.schema.space()

    def action_space(self) -> BoxSpace:
        return self.action_spec.space()

    # -- lifecycle --------------------------------------------------------

    def reset(self, *, seed: int, options: Mapping[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        config = self.config
        streams = SeedStreams(episode_seed=seed)
        resolved = resolve_scene(
            scene_ref=self.scene_ref,
            objects=config.objects,
            virtual_scene_config=config.virtual_scene,
            seed=int(streams.generator("scene").integers(0, 2**31 - 1)),
            scene_id=f"acquire-{seed}",
        )
        model, robot = build_hand_scene_model(resolved.spec, config.robot_profile)
        indices = resolve_hand_scene_indices(model, robot, resolved.spec)

        sample = config.randomization.sample(streams.generator("physics"))
        object_ids = [item.object_id for item in resolved.spec.objects]
        target_geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for object_id in object_ids
            for geom_id in indices.object_geoms[object_id]
        ]
        applied = apply_randomization(model, object_ids, [n for n in target_geom_names if n], sample)
        mujoco.mj_setConst(model, mujoco.MjData(model))

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        self._model, self._data, self._indices, self._robot = model, data, indices, robot
        self._resolved = resolved

        # The mocap weld drives the hand *root*, but every command in this
        # environment is expressed at the palm.  Measure the root->palm offset
        # once, at the posture the episode starts from, and convert; writing the
        # palm target straight onto the root would misplace any hand whose root
        # is not its palm.
        mujoco.mj_forward(model, data)
        root_rotation = np.array(data.xmat[indices.root_body]).reshape(3, 3)
        palm_rotation = np.array(data.xmat[indices.palm_body]).reshape(3, 3)
        self._root_to_palm_rotation = root_rotation.T @ palm_rotation
        self._root_to_palm_position = root_rotation.T @ (
            np.array(data.xpos[indices.palm_body]) - np.array(data.xpos[indices.root_body])
        )

        # Park the hand clear of the drop before anything falls, so the settle
        # that follows is the scene's own and not a collision with the robot.
        rest = np.array([0.0, 0.0, config.approach_height_m + 0.4], dtype=np.float64)
        self._write_palm_command(rest, np.eye(3), teleport=True)
        data.ctrl[list(indices.actuator_ids)] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(config.settle_steps):
            mujoco.mj_step(model, data)

        target_body = indices.object_bodies[config.target_object_id]
        self._settled_height = float(data.xpos[target_body][2])
        self._settled_positions = {
            object_id: np.array(data.xpos[body_id], dtype=np.float64)
            for object_id, body_id in indices.object_bodies.items()
        }
        target_position = np.array(data.xpos[target_body], dtype=np.float64)
        start = target_position + np.array([0.0, 0.0, config.approach_height_m])
        start_rotation = initial_palm_rotation(robot)
        self._write_palm_command(start, start_rotation, teleport=True)
        mujoco.mj_forward(model, data)

        joint_count = len(indices.hand_joint_ids)
        self._action_spec = RlActionSpec(
            joint_names=tuple(robot.actuated_joint_names),
            active_joint_mask=tuple([True] * joint_count),
            control_dt=float(model.opt.timestep) * config.substeps,
            palm_command="delta_pose_6d",
            joint_command="named_delta_target",
            palm_translation_limit_m=config.palm_translation_limit_m,
            palm_rotation_limit_rad=config.palm_rotation_limit_rad,
            joint_delta_limit_rad=config.joint_delta_limit_rad,
        )
        self._action_spec.validate()
        self.schema = build_acquire_observation_schema(
            joint_count, len(indices.fingertip_bodies), len(config.non_target_ids)
        )
        self._joint_lower = np.array([model.jnt_range[j, 0] for j in indices.hand_joint_ids], dtype=np.float64)
        self._joint_upper = np.array([model.jnt_range[j, 1] for j in indices.hand_joint_ids], dtype=np.float64)
        self._owners = geom_owner_map(indices)
        self._palm_target = start.copy()
        self._palm_rotation = start_rotation
        self._joint_target = np.zeros(joint_count, dtype=np.float64)
        self._previous_action = np.zeros(self._action_spec.dimension, dtype=np.float64)
        self._step_index = 0
        self._hold_steps = 0
        self._max_lift = 0.0
        self._deep_steps = 0
        self._ever_lifted = False
        self._done = False
        self._contact = self._read_contacts()
        if self._contact["links_in_contact"] or self._contact["penetration"] > 0.0:
            raise RuntimeError(
                f"the hand already touches the target at reset (penetration "
                f"{self._contact['penetration'] * 1e3:.2f} mm); an episode that starts in contact measures "
                "the placement, not the policy. Raise approach_height_m."
            )

        info = {
            "scene_source": resolved.source.value,
            "scene_signature": scene_signature(resolved.spec, robot_profile=config.robot_profile),
            "robot_profile": config.robot_profile,
            "target_object_id": config.target_object_id,
            "non_target_object_ids": list(config.non_target_ids),
            "randomization": sample,
            "randomization_applied": applied,
            "observation_schema_hash": self.schema.content_hash(),
            "action_spec_hash": self._action_spec.content_hash(),
            "settled_height_m": self._settled_height,
        }
        return self._observation(), info

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        config = self.config
        model, data, indices = self._model, self._data, self._indices
        assert model is not None and data is not None

        spec = self._action_spec
        unit = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if unit.shape[0] != spec.dimension:
            raise ValueError(f"action must have {spec.dimension} entries, got {unit.shape[0]}")

        translation = unit[:3] * spec.palm_translation_limit_m
        rotation = unit[3:6] * spec.palm_rotation_limit_rad
        joint_delta = unit[6:] * spec.joint_delta_limit_rad

        centre = self._settled_positions[config.target_object_id]
        proposed = self._palm_target + translation
        offset = proposed - centre
        radius = float(np.linalg.norm(offset))
        if radius > config.workspace_radius_m:
            proposed = centre + offset * (config.workspace_radius_m / radius)
        self._palm_target = proposed
        self._palm_rotation = Rotation.from_rotvec(rotation).as_matrix() @ self._palm_rotation
        self._joint_target = np.clip(self._joint_target + joint_delta, self._joint_lower, self._joint_upper)

        self._write_palm_command(self._palm_target, self._palm_rotation)
        data.ctrl[list(indices.actuator_ids)] = self._joint_target

        physics_dt = float(model.opt.timestep)
        step_impulse = 0.0
        step_penetration = 0.0
        for _ in range(config.substeps):
            mujoco.mj_step(model, data)
            contact = self._read_contacts()
            step_penetration = max(step_penetration, contact["penetration"])
            step_impulse += contact["total_force"] * physics_dt
        self._contact = self._read_contacts()
        self._step_index += 1

        target_body = indices.object_bodies[config.target_object_id]
        position = np.array(data.xpos[target_body], dtype=np.float64)
        lift = float(position[2] - self._settled_height)
        previous_max_lift = self._max_lift
        self._max_lift = max(self._max_lift, lift)

        contacts = int(self._contact["links_in_contact"])
        support = bool(self._contact["support"])
        held = lift >= config.success.lift_height_m and not support
        self._hold_steps = self._hold_steps + 1 if held else 0
        if held:
            self._ever_lifted = True

        displacement = {
            object_id: float(
                np.linalg.norm(
                    np.array(data.xpos[indices.object_bodies[object_id]], dtype=np.float64)
                    - self._settled_positions[object_id]
                )
            )
            for object_id in config.non_target_ids
        }

        self._deep_steps = self._deep_steps + 1 if step_penetration > config.safety.max_penetration_m else 0
        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        reason = TerminalReason.NONE
        if not finite:
            reason = TerminalReason.INVALID_STATE
        elif (
            step_penetration > config.safety.max_transient_penetration_m
            or self._deep_steps > config.safety.penetration_grace_steps
        ):
            reason = TerminalReason.SAFETY_PENETRATION
        elif step_impulse > config.safety.max_step_impulse_ns:
            reason = TerminalReason.SAFETY_IMPULSE
        elif any(value > config.safety.max_non_target_displacement_m for value in displacement.values()):
            reason = TerminalReason.NON_TARGET_DISTURBED
        elif self._hold_steps >= config.success.retain_steps and contacts >= config.success.min_contact_links:
            reason = TerminalReason.SUCCESS
        elif self._ever_lifted and lift < config.success.lift_height_m and contacts == 0:
            reason = TerminalReason.OBJECT_DROPPED

        terminated = reason is not TerminalReason.NONE
        truncated = (not terminated) and self._step_index >= config.max_steps
        if truncated:
            reason = TerminalReason.HORIZON
        self._done = terminated or truncated

        reward = self._reward(
            unit, lift, previous_max_lift, contacts, step_penetration, step_impulse, held, displacement
        )
        self._previous_action = unit
        info: dict[str, Any] = {
            "terminal_reason": reason,
            "reward_terms": reward.to_document(),
            "lift_m": lift,
            "links_in_contact": contacts,
            "support_contact": support,
            "hold_steps": self._hold_steps,
            "max_penetration_m": step_penetration,
            "sustained_penetration_steps": self._deep_steps,
            "step_impulse_ns": step_impulse,
            "non_target_displacement_m": displacement,
            "success": reason is TerminalReason.SUCCESS,
        }
        result = StepResult(self._observation(), reward.total, terminated, truncated, info)
        return result.as_tuple()

    def close(self) -> None:
        self._model = None
        self._data = None

    # -- control ----------------------------------------------------------

    def _write_palm_command(
        self, palm_position: np.ndarray, palm_rotation: np.ndarray, *, teleport: bool = False
    ) -> None:
        model, data, indices = self._model, self._data, self._indices
        assert model is not None and data is not None
        root_rotation = palm_rotation @ self._root_to_palm_rotation.T
        root_position = palm_position - root_rotation @ self._root_to_palm_position
        quat_xyzw = Rotation.from_matrix(root_rotation).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        data.mocap_pos[indices.mocap_index] = root_position
        data.mocap_quat[indices.mocap_index] = quat_wxyz
        if teleport:
            address = indices.root_qpos_adr
            data.qpos[address : address + 3] = root_position
            data.qpos[address + 3 : address + 7] = quat_wxyz

    # -- measurement ------------------------------------------------------

    def _read_contacts(self) -> dict[str, Any]:
        model, data, indices = self._model, self._data, self._indices
        assert model is not None and data is not None
        target_geoms = set(indices.object_geoms[self.config.target_object_id])
        support_geoms = set(indices.support_geoms)
        forces = np.zeros(len(indices.fingertip_bodies), dtype=np.float64)
        links: set[int] = set()
        total_force = 0.0
        penetration = 0.0
        support = False
        wrench = np.zeros(6, dtype=np.float64)
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if not (geom1 in target_geoms or geom2 in target_geoms):
                continue
            other = geom2 if geom1 in target_geoms else geom1
            if other in support_geoms:
                if contact.dist <= self.config.success.support_clearance_m:
                    support = True
                continue
            if self._owners.get(other) != "hand":
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            normal = abs(float(wrench[0]))
            total_force += normal
            penetration = max(penetration, float(-contact.dist))
            body_id = int(model.geom_bodyid[other])
            links.add(body_id)
            for tip_index, tip_body in enumerate(indices.fingertip_bodies):
                if body_id == tip_body:
                    forces[tip_index] += normal
        return {
            "forces": forces,
            "links_in_contact": len(links),
            "total_force": total_force,
            "penetration": penetration,
            "support": support,
        }

    def _reward(
        self,
        action: np.ndarray,
        lift: float,
        previous_max_lift: float,
        contacts: int,
        penetration: float,
        impulse: float,
        held: bool,
        displacement: Mapping[str, float],
    ) -> RewardBreakdown:
        weights = self.config.reward
        model, data, indices = self._model, self._data, self._indices
        assert model is not None and data is not None
        target = np.array(data.xpos[indices.object_bodies[self.config.target_object_id]], dtype=np.float64)
        tips = np.array([data.xpos[body] for body in indices.fingertip_bodies], dtype=np.float64)
        distance = float(np.mean(np.linalg.norm(tips - target, axis=1)))
        previous = getattr(self, "_previous_distance", distance)
        self._previous_distance = distance

        joint_positions = np.array([data.qpos[address] for address in indices.hand_qpos_adr], dtype=np.float64)
        span = np.maximum(self._joint_upper - self._joint_lower, 1e-6)
        overshoot = np.maximum(
            (self._joint_lower - joint_positions) / span, (joint_positions - self._joint_upper) / span
        )
        limit_violation = float(np.sum(np.maximum(overshoot, 0.0)))

        terms = {
            "reach": weights.reach * float(np.clip(previous - distance, -0.05, 0.05)),
            "contact_progress": weights.contact_progress * (1.0 if contacts > 0 else 0.0),
            "enclosure": weights.enclosure
            * min(contacts, self.config.success.min_contact_links)
            / self.config.success.min_contact_links,
            "lift": weights.lift * float(np.clip(max(0.0, lift) - max(0.0, previous_max_lift), 0.0, 0.05)),
            "retention": weights.retention * (1.0 if held else 0.0),
            "penetration": -weights.penetration
            * max(0.0, penetration - self.config.safety.max_penetration_m)
            / self.config.safety.max_penetration_m,
            "unsafe_impulse": -weights.unsafe_impulse
            * max(0.0, impulse - self.config.safety.max_step_impulse_ns)
            / self.config.safety.max_step_impulse_ns,
            "non_target_disturbance": -weights.non_target_disturbance
            * float(sum(displacement.values()))
            / max(self.config.safety.max_non_target_displacement_m, 1e-9),
            "action_rate": -weights.action_rate * float(np.sum((action - self._previous_action) ** 2)),
            "joint_limit": -weights.joint_limit * limit_violation,
            "drop": -weights.drop * (1.0 if (self._ever_lifted and not held and contacts == 0) else 0.0),
        }
        return RewardBreakdown(terms=terms)

    def _observation(self) -> np.ndarray:
        config = self.config
        model, data, indices = self._model, self._data, self._indices
        assert model is not None and data is not None
        joint_position = np.array([data.qpos[address] for address in indices.hand_qpos_adr], dtype=np.float64)
        joint_velocity = np.array([data.qvel[address] for address in indices.hand_dof_adr], dtype=np.float64)
        palm_position = np.array(data.xpos[indices.palm_body], dtype=np.float64)
        palm_rotation = np.array(data.xmat[indices.palm_body]).reshape(3, 3)
        palm_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, indices.palm_body, palm_velocity, 0)

        target_body = indices.object_bodies[config.target_object_id]
        target_position = np.array(data.xpos[target_body], dtype=np.float64)
        target_rotation = np.array(data.xmat[target_body]).reshape(3, 3)
        target_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, target_body, target_velocity, 0)
        tips = np.array([data.xpos[body] for body in indices.fingertip_bodies], dtype=np.float64)

        parts: dict[str, np.ndarray] = {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "joint_mask": np.ones(joint_position.shape[0], dtype=np.float64),
            "palm_position": palm_position - self._settled_positions[config.target_object_id],
            "palm_rotation_6d": palm_rotation[:, :2].T.reshape(-1),
            "palm_linear_velocity": palm_velocity[3:],
            "palm_angular_velocity": palm_velocity[:3],
            "target_position": palm_rotation.T @ (target_position - palm_position),
            "target_rotation_6d": (palm_rotation.T @ target_rotation)[:, :2].T.reshape(-1),
            "target_linear_velocity": palm_rotation.T @ target_velocity[3:],
            "target_angular_velocity": palm_rotation.T @ target_velocity[:3],
            "target_lift": np.array([target_position[2] - self._settled_height]),
            "fingertip_position": (palm_rotation.T @ (tips - palm_position).T).T.reshape(-1),
            "fingertip_contact": (self._contact["forces"] > 0.0).astype(np.float64),
            "contact_force": np.log1p(self._contact["forces"]),
            "support_contact": np.array([1.0 if self._contact["support"] else 0.0]),
            "previous_action": self._previous_action,
            "time_remaining": np.array([1.0 - self._step_index / config.max_steps]),
        }
        if config.non_target_ids:
            positions = []
            displacements = []
            for object_id in config.non_target_ids:
                body_id = indices.object_bodies[object_id]
                position = np.array(data.xpos[body_id], dtype=np.float64)
                positions.append(palm_rotation.T @ (position - palm_position))
                displacements.append(float(np.linalg.norm(position - self._settled_positions[object_id])))
            parts["non_target_position"] = np.concatenate(positions)
            parts["non_target_displacement"] = np.asarray(displacements, dtype=np.float64)
        observation = self.schema.assemble(parts)
        return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)


class DexAcquireSceneEnv(DexAcquireEnv):
    """The clutter variant: same task, with non-target accounting required."""

    environment_id = "QDGrasp-DexAcquireScene-v0"

    def __init__(self, config: DexAcquireConfig, *, scene_ref: str | None = None) -> None:
        if not config.non_target_ids:
            raise ValueError(
                "QDGrasp-DexAcquireScene-v0 is the clutter environment and needs at least one non-target object; "
                "use QDGrasp-DexAcquire-v0 for a single-object scene"
            )
        super().__init__(config, scene_ref=scene_ref)
