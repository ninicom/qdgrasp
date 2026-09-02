"""``QDGrasp-DexAcquire-MVP-v0``: one hand, one table, one cuboid, state input.

The environment is deliberately small and deliberately honest about the two
things an MVP is easiest to fool itself with.

*The verdict is measured, not shaped.*  :func:`EpisodeResult.success` is decided
by the predicate in ``ROADMAP-MVP-001`` §4 -- lift, sustained hold, terminal
multi-finger contact, no support assistance, no invalid state, no timeout -- and
the reward never touches it.  Timeouts, safety terminations and simulator errors
all stay in the denominator.

*The policy cannot write the world.*  An action is a bounded residual on the
controller prior's palm target and finger synergy; it is clamped by workspace,
joint limits and the safety budget before it reaches ``ctrl``.  There is no path
from an action to the target's pose, to a termination flag, or to the verdict.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from qdgrasp.mvp.challenge import ChallengeDomain
from qdgrasp.mvp.config import EpisodeSplit, MvpScopeConfig, ObjectVariant, tier_of_split
from qdgrasp.mvp.prior import PinchPriorTable
from qdgrasp.mvp.scene import SceneIndices, build_mvp_scene, resolve_indices
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

OBSERVATION_SCHEMA_V0 = "qdgrasp/mvp-observation/v0"

#: Ordered observation layout.  Every field states its own frame and unit in the
#: comment beside it; the schema hash is what a checkpoint pins.
OBSERVATION_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("joint_position", 16, "rad, actuated hand joints in profile order"),
    ("joint_velocity", 16, "rad/s, actuated hand joints in profile order"),
    ("palm_position", 3, "m, palm origin relative to the settled target centre, world axes"),
    ("palm_rotation_6d", 6, "first two columns of the world palm rotation"),
    ("palm_linear_velocity", 3, "m/s, world"),
    ("palm_angular_velocity", 3, "rad/s, world"),
    ("target_position", 3, "m, target centre in the palm frame"),
    ("target_rotation_6d", 6, "first two columns of the target rotation in the palm frame"),
    ("target_linear_velocity", 3, "m/s, palm frame"),
    ("target_angular_velocity", 3, "rad/s, palm frame"),
    ("target_half_extents", 3, "m, privileged cuboid half extents"),
    ("fingertip_position", 12, "m, four fingertip contact points in the palm frame"),
    ("fingertip_contact", 4, "0/1 contact bit per finger group"),
    ("fingertip_force", 4, "log1p of summed normal force per finger group, N"),
    ("previous_action", 8, "unit residual issued at the previous control step"),
    ("phase_one_hot", 4, "approach / enclose / lift / retain"),
    ("time_remaining", 1, "control steps left, normalised by the episode length"),
    ("lift_progress", 1, "target rise above settled height, normalised by the gate"),
)

OBSERVATION_DIMENSION = sum(size for _, size, _ in OBSERVATION_FIELDS)

#: Finger-group prefixes, in observation and synergy order.
FINGER_GROUP_PREFIXES: tuple[str, ...] = ("if_", "mf_", "rf_", "th_")

#: Phases the controller prior walks through, in order.
PHASES: tuple[str, ...] = ("approach", "enclose", "lift", "retain")

#: Below this the target has left the table entirely.
OFF_TABLE_Z_M = -0.05


def observation_schema_hash() -> str:
    payload = json.dumps(
        {"schema": OBSERVATION_SCHEMA_V0, "fields": [[name, size] for name, size, _ in OBSERVATION_FIELDS]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def smoothstep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return 3.0 * clamped**2 - 2.0 * clamped**3


def _rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[:, :2].T.reshape(-1)


@dataclasses.dataclass(frozen=True)
class EpisodeSetup:
    """Everything sampled for one episode, before any physics runs."""

    seed: int
    split: EpisodeSplit
    variant_id: str
    randomized: bool
    position: tuple[float, float]
    yaw: float
    density: float
    friction_slide: float
    drop_height: float
    mass: float
    #: Whether the parameters above were drawn from the challenge domain
    #: rather than from the scope's own randomization.
    challenged: bool = False

    def to_document(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class EpisodeResult:
    """The measured outcome of one episode."""

    setup: EpisodeSetup
    success: bool
    failure_bucket: str
    steps: int
    max_lift_m: float
    terminal_lift_m: float
    hold_steps: int
    terminal_contact_groups: int
    max_penetration_m: float
    max_contact_force_n: float
    total_contact_impulse_ns: float
    max_step_impulse_ns: float
    support_assisted_terminal: bool
    support_assisted_in_retain: bool
    invalid_state: bool
    safety_violation: bool
    reward_total: float
    reward_components: dict[str, float]

    def to_document(self) -> dict[str, Any]:
        document = dataclasses.asdict(self)
        document["setup"] = self.setup.to_document()
        return document


class DexAcquireMvpEnv:
    """The MVP environment.  ``reset`` then ``step`` until it says it is done."""

    environment_id = "QDGrasp-DexAcquire-MVP-v0"
    observation_schema = OBSERVATION_SCHEMA_V0

    def __init__(
        self,
        scope: MvpScopeConfig,
        prior: PinchPriorTable,
        *,
        robot_spec: RobotSpec | None = None,
        challenge: ChallengeDomain | None = None,
    ) -> None:
        self.scope = scope
        self.prior = prior
        # Only the challenge split draws from this.  Every other split keeps the
        # scope's own ranges, so supplying a domain cannot silently move the
        # ground under tiers A, B and C.
        if challenge is not None:
            challenge.validate_against(scope)
        self.challenge = challenge
        self.spec = robot_spec or RobotSpec.from_config(scope.robot_profile, sample_anchors=False)
        self.joint_names = tuple(self.spec.actuated_joint_names)
        self.hand_xml_path = str(resolve_robot_asset(self.spec.config.source_asset))
        self._synergy = prior.synergy_directions()
        self._models: dict[str, mujoco.MjModel] = {}
        self._indices: dict[str, SceneIndices] = {}
        self._reference: dict[str, tuple[float, np.ndarray]] = {}
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._index: SceneIndices | None = None
        self._action_scale = np.asarray(scope.action.scale_vector(), dtype=np.float64)

    # -- model management -------------------------------------------------

    def _model_for(self, variant: ObjectVariant) -> tuple[mujoco.MjModel, SceneIndices]:
        cached = self._models.get(variant.variant_id)
        if cached is not None:
            return cached, self._indices[variant.variant_id]
        volume = 8.0 * variant.half_width * variant.half_depth * variant.half_height
        reference_mass = float(np.mean(self.scope.randomization.density) * volume)
        model = build_mvp_scene(self.hand_xml_path, variant.half_extents, reference_mass=reference_mass)
        indices = resolve_indices(model, self.joint_names, self.spec.fingertip_links, FINGER_GROUP_PREFIXES)
        expected_dt = self.scope.episode.control_dt / self.scope.episode.physics_substeps
        if not np.isclose(model.opt.timestep, expected_dt, rtol=0.0, atol=1e-12):
            raise ValueError(
                "compiled integrator timestep does not match the locked control rate: "
                f"model dt={model.opt.timestep!r}, config expects {expected_dt!r}"
            )
        self._models[variant.variant_id] = model
        self._indices[variant.variant_id] = indices
        self._reference[variant.variant_id] = (
            float(model.body_mass[indices.target_body]),
            np.array(model.body_inertia[indices.target_body], dtype=np.float64),
        )
        return model, indices

    # -- episode sampling -------------------------------------------------

    def _is_challenge_split(self, split: EpisodeSplit) -> bool:
        """Whether this split is the tier the scope marked as the challenge."""

        tier = tier_of_split(split)
        return tier is not None and self.scope.tier(tier).challenge_domain

    def sample_setup(
        self,
        seed: int,
        split: EpisodeSplit,
        *,
        randomized: bool | None = None,
        variant_id: str | None = None,
        challenged: bool | None = None,
    ) -> EpisodeSetup:
        """Draw one episode's parameters from the locked ranges.

        ``challenged`` overrides the split-based rule.  Demonstrations are
        collected on train and dev seeds but have to cover the region where the
        prior fails, or the expert has nothing to teach; the locked tiers never
        pass this, so the default remains "only the challenge split".
        """

        if challenged is None:
            challenged = self.challenge is not None and self._is_challenge_split(split)
        elif challenged and self.challenge is None:
            raise ValueError("challenge sampling was requested but this environment has no challenge domain")
        if challenged:
            assert self.challenge is not None
            variants = self.challenge.variants(self.scope)
        else:
            variants = self.scope.variants_for_split(split)
        if variant_id is not None:
            variant = self.scope.variant(variant_id)
            if variant not in variants:
                raise ValueError(f"variant {variant_id!r} is not available to split {split!r}")
        randomize = bool(randomized) if randomized is not None else split != "eval_a"
        rng = np.random.default_rng(seed)
        if variant_id is None:
            variant = variants[int(rng.integers(len(variants)))]
        ranges = self.scope.randomization

        def interval(axis: str) -> tuple[float, float]:
            if challenged:
                assert self.challenge is not None
                return self.challenge.range_for(axis, self.scope)
            return tuple(getattr(ranges, axis))  # type: ignore[return-value]

        if randomize:
            position = (float(rng.uniform(*interval("position_x"))), float(rng.uniform(*interval("position_y"))))
            yaw = float(rng.uniform(*interval("yaw")))
            density = float(rng.uniform(*interval("density")))
            friction = float(rng.uniform(*interval("friction_slide")))
            # Drop height is not a challenge axis, so it always keeps the
            # scope's range.
            drop = float(rng.uniform(*ranges.drop_height))
        else:
            position = (0.0, 0.0)
            yaw = 0.0
            density = float(np.mean(ranges.density))
            friction = float(np.mean(ranges.friction_slide))
            drop = float(np.mean(ranges.drop_height))
        volume = 8.0 * variant.half_width * variant.half_depth * variant.half_height
        return EpisodeSetup(
            seed=int(seed),
            split=split,
            variant_id=variant.variant_id,
            randomized=randomize,
            position=position,
            yaw=yaw,
            density=density,
            friction_slide=friction,
            drop_height=drop,
            mass=float(density * volume),
            challenged=bool(challenged),
        )

    # -- lifecycle --------------------------------------------------------

    def reset(
        self,
        seed: int,
        split: EpisodeSplit = "train",
        *,
        randomized: bool | None = None,
        variant_id: str | None = None,
        setup: EpisodeSetup | None = None,
        challenged: bool | None = None,
    ) -> np.ndarray:
        """Place the scene, settle the target, and return the first observation."""

        self.setup = setup or self.sample_setup(
            seed, split, randomized=randomized, variant_id=variant_id, challenged=challenged
        )
        variant = self.scope.variant(self.setup.variant_id)
        model, indices = self._model_for(variant)
        self._model, self._index = model, indices
        self._data = mujoco.MjData(model)
        data = self._data
        self._variant = variant

        reference_mass, reference_inertia = self._reference[variant.variant_id]
        scale = self.setup.mass / reference_mass
        model.body_mass[indices.target_body] = self.setup.mass
        # Inertia is linear in mass for fixed geometry, so one factor rescales
        # both.  What is *not* automatic is everything MuJoCo derives from mass
        # at compile time -- ``body_invweight0`` and friends, which scale the
        # constraint solver's reference impedance.  Leaving them stale would
        # give a heavy target the contact compliance of the reference mass,
        # which is exactly the variation this environment randomises over, so
        # the ``mj_setConst`` below is a correctness requirement, not hygiene.
        model.body_inertia[indices.target_body] = reference_inertia * scale
        # MuJoCo mixes a contact pair's friction by taking the elementwise
        # maximum, so randomising the target alone would be a no-op against the
        # hand's own coefficient.  The sampled value is the scene's tangential
        # friction, and it is stamped on every geom that can touch the target.
        for geom_id in (indices.target_geom, indices.table_geom, *indices.hand_geoms):
            model.geom_friction[geom_id, 0] = self.setup.friction_slide

        mujoco.mj_resetData(model, data)
        mujoco.mj_setConst(model, data)
        command = self.prior.command(variant.half_width)
        self._command = command
        self._joint_lower = np.array([model.jnt_range[j, 0] for j in indices.hand_joint_ids], dtype=np.float64)
        self._joint_upper = np.array([model.jnt_range[j, 1] for j in indices.hand_joint_ids], dtype=np.float64)

        for address, value in zip(indices.hand_qpos_adr, command.open_q):
            data.qpos[address] = float(value)
        mujoco.mj_forward(model, data)
        root_rot = np.array(data.xmat[indices.root_body]).reshape(3, 3)
        palm_rot = np.array(data.xmat[indices.palm_body]).reshape(3, 3)
        self._root_to_palm_rot = root_rot.T @ palm_rot
        self._root_to_palm_pos = root_rot.T @ (
            np.array(data.xpos[indices.palm_body]) - np.array(data.xpos[indices.root_body])
        )

        spawn_centre = np.array(
            [self.setup.position[0], self.setup.position[1], variant.half_height + self.setup.drop_height],
            dtype=np.float64,
        )
        spawn_rot = Rotation.from_euler("z", self.setup.yaw).as_matrix()
        grasp_pos, grasp_rot = self._grasp_pose(spawn_centre, spawn_rot)
        pregrasp_pos = grasp_pos + np.array([0.0, 0.0, self.scope.controller.pregrasp_height_m])
        self._write_palm_command(pregrasp_pos, grasp_rot, teleport=True)

        quat = Rotation.from_matrix(spawn_rot).as_quat()
        data.qpos[indices.target_qpos_adr : indices.target_qpos_adr + 3] = spawn_centre
        data.qpos[indices.target_qpos_adr + 3 : indices.target_qpos_adr + 7] = [quat[3], quat[0], quat[1], quat[2]]
        data.ctrl[list(indices.actuator_ids)] = command.open_q
        mujoco.mj_forward(model, data)

        for _ in range(self.scope.episode.settle_steps):
            for _ in range(self.scope.episode.physics_substeps):
                mujoco.mj_step(model, data)

        settled = np.array(data.xpos[indices.target_body], dtype=np.float64)
        settled_rot = np.array(data.xmat[indices.target_body]).reshape(3, 3)
        self._settled_centre = settled
        self._settled_height = float(settled[2])
        self._settled_yaw = float(Rotation.from_matrix(settled_rot).as_euler("zyx")[0])
        settled_yaw_rot = Rotation.from_euler("z", self._settled_yaw).as_matrix()
        self._grasp_pos, self._grasp_rot = self._grasp_pose(settled, settled_yaw_rot)
        self._start_palm_pos = np.array(data.xpos[indices.palm_body], dtype=np.float64)

        self._step_index = 0
        self._closure_bound = 1.0
        self._previous_action = np.zeros(self.scope.action.dimension, dtype=np.float64)
        self._filtered_action = np.zeros(self.scope.action.dimension, dtype=np.float64)
        self._previous_target_pos = settled.copy()
        self._hold_steps = 0
        self._max_lift = 0.0
        self._max_penetration = 0.0
        self._max_force = 0.0
        self._impulse = 0.0
        self._max_step_impulse = 0.0
        self._ever_contacted = False
        self._ever_lifted = False
        self._support_assisted_in_retain = False
        self._invalid = False
        self._violation = False
        self._reward_total = 0.0
        self._reward_components = {key: 0.0 for key in self.scope.reward.model_dump()}
        self._previous_approach_distance = self._approach_distance()
        self._done = False
        self._result: EpisodeResult | None = None
        self._contact = self._read_contacts()
        return self._observation()

    # -- prior and command ------------------------------------------------

    def _grasp_pose(self, centre: np.ndarray, yaw_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        command = self._command
        return (
            centre + yaw_rotation @ command.palm_offset,
            yaw_rotation @ command.palm_rotation,
        )

    def prior_command(self, step_index: int) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Palm target, palm rotation, open-loop closure fraction and phase.

        The closure fraction is what the *schedule* asks for.  What actually
        reaches the actuators is that fraction capped by the grip regulator, so
        the state machine stays a clean function of the step index and the force
        feedback stays a separate, inspectable term.
        """

        episode = self.scope.episode
        controller = self.scope.controller
        bounds = (
            episode.approach_steps,
            episode.approach_steps + episode.enclose_steps,
            episode.approach_steps + episode.enclose_steps + episode.lift_steps,
        )
        lifted = self._grasp_pos + np.array([0.0, 0.0, controller.lift_travel_m])
        if step_index < bounds[0]:
            progress = smoothstep((step_index + 1) / episode.approach_steps)
            start = self._grasp_pos + np.array([0.0, 0.0, controller.pregrasp_height_m])
            spread = controller.approach_closure
            return start + progress * (self._grasp_pos - start), self._grasp_rot, spread, 0
        if step_index < bounds[1]:
            progress = smoothstep((step_index - bounds[0] + 1) / episode.enclose_steps)
            spread = controller.approach_closure
            return self._grasp_pos, self._grasp_rot, float(spread + progress * (1.0 - spread)), 1
        if step_index < bounds[2]:
            progress = smoothstep((step_index - bounds[1] + 1) / episode.lift_steps)
            travel = controller.lift_travel_m * progress
            return self._grasp_pos + np.array([0.0, 0.0, travel]), self._grasp_rot, 1.0, 2
        return lifted, self._grasp_rot, 1.0, 3

    def _regulate_closure(self, scheduled: float) -> float:
        """Cap the scheduled closure by the grip-force regulator's own bound."""

        controller = self.scope.controller
        measured = float(np.max(self._contact["forces"])) if self._contact["forces"].size else 0.0
        error = controller.grip_force_target_n - measured
        delta = float(
            np.clip(
                controller.grip_gain * error * self.scope.episode.control_dt,
                -controller.closure_rate_limit,
                controller.closure_rate_limit,
            )
        )
        self._closure_bound = float(np.clip(self._closure_bound + delta, controller.closure_min, 1.0))
        # The bound limits how far the hand may *close*; a scheduled spread is
        # always allowed through, or the approach could never open the fingers.
        return min(scheduled, self._closure_bound)

    def _write_palm_command(self, palm_pos: np.ndarray, palm_rot: np.ndarray, *, teleport: bool = False) -> None:
        model, data, indices = self._model, self._data, self._index
        assert model is not None and data is not None and indices is not None
        root_rot = palm_rot @ self._root_to_palm_rot.T
        root_pos = palm_pos - root_rot @ self._root_to_palm_pos
        quat_xyzw = Rotation.from_matrix(root_rot).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        data.mocap_pos[indices.mocap_index] = root_pos
        data.mocap_quat[indices.mocap_index] = quat_wxyz
        if teleport:
            address = indices.root_qpos_adr
            data.qpos[address : address + 3] = root_pos
            data.qpos[address + 3 : address + 7] = quat_wxyz

    # -- stepping ---------------------------------------------------------

    def step(self, action: Sequence[float] | np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply one bounded residual and advance one control period."""

        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        model, data, indices = self._model, self._data, self._index
        assert model is not None and data is not None and indices is not None

        unit_action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if unit_action.shape[0] != self.scope.action.dimension:
            raise ValueError(f"action must have {self.scope.action.dimension} entries, got {unit_action.shape[0]}")
        # Low-pass the residual before it becomes a command.  The policy still
        # chooses freely every step; the interface simply refuses to slew the
        # palm target at the control rate.
        alpha = self.scope.action.residual_low_pass
        self._filtered_action = (1.0 - alpha) * self._filtered_action + alpha * unit_action
        residual = self._filtered_action * self._action_scale

        palm_pos, palm_rot, scheduled_closure, phase = self.prior_command(self._step_index)
        closure = self._regulate_closure(scheduled_closure)
        command = self._command
        joint_target = command.open_q + closure * (command.squeeze_q - command.open_q)
        radius = self.scope.action.workspace_radius_m
        commanded_pos = palm_pos + np.clip(residual[:3], -radius, radius)
        commanded_rot = Rotation.from_rotvec(residual[3:6]).as_matrix() @ palm_rot
        commanded_q = joint_target + self._synergy.T @ residual[6:8]
        commanded_q = np.clip(commanded_q, self._joint_lower, self._joint_upper)

        self._write_palm_command(commanded_pos, commanded_rot)
        data.ctrl[list(indices.actuator_ids)] = commanded_q

        substeps = self.scope.episode.physics_substeps
        physics_dt = float(model.opt.timestep)
        step_force = 0.0
        step_penetration = 0.0
        step_impulse = 0.0
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            contact = self._read_contacts()
            step_force = max(step_force, contact["max_force"])
            step_penetration = max(step_penetration, contact["penetration"])
            step_impulse += contact["total_force"] * physics_dt
        self._contact = self._read_contacts()
        self._max_force = max(self._max_force, step_force)
        self._max_penetration = max(self._max_penetration, step_penetration)
        self._impulse += step_impulse
        self._max_step_impulse = max(self._max_step_impulse, step_impulse)

        target_pos = np.array(data.xpos[indices.target_body], dtype=np.float64)
        state = np.concatenate([data.qpos, data.qvel])
        if not np.all(np.isfinite(state)):
            self._invalid = True
        jump = float(np.linalg.norm(target_pos - self._previous_target_pos))
        if jump > self.scope.success.max_pose_jump_m:
            self._invalid = True
        self._previous_target_pos = target_pos

        success = self.scope.success
        if step_penetration > success.max_penetration_m:
            self._violation = True
        if step_force > success.max_contact_force_n:
            self._violation = True
        if step_impulse > success.max_contact_impulse_ns:
            self._violation = True

        lift = float(target_pos[2] - self._settled_height)
        previous_max_lift = self._max_lift
        self._max_lift = max(self._max_lift, lift)
        groups = int(self._contact["groups"])
        if groups > 0:
            self._ever_contacted = True
        support_assisted = bool(self._contact["support"])
        # ``ROADMAP-MVP-001`` §4 asks for the *height* to hold continuously for
        # half a second, for at least two finger groups to be in contact at the
        # end, and for the target not to be support-assisted during the retain
        # window.  Those are three separate conditions.  Folding contact into
        # the continuity counter -- as this did at first -- makes one step of
        # contact chatter reset half a second of a perfectly good lift, and
        # measured a third of a millimetre of palm jitter as a failed grasp.
        held = lift >= success.lift_height_m
        if held:
            self._hold_steps += 1
            self._ever_lifted = True
        else:
            self._hold_steps = 0
        if phase == 3 and support_assisted:
            self._support_assisted_in_retain = True
        # Reward retention only when the grasp is doing the work, so the
        # shaping term cannot be earned by resting the target on the table.
        rewarded_hold = held and not support_assisted and groups >= success.min_finger_groups

        reward, components = self._reward(
            unit_action, lift, groups, step_penetration, step_force, rewarded_hold, previous_max_lift
        )
        self._reward_total += reward
        for key, value in components.items():
            self._reward_components[key] += value

        self._previous_action = unit_action
        self._step_index += 1
        off_table = bool(target_pos[2] < OFF_TABLE_Z_M)
        terminated = self._invalid or self._violation or off_table
        truncated = self._step_index >= self.scope.episode.max_steps
        self._done = bool(terminated or truncated)
        observation = self._observation()
        info: dict[str, Any] = {
            "phase": PHASES[phase],
            "closure": closure,
            "lift_m": lift,
            "contact_groups": groups,
            "support_assisted": support_assisted,
            "hold_steps": self._hold_steps,
            "reward_components": components,
        }
        if self._done:
            self._result = self._finish(lift, groups, support_assisted, off_table)
            info["result"] = self._result
        return observation, float(reward), self._done, info

    # -- measurement ------------------------------------------------------

    def _read_contacts(self) -> dict[str, Any]:
        model, data, indices = self._model, self._data, self._index
        assert model is not None and data is not None and indices is not None
        forces = np.zeros(len(FINGER_GROUP_PREFIXES), dtype=np.float64)
        penetration = 0.0
        total_force = 0.0
        support = False
        wrench = np.zeros(6, dtype=np.float64)
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if indices.target_geom not in (geom1, geom2):
                continue
            other = geom2 if geom1 == indices.target_geom else geom1
            if other == indices.table_geom:
                if contact.dist <= self.scope.success.support_clearance_m:
                    support = True
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            normal = abs(float(wrench[0]))
            total_force += normal
            penetration = max(penetration, float(-contact.dist))
            group = int(indices.geom_group[other])
            if group >= 0:
                forces[group] += normal
        return {
            "forces": forces,
            "groups": int(np.count_nonzero(forces > 0.0)),
            "max_force": float(np.max(forces)) if forces.size else 0.0,
            "total_force": total_force,
            "penetration": penetration,
            "support": support,
        }

    def _approach_distance(self) -> float:
        data, indices = self._data, self._index
        assert data is not None and indices is not None
        tips = np.array([data.xpos[body] for body in indices.fingertip_bodies], dtype=np.float64)
        pinch_centre = 0.5 * (tips[0] + tips[3])
        target = np.array(data.xpos[indices.target_body], dtype=np.float64)
        return float(np.linalg.norm(pinch_centre - target))

    def _reward(
        self,
        action: np.ndarray,
        lift: float,
        groups: int,
        penetration: float,
        force: float,
        held: bool,
        previous_max_lift: float,
    ) -> tuple[float, dict[str, float]]:
        weights = self.scope.reward
        success = self.scope.success
        distance = self._approach_distance()
        approach = self._previous_approach_distance - distance
        self._previous_approach_distance = distance
        components = {
            "approach_progress": weights.approach_progress * float(np.clip(approach, -0.05, 0.05)),
            "target_contact": weights.target_contact * (1.0 if groups > 0 else 0.0),
            "enclosure": weights.enclosure * min(groups, success.min_finger_groups) / success.min_finger_groups,
            "lift_progress": weights.lift_progress
            * float(np.clip(max(0.0, lift) - max(0.0, previous_max_lift), 0.0, 0.05)),
            "retain_bonus": weights.retain_bonus * (1.0 if held else 0.0),
            "penetration_penalty": -weights.penetration_penalty
            * max(0.0, penetration - success.max_penetration_m)
            / success.max_penetration_m,
            "excess_force_penalty": -weights.excess_force_penalty
            * max(0.0, force - success.max_contact_force_n)
            / success.max_contact_force_n,
            "action_rate_penalty": -weights.action_rate_penalty * float(np.sum((action - self._previous_action) ** 2)),
            "drop_penalty": -weights.drop_penalty
            * (1.0 if (self._ever_lifted and lift < success.lift_height_m) else 0.0),
            "timeout_penalty": 0.0,
        }
        return float(sum(components.values())), components

    def _finish(self, lift: float, groups: int, support_assisted: bool, off_table: bool) -> EpisodeResult:
        success_spec = self.scope.success
        required_hold = round(success_spec.retain_duration_s * self.scope.episode.control_hz)
        success = (
            not self._invalid
            and not self._violation
            and not off_table
            and self._hold_steps >= required_hold
            and groups >= success_spec.min_finger_groups
            and not self._support_assisted_in_retain
            and not support_assisted
            and lift >= success_spec.lift_height_m
        )
        if success:
            bucket = "none"
        elif self._invalid:
            bucket = "simulator_error"
        elif self._violation:
            bucket = "penetration" if self._max_penetration > success_spec.max_penetration_m else "excess_force"
        elif off_table or (self._ever_lifted and lift < success_spec.lift_height_m):
            bucket = "drop"
        elif not self._ever_contacted:
            bucket = "approach_miss"
        elif groups < success_spec.min_finger_groups:
            bucket = "contact_loss"
        else:
            bucket = "timeout"
        if bucket == "timeout":
            self._reward_total -= self.scope.reward.timeout_penalty
            self._reward_components["timeout_penalty"] -= self.scope.reward.timeout_penalty
        return EpisodeResult(
            setup=self.setup,
            success=bool(success),
            failure_bucket=bucket,
            steps=self._step_index,
            max_lift_m=self._max_lift,
            terminal_lift_m=lift,
            hold_steps=self._hold_steps,
            terminal_contact_groups=groups,
            max_penetration_m=self._max_penetration,
            max_contact_force_n=self._max_force,
            total_contact_impulse_ns=self._impulse,
            max_step_impulse_ns=self._max_step_impulse,
            support_assisted_terminal=support_assisted,
            support_assisted_in_retain=self._support_assisted_in_retain,
            invalid_state=self._invalid,
            safety_violation=self._violation,
            reward_total=self._reward_total,
            reward_components=dict(self._reward_components),
        )

    # -- observation ------------------------------------------------------

    def _observation(self) -> np.ndarray:
        model, data, indices = self._model, self._data, self._index
        assert model is not None and data is not None and indices is not None
        variant = self._variant
        episode = self.scope.episode

        joint_position = np.array([data.qpos[address] for address in indices.hand_qpos_adr], dtype=np.float64)
        joint_velocity = np.array([data.qvel[address] for address in indices.hand_dof_adr], dtype=np.float64)
        palm_pos = np.array(data.xpos[indices.palm_body], dtype=np.float64)
        palm_rot = np.array(data.xmat[indices.palm_body]).reshape(3, 3)
        palm_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, indices.palm_body, palm_velocity, 0)
        target_pos = np.array(data.xpos[indices.target_body], dtype=np.float64)
        target_rot = np.array(data.xmat[indices.target_body]).reshape(3, 3)
        target_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, indices.target_body, target_velocity, 0)

        tips = np.array([data.xpos[body] for body in indices.fingertip_bodies], dtype=np.float64)
        contact = self._contact
        _, _, _, phase = self.prior_command(min(self._step_index, episode.max_steps - 1))
        phase_one_hot = np.zeros(len(PHASES), dtype=np.float64)
        phase_one_hot[phase] = 1.0

        parts = [
            joint_position,
            joint_velocity,
            palm_pos - self._settled_centre,
            _rot6d(palm_rot),
            palm_velocity[3:],
            palm_velocity[:3],
            palm_rot.T @ (target_pos - palm_pos),
            _rot6d(palm_rot.T @ target_rot),
            palm_rot.T @ target_velocity[3:],
            palm_rot.T @ target_velocity[:3],
            np.asarray(variant.half_extents, dtype=np.float64),
            (palm_rot.T @ (tips - palm_pos).T).T.reshape(-1),
            (contact["forces"] > 0.0).astype(np.float64),
            np.log1p(contact["forces"]),
            self._previous_action,
            phase_one_hot,
            np.array([1.0 - self._step_index / episode.max_steps], dtype=np.float64),
            np.array([(target_pos[2] - self._settled_height) / self.scope.success.lift_height_m], dtype=np.float64),
        ]
        observation = np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in parts])
        if observation.shape[0] != OBSERVATION_DIMENSION:
            raise AssertionError(
                f"observation layout drift: built {observation.shape[0]}, schema declares {OBSERVATION_DIMENSION}"
            )
        return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # -- convenience ------------------------------------------------------

    @property
    def result(self) -> EpisodeResult | None:
        return self._result

    @property
    def done(self) -> bool:
        return self._done

    @property
    def step_index(self) -> int:
        """Control steps taken since ``reset``; the episode's own clock."""

        return self._step_index

    def run_episode(
        self,
        seed: int,
        split: EpisodeSplit = "train",
        *,
        policy: Any | None = None,
        randomized: bool | None = None,
        variant_id: str | None = None,
    ) -> EpisodeResult:
        """Run one episode to completion under ``policy`` (zero residual if None)."""

        observation = self.reset(seed, split, randomized=randomized, variant_id=variant_id)
        zeros = np.zeros(self.scope.action.dimension, dtype=np.float64)
        while not self._done:
            action = zeros if policy is None else np.asarray(policy(observation), dtype=np.float64)
            observation, _, _, _ = self.step(action)
        assert self._result is not None
        return self._result


def environment_fingerprint(scope: MvpScopeConfig, prior: PinchPriorTable) -> dict[str, str]:
    """Hashes a checkpoint must carry to be replayed against the same world."""

    return {
        # The scope names the world, not the class: a v1 scope run through the
        # same environment class is not the v0 environment, and a fingerprint
        # that said otherwise would let an artifact claim the wrong identity.
        "environment_id": scope.environment_id,
        "scope_hash": scope.content_hash(),
        "eval_manifest_hash": scope.eval_manifest_hash(),
        "prior_hash": prior.content_hash(),
        "observation_schema": OBSERVATION_SCHEMA_V0,
        "observation_schema_hash": observation_schema_hash(),
    }


def write_episode_ledger(path: str | Path, results: Sequence[EpisodeResult]) -> Path:
    """Write the raw per-episode ledger as newline-delimited JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_document(), sort_keys=True, separators=(",", ":")) + "\n")
    return target
