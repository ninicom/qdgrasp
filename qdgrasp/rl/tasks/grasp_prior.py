"""A fitted pinch prior that actually acquires the target (P3.5-11/17).

The open-loop descend-and-close fixture in :mod:`~qdgrasp.rl.tasks.scripted` runs
both hands safely to the horizon and never picks anything up.  That was a
structural result, not a tuning gap: closing the fingers from wherever the hand
happens to be does not enclose a box.  What does is the thing Phase 3.2/3.3
already established and the temporary MVP had to rebuild -- an opposed pinch
whose aperture is solved for the *target's* width and whose palm is placed on the
target's own frame.

So this module fits that prior, per hand and per width, and drives it through the
environment's ordinary bounded action.  Two things are worth being explicit
about.

*The prior reads privileged state.*  It takes the target's settled pose and
extents from the simulator, the way the MVP's controller prior does.  That is
what makes it a fixture rather than a policy: it is here to prove the
environment supports an acquire, not to demonstrate that one can be learned from
observations.

*It still cannot cheat.*  Every command leaves through the same eight-plus-joint
bounded action the learner gets, clamped by the same workspace, joint limits and
safety budget.  If the prior acquires the target, the action space admits an
acquire.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.robot.spec import RobotSpec

#: Fingertip indices that oppose each other.  Both active profiles order their
#: fingertips with the index finger first and the thumb last, and it is those two
#: that form the pinch.
PINCH_TIP_INDICES: tuple[int, int] = (0, 3)

#: Contact postures taken from the recipes that already have measured physical
#: positives (``scene_pinch_leap_v1`` and ``scene_pinch_allegro_v1``).  They are
#: copied rather than imported so this module does not depend on the scene
#: release pipeline's own evolution.
PINCH_POSTURES: dict[str, tuple[float, ...]] = {
    "leap_hand.yaml": (
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
    ),
    "wonik_allegro.yaml": (
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
    ),
}

#: How far outside the target face the open posture sits, and how far inside the
#: squeeze posture aims.  The overshoot is what generates grip force.
OPEN_CLEARANCE_M = 0.006
SQUEEZE_OVERSHOOT_M = 0.003


@dataclasses.dataclass(frozen=True)
class PinchPrior:
    """A grasp expressed in the *target's* frame, so it follows the target."""

    profile: str
    half_width: float
    palm_offset: np.ndarray
    palm_rotation: np.ndarray
    open_q: np.ndarray
    squeeze_q: np.ndarray
    contact_residual_m: float


def _pinch_frame(robot: RobotSpec, posture: np.ndarray):
    """Palm placement and fingertip axes at the pinned contact posture."""

    local = robot.fingertip_positions(torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(posture[None]))[0].numpy()
    index_tip, thumb_tip = local[PINCH_TIP_INDICES[0]], local[PINCH_TIP_INDICES[1]]
    axis = thumb_tip - index_tip
    axis = axis / np.linalg.norm(axis)
    # Put the thumb-to-index axis on -x, so a half-width `a` places the index
    # contact at +a and the thumb contact at -a in the target's frame.
    palm_rotation = Rotation.align_vectors(np.array([[-1.0, 0.0, 0.0]]), axis[None])[0].as_matrix()
    palm_offset = -palm_rotation @ (0.5 * (index_tip + thumb_tip))
    palm_pos_batch = palm_offset.astype(np.float32)[None]
    palm_rot_batch = palm_rotation.astype(np.float32)[None]
    posture_batch = torch.from_numpy(posture[None])
    contacts = robot.fingertip_positions(
        torch.from_numpy(palm_pos_batch), torch.from_numpy(palm_rot_batch), posture_batch
    )[0].numpy()
    axes = robot.fingertip_contact_directions(
        torch.from_numpy(palm_pos_batch), torch.from_numpy(palm_rot_batch), posture_batch
    )[0].numpy()
    return palm_offset.astype(np.float64), palm_rotation.astype(np.float64), contacts, axes


def build_pinch_prior(robot: RobotSpec, profile: str, half_width: float) -> PinchPrior:
    """Solve the open and squeeze postures for one target half-width."""

    if profile not in PINCH_POSTURES:
        raise KeyError(f"no pinned pinch posture for {profile!r}; known: {sorted(PINCH_POSTURES)}")
    posture = np.asarray(PINCH_POSTURES[profile], dtype=np.float32)
    palm_offset, palm_rotation, contacts, axes = _pinch_frame(robot, posture)

    surface = contacts.copy()
    surface[PINCH_TIP_INDICES[0]] = np.array([+float(half_width), 0.0, 0.0])
    surface[PINCH_TIP_INDICES[1]] = np.array([-float(half_width), 0.0, 0.0])
    active = np.array(PINCH_TIP_INDICES)
    open_targets = surface.copy()
    squeeze_targets = surface.copy()
    open_targets[active] -= OPEN_CLEARANCE_M * axes[active]
    squeeze_targets[active] += SQUEEZE_OVERSHOOT_M * axes[active]

    palm_pos_batch = palm_offset.astype(np.float32)[None]
    palm_rot_batch = palm_rotation.astype(np.float32)[None]
    solution = solve_dls_ik_batch(
        robot,
        np.repeat(palm_pos_batch, 2, axis=0),
        np.repeat(palm_rot_batch, 2, axis=0),
        np.stack([open_targets, squeeze_targets]).astype(np.float32),
        np.repeat(axes[None].astype(np.float32), 2, axis=0),
        init_q=np.repeat(posture[None], 2, axis=0),
        active_fingers=np.array([True, False, False, True]),
        max_iter=120,
        pos_tolerance=0.0007,
        normal_tolerance_dot=0.8,
        require_normal_alignment=False,
    )
    reached = robot.fingertip_positions(
        torch.from_numpy(np.repeat(palm_pos_batch, 2, axis=0)),
        torch.from_numpy(np.repeat(palm_rot_batch, 2, axis=0)),
        torch.as_tensor(solution.q, dtype=torch.float32),
    ).numpy()
    requested = np.stack([open_targets, squeeze_targets])
    residual = float(np.max(np.linalg.norm(reached[:, active] - requested[:, active], axis=-1)))
    return PinchPrior(
        profile=profile,
        half_width=float(half_width),
        palm_offset=palm_offset,
        palm_rotation=palm_rotation,
        open_q=np.asarray(solution.q[0], dtype=np.float64),
        squeeze_q=np.asarray(solution.q[1], dtype=np.float64),
        contact_residual_m=residual,
    )


def target_pinch_frame(rotation: np.ndarray, half_extents: np.ndarray) -> tuple[np.ndarray, float]:
    """Pick which of the target's axes to pinch across, and how wide it is.

    A box that settled on an arbitrary face has no privileged "width".  The one
    worth pinching is the narrowest horizontal axis: narrowest because the hand's
    aperture is finite, horizontal because a top-down pinch closes across it.
    """

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    half_extents = np.asarray(half_extents, dtype=np.float64).reshape(3)
    best_index = -1
    best_score = np.inf
    for index in range(3):
        axis = rotation[:, index]
        horizontality = abs(float(axis[2]))
        if horizontality > 0.5:
            continue  # too close to vertical to pinch across from above
        score = float(half_extents[index]) + horizontality * 0.01
        if score < best_score:
            best_score = score
            best_index = index
    if best_index < 0:
        # Degenerate orientation: fall back to the narrowest axis outright.
        best_index = int(np.argmin(half_extents))
    axis = rotation[:, best_index].copy()
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / norm
    return axis, float(half_extents[best_index])


def prior_frame_to_world(pinch_axis: np.ndarray) -> np.ndarray:
    """Rotation taking the prior's frame (pinch on x, up on z) into the world."""

    up = np.array([0.0, 0.0, 1.0])
    x = np.asarray(pinch_axis, dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = np.cross(up, x)
    y = y / np.linalg.norm(y)
    return np.stack([x, y, up], axis=1)


@dataclasses.dataclass(frozen=True)
class GraspPriorSpec:
    """Phase budgets and rates for the prior-driven fixture."""

    align_steps: int = 25
    descend_steps: int = 40
    close_steps: int = 35
    lift_steps: int = 30
    hold_steps: int = 35
    #: Height above the grasp pose the hand aligns at before descending.
    approach_clearance_m: float = 0.09
    #: Vertical travel of the lift, above the success gate so a success is not
    #: sitting exactly on the threshold.
    lift_travel_m: float = 0.09
    #: Fraction of the palm translation limit used while descending.  Full rate
    #: is an impact, which the safety budget correctly refuses.
    descend_rate: float = 0.35
    align_rate: float = 1.0
    lift_rate: float = 0.5
    #: How far past the fitted open posture the fingers spread on approach, so a
    #: descending fingertip clears the target's side instead of scraping it.
    approach_spread: float = -0.8

    @property
    def total_steps(self) -> int:
        return self.align_steps + self.descend_steps + self.close_steps + self.lift_steps + self.hold_steps


class GraspPriorPolicy:
    """Drive the fitted pinch through the environment's bounded action.

    The controller is a saturating proportional servo on the *commanded* palm
    target and joint targets, which is what the mocap weld follows.  Saturating
    at the interface's own limits means the fixture can never ask for more
    authority than a learner has.
    """

    def __init__(self, env: Any, prior: PinchPrior, spec: GraspPriorSpec | None = None) -> None:
        self.env = env
        self.prior = prior
        self.spec = spec or GraspPriorSpec()
        self._step = 0

        indices = env._indices
        data = env._data
        target_body = indices.object_bodies[env.config.target_object_id]
        rotation = np.array(data.xmat[target_body]).reshape(3, 3)
        centre = np.array(data.xpos[target_body], dtype=np.float64)
        axis, _half_width = target_pinch_frame(rotation, self._half_extents())
        frame = prior_frame_to_world(axis)
        self.grasp_position = centre + frame @ prior.palm_offset
        self.grasp_rotation = frame @ prior.palm_rotation

    def _half_extents(self) -> np.ndarray:
        env = self.env
        indices = env._indices
        model = env._model
        geom_ids = indices.object_geoms[env.config.target_object_id]
        return np.asarray(model.geom_size[geom_ids[0]], dtype=np.float64)

    def phase(self) -> str:
        spec = self.spec
        bounds = np.cumsum([spec.align_steps, spec.descend_steps, spec.close_steps, spec.lift_steps])
        if self._step < bounds[0]:
            return "align"
        if self._step < bounds[1]:
            return "descend"
        if self._step < bounds[2]:
            return "close"
        if self._step < bounds[3]:
            return "lift"
        return "hold"

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        del observation
        env = self.env
        spec = self.spec
        prior = self.prior
        phase = self.phase()

        if phase == "align":
            desired_position = self.grasp_position + np.array([0.0, 0.0, spec.approach_clearance_m])
            closure = spec.approach_spread
            rate = spec.align_rate
        elif phase == "descend":
            desired_position = self.grasp_position
            closure = spec.approach_spread
            rate = spec.descend_rate
        elif phase == "close":
            desired_position = self.grasp_position
            progress = (self._step - spec.align_steps - spec.descend_steps + 1) / max(spec.close_steps, 1)
            closure = spec.approach_spread + min(progress, 1.0) * (1.0 - spec.approach_spread)
            rate = spec.descend_rate
        elif phase == "lift":
            progress = (self._step - spec.align_steps - spec.descend_steps - spec.close_steps + 1) / max(
                spec.lift_steps, 1
            )
            desired_position = self.grasp_position + np.array([0.0, 0.0, spec.lift_travel_m * min(progress, 1.0)])
            closure = 1.0
            rate = spec.lift_rate
        else:
            desired_position = self.grasp_position + np.array([0.0, 0.0, spec.lift_travel_m])
            closure = 1.0
            rate = spec.lift_rate

        desired_joints = prior.open_q + closure * (prior.squeeze_q - prior.open_q)

        action_spec = env.action_spec
        translation_error = desired_position - env._palm_target
        translation = np.clip(translation_error / action_spec.palm_translation_limit_m, -rate, rate)

        rotation_error = Rotation.from_matrix(self.grasp_rotation @ env._palm_rotation.T)
        rotation = np.clip(rotation_error.as_rotvec() / action_spec.palm_rotation_limit_rad, -1.0, 1.0)

        joint_error = desired_joints - env._joint_target
        joints = np.clip(joint_error / action_spec.joint_delta_limit_rad, -1.0, 1.0)

        self._step += 1
        return np.clip(np.concatenate([translation, rotation, joints]), -1.0, 1.0)


def run_prior_episode(
    env: Any,
    *,
    seed: int,
    spec: GraspPriorSpec | None = None,
    prior_cache: dict[tuple[str, float], PinchPrior] | None = None,
) -> dict[str, Any]:
    """Reset, fit the prior for this target, and run the acquire to completion."""

    settings = spec or GraspPriorSpec()
    observation, reset_info = env.reset(seed=seed)

    indices = env._indices
    data = env._data
    model = env._model
    target_body = indices.object_bodies[env.config.target_object_id]
    rotation = np.array(data.xmat[target_body]).reshape(3, 3)
    geom_id = indices.object_geoms[env.config.target_object_id][0]
    half_extents = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    _axis, half_width = target_pinch_frame(rotation, half_extents)

    key = (env.config.robot_profile, round(half_width, 4))
    cache = prior_cache if prior_cache is not None else {}
    if key not in cache:
        cache[key] = build_pinch_prior(env._robot, env.config.robot_profile, half_width)
    prior = cache[key]

    policy = GraspPriorPolicy(env, prior, settings)
    reward_total = 0.0
    steps = 0
    max_lift = 0.0
    finite = True
    info: dict[str, Any] = {}
    terminated = truncated = False
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        finite = finite and bool(np.all(np.isfinite(observation)))
        reward_total += float(reward)
        max_lift = max(max_lift, float(info.get("lift_m", 0.0)))
        steps += 1
        if terminated or truncated:
            break
    reason = info.get("terminal_reason")
    return {
        "seed": seed,
        "robot_profile": env.config.robot_profile,
        "half_width_m": half_width,
        "prior_contact_residual_m": prior.contact_residual_m,
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminal_reason": getattr(reason, "value", str(reason)),
        "success": bool(info.get("success", False)),
        "max_lift_m": max_lift,
        "hold_steps": int(info.get("hold_steps", 0)),
        "links_in_contact": int(info.get("links_in_contact", 0)),
        "reward_total": reward_total,
        "observations_finite": finite,
        "scene_source": reset_info.get("scene_source"),
        "observation_schema_hash": reset_info.get("observation_schema_hash"),
    }
