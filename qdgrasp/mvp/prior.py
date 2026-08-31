"""The LEAP pinch prior the MVP policy writes residuals on top of.

``ROADMAP-MVP-001`` §3.1 says to take the state machine from the LEAP recipe
that already has a measured physical positive and use it as a prior.  That
recipe (``qdgrasp.scenes.release_recipes._leap_recipe``) is fixed at one
aperture; the MVP needs a family.  So the geometry is generalised here: the
palm-relative pinch frame and the fingertip approach axes are taken from the
same pinned contact posture, and the open/close joint targets are re-solved by
the same DLS IK for each *train* half-width.

Two properties are deliberate.

The table is fitted **only at the train widths**.  A held-out width is served by
interpolating between neighbouring knots and by clamping outside them, so Tier C
measures something the prior genuinely does not know rather than a solver call
it could have made at runtime.

The prior commands *joint targets*, never an object pose.  Nothing in this
module can move the target; the target moves because a finger pushed it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.mvp.config import MvpScopeConfig
from qdgrasp.robot.spec import RobotSpec

PINCH_PRIOR_SCHEMA_V0 = "qdgrasp/mvp-pinch-prior/v0"

#: The pinned LEAP contact posture the pinch frame is measured from.  It is the
#: posture behind ``scene_pinch_leap_v1``, the recipe with the measured positive;
#: copying the numbers rather than importing the recipe keeps this module
#: independent of the scene release pipeline's own evolution.
LEAP_PINCH_POSTURE = (
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
)

#: Fingertip indices of the two groups that form the pinch: index and thumb.
PINCH_TIP_INDICES = (0, 3)

#: Joint-name prefixes of the two synergy groups, in action order.
SYNERGY_GROUPS: tuple[tuple[str, ...], ...] = (("if_",), ("th_",))

#: How far outside the target face the open posture sits, and how far inside the
#: squeeze posture aims.  The squeeze overshoot is what generates normal force.
OPEN_CLEARANCE_M = 0.004
SQUEEZE_OVERSHOOT_M = 0.003


@dataclasses.dataclass(frozen=True)
class PinchKnot:
    """The prior fitted at one train half-width."""

    half_width: float
    #: Palm target expressed in the target's own frame, relative to its centre.
    palm_offset: np.ndarray  # [3]
    palm_rotation: np.ndarray  # [3, 3]
    open_q: np.ndarray  # [J]
    squeeze_q: np.ndarray  # [J]
    #: Worst residual, in metres, between the requested contact points and the
    #: ones the solved posture actually reaches.  Recorded rather than asserted
    #: away: the rollout, not the solver, decides whether a knot is good enough.
    contact_residual_m: float


@dataclasses.dataclass(frozen=True)
class PinchCommand:
    """What the prior asks for at one instant, in the target's frame."""

    palm_offset: np.ndarray  # [3]
    palm_rotation: np.ndarray  # [3, 3]
    open_q: np.ndarray  # [J]
    squeeze_q: np.ndarray  # [J]


class PinchPriorTable:
    """Piecewise-linear pinch prior over the fitted half-widths."""

    def __init__(self, robot_profile: str, joint_names: Sequence[str], knots: Sequence[PinchKnot]) -> None:
        if not knots:
            raise ValueError("pinch prior table needs at least one knot")
        ordered = sorted(knots, key=lambda knot: knot.half_width)
        widths = [knot.half_width for knot in ordered]
        if len(set(widths)) != len(widths):
            raise ValueError("pinch prior knots must have distinct half widths")
        self.robot_profile = robot_profile
        self.joint_names = tuple(joint_names)
        self.knots = tuple(ordered)
        self._widths = np.asarray(widths, dtype=np.float64)

    # -- lookup -----------------------------------------------------------

    def command(self, half_width: float) -> PinchCommand:
        """Prior for a half-width, interpolated inside the fitted range.

        Outside the range the nearest knot is held.  That is the honest
        behaviour for an extrapolation the fit has no evidence for, and it is
        what makes the widest held-out variant a real test.
        """

        lower, upper, weight = self._bracket(float(half_width))
        left, right = self.knots[lower], self.knots[upper]
        return PinchCommand(
            palm_offset=_lerp(left.palm_offset, right.palm_offset, weight),
            palm_rotation=_slerp_rotation(left.palm_rotation, right.palm_rotation, weight),
            open_q=_lerp(left.open_q, right.open_q, weight),
            squeeze_q=_lerp(left.squeeze_q, right.squeeze_q, weight),
        )

    def _bracket(self, half_width: float) -> tuple[int, int, float]:
        widths = self._widths
        if half_width <= widths[0]:
            return 0, 0, 0.0
        if half_width >= widths[-1]:
            last = len(widths) - 1
            return last, last, 0.0
        upper = int(np.searchsorted(widths, half_width, side="left"))
        lower = upper - 1
        span = widths[upper] - widths[lower]
        return lower, upper, float((half_width - widths[lower]) / span)

    def synergy_directions(self) -> np.ndarray:
        """Unit closing direction per synergy group, in joint space.

        The direction is the prior's own open-to-squeeze delta restricted to the
        group's joints, so a positive residual means "close further along the
        motion this prior was already making" rather than an invented axis.
        """

        reference = self.knots[len(self.knots) // 2]
        delta = reference.squeeze_q - reference.open_q
        directions = np.zeros((len(SYNERGY_GROUPS), delta.shape[0]), dtype=np.float64)
        for group_index, prefixes in enumerate(SYNERGY_GROUPS):
            mask = np.array(
                [any(name.startswith(prefix) for prefix in prefixes) for name in self.joint_names],
                dtype=bool,
            )
            masked = np.where(mask, delta, 0.0)
            norm = float(np.linalg.norm(masked))
            if norm <= 1e-9:
                raise ValueError(f"synergy group {prefixes} has no motion in the prior")
            directions[group_index] = masked / norm
        return directions

    # -- serialization ----------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": PINCH_PRIOR_SCHEMA_V0,
            "robot_profile": self.robot_profile,
            "joint_names": list(self.joint_names),
            "knots": [
                {
                    "half_width": knot.half_width,
                    "palm_offset": [float(v) for v in knot.palm_offset],
                    "palm_rotation": [[float(v) for v in row] for row in knot.palm_rotation],
                    "open_q": [float(v) for v in knot.open_q],
                    "squeeze_q": [float(v) for v in knot.squeeze_q],
                    "contact_residual_m": knot.contact_residual_m,
                }
                for knot in self.knots
            ],
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> PinchPriorTable:
        if document.get("schema") != PINCH_PRIOR_SCHEMA_V0:
            raise ValueError(f"unsupported pinch prior schema: {document.get('schema')!r}")
        knots = [
            PinchKnot(
                half_width=float(entry["half_width"]),
                palm_offset=np.asarray(entry["palm_offset"], dtype=np.float64),
                palm_rotation=np.asarray(entry["palm_rotation"], dtype=np.float64),
                open_q=np.asarray(entry["open_q"], dtype=np.float64),
                squeeze_q=np.asarray(entry["squeeze_q"], dtype=np.float64),
                contact_residual_m=float(entry["contact_residual_m"]),
            )
            for entry in document["knots"]
        ]
        return cls(str(document["robot_profile"]), tuple(document["joint_names"]), knots)

    @classmethod
    def load(cls, path: str | Path) -> PinchPriorTable:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"pinch prior table not found: {resolved}")
        return cls.from_document(json.loads(resolved.read_text(encoding="utf-8")))


def _lerp(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    return np.asarray(left, dtype=np.float64) * (1.0 - weight) + np.asarray(right, dtype=np.float64) * weight


def _slerp_rotation(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0:
        return np.array(left, dtype=np.float64)
    if weight >= 1.0:
        return np.array(right, dtype=np.float64)
    rotations = Rotation.from_matrix(np.stack([left, right]))
    relative = rotations[0].inv() * rotations[1]
    return (rotations[0] * Rotation.from_rotvec(relative.as_rotvec() * weight)).as_matrix()


@dataclasses.dataclass(frozen=True)
class _PinchFrame:
    """Palm placement and fingertip approach axes at the pinned posture."""

    palm_offset: np.ndarray
    palm_rotation: np.ndarray
    nominal_contacts: np.ndarray  # [K, 3] in the target frame, centre at origin
    contact_axes: np.ndarray  # [K, 3]
    natural_half_width: float


def _pinch_frame(spec: RobotSpec) -> _PinchFrame:
    posture = np.asarray(LEAP_PINCH_POSTURE, dtype=np.float32)
    if posture.shape[0] != len(spec.actuated_joint_names):
        raise ValueError(
            f"pinned pinch posture has {posture.shape[0]} joints, profile has {len(spec.actuated_joint_names)}"
        )
    local = spec.fingertip_positions(
        torch.zeros(1, 3),
        torch.eye(3)[None],
        torch.from_numpy(posture[None]),
    )[0].numpy()
    index_tip, thumb_tip = local[PINCH_TIP_INDICES[0]], local[PINCH_TIP_INDICES[1]]
    axis = thumb_tip - index_tip
    natural_half_width = 0.5 * float(np.linalg.norm(axis))
    axis = axis / np.linalg.norm(axis)
    # Put the thumb-to-index axis on -x of the target frame, so a half-width `a`
    # places the index contact at +a and the thumb contact at -a.
    palm_rotation = Rotation.align_vectors(np.array([[-1.0, 0.0, 0.0]]), axis[None])[0].as_matrix()
    palm_offset = -palm_rotation @ (0.5 * (index_tip + thumb_tip))

    palm_pos_batch = palm_offset.astype(np.float32)[None]
    palm_rot_batch = palm_rotation.astype(np.float32)[None]
    posture_batch = torch.from_numpy(posture[None])
    contacts = spec.fingertip_positions(
        torch.from_numpy(palm_pos_batch), torch.from_numpy(palm_rot_batch), posture_batch
    )[0].numpy()
    axes = spec.fingertip_contact_directions(
        torch.from_numpy(palm_pos_batch), torch.from_numpy(palm_rot_batch), posture_batch
    )[0].numpy()
    return _PinchFrame(
        palm_offset=palm_offset.astype(np.float64),
        palm_rotation=palm_rotation.astype(np.float64),
        nominal_contacts=contacts.astype(np.float64),
        contact_axes=axes.astype(np.float64),
        natural_half_width=natural_half_width,
    )


def build_pinch_prior_table(
    spec: RobotSpec,
    half_widths: Sequence[float],
    *,
    max_iter: int = 120,
    pos_tolerance: float = 0.0007,
) -> PinchPriorTable:
    """Fit one knot per half-width by re-solving the pinch contact IK."""

    frame = _pinch_frame(spec)
    posture = np.asarray(LEAP_PINCH_POSTURE, dtype=np.float32)
    palm_pos_batch = frame.palm_offset.astype(np.float32)[None]
    palm_rot_batch = frame.palm_rotation.astype(np.float32)[None]

    knots: list[PinchKnot] = []
    for half_width in half_widths:
        surface = frame.nominal_contacts.copy()
        surface[PINCH_TIP_INDICES[0]] = np.array([+float(half_width), 0.0, 0.0])
        surface[PINCH_TIP_INDICES[1]] = np.array([-float(half_width), 0.0, 0.0])
        active = np.array(PINCH_TIP_INDICES)
        open_targets = surface.copy()
        squeeze_targets = surface.copy()
        open_targets[active] -= OPEN_CLEARANCE_M * frame.contact_axes[active]
        squeeze_targets[active] += SQUEEZE_OVERSHOOT_M * frame.contact_axes[active]

        solution = solve_dls_ik_batch(
            spec,
            np.repeat(palm_pos_batch, 2, axis=0),
            np.repeat(palm_rot_batch, 2, axis=0),
            np.stack([open_targets, squeeze_targets]).astype(np.float32),
            np.repeat(frame.contact_axes[None].astype(np.float32), 2, axis=0),
            init_q=np.repeat(posture[None], 2, axis=0),
            max_iter=max_iter,
            pos_tolerance=pos_tolerance,
            normal_tolerance_dot=0.8,
            require_normal_alignment=False,
        )
        reached = spec.fingertip_positions(
            torch.from_numpy(np.repeat(palm_pos_batch, 2, axis=0)),
            torch.from_numpy(np.repeat(palm_rot_batch, 2, axis=0)),
            torch.as_tensor(solution.q, dtype=torch.float32),
        ).numpy()
        requested = np.stack([open_targets, squeeze_targets])
        residual = float(np.max(np.linalg.norm(reached[:, active] - requested[:, active], axis=-1)))
        knots.append(
            PinchKnot(
                half_width=float(half_width),
                palm_offset=frame.palm_offset,
                palm_rotation=frame.palm_rotation,
                open_q=np.asarray(solution.q[0], dtype=np.float64),
                squeeze_q=np.asarray(solution.q[1], dtype=np.float64),
                contact_residual_m=residual,
            )
        )
    return PinchPriorTable(spec.config.name, spec.actuated_joint_names, knots)


#: Repository-relative home of the fitted prior shipped with the MVP.
DEFAULT_PRIOR_PATH = Path("configs/mvp/leap-pinch-prior-v0.json")


def load_or_build_prior(scope: MvpScopeConfig, spec: RobotSpec, path: str | Path | None = None) -> PinchPriorTable:
    """Load the shipped prior, or fit it if the artifact is absent."""

    resolved = Path(path) if path is not None else DEFAULT_PRIOR_PATH
    if resolved.is_file():
        return PinchPriorTable.load(resolved)
    return build_pinch_prior_table(spec, [variant.half_width for variant in scope.train_variants])
