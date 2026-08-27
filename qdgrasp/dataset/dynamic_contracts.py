"""Typed contracts for Phase 3.4 contact-rich dynamic grasp synthesis.

Static grasp treats every contact with the table or a neighbouring object as a
rejection.  Phase 3.4 allows those contacts under an explicit, measured budget,
so the contracts here carry three things the static ones do not: a class per
contact, a multi-quantity safety budget, and the target's own motion over time.

Nothing in this module admits a grasp.  Admission needs measured evidence from a
backend rollout and a CPU replay; these are the shapes that evidence takes.

Version 2 of these contracts (P3.4.3 C01, blocker B-11) closes the gap that made
the first version unsafe to release from: a malformed trajectory could reach the
writer and the certifier unchallenged.  Every array is now checked for rank,
shape and finiteness; time has to be strictly increasing and to agree with a
declared sample period; quaternions have to be normalised and their order
declared; stages have to be typed, and a positive's required stages have to
appear in canonical order; contact events have to sit inside the rollout they
claim to belong to; and a positive outcome has to carry a typed CPU certificate
rather than any truthy object.

Nothing here is lenient on purpose.  A contract that repairs its input is a
contract that hides the bug it was meant to catch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Schema identity
# --------------------------------------------------------------------------

#: The first release contracts.  Readable in legacy mode, never release-ready.
DYNAMIC_TRAJECTORY_SCHEMA_V1 = "qdgrasp/dynamic-trajectory/v1"

#: Contracts with complete fail-closed validation, declared timebase and frames.
DYNAMIC_TRAJECTORY_SCHEMA_V2 = "qdgrasp/dynamic-trajectory/v2"
CONTACT_EVENT_SCHEMA_V2 = "qdgrasp/dynamic-contact-event/v2"
SEARCH_OUTCOME_SCHEMA_V2 = "qdgrasp/dynamic-search-outcome/v2"
REPLAY_CAPSULE_SCHEMA_V1 = "qdgrasp/replay-capsule/v1"
CPU_REPLAY_CERTIFICATE_SCHEMA_V1 = "qdgrasp/cpu-replay-certificate/v1"
CONTACTRICH_MANIFEST_SCHEMA_V2 = "qdgrasp/contactrich-manifest/v2"
GATE_EVIDENCE_SCHEMA_V1 = "qdgrasp/phase3_4_3-gate-evidence/v1"

#: Only these may back a release artifact.  A v1 payload is readable so that old
#: evidence stays inspectable, but reading it never promotes it (C01.5).
RELEASE_TRAJECTORY_SCHEMAS: frozenset[str] = frozenset({DYNAMIC_TRAJECTORY_SCHEMA_V2})
LEGACY_TRAJECTORY_SCHEMAS: frozenset[str] = frozenset({DYNAMIC_TRAJECTORY_SCHEMA_V1})

#: Quaternion order used everywhere in these contracts.  Written down because
#: half of robotics uses the other one, and a silent swap is a rotation bug that
#: looks like a physics bug.
QUATERNION_ORDER = "wxyz"

#: Tolerances for the geometric invariants.  Pinned rather than passed in, so a
#: caller cannot loosen them to get a trajectory accepted.
QUATERNION_NORM_ATOL = 1e-6
FRAME_ORTHONORMAL_ATOL = 1e-6
TIME_PERIOD_RTOL = 1e-6


class ContractViolation(ValueError):
    """A payload does not satisfy the contract it claims to satisfy.

    Deliberately a ``ValueError`` so that existing ``except ValueError`` paths
    keep working; the distinct type exists so a caller can tell a contract
    breach from an ordinary bad argument.
    """


# --------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------


class ContactClass(str, Enum):
    """How a contact is treated by the safety accounting.

    ``ALLOWED`` classes are still penalised by the search objective: permitting
    a contact is not the same as being indifferent to it.
    """

    TARGET_INTENTIONAL = "target_intentional"
    SUPPORT_ASSISTED = "support_assisted"
    NEIGHBOR_INCIDENTAL = "neighbor_incidental"
    SELF_CONTACT_ALLOWED = "self_contact_allowed"
    FORBIDDEN = "forbidden"
    DAMAGING = "damaging"


class ContactPairKind(str, Enum):
    """Exactly which two roles are touching.

    :class:`ContactClass` says how a contact is *treated*; this says what it
    *is*. v1 had only the treatment, so a hand resting on the table and a target
    resting on the table were both ``support_assisted`` -- which made "has the
    target left its support" unanswerable, and let a hand-floor contact keep a
    lifted object classified as still supported (blocker B-12).
    """

    TARGET_SUPPORT = "target_support"
    ROBOT_SUPPORT = "robot_support"
    TARGET_ROBOT = "target_robot"
    NON_TARGET_SUPPORT = "non_target_support"
    NON_TARGET_ROBOT = "non_target_robot"
    NON_TARGET_TARGET = "non_target_target"
    NON_TARGET_NON_TARGET = "non_target_non_target"
    ROBOT_SELF = "robot_self"
    TARGET_TARGET = "target_target"
    SUPPORT_SUPPORT = "support_support"
    UNKNOWN = "unknown"


#: Pair kinds that mean the target is resting on something. Only these answer
#: the support-release question; a robot link on the floor does not.
TARGET_SUPPORTING_PAIRS: frozenset[ContactPairKind] = frozenset(
    {ContactPairKind.TARGET_SUPPORT}
)

#: Pair kinds the robot's contact budget governs.
#:
#: ``ContactSafetyBudget`` is a budget for one ``robot_profile``: it bounds what
#: the *hand* does. A neighbouring box resting on the table is scene physics the
#: hand is not part of, and charging it against the hand's peak-force limit
#: rejects trajectories for the weight of the furniture. Scene damage is bounded
#: instead by the non-target translation, rotation and velocity limits, which are
#: measured at trajectory scope. ``TARGET_SUPPORT`` is in scope because the hand
#: really can crush the target against its support.
ROBOT_BUDGETED_PAIRS: frozenset[ContactPairKind] = frozenset(
    {
        ContactPairKind.TARGET_ROBOT,
        ContactPairKind.ROBOT_SUPPORT,
        ContactPairKind.NON_TARGET_ROBOT,
        ContactPairKind.ROBOT_SELF,
        ContactPairKind.TARGET_SUPPORT,
        ContactPairKind.UNKNOWN,
    }
)


#: Pair kinds that involve an object which is not the target. Damage to these is
#: scene damage, not acquisition progress.
NON_TARGET_PAIRS: frozenset[ContactPairKind] = frozenset(
    {
        ContactPairKind.NON_TARGET_SUPPORT,
        ContactPairKind.NON_TARGET_ROBOT,
        ContactPairKind.NON_TARGET_TARGET,
        ContactPairKind.NON_TARGET_NON_TARGET,
    }
)


#: Classes that reject a trajectory outright.  A high terminal score can never
#: buy one of these back (plan section 8).
HARD_REJECT_CLASSES: frozenset[ContactClass] = frozenset(
    {ContactClass.FORBIDDEN, ContactClass.DAMAGING}
)


class TrajectoryStage(str, Enum):
    """Phase of the acquisition a timestep belongs to.

    The order of declaration is the canonical order of an acquisition.
    ``SUPPORT_RELEASE`` and ``RETAIN`` are v2 additions, because a positive has
    to show the target leaving its support and still being held afterwards --
    neither of which the original five stages could express (C03.7).
    """

    APPROACH = "approach"
    REPOSITION = "reposition"
    ENCLOSE = "enclose"
    SUPPORT_RELEASE = "support_release"
    LIFT = "lift"
    PERTURB = "perturb"
    RETAIN = "retain"


#: Canonical rank of each stage, used to check the order a positive's required
#: stages first appear in.
STAGE_ORDER: dict[TrajectoryStage, int] = {
    stage: index for index, stage in enumerate(TrajectoryStage)
}

#: Stages a positive acquisition has to reach.  Checked at certification rather
#: than at construction: a failed trajectory legitimately stops early, and
#: refusing to build it would destroy the negative evidence.
REQUIRED_TERMINAL_STAGES: tuple[TrajectoryStage, ...] = (
    TrajectoryStage.ENCLOSE,
    TrajectoryStage.SUPPORT_RELEASE,
    TrajectoryStage.LIFT,
    TrajectoryStage.PERTURB,
)


class FailureStage(str, Enum):
    """Where an attempt stopped.  Closed, because ``failure_stage`` is grouped
    on in every ledger and a free string makes those counts meaningless."""

    NONE = "none"
    REQUEST = "request"
    COMPILE = "compile"
    BACKEND = "backend"
    SEARCH = "search"
    ROLLOUT = "rollout"
    CONTACT = "contact"
    APPROACH = "approach"
    REPOSITION = "reposition"
    ENCLOSE = "enclose"
    SUPPORT_RELEASE = "support_release"
    LIFT = "lift"
    PERTURB = "perturb"
    ACQUISITION = "acquisition"
    SCENE = "scene"
    TERMINAL = "terminal"
    CPU_REPLAY = "cpu_replay"
    DYNAMIC = "dynamic"


#: Reasons an attempt can fail.  A reason outside this set is a bug in the
#: producer, not a new category: whoever adds one adds it here so the ledger
#: keeps reconciling.
FAILURE_REASONS: frozenset[str] = frozenset(
    {
        "none",
        # rollout and numerics
        "empty_trajectory",
        "non_finite_state",
        "simulation_instability",
        "contact_buffer_overflow",
        "truncated_contact_stream",
        "world_rejected",
        "validated_rollout_failed",
        # contact and safety
        "forbidden_contact",
        "damaging_contact",
        "safety_budget_violation",
        "scene_damage",
        "non_target_disturbance",
        # acquisition semantics
        "insufficient_enclosure",
        "support_not_released",
        "insufficient_lift",
        "wrong_object_lift",
        "target_teleported",
        "perturbation_slip",
        "perturbation_failed",
        "no_environmental_assistance",
        "no_closure",
        # search and control
        "budget_exhausted",
        "no_feasible_elite",
        "missing_objective_term",
        "non_finite_objective",
        "dynamic_skipped",
        "unexpected_control_outcome",
        # parity and provenance
        "backend_divergence",
        "replay_violates_safety_budget",
        "gpu_only_evidence",
        "evidence_hash_mismatch",
        "stale_schema",
    }
)

#: Reason prefixes that carry a qualifier after a colon.  A primitive that timed
#: out has to say *which* condition never arrived, and a wrapped validator has to
#: say which of its own stages failed; neither may collapse into a bare string
#: (C03.6).
NAMESPACED_FAILURE_REASONS: tuple[str, ...] = ("transition_timeout", "validated_rollout")


def is_known_failure_reason(reason: str) -> bool:
    """True when ``reason`` is in the closed set or a declared namespace."""
    if reason in FAILURE_REASONS:
        return True
    head, _, tail = reason.partition(":")
    return bool(tail) and head in NAMESPACED_FAILURE_REASONS


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _require_ref(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(f"{field} must be a non-empty reference, got {value!r}")
    return value


def _require_finite_array(array: Any, *, field: str, rank: int) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype.kind not in "fiu":
        raise ContractViolation(f"{field} must be numeric, got dtype {values.dtype}")
    values = values.astype(np.float64, copy=False)
    if values.ndim != rank:
        raise ContractViolation(f"{field} must have rank {rank}, got {values.ndim}")
    if values.size and not np.all(np.isfinite(values)):
        raise ContractViolation(f"{field} contains non-finite values")
    return values


def _require_non_negative(value: float, *, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ContractViolation(f"{field} must be finite, got {value!r}")
    if number < 0.0:
        raise ContractViolation(f"{field} must not be negative, got {number}")
    return number


def _require_finite(value: float, *, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ContractViolation(f"{field} must be finite, got {value!r}")
    return number


def _require_unit_quaternions(quaternions: np.ndarray, *, field: str) -> None:
    if not quaternions.size:
        return
    norms = np.linalg.norm(quaternions, axis=-1)
    if not np.all(np.abs(norms - 1.0) <= QUATERNION_NORM_ATOL):
        worst = float(np.max(np.abs(norms - 1.0)))
        raise ContractViolation(
            f"{field} quaternions must be normalised in {QUATERNION_ORDER} order; "
            f"worst deviation {worst:.3e} exceeds {QUATERNION_NORM_ATOL:.1e}"
        )


def _require_finite_metrics(metrics: Mapping[str, float], *, field: str) -> dict[str, float]:
    checked: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ContractViolation(f"{field} has a non-string key {key!r}")
        checked[key] = _require_finite(value, field=f"{field}[{key!r}]")
    return checked


def canonical_hash(payload: Any) -> str:
    """Stable sha256 over a JSON-serialisable payload."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Safety budget
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContactSafetyBudget:
    """Conservative simulation limits for one robot profile and material class.

    These are research constraints.  They are **not** a claim that a physical
    hand survives them: a hardware claim needs manufacturer limits, calibration,
    a safety factor and its own revision record.
    """

    budget_id: str
    robot_profile: str

    peak_normal_force_N: float
    peak_tangential_force_N: float
    normal_impulse_Ns: float
    tangential_impulse_Ns: float
    contact_duration_s: float
    contact_work_J: float
    max_penetration_m: float
    max_wrist_force_N: float
    max_wrist_torque_Nm: float
    max_joint_or_tendon_load: float
    max_non_target_translation_m: float
    max_non_target_rotation_rad: float
    max_non_target_velocity_mps: float

    #: Impulse is judged over a rolling window, not over the whole rollout.
    #: Impulse is force times time, so a cumulative limit rejects every
    #: sustained hold no matter how gentle -- it would measure grasp duration
    #: rather than safety, and make hands incomparable by how long they held on.
    #: Sustained load is bounded by peak force and contact_duration_s instead.
    impulse_window_s: float = 0.1

    material_class: str = "rigid_default"
    environment_class: str = "table"

    def __post_init__(self) -> None:
        _require_ref(self.budget_id, field="budget_id")
        _require_ref(self.robot_profile, field="robot_profile")
        for field in dataclasses.fields(self):
            if field.type is float or field.name.endswith(
                ("_N", "_Ns", "_s", "_J", "_m", "_Nm", "_rad", "_mps", "_load")
            ):
                value = getattr(self, field.name)
                if isinstance(value, float) and not (np.isfinite(value) and value > 0.0):
                    raise ValueError(f"{field.name} must be finite and positive, got {value!r}")

    @property
    def limit_fields(self) -> tuple[str, ...]:
        """Every declared limit, in declaration order.

        Used by the safety observer to prove that each one has a sensor behind
        it: a limit nobody measures is worse than no limit, because it reads as
        a guarantee (blocker B-01).
        """
        return tuple(
            field.name
            for field in dataclasses.fields(self)
            if field.name.startswith(("peak_", "max_"))
            or field.name in {"normal_impulse_Ns", "tangential_impulse_Ns", "contact_duration_s", "contact_work_J"}
        )

    @property
    def budget_hash(self) -> str:
        return canonical_hash(dataclasses.asdict(self))


# --------------------------------------------------------------------------
# Contact events
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContactEvent:
    """One measured contact interval, not a boolean.

    ``budget_margin`` is the smallest headroom across every quantity of the
    governing :class:`ContactSafetyBudget`; negative means the budget is blown.

    ``time_index`` addresses the sampled state stream, while ``simulator_step``
    addresses the integrator step the contact was actually read on.  They are
    different clocks and v1 conflated them, which is how tail events all ended
    up pinned to the last sample (blocker B-06).
    """

    time_index: int
    contact_class: ContactClass
    geom_a: str
    geom_b: str
    body_a: str
    body_b: str

    point: np.ndarray  # [3] world frame
    frame: np.ndarray  # [3, 3] contact frame, first row is the normal
    normal_force_N: float
    tangential_force_N: float
    normal_impulse_Ns: float
    tangential_impulse_Ns: float
    penetration_m: float
    relative_velocity_mps: float
    slip_m: float
    work_J: float
    budget_margin: float

    duration_s: float = 0.0
    link_class: str = "unknown"
    #: Which two roles are touching. ``UNKNOWN`` is legal to construct so that a
    #: diagnostic reading is not lost, but a release path refuses it.
    pair_kind: ContactPairKind = ContactPairKind.UNKNOWN
    #: Integrator step this reading came from; ``-1`` means "not recorded",
    #: which a v2 release payload does not accept.
    simulator_step: int = -1
    #: Contact episodes are separated when the pair stops touching, so a
    #: recontact does not inherit the previous episode's duration or work.
    episode_index: int = 0

    def __post_init__(self) -> None:
        if int(self.time_index) < 0:
            raise ContractViolation(f"time_index must be non-negative, got {self.time_index}")
        if not isinstance(self.contact_class, ContactClass):
            raise ContractViolation(f"contact_class must be a ContactClass, got {self.contact_class!r}")
        for name in ("geom_a", "geom_b", "body_a", "body_b"):
            _require_ref(getattr(self, name), field=name)

        point = _require_finite_array(self.point, field="point", rank=1)
        if point.shape != (3,):
            raise ContractViolation(f"point must be [3], got {point.shape}")
        frame = _require_finite_array(self.frame, field="frame", rank=2)
        if frame.shape != (3, 3):
            raise ContractViolation(f"frame must be [3, 3], got {frame.shape}")
        residual = float(np.max(np.abs(frame @ frame.T - np.eye(3))))
        if residual > FRAME_ORTHONORMAL_ATOL:
            raise ContractViolation(
                f"contact frame must be orthonormal; residual {residual:.3e} exceeds "
                f"{FRAME_ORTHONORMAL_ATOL:.1e}"
            )

        for name in (
            "normal_force_N",
            "tangential_force_N",
            "normal_impulse_Ns",
            "tangential_impulse_Ns",
            "penetration_m",
            "relative_velocity_mps",
            "slip_m",
            "work_J",
            "duration_s",
        ):
            _require_non_negative(getattr(self, name), field=name)
        _require_finite(self.budget_margin, field="budget_margin")
        if int(self.episode_index) < 0:
            raise ContractViolation(f"episode_index must be non-negative, got {self.episode_index}")
        if not isinstance(self.pair_kind, ContactPairKind):
            raise ContractViolation(f"pair_kind must be a ContactPairKind, got {self.pair_kind!r}")

    @property
    def is_hard_reject(self) -> bool:
        return self.contact_class in HARD_REJECT_CLASSES

    @property
    def robot_budgeted(self) -> bool:
        """Whether the robot's contact budget governs this contact."""
        return self.pair_kind in ROBOT_BUDGETED_PAIRS

    @property
    def supports_target(self) -> bool:
        """Whether this contact is the target resting on something."""
        return self.pair_kind in TARGET_SUPPORTING_PAIRS

    @property
    def involves_non_target(self) -> bool:
        return self.pair_kind in NON_TARGET_PAIRS

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.geom_a, self.geom_b) if self.geom_a <= self.geom_b else (self.geom_b, self.geom_a)


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------

#: Backends a request may name.  ``cuda`` here is a *request*; whether it can be
#: served is the backend's decision, and it never falls back to CPU silently.
ALLOWED_BACKENDS: frozenset[str] = frozenset({"cpu", "mujoco_cpu", "cuda", "mjwarp_cuda"})


@dataclasses.dataclass(frozen=True)
class DynamicGraspRequest:
    """One search request against a compiled scene bucket."""

    scene_state_ref: str
    observation_ref: str
    target_object_id: str
    robot_profile: str
    strategy_id: str
    safety_budget_id: str

    horizon: int
    control_dt: float
    seed: int
    backend_request: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "scene_state_ref",
            "observation_ref",
            "target_object_id",
            "robot_profile",
            "strategy_id",
            "safety_budget_id",
        ):
            _require_ref(getattr(self, name), field=name)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if not (np.isfinite(self.control_dt) and self.control_dt > 0.0):
            raise ValueError(f"control_dt must be finite and positive, got {self.control_dt}")
        if self.backend_request not in ALLOWED_BACKENDS:
            raise ContractViolation(
                f"backend_request {self.backend_request!r} is not one of {sorted(ALLOWED_BACKENDS)}"
            )
        # A negative seed is not an error, but a non-integer one silently
        # changes which candidates get sampled on a rerun.
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ContractViolation(f"seed must be an int, got {self.seed!r}")

    @property
    def request_hash(self) -> str:
        return canonical_hash(dataclasses.asdict(self))


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrajectoryTimebase:
    """How the sampled stream relates to simulator time.

    v1 wrote ``index * control_dt`` and called it time, which is only correct
    when the sampler runs every step at the control rate.  It did not, so the
    recorded duration disagreed with the rollout it came from (blocker B-06).
    Stating all four numbers makes that disagreement checkable instead.
    """

    simulator_dt: float
    sample_every: int
    start_time_s: float = 0.0
    quaternion_order: str = QUATERNION_ORDER
    world_frame: str = "world"
    object_frame: str = "world"

    def __post_init__(self) -> None:
        if not (np.isfinite(self.simulator_dt) and self.simulator_dt > 0.0):
            raise ContractViolation(f"simulator_dt must be finite and positive, got {self.simulator_dt}")
        if int(self.sample_every) <= 0:
            raise ContractViolation(f"sample_every must be positive, got {self.sample_every}")
        _require_finite(self.start_time_s, field="start_time_s")
        if self.quaternion_order != QUATERNION_ORDER:
            raise ContractViolation(
                f"quaternion_order must be {QUATERNION_ORDER!r}, got {self.quaternion_order!r}"
            )

    @property
    def sample_period_s(self) -> float:
        return float(self.simulator_dt) * int(self.sample_every)

    def as_dict(self) -> dict[str, Any]:
        return {
            "simulator_dt": float(self.simulator_dt),
            "sample_every": int(self.sample_every),
            "sample_period_s": self.sample_period_s,
            "start_time_s": float(self.start_time_s),
            "quaternion_order": self.quaternion_order,
            "world_frame": self.world_frame,
            "object_frame": self.object_frame,
        }


@dataclasses.dataclass(frozen=True)
class DynamicGraspTrajectory:
    """Fixed-rate state samples plus a sparse contact stream.

    Storage is deliberately not one array per simulator step: the release
    dataset must not grow with the integrator timestep (plan section 5).
    """

    time: np.ndarray  # [T]
    palm_pose: np.ndarray  # [T, 7] position + quaternion (w, x, y, z)
    joint_state: np.ndarray  # [T, J]
    actuator_command: np.ndarray  # [T, U]
    object_pose: np.ndarray  # [T, O, 7]
    object_velocity: np.ndarray  # [T, O, 6]
    stage: tuple[TrajectoryStage, ...]  # [T]
    timebase: TrajectoryTimebase
    contact_graph: tuple[ContactEvent, ...] = ()
    terminal_grasp: dict[str, Any] | None = None
    schema: str = DYNAMIC_TRAJECTORY_SCHEMA_V2
    robot_profile: str = ""
    palm_body: str = ""

    def __post_init__(self) -> None:
        if self.schema not in RELEASE_TRAJECTORY_SCHEMAS | LEGACY_TRAJECTORY_SCHEMAS:
            raise ContractViolation(f"unknown trajectory schema {self.schema!r}")

        time = _require_finite_array(self.time, field="time", rank=1)
        steps = int(time.shape[0])

        for name, expected_rank in (
            ("palm_pose", 2),
            ("joint_state", 2),
            ("actuator_command", 2),
            ("object_pose", 3),
            ("object_velocity", 3),
        ):
            array = getattr(self, name)
            values = _require_finite_array(array, field=name, rank=expected_rank)
            if values.shape[0] != steps:
                raise ValueError(f"{name} has {values.shape[0]} steps but time has {steps}")

        if self.palm_pose.shape[1] != 7:
            raise ValueError(f"palm_pose must be [T, 7], got {self.palm_pose.shape}")
        if self.object_pose.shape[2] != 7:
            raise ValueError(f"object_pose must be [T, O, 7], got {self.object_pose.shape}")
        if self.object_velocity.shape[2] != 6:
            raise ValueError(
                f"object_velocity must be [T, O, 6], got {self.object_velocity.shape}"
            )
        if self.object_pose.shape[1] != self.object_velocity.shape[1]:
            raise ValueError("object_pose and object_velocity disagree on object count")
        if len(self.stage) != steps:
            raise ValueError(f"stage has {len(self.stage)} entries but time has {steps}")

        _require_unit_quaternions(np.asarray(self.palm_pose)[:, 3:7], field="palm_pose")
        _require_unit_quaternions(np.asarray(self.object_pose)[:, :, 3:7], field="object_pose")

        self._validate_time(time, steps)
        self._validate_stages()

        for event in self.contact_graph:
            if not (0 <= event.time_index < max(steps, 1)) or steps == 0:
                raise ValueError(
                    f"contact event time_index {event.time_index} outside [0, {steps})"
                )

    def _validate_time(self, time: np.ndarray, steps: int) -> None:
        if steps == 0:
            return
        expected_start = float(self.timebase.start_time_s)
        if abs(float(time[0]) - expected_start) > max(1e-9, TIME_PERIOD_RTOL * abs(expected_start)):
            raise ContractViolation(
                f"time starts at {float(time[0])!r} but the timebase declares {expected_start!r}"
            )
        if steps == 1:
            return
        deltas = np.diff(time)
        if not np.all(deltas > 0.0):
            raise ContractViolation("time must be strictly increasing")
        period = self.timebase.sample_period_s
        if not np.allclose(deltas, period, rtol=TIME_PERIOD_RTOL, atol=1e-12):
            worst = float(np.max(np.abs(deltas - period)))
            raise ContractViolation(
                f"time steps must equal the declared sample period {period!r}; "
                f"worst deviation {worst:.3e}"
            )

    def _validate_stages(self) -> None:
        # Typing is enforced here; canonical *ordering* is not. A contact-rich
        # strategy may legitimately reposition after enclosing, so a
        # monotonicity rule at construction would reject real physics. What a
        # positive has to show -- the required stages, in canonical order of
        # first appearance -- is checked at certification, where it belongs
        # (C03.7); a failed trajectory keeps its evidence either way.
        for index, stage in enumerate(self.stage):
            if not isinstance(stage, TrajectoryStage):
                raise ContractViolation(f"stage[{index}] must be a TrajectoryStage, got {stage!r}")

    @property
    def num_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def num_objects(self) -> int:
        return int(self.object_pose.shape[1])

    @property
    def duration_s(self) -> float:
        if self.num_steps < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    @property
    def is_release_schema(self) -> bool:
        return self.schema in RELEASE_TRAJECTORY_SCHEMAS

    @property
    def reached_stages(self) -> frozenset[TrajectoryStage]:
        return frozenset(self.stage)

    @property
    def has_required_terminal_stages(self) -> bool:
        """Whether the acquisition went through every stage a positive needs."""
        return all(stage in self.reached_stages for stage in REQUIRED_TERMINAL_STAGES)

    @property
    def terminal_stages_in_canonical_order(self) -> bool:
        """Whether the required stages first appear in the canonical order.

        Enclosing after the lift, or lifting before the support was released,
        is not an acquisition -- it is a mislabelled one.
        """
        if not self.has_required_terminal_stages:
            return False
        firsts = [self.stage.index(stage) for stage in REQUIRED_TERMINAL_STAGES]
        return all(a < b for a, b in itertools.pairwise(firsts))

    def events_of_class(self, contact_class: ContactClass) -> tuple[ContactEvent, ...]:
        return tuple(e for e in self.contact_graph if e.contact_class is contact_class)

    @property
    def hard_reject_events(self) -> tuple[ContactEvent, ...]:
        return tuple(e for e in self.contact_graph if e.is_hard_reject)


# --------------------------------------------------------------------------
# CPU replay certificate
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CpuReplayCertificate:
    """Typed evidence that a CPU oracle replayed this candidate and agreed.

    v1 accepted any truthy ``cpu_replay_evidence``, so ``{"confirmed": True}``
    admitted a release positive with nothing behind it (blockers B-05, B-11).
    The fields here are the ones a reviewer needs to re-run the replay and get
    the same answer: which backend, from which capsule, under which commands,
    against which compiled model, at which timestep.
    """

    backend_id: str
    capsule_sha256: str
    command_sha256: str
    model_sha256: str
    timestep_s: float
    terminal_certified: bool
    safety_certified: bool
    outcome_class: str
    schema: str = CPU_REPLAY_CERTIFICATE_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != CPU_REPLAY_CERTIFICATE_SCHEMA_V1:
            raise ContractViolation(f"unknown certificate schema {self.schema!r}")
        _require_ref(self.backend_id, field="backend_id")
        if "cuda" in self.backend_id:
            raise ContractViolation(
                f"a CPU replay certificate cannot name a CUDA backend ({self.backend_id!r}); "
                "GPU search never admits a release positive on its own"
            )
        for name in ("capsule_sha256", "command_sha256", "model_sha256"):
            value = getattr(self, name)
            _require_ref(value, field=name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractViolation(f"{name} must be a sha256 hex digest, got {value!r}")
        if not (np.isfinite(self.timestep_s) and self.timestep_s > 0.0):
            raise ContractViolation(f"timestep_s must be finite and positive, got {self.timestep_s}")
        _require_ref(self.outcome_class, field="outcome_class")

    @property
    def is_positive(self) -> bool:
        return self.terminal_certified and self.safety_certified

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
# Outcome
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DynamicSearchOutcome:
    """Result of one search request, positive or not.

    Failure trajectories are kept deliberately: a critic or safety model trained
    later needs them (plan section 11).
    """

    trajectory_ref: str
    passed: bool
    failure_stage: str
    failure_reason: str

    objective_terms: dict[str, float] = dataclasses.field(default_factory=dict)
    peak_safety_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    cumulative_safety_metrics: dict[str, float] = dataclasses.field(default_factory=dict)

    cpu_replay_evidence: CpuReplayCertificate | None = None
    gpu_search_evidence: dict[str, Any] | None = None
    schema: str = SEARCH_OUTCOME_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema != SEARCH_OUTCOME_SCHEMA_V2:
            raise ContractViolation(f"unknown outcome schema {self.schema!r}")

        if self.failure_stage not in {stage.value for stage in FailureStage}:
            raise ContractViolation(
                f"failure_stage {self.failure_stage!r} is not one of "
                f"{sorted(stage.value for stage in FailureStage)}"
            )
        if not is_known_failure_reason(self.failure_reason):
            raise ContractViolation(
                f"failure_reason {self.failure_reason!r} is not a known reason; add it to "
                "FAILURE_REASONS rather than inventing one at the call site"
            )

        _require_finite_metrics(self.objective_terms, field="objective_terms")
        _require_finite_metrics(self.peak_safety_metrics, field="peak_safety_metrics")
        _require_finite_metrics(self.cumulative_safety_metrics, field="cumulative_safety_metrics")

        if self.passed:
            if self.failure_reason != "none":
                raise ValueError(
                    f"a passed outcome must carry failure_reason 'none', got {self.failure_reason!r}"
                )
            if self.failure_stage != FailureStage.NONE.value:
                raise ContractViolation(
                    f"a passed outcome must carry failure_stage 'none', got {self.failure_stage!r}"
                )
            if not isinstance(self.cpu_replay_evidence, CpuReplayCertificate):
                raise ValueError(
                    "a passed outcome requires cpu_replay_evidence as a typed "
                    "CpuReplayCertificate: GPU search alone never admits a release "
                    f"positive, and a truthy object is not evidence (got {self.cpu_replay_evidence!r})"
                )
            if not self.cpu_replay_evidence.is_positive:
                raise ContractViolation(
                    "a passed outcome needs a certificate whose terminal and safety "
                    "checks both certified"
                )
        else:
            if self.failure_reason == "none":
                raise ContractViolation("a failed outcome must name a failure_reason")
            if self.failure_stage == FailureStage.NONE.value:
                raise ContractViolation("a failed outcome must name a failure_stage")

    @property
    def outcome_class(self) -> str:
        return "pass" if self.passed else f"fail:{self.failure_reason}"

    @property
    def is_release_positive(self) -> bool:
        return bool(self.passed and isinstance(self.cpu_replay_evidence, CpuReplayCertificate))


def outcome_evidence_dict(outcome: DynamicSearchOutcome) -> dict[str, Any] | None:
    """The certificate of ``outcome`` as a plain dict, for serialisation."""
    certificate = outcome.cpu_replay_evidence
    return certificate.as_dict() if certificate is not None else None


def certificate_from_dict(payload: Mapping[str, Any] | None) -> CpuReplayCertificate | None:
    """Rebuild a certificate, rejecting unknown keys rather than ignoring them."""
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ContractViolation(f"cpu_replay_evidence must be a mapping, got {type(payload).__name__}")
    known = {field.name for field in dataclasses.fields(CpuReplayCertificate)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ContractViolation(f"cpu_replay_evidence carries unknown keys {unknown}")
    return CpuReplayCertificate(**dict(payload))


def sequence_hash(values: Sequence[Any] | np.ndarray) -> str:
    """Hash a command or state tensor by value, shape and dtype.

    Shape and dtype are in the digest on purpose: the same numbers at a
    different shape are a different command sequence, and a silent dtype change
    is exactly the kind of drift a replay is supposed to catch.
    """
    array = np.asarray(values)
    payload = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.astype(np.float64, copy=False).ravel().tolist()
        if array.size
        else [],
    }
    return canonical_hash(payload)
