"""Typed contracts for Phase 3.4 contact-rich dynamic grasp synthesis.

Static grasp treats every contact with the table or a neighbouring object as a
rejection.  Phase 3.4 allows those contacts under an explicit, measured budget,
so the contracts here carry three things the static ones do not: a class per
contact, a multi-quantity safety budget, and the target's own motion over time.

Nothing in this module admits a grasp.  Admission needs measured evidence from a
backend rollout and a CPU replay; these are the shapes that evidence takes.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

import numpy as np


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


#: Classes that reject a trajectory outright.  A high terminal score can never
#: buy one of these back (plan section 8).
HARD_REJECT_CLASSES: frozenset[ContactClass] = frozenset(
    {ContactClass.FORBIDDEN, ContactClass.DAMAGING}
)


class TrajectoryStage(str, Enum):
    """Phase of the acquisition a timestep belongs to."""

    APPROACH = "approach"
    REPOSITION = "reposition"
    ENCLOSE = "enclose"
    LIFT = "lift"
    PERTURB = "perturb"


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

    material_class: str = "rigid_default"
    environment_class: str = "table"

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.type is float or field.name.endswith(
                ("_N", "_Ns", "_s", "_J", "_m", "_Nm", "_rad", "_mps", "_load")
            ):
                value = getattr(self, field.name)
                if isinstance(value, float) and not (np.isfinite(value) and value > 0.0):
                    raise ValueError(f"{field.name} must be finite and positive, got {value!r}")


@dataclasses.dataclass(frozen=True)
class ContactEvent:
    """One measured contact interval, not a boolean.

    ``budget_margin`` is the smallest headroom across every quantity of the
    governing :class:`ContactSafetyBudget`; negative means the budget is blown.
    """

    time_index: int
    contact_class: ContactClass
    geom_a: str
    geom_b: str
    body_a: str
    body_b: str

    point: np.ndarray  # [3] world frame
    frame: np.ndarray  # [3, 3] contact frame, first column is the normal
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

    @property
    def is_hard_reject(self) -> bool:
        return self.contact_class in HARD_REJECT_CLASSES


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
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if not (np.isfinite(self.control_dt) and self.control_dt > 0.0):
            raise ValueError(f"control_dt must be finite and positive, got {self.control_dt}")


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
    contact_graph: tuple[ContactEvent, ...] = ()
    terminal_grasp: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        steps = int(self.time.shape[0])
        for name, expected_rank in (
            ("palm_pose", 2),
            ("joint_state", 2),
            ("actuator_command", 2),
            ("object_pose", 3),
            ("object_velocity", 3),
        ):
            array = getattr(self, name)
            if array.ndim != expected_rank:
                raise ValueError(f"{name} must have rank {expected_rank}, got {array.ndim}")
            if array.shape[0] != steps:
                raise ValueError(
                    f"{name} has {array.shape[0]} steps but time has {steps}"
                )
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
        for event in self.contact_graph:
            if not (0 <= event.time_index < steps):
                raise ValueError(
                    f"contact event time_index {event.time_index} outside [0, {steps})"
                )

    @property
    def num_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def num_objects(self) -> int:
        return int(self.object_pose.shape[1])

    def events_of_class(self, contact_class: ContactClass) -> tuple[ContactEvent, ...]:
        return tuple(e for e in self.contact_graph if e.contact_class is contact_class)

    @property
    def hard_reject_events(self) -> tuple[ContactEvent, ...]:
        return tuple(e for e in self.contact_graph if e.is_hard_reject)


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

    cpu_replay_evidence: dict[str, Any] = dataclasses.field(default_factory=dict)
    gpu_search_evidence: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.passed and self.failure_reason != "none":
            raise ValueError(
                f"a passed outcome must carry failure_reason 'none', got {self.failure_reason!r}"
            )
        if self.passed and not self.cpu_replay_evidence:
            raise ValueError(
                "a passed outcome requires cpu_replay_evidence: GPU search alone "
                "never admits a release positive"
            )
