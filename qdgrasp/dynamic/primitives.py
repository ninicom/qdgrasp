"""Parameterised control primitives and their sequencing (P3.4-07).

A primitive is a **control prior**, not a target pose. `push` says "drive the
hand along this direction at this speed for at most this long"; it never writes
where the object should end up. The object moves because contact moved it, which
is the property the whole phase rests on -- a primitive that set object pose
would manufacture the result it is meant to measure.

Transitions are driven by observed contact and object state, so a sequence
advances when the physics says the precondition is met, not on a fixed clock.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from enum import Enum

import numpy as np

from qdgrasp.dataset.dynamic_contracts import ContactClass, ContactEvent, TrajectoryStage


class PrimitiveKind(str, Enum):
    """The v1 primitive vocabulary from the plan."""

    PUSH = "push"
    SLIDE = "slide"
    ROLL = "roll"
    PIVOT_ON_SUPPORT = "pivot_on_support"
    HOOK = "hook"
    CAGE = "cage"
    SQUEEZE = "squeeze"
    SUPPORT_RELEASE = "support_release"
    LIFT = "lift"
    PERTURB = "perturb"


#: Which acquisition stage each primitive belongs to.
_STAGE_OF: dict[PrimitiveKind, TrajectoryStage] = {
    PrimitiveKind.PUSH: TrajectoryStage.REPOSITION,
    PrimitiveKind.SLIDE: TrajectoryStage.REPOSITION,
    PrimitiveKind.ROLL: TrajectoryStage.REPOSITION,
    PrimitiveKind.PIVOT_ON_SUPPORT: TrajectoryStage.REPOSITION,
    PrimitiveKind.HOOK: TrajectoryStage.REPOSITION,
    PrimitiveKind.CAGE: TrajectoryStage.ENCLOSE,
    PrimitiveKind.SQUEEZE: TrajectoryStage.ENCLOSE,
    PrimitiveKind.SUPPORT_RELEASE: TrajectoryStage.SUPPORT_RELEASE,
    PrimitiveKind.LIFT: TrajectoryStage.LIFT,
    PrimitiveKind.PERTURB: TrajectoryStage.PERTURB,
}


class TransitionCondition(str, Enum):
    """What must be observed before a primitive yields to the next."""

    #: Ran its full duration.
    DURATION_ELAPSED = "duration_elapsed"
    #: At least one intentional contact with the target exists.
    TARGET_CONTACT_MADE = "target_contact_made"
    #: No contact of any class with the target remains.
    TARGET_CONTACT_LOST = "target_contact_lost"
    #: The target no longer touches any support geom.
    SUPPORT_RELEASED = "support_released"
    #: At least this many distinct robot links touch the target.
    ENCLOSURE_REACHED = "enclosure_reached"


@dataclasses.dataclass(frozen=True)
class Primitive:
    """One parameterised control prior.

    ``direction`` is a unit vector in the world frame and ``speed`` a magnitude,
    together forming a velocity command for the wrist. There is deliberately no
    field for a desired object pose.
    """

    kind: PrimitiveKind
    direction: np.ndarray  # [3] unit vector
    speed: float  # m/s
    max_duration_s: float
    until: TransitionCondition = TransitionCondition.DURATION_ELAPSED
    grip: float = 0.0  # normalised finger closure command in [0, 1]
    required_contacts: int = 2

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=np.float64)
        if direction.shape != (3,):
            raise ValueError(f"direction must be [3], got {direction.shape}")
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-9:
            raise ValueError("direction must be a finite non-degenerate vector")
        object.__setattr__(self, "direction", direction / norm)
        if not (np.isfinite(self.speed) and self.speed >= 0.0):
            raise ValueError(f"speed must be finite and non-negative, got {self.speed}")
        if not (np.isfinite(self.max_duration_s) and self.max_duration_s > 0.0):
            raise ValueError(
                f"max_duration_s must be finite and positive, got {self.max_duration_s}"
            )
        if not 0.0 <= self.grip <= 1.0:
            raise ValueError(f"grip must lie in [0, 1], got {self.grip}")
        if self.required_contacts < 1:
            raise ValueError(f"required_contacts must be >= 1, got {self.required_contacts}")

    @property
    def stage(self) -> TrajectoryStage:
        return _STAGE_OF[self.kind]

    def wrist_velocity(self) -> np.ndarray:
        return self.direction * self.speed


def _target_contacts(events: Sequence[ContactEvent]) -> tuple[ContactEvent, ...]:
    return tuple(
        e
        for e in events
        if e.contact_class
        in (ContactClass.TARGET_INTENTIONAL, ContactClass.DAMAGING)
    )


def condition_met(
    condition: TransitionCondition,
    *,
    events: Sequence[ContactEvent],
    elapsed_s: float,
    max_duration_s: float,
    required_contacts: int,
) -> bool:
    """Evaluate one transition condition against observed state."""
    if condition is TransitionCondition.DURATION_ELAPSED:
        return elapsed_s >= max_duration_s
    if condition is TransitionCondition.TARGET_CONTACT_MADE:
        return bool(_target_contacts(events))
    if condition is TransitionCondition.TARGET_CONTACT_LOST:
        return not _target_contacts(events)
    if condition is TransitionCondition.SUPPORT_RELEASED:
        # Only the *target* resting on something answers this. A robot link on
        # the table is support-assisted too, and counting it kept every grasp
        # "still supported" no matter how high the object had been lifted
        # (blocker B-12).
        return not any(e.supports_target for e in events)
    if condition is TransitionCondition.ENCLOSURE_REACHED:
        links = {e.body_a for e in _target_contacts(events)} | {
            e.body_b for e in _target_contacts(events)
        }
        # Discount the target's own body from the link count.
        return len(links) - 1 >= required_contacts
    raise ValueError(f"unhandled transition condition: {condition}")


@dataclasses.dataclass(frozen=True)
class PrimitiveStep:
    """What the controller commands for one timestep."""

    index: int
    primitive: Primitive
    stage: TrajectoryStage
    wrist_velocity: np.ndarray
    grip: float
    advanced: bool
    finished: bool
    #: True when the primitive yielded because its clock ran out rather than
    #: because its condition was observed. The two are not the same outcome and
    #: must not be recorded as one (blocker B-15).
    timed_out: bool = False
    #: The condition that never arrived, as a typed failure reason.
    timeout_reason: str = ""


class PrimitiveSequenceController:
    """Walks a primitive sequence, advancing on observed state.

    A primitive that never meets its condition still yields at
    ``max_duration_s``: without that ceiling a search could stall forever on a
    precondition the physics will not produce.
    """

    def __init__(self, sequence: Sequence[Primitive], control_dt: float) -> None:
        if not sequence:
            raise ValueError("a primitive sequence must contain at least one primitive")
        if not (np.isfinite(control_dt) and control_dt > 0.0):
            raise ValueError(f"control_dt must be finite and positive, got {control_dt}")
        self._sequence = tuple(sequence)
        self._dt = float(control_dt)
        self._index = 0
        self._elapsed = 0.0
        self._timeouts: list[str] = []

    @property
    def sequence(self) -> tuple[Primitive, ...]:
        return self._sequence

    @property
    def finished(self) -> bool:
        return self._index >= len(self._sequence)

    @property
    def current(self) -> Primitive | None:
        return None if self.finished else self._sequence[self._index]

    def reset(self) -> None:
        self._index = 0
        self._elapsed = 0.0
        self._timeouts = []

    def step(self, events: Sequence[ContactEvent]) -> PrimitiveStep:
        """Command one timestep and decide whether to advance."""
        if self.finished:
            raise RuntimeError("the primitive sequence is exhausted; call reset() first")

        primitive = self._sequence[self._index]
        self._elapsed += self._dt

        satisfied = condition_met(
            primitive.until,
            events=events,
            elapsed_s=self._elapsed,
            max_duration_s=primitive.max_duration_s,
            required_contacts=primitive.required_contacts,
        )
        expired = self._elapsed >= primitive.max_duration_s
        # A timeout is only a timeout when the condition is something other than
        # the clock: a DURATION_ELAPSED primitive that runs its course did what
        # it was asked.
        timed_out = bool(
            expired
            and not satisfied
            and primitive.until is not TransitionCondition.DURATION_ELAPSED
        )
        advance = satisfied or expired
        if timed_out:
            self._timeouts.append(primitive.until.value)

        step = PrimitiveStep(
            index=self._index,
            primitive=primitive,
            stage=primitive.stage,
            wrist_velocity=primitive.wrist_velocity(),
            grip=primitive.grip,
            advanced=advance,
            finished=advance and self._index + 1 >= len(self._sequence),
            timed_out=timed_out,
            timeout_reason=f"transition_timeout:{primitive.until.value}" if timed_out else "",
        )
        if advance:
            self._index += 1
            self._elapsed = 0.0
        return step

    @property
    def timeouts(self) -> tuple[str, ...]:
        """Conditions that never arrived, in the order they expired.

        A sequence that finished only because every clock ran out has not shown
        the acquisition it claims; the caller turns this into a typed negative
        rather than letting the sequence end quietly (blocker B-15).
        """
        return tuple(self._timeouts)

    @property
    def first_timeout_reason(self) -> str:
        return f"transition_timeout:{self._timeouts[0]}" if self._timeouts else ""


def table_pivot_sequence(approach_axis: np.ndarray) -> tuple[Primitive, ...]:
    """A reference sequence: push against the support, then enclose and lift.

    Exists as a declared prior for the search to start from, not as a solution.
    """
    axis = np.asarray(approach_axis, dtype=np.float64)
    return (
        Primitive(
            kind=PrimitiveKind.PUSH,
            direction=axis,
            speed=0.05,
            max_duration_s=0.6,
            until=TransitionCondition.TARGET_CONTACT_MADE,
        ),
        Primitive(
            kind=PrimitiveKind.PIVOT_ON_SUPPORT,
            direction=axis,
            speed=0.03,
            max_duration_s=0.8,
            grip=0.3,
        ),
        Primitive(
            kind=PrimitiveKind.CAGE,
            direction=np.array([0.0, 0.0, 1.0]),
            speed=0.0,
            max_duration_s=0.5,
            grip=0.6,
            until=TransitionCondition.ENCLOSURE_REACHED,
            required_contacts=2,
        ),
        Primitive(
            kind=PrimitiveKind.SQUEEZE,
            direction=np.array([0.0, 0.0, 1.0]),
            speed=0.0,
            max_duration_s=0.5,
            grip=1.0,
        ),
        Primitive(
            kind=PrimitiveKind.LIFT,
            direction=np.array([0.0, 0.0, 1.0]),
            speed=0.08,
            max_duration_s=1.0,
            grip=1.0,
            until=TransitionCondition.SUPPORT_RELEASED,
        ),
    )
