"""Simulator-neutral RL contracts (P3.5-09).

Gymnasium's ``reset``/``step`` signatures are the boundary, but Gymnasium itself
is not a dependency here: the base install must not pull a package that has not
been through reference intake, and nothing in these contracts needs it.  Spaces
are described declaratively and :func:`to_gymnasium_space` converts them if the
caller has Gymnasium installed.

Two invariants are enforced rather than documented.

``terminated`` means the task ended -- success or failure.  Running out of
horizon or compute budget is ``truncated``.  Collapsing the two teaches a value
function that the end of the clock is a property of the state.

The reward total is the sum of its logged terms, checked on construction.  A
reward with an unlogged component cannot be audited, and a positive term must
never be able to pay for a safety barrier, so barriers terminate instead of
subtracting.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Literal, Protocol

import numpy as np

RL_CONTRACT_SCHEMA_V1 = "qdgrasp/rl-contract/v1"


@dataclasses.dataclass(frozen=True)
class BoxSpace:
    """A bounded continuous space, described without importing a framework."""

    name: str
    shape: tuple[int, ...]
    low: float = -np.inf
    high: float = np.inf
    dtype: str = "float32"

    def validate(self) -> None:
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError(f"space {self.name!r} must have a positive shape, got {self.shape}")
        if not self.low < self.high:
            raise ValueError(f"space {self.name!r} needs low < high, got [{self.low}, {self.high}]")

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def contains(self, value: np.ndarray) -> bool:
        array = np.asarray(value)
        return bool(
            array.shape == self.shape
            and np.all(np.isfinite(array))
            and np.all(array >= self.low - 1e-6)
            and np.all(array <= self.high + 1e-6)
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "low": None if np.isinf(self.low) else self.low,
            "high": None if np.isinf(self.high) else self.high,
            "dtype": self.dtype,
        }


def to_gymnasium_space(space: BoxSpace) -> Any:
    """Convert to ``gymnasium.spaces.Box`` when Gymnasium is available."""

    try:
        from gymnasium import spaces  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - exercised only where installed
        raise ImportError(
            "gymnasium is not installed; the RL contracts do not require it, and it is not a base dependency"
        ) from error
    return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=np.dtype(space.dtype))


@dataclasses.dataclass(frozen=True)
class ObservationField:
    """One named block of the observation, with its frame and unit stated."""

    name: str
    size: int
    unit: str
    frame: str
    description: str = ""

    def validate(self) -> None:
        if self.size <= 0:
            raise ValueError(f"observation field {self.name!r} must have positive size")
        if not self.unit or not self.frame:
            raise ValueError(f"observation field {self.name!r} must declare a unit and a frame")


@dataclasses.dataclass(frozen=True)
class ObservationSchema:
    """The ordered observation layout, hashable so a checkpoint can pin it."""

    fields: tuple[ObservationField, ...]
    schema_version: str = RL_CONTRACT_SCHEMA_V1

    def validate(self) -> None:
        if not self.fields:
            raise ValueError("an observation schema needs at least one field")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("observation field names must be unique")
        for field in self.fields:
            field.validate()

    @property
    def dimension(self) -> int:
        return sum(field.size for field in self.fields)

    def offset_of(self, name: str) -> slice:
        offset = 0
        for field in self.fields:
            if field.name == name:
                return slice(offset, offset + field.size)
            offset += field.size
        raise KeyError(f"observation schema has no field {name!r}")

    def space(self) -> BoxSpace:
        return BoxSpace(name="observation", shape=(self.dimension,), low=-1e6, high=1e6, dtype="float32")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "dimension": self.dimension,
            "fields": [dataclasses.asdict(field) for field in self.fields],
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assemble(self, parts: Mapping[str, np.ndarray]) -> np.ndarray:
        """Concatenate named blocks in schema order, checking each size."""

        chunks: list[np.ndarray] = []
        for field in self.fields:
            if field.name not in parts:
                raise KeyError(f"observation part {field.name!r} was not produced")
            chunk = np.asarray(parts[field.name], dtype=np.float64).reshape(-1)
            if chunk.shape[0] != field.size:
                raise ValueError(
                    f"observation part {field.name!r} has {chunk.shape[0]} entries, schema declares {field.size}"
                )
            chunks.append(chunk)
        return np.concatenate(chunks).astype(np.float32)


PalmCommand = Literal["delta_pose_6d", "twist_6d", "fixed"]
JointCommand = Literal["named_position_target", "named_delta_target"]


@dataclasses.dataclass(frozen=True)
class RlActionSpec:
    """The engine-independent action contract (``ROADMAP-P3.5-001`` §6.3)."""

    joint_names: tuple[str, ...]
    active_joint_mask: tuple[bool, ...]
    control_dt: float
    palm_command: PalmCommand = "delta_pose_6d"
    joint_command: JointCommand = "named_delta_target"
    palm_translation_limit_m: float = 0.01
    palm_rotation_limit_rad: float = 0.1
    joint_delta_limit_rad: float = 0.1

    def validate(self) -> None:
        if len(self.joint_names) != len(self.active_joint_mask):
            raise ValueError("active_joint_mask must have one entry per joint")
        if not any(self.active_joint_mask):
            raise ValueError("at least one joint must be active")
        if not np.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError(f"control_dt must be finite and positive, got {self.control_dt!r}")
        if self.palm_command not in ("delta_pose_6d", "twist_6d", "fixed"):
            raise ValueError(f"palm_command={self.palm_command!r}")
        if self.joint_command not in ("named_position_target", "named_delta_target"):
            raise ValueError(f"joint_command={self.joint_command!r}")

    @property
    def palm_dimension(self) -> int:
        return 0 if self.palm_command == "fixed" else 6

    @property
    def dimension(self) -> int:
        return self.palm_dimension + int(sum(self.active_joint_mask))

    def space(self) -> BoxSpace:
        return BoxSpace(name="action", shape=(self.dimension,), low=-1.0, high=1.0, dtype="float32")

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "joint_names": list(self.joint_names),
                "active_joint_mask": [bool(flag) for flag in self.active_joint_mask],
                "control_dt": self.control_dt,
                "palm_command": self.palm_command,
                "joint_command": self.joint_command,
                "palm_translation_limit_m": self.palm_translation_limit_m,
                "palm_rotation_limit_rad": self.palm_rotation_limit_rad,
                "joint_delta_limit_rad": self.joint_delta_limit_rad,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TerminalReason(str, Enum):
    """Exactly one of these is reported when an episode ends."""

    NONE = "none"
    SUCCESS = "success"
    OBJECT_DROPPED = "object_dropped"
    CONTACT_LOST = "contact_lost"
    SAFETY_PENETRATION = "safety_penetration"
    SAFETY_IMPULSE = "safety_impulse"
    NON_TARGET_DISTURBED = "non_target_disturbed"
    INVALID_STATE = "invalid_state"
    SIMULATOR_ERROR = "simulator_error"
    #: Only ever paired with ``truncated``.
    HORIZON = "horizon"


#: The reward terms of §6.4.  Positive terms first, penalties second; the split
#: is not cosmetic, because a barrier must never be payable by a bonus.
REWARD_TERMS: tuple[str, ...] = (
    "reach",
    "contact_progress",
    "enclosure",
    "lift",
    "retention",
    "penetration",
    "unsafe_impulse",
    "non_target_disturbance",
    "action_rate",
    "joint_limit",
    "drop",
)

PENALTY_TERMS: frozenset[str] = frozenset(
    {"penetration", "unsafe_impulse", "non_target_disturbance", "action_rate", "joint_limit", "drop"}
)


@dataclasses.dataclass(frozen=True)
class RewardBreakdown:
    """Per-term reward, with the total derived rather than asserted."""

    terms: dict[str, float]

    def __post_init__(self) -> None:
        unknown = sorted(set(self.terms) - set(REWARD_TERMS))
        if unknown:
            raise ValueError(f"unknown reward terms: {unknown}")
        for name, value in self.terms.items():
            if not np.isfinite(value):
                raise ValueError(f"reward term {name!r} is not finite: {value!r}")
            if name in PENALTY_TERMS and value > 0.0:
                raise ValueError(f"penalty term {name!r} must be non-positive, got {value}")

    @property
    def total(self) -> float:
        return float(sum(self.terms.values()))

    def to_document(self) -> dict[str, Any]:
        document = {name: float(self.terms.get(name, 0.0)) for name in REWARD_TERMS}
        document["total"] = self.total
        return document


@dataclasses.dataclass
class StepResult:
    """One environment step, in the shape Gymnasium's API expects."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated are mutually exclusive: the task either ended or the clock did")
        reason = self.info.get("terminal_reason")
        if self.terminated and reason in (None, TerminalReason.NONE, TerminalReason.NONE.value):
            raise ValueError("a terminated step must report a terminal reason")
        if self.truncated and reason not in (TerminalReason.HORIZON, TerminalReason.HORIZON.value):
            raise ValueError("a truncated step must report the horizon as its reason")

    def as_tuple(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.observation, self.reward, self.terminated, self.truncated, self.info


class RlEnvironment(Protocol):
    """The environment surface every QDGrasp RL environment presents."""

    environment_id: str

    def reset(self, *, seed: int, options: Mapping[str, Any] | None = ...) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...

    def observation_space(self) -> BoxSpace: ...

    def action_space(self) -> BoxSpace: ...

    def close(self) -> None: ...
