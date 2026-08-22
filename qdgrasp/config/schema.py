"""Declarative QDGrasp configuration schemas.

Every schema is written from QDGrasp requirements; no key, default or docstring
is copied from another framework's configuration surface.  Unknown and dead keys
are hard errors (``extra="forbid"``) and each document carries an explicit
``schema`` version so migrations stay testable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .registry import register_document_schema


MODEL_SCHEMA_V1 = "qdgrasp/model/v1"
ROBOT_SCHEMA_V1 = "qdgrasp/robot/v1"
ROBOT_SCHEMA_V2 = "qdgrasp/robot/v2"
DATA_SCHEMA_V1 = "qdgrasp/data/v1"
RUN_SCHEMA_V1 = "qdgrasp/run/v1"

SUPPORTED_SCHEMAS: dict[str, frozenset[str]] = {
    "model": frozenset({MODEL_SCHEMA_V1}),
    "robot": frozenset({ROBOT_SCHEMA_V1, ROBOT_SCHEMA_V2}),
    "data": frozenset({DATA_SCHEMA_V1}),
    "run": frozenset({RUN_SCHEMA_V1}),
}


class ConfigError(ValueError):
    """Raised when a configuration document violates the QDGrasp schema."""


class _Document(BaseModel):
    """Base for every YAML-backed document: strict, immutable and versioned."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, protected_namespaces=())

    schema_version: str = Field(alias="schema")

    def to_document(self) -> dict[str, Any]:
        """Round-trip back to the YAML-shaped mapping, aliases included."""

        return self.model_dump(by_alias=True, mode="json")

    def content_hash(self) -> str:
        """Stable SHA-256 over the canonical JSON form of this document."""

        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelConfig(_Document):
    """Selects a registered model builder and its declared parameters."""

    schema_version: Literal[MODEL_SCHEMA_V1] = Field(alias="schema")
    name: str
    type: str
    params: dict[str, int | float | bool | str] = Field(default_factory=dict)


class RobotConfig(_Document):
    """Minimal named-joint profile consumed by heads, bundles and results.

    The full robot layer (URDF/MJCF, FK, mesh resolution) is Phase 2 scope; this
    document only fixes the contract Phase 1 needs: ordered named joints with
    finite limits plus the palm frame the poses are expressed in.
    """

    schema_version: Literal[ROBOT_SCHEMA_V1] = Field(alias="schema")
    name: str
    palm_link: str
    frame: str = "palm"
    joints: tuple[str, ...]
    joint_limits: dict[str, tuple[float, float]]

    @field_validator("joints")
    @classmethod
    def _joints_are_unique(cls, joints: tuple[str, ...]) -> tuple[str, ...]:
        if not joints:
            raise ValueError("robot profile needs at least one actuated joint")
        if len(set(joints)) != len(joints):
            raise ValueError("joint names must be unique and ordered")
        return joints

    @model_validator(mode="after")
    def _limits_cover_joints(self) -> "RobotConfig":
        missing = [name for name in self.joints if name not in self.joint_limits]
        if missing:
            raise ValueError(f"missing finite joint limits for {missing}")
        extra = sorted(set(self.joint_limits) - set(self.joints))
        if extra:
            raise ValueError(f"joint_limits declares unknown joints {extra}")
        for name in self.joints:
            lower, upper = self.joint_limits[name]
            if not (lower == lower and upper == upper) or lower in (float("-inf"),) or upper in (float("inf"),):
                raise ValueError(f"joint '{name}' needs finite limits")
            if lower >= upper:
                raise ValueError(f"joint '{name}' has an empty limit range [{lower}, {upper}]")
        return self

    @property
    def lower_limits(self) -> tuple[float, ...]:
        return tuple(self.joint_limits[name][0] for name in self.joints)

    @property
    def upper_limits(self) -> tuple[float, ...]:
        return tuple(self.joint_limits[name][1] for name in self.joints)


class DataConfig(_Document):
    """Selects a registered dataset builder and its declared parameters."""

    schema_version: Literal[DATA_SCHEMA_V1] = Field(alias="schema")
    name: str
    type: str
    params: dict[str, int | float | bool | str] = Field(default_factory=dict)


class RunConfig(_Document):
    """Runtime request for a train/val/predict/export invocation.

    Values here are *requested*; the effective values after device policy are
    produced by :func:`qdgrasp.config.policy.resolve_runtime`.
    """

    schema_version: Literal[RUN_SCHEMA_V1] = Field(alias="schema", default=RUN_SCHEMA_V1)
    device: str = "cpu"
    amp: bool = False
    seed: int = 0
    deterministic: bool = True
    max_steps: int = 100
    stop_after_steps: int | None = None
    val_interval: int = 0
    batch_size: int = 4
    learning_rate: float = 1e-3
    ema_decay: float = 0.0
    workers: int = 0
    grad_clip: float = 0.0
    project_dir: str = "runs"
    run_name: str = "phase1"
    resume: str | None = None

    @field_validator("device")
    @classmethod
    def _known_device(cls, device: str) -> str:
        if device == "cpu" or device == "cuda" or device.startswith("cuda:"):
            return device
        raise ValueError(f"unsupported device '{device}'; QDGrasp v1 supports 'cpu' and 'cuda[:index]'")

    @field_validator("max_steps", "batch_size")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("value must be >= 1")
        return value

    @field_validator("stop_after_steps")
    @classmethod
    def _positive_budget(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("stop_after_steps must be >= 1 when set")
        return value

    @field_validator("workers", "val_interval")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be >= 0")
        return value

    @field_validator("ema_decay")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("ema_decay must be in [0.0, 1.0)")
        return value

    @field_validator("project_dir")
    @classmethod
    def _relative_project_dir(cls, value: str) -> str:
        if value.startswith("/") or value.startswith("~"):
            raise ValueError("project_dir must be a relative path from the working directory")
        return value


register_document_schema("model", MODEL_SCHEMA_V1, ModelConfig)
register_document_schema("robot", ROBOT_SCHEMA_V1, RobotConfig)
register_document_schema("data", DATA_SCHEMA_V1, DataConfig)
register_document_schema("run", RUN_SCHEMA_V1, RunConfig)
