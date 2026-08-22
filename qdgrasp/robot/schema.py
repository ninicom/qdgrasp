"""Robot profile schema v2 for cross-embodiment grasping."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config.schema import ROBOT_SCHEMA_V1, _Document


ROBOT_SCHEMA_V2 = "qdgrasp/robot/v2"


class MimicSpec(BaseModel):
    """Specification of a mimic/coupled joint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_joint: str
    multiplier: float = 1.0
    offset: float = 0.0


class ActuatorSpec(BaseModel):
    """Specification of an actuator or squeeze configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    joint: str | None = None
    type: str = "position"
    kp: float = 1.0
    kv: float = 0.1
    ctrl_range: tuple[float, float] | None = None
    force_range: tuple[float, float] | None = None
    squeeze_target: float | None = None


class RobotConfigV2(_Document):
    """Rich robot profile for cross-embodiment kinematics, meshes and simulation."""

    schema_version: Literal[ROBOT_SCHEMA_V2] = Field(alias="schema", default=ROBOT_SCHEMA_V2)
    name: str
    format: Literal["mjcf", "urdf"] = "mjcf"
    source_asset: str
    palm_link: str
    base_link: str | None = None
    wrist_link: str | None = None
    fingertip_links: tuple[str, ...] = Field(default_factory=tuple)
    contact_links: tuple[str, ...] = Field(default_factory=tuple)
    joints: tuple[str, ...]
    joint_limits: dict[str, tuple[float, float]]
    mimic_joints: dict[str, MimicSpec] = Field(default_factory=dict)
    actuators: dict[str, ActuatorSpec] = Field(default_factory=dict)
    mesh_root: str | None = None
    package_roots: dict[str, str] = Field(default_factory=dict)
    frame: str = "palm"
    release_blocked: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("joints")
    @classmethod
    def _joints_are_unique(cls, joints: tuple[str, ...]) -> tuple[str, ...]:
        if not joints:
            raise ValueError("robot profile needs at least one actuated joint")
        if len(set(joints)) != len(joints):
            raise ValueError("joint names must be unique and ordered")
        return joints

    @field_validator("fingertip_links", "contact_links")
    @classmethod
    def _links_are_unique(cls, links: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(links)) != len(links):
            raise ValueError("link names must be unique")
        return links

    @model_validator(mode="after")
    def _limits_cover_joints(self) -> "RobotConfigV2":
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

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def num_fingertips(self) -> int:
        return len(self.fingertip_links)
