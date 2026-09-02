"""Robot profile schema v2 for cross-embodiment grasping."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import schema as _config_schema
from ..config.registry import register_document_schema

_Document = _config_schema._Document
ROBOT_SCHEMA_V1 = _config_schema.ROBOT_SCHEMA_V1

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


class FingertipContactSpec(BaseModel):
    """Pinned contact anchor and approach axis in a fingertip body's frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    offset: tuple[float, float, float]
    approach_axis: tuple[float, float, float]

    @field_validator("offset", "approach_axis")
    @classmethod
    def _finite_vector(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(float(component) == float(component) and abs(float(component)) != float("inf") for component in value):
            raise ValueError("contact vectors must be finite")
        return value

    @field_validator("approach_axis")
    @classmethod
    def _nonzero_axis(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        norm = sum(float(component) ** 2 for component in value) ** 0.5
        if norm < 1e-8:
            raise ValueError("approach_axis must be non-zero")
        return tuple(float(component) / norm for component in value)


class RobotConfigV2(_Document):
    """Rich robot profile for cross-embodiment kinematics, meshes and simulation."""

    schema_version: Literal["qdgrasp/robot/v2"] = Field(alias="schema", default=ROBOT_SCHEMA_V2)
    name: str
    format: Literal["mjcf", "urdf"] = "mjcf"
    source_asset: str
    palm_link: str
    base_link: str | None = None
    wrist_link: str | None = None
    fingertip_links: tuple[str, ...] = Field(default_factory=tuple)
    fingertip_contacts: dict[str, FingertipContactSpec] = Field(default_factory=dict)
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
    def _limits_cover_joints(self) -> RobotConfigV2:
        missing = [name for name in self.joints if name not in self.joint_limits]
        if missing:
            raise ValueError(f"missing finite joint limits for {missing}")
        extra = sorted(set(self.joint_limits) - set(self.joints))
        if extra:
            raise ValueError(f"joint_limits declares unknown joints {extra}")
        for name in self.joints:
            lower, upper = self.joint_limits[name]
            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError(f"joint '{name}' needs finite limits")
            if lower >= upper:
                raise ValueError(f"joint '{name}' has an empty limit range [{lower}, {upper}]")
        for mimic_name, mimic in self.mimic_joints.items():
            if mimic_name in self.joints:
                raise ValueError(f"mimic joint '{mimic_name}' must not also be an actuated joint")
            if mimic.target_joint not in self.joints:
                raise ValueError(
                    f"mimic joint '{mimic_name}' targets unknown actuated joint '{mimic.target_joint}'"
                )
        missing_contacts = sorted(set(self.fingertip_links) - set(self.fingertip_contacts))
        extra_contacts = sorted(set(self.fingertip_contacts) - set(self.fingertip_links))
        if missing_contacts or extra_contacts:
            raise ValueError(
                "fingertip_contacts must cover fingertip_links exactly: "
                f"missing={missing_contacts}, extra={extra_contacts}"
            )
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


register_document_schema("robot", ROBOT_SCHEMA_V2, RobotConfigV2)
