"""Pydantic configuration schema for dataset profile v2."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from ..config.active_scope import DEFAULT_ROBOT_PROFILES
from ..config.registry import register_document_schema
from ..config.schema import _Document

DATA_SCHEMA_V2 = "qdgrasp/data/v2"


class DataConfigV2(_Document):
    """Configuration profile for cross-embodiment datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["qdgrasp/data/v2"] = Field(default=DATA_SCHEMA_V2, alias="schema")
    name: str
    dataset_root: str
    manifest_file: str = "dataset_manifest.json"
    protocol_file: str | None = None
    point_count: int = 1024
    # Used for deterministic resampling when a cloud does not have point_count
    # points.  Batch/DataLoader knobs belong to RunConfig and are deliberately
    # rejected here instead of being accepted as silent no-ops.
    seed: int = 42
    #: Defaults to the active corpus of ADR-0008. A paused hand may still be
    #: listed explicitly in a config file, but nothing selects one by default.
    robot_profiles: tuple[str, ...] = DEFAULT_ROBOT_PROFILES

    @field_validator("point_count")
    @classmethod
    def _validate_positive(cls, value: int, info: Any) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value


register_document_schema("data", DATA_SCHEMA_V2, DataConfigV2)
