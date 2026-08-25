"""Manifest schema for immutable scene datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qdgrasp.config.schema import ConfigError

SCENE_MANIFEST_SCHEMA_V1 = "qdgrasp/scene-dataset-manifest/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SceneShardMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    sha256: str
    num_records: int
    record_type: Literal["scene_state", "observation", "grasp"]
    split: str

    @field_validator("filename")
    @classmethod
    def _relative_filename(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("scene shard filename must be a normalized relative POSIX path")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("scene shard sha256 must contain 64 lowercase hex characters")
        return value

    @field_validator("num_records")
    @classmethod
    def _positive_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("scene shard num_records cannot be negative")
        return value


class SceneDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["qdgrasp/scene-dataset-manifest/v1"] = Field(
        default=SCENE_MANIFEST_SCHEMA_V1, alias="schema"
    )
    dataset_id: str
    generator_version: str
    generator_commit: str
    generator_worktree_dirty: bool
    seed: int
    splits: dict[str, list[str]]
    scene_spec_hashes: dict[str, str]
    camera_calibration_hashes: dict[str, str]
    environment_hashes: dict[str, str]
    object_asset_hashes: dict[str, str] = Field(default_factory=dict)
    robot_profile_hashes: dict[str, str] = Field(default_factory=dict)
    split_hashes: dict[str, str] = Field(default_factory=dict)
    release_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    source_licenses: dict[str, str]
    shards: tuple[SceneShardMetadata, ...]
    success_criteria: dict[str, float]
    coverage: dict[str, Any] = Field(default_factory=dict)
    resource_policy: dict[str, Any] = Field(default_factory=dict)
    release_blocked: bool = True
    invalidated: bool = False
    invalidation_reason: str = ""

    @field_validator(
        "scene_spec_hashes",
        "camera_calibration_hashes",
        "environment_hashes",
        "object_asset_hashes",
        "robot_profile_hashes",
        "split_hashes",
        "release_artifact_hashes",
    )
    @classmethod
    def _hash_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        bad = sorted(key for key, digest in value.items() if not _SHA256.fullmatch(digest))
        if bad:
            raise ValueError(f"invalid SHA-256 entries: {bad}")
        return value

    @model_validator(mode="after")
    def _consistent_release(self) -> SceneDatasetManifest:
        scene_ids = [scene_id for values in self.splits.values() for scene_id in values]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene IDs must not leak across splits")
        if set(scene_ids) != set(self.scene_spec_hashes):
            raise ValueError("split scene IDs must exactly match scene_spec_hashes")
        if self.split_hashes and set(self.split_hashes) != set(self.splits):
            raise ValueError("split_hashes keys must exactly match manifest splits")
        if self.invalidated and not self.invalidation_reason:
            raise ValueError("invalidated scene release requires invalidation_reason")
        filenames = [shard.filename for shard in self.shards]
        if len(filenames) != len(set(filenames)):
            raise ValueError("scene shard filenames must be unique")
        return self


def save_scene_manifest(manifest: SceneDatasetManifest, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.model_dump(by_alias=True, mode="json"), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def load_scene_manifest(manifest_path: str | Path) -> SceneDatasetManifest:
    path = Path(manifest_path)
    if not path.is_file():
        raise ConfigError(f"scene dataset manifest not found: {path}")
    try:
        return SceneDatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"invalid scene dataset manifest at {path}: {exc}") from exc
