"""Pydantic schema and utilities for dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..config.schema import ConfigError
from .artifact_io import atomic_write_text, validate_relative_artifact_path

DATASET_MANIFEST_SCHEMA_V1 = "qdgrasp/dataset-manifest/v1"
DATASET_MANIFEST_SCHEMA_V2 = "qdgrasp/dataset-manifest/v2"

RelativeArtifactPath = Annotated[str, AfterValidator(validate_relative_artifact_path)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ShardMetadata(BaseModel):
    """Metadata and cryptographic checksum of an immutable dataset shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: RelativeArtifactPath
    sha256: Sha256Hex
    num_samples: int = Field(ge=0)
    positive_samples: int = Field(ge=0)
    robot_name: str
    split: str
    recipe_id: str = "legacy"

    @model_validator(mode="after")
    def _positive_count_fits_shard(self) -> "ShardMetadata":
        if self.positive_samples > self.num_samples:
            raise ValueError("positive_samples may not exceed num_samples")
        return self


class DatasetManifestSpec(BaseModel):
    """Complete top-level manifest defining a reproducible dataset release."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["qdgrasp/dataset-manifest/v2"] = Field(
        default=DATASET_MANIFEST_SCHEMA_V2,
        alias="schema",
    )
    dataset_id: str
    generator_version: str
    generator_commit: str = "legacy"
    generator_worktree_dirty: bool = True
    seed: int
    environment_fingerprint: dict[str, Any]
    robot_profile_hashes: dict[str, Sha256Hex]
    object_manifest_hashes: dict[str, Sha256Hex] = Field(default_factory=dict)
    generator_source_hashes: dict[RelativeArtifactPath, Sha256Hex] = Field(default_factory=dict)
    recipe_id: str = "legacy"
    proposal_module: str = "legacy"
    solver_module: str = "legacy"
    certifier_version: str = "legacy"
    dynamic_protocol_version: str = "legacy"
    splits: dict[str, list[str]]  # split_name -> list of object_ids
    shards: list[ShardMetadata]
    success_criteria: dict[str, float]
    license: str = "CC0-1.0"
    release_blocked: bool = False
    invalidated: bool = False
    invalidation_reason: str = ""

    @model_validator(mode="after")
    def _manifest_references_are_coherent(self) -> "DatasetManifestSpec":
        split_names = set(self.splits)
        if not split_names:
            raise ValueError("dataset manifest must declare at least one split")
        for split_name, object_ids in self.splits.items():
            if not split_name or split_name in {".", ".."} or "/" in split_name or "\\" in split_name:
                raise ValueError(f"unsafe split name {split_name!r}")
            if len(object_ids) != len(set(object_ids)):
                raise ValueError(f"split {split_name!r} repeats object ids")
            for object_id in object_ids:
                if not object_id or object_id in {".", ".."} or "/" in object_id or "\\" in object_id:
                    raise ValueError(f"unsafe object id {object_id!r}")

        filenames: set[str] = set()
        pairs: set[tuple[str, str]] = set()
        for shard in self.shards:
            if shard.filename in filenames:
                raise ValueError(f"duplicate shard filename {shard.filename!r}")
            filenames.add(shard.filename)
            pair = (shard.split, shard.robot_name)
            if pair in pairs:
                raise ValueError(f"duplicate shard for split/robot pair {pair!r}")
            pairs.add(pair)
            if shard.split not in split_names:
                raise ValueError(f"shard {shard.filename!r} names unknown split {shard.split!r}")
            if shard.robot_name not in self.robot_profile_hashes:
                raise ValueError(f"shard {shard.filename!r} names unknown robot {shard.robot_name!r}")
        return self


def load_dataset_manifest(manifest_path: str | Path) -> DatasetManifestSpec:
    """Load and validate dataset manifest from disk."""
    p = Path(manifest_path)
    if not p.is_file():
        raise ConfigError(f"dataset manifest not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return DatasetManifestSpec.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"invalid dataset manifest at {p}: {exc}") from exc


def save_dataset_manifest(manifest: DatasetManifestSpec, output_path: str | Path) -> None:
    """Save dataset manifest to disk with sorted keys."""
    p = Path(output_path)
    data = manifest.model_dump(by_alias=True)
    atomic_write_text(p, json.dumps(data, indent=2, sort_keys=True) + "\n")
