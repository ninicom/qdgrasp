"""Pydantic schema and utilities for dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from pydantic import BaseModel, ConfigDict, Field

from ..config.schema import ConfigError

DATASET_MANIFEST_SCHEMA_V1 = "qdgrasp/dataset-manifest/v1"


class ShardMetadata(BaseModel):
    """Metadata and cryptographic checksum of an immutable dataset shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    sha256: str
    num_samples: int
    positive_samples: int
    robot_name: str
    split: str


class DatasetManifestSpec(BaseModel):
    """Complete top-level manifest defining a reproducible dataset release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=DATASET_MANIFEST_SCHEMA_V1, alias="schema")
    dataset_id: str
    generator_version: str
    seed: int
    environment_fingerprint: dict[str, Any]
    robot_profile_hashes: dict[str, str]
    splits: dict[str, list[str]]  # split_name -> list of object_ids
    shards: list[ShardMetadata]
    success_criteria: dict[str, float]
    license: str = "CC0-1.0"
    release_blocked: bool = False


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
    p.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.model_dump(by_alias=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
