"""Immutable shard container reading and writing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from ..config.schema import ConfigError


def write_shard_file(
    samples: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """Serialize a list of grasp sample dictionaries to a PyTorch file and return its SHA-256."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Use deterministic torch save format
    torch.save(samples, p, _use_new_zipfile_serialization=True)

    data = p.read_bytes()
    return hashlib.sha256(data).hexdigest()


def read_shard_file(
    shard_path: str | Path,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Load and verify an immutable shard from disk."""
    p = Path(shard_path)
    if not p.is_file():
        raise ConfigError(f"shard file not found: {p}")

    if expected_sha256 is not None:
        data = p.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_sha256:
            raise ConfigError(
                f"integrity mismatch for shard {p}: expected {expected_sha256}, got {actual_hash}"
            )

    loaded = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(loaded, list):
        raise ConfigError(f"invalid shard format in {p}: expected list of samples")
    return loaded
