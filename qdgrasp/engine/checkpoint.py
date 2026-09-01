"""Checkpoint contract: a public weight bundle and a separate resume state.

The public bundle never pickles a module.  It holds safetensors weights plus a
JSON manifest with the configuration snapshot, the preprocessing schema, the
robot profile and content hashes.  Optimiser/scheduler/scaler/RNG/global-step
state lives in a distinct resume artifact that is not part of a public release.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn

from .. import __version__
from ..config.schema import ConfigError, ModelConfig, RobotConfig


BUNDLE_SCHEMA = "qdgrasp/bundle/v1"
RESUME_SCHEMA = "qdgrasp/resume/v1"
WEIGHTS_FILE = "weights.safetensors"
MANIFEST_FILE = "bundle.json"
RESUME_FILE = "resume.pt"


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BundleInfo:
    """Where a public bundle was written and what it hashes to."""

    directory: Path
    manifest: dict[str, Any]

    @property
    def bundle_hash(self) -> str:
        return str(self.manifest["hashes"]["bundle"])

    @property
    def model_hash(self) -> str:
        return str(self.manifest["hashes"]["model_config"])

    @property
    def robot_hash(self) -> str:
        return str(self.manifest["hashes"]["robot_config"])


def save_public_bundle(
    directory: str | Path,
    *,
    model: nn.Module,
    model_config: ModelConfig,
    robot_config: RobotConfig,
    preprocess: dict[str, Any],
    data_manifest: dict[str, Any] | None = None,
) -> BundleInfo:
    """Write ``weights.safetensors`` plus ``bundle.json`` into ``directory``."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    weights_path = target / WEIGHTS_FILE
    save_safetensors(tensors, str(weights_path))

    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "qdgrasp_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_config": model_config.to_document(),
        "robot_config": robot_config.to_document(),
        "preprocess": preprocess,
        "data_manifest": data_manifest or {},
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in tensors.items()},
        "hashes": {
            "model_config": model_config.content_hash(),
            "robot_config": robot_config.content_hash(),
            "weights": sha256_file(weights_path),
        },
    }
    manifest["hashes"]["bundle"] = _canonical_hash(manifest)
    (target / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BundleInfo(directory=target, manifest=manifest)


def read_bundle_manifest(directory: str | Path) -> dict[str, Any]:
    """Read and integrity-check ``bundle.json`` in ``directory``."""

    target = Path(directory)
    manifest_path = target / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ConfigError(f"{target}: missing {MANIFEST_FILE}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ConfigError(
            f"{target}: unsupported bundle schema {manifest.get('schema')!r}; this build reads "
            f"{BUNDLE_SCHEMA!r}. A bundle written under another schema was produced by different "
            "semantics and is not loadable as if it were this one"
        )
    recorded = dict(manifest["hashes"])
    declared_bundle = recorded.pop("bundle")
    probe = dict(manifest)
    probe["hashes"] = recorded
    if _canonical_hash(probe) != declared_bundle:
        raise ConfigError(f"{target}: bundle manifest hash mismatch")
    weights_hash = sha256_file(target / WEIGHTS_FILE)
    if weights_hash != recorded["weights"]:
        raise ConfigError(f"{target}: weights hash mismatch")
    return manifest


def load_public_bundle(directory: str | Path, model: nn.Module, *, robot_config: RobotConfig | None = None) -> dict[str, Any]:
    """Verify a bundle, optionally gate it on a robot profile, and load the weights."""

    target = Path(directory)
    manifest = read_bundle_manifest(target)
    if robot_config is not None and robot_config.content_hash() != manifest["hashes"]["robot_config"]:
        raise ConfigError(
            f"{target}: robot profile hash mismatch; bundle was produced for "
            f"'{manifest['robot_config']['name']}' with joints {manifest['robot_config']['joints']}"
        )
    tensors = load_safetensors(str(target / WEIGHTS_FILE))
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if missing or unexpected:
        raise ConfigError(f"{target}: weight/state mismatch (missing={list(missing)}, unexpected={list(unexpected)})")
    return manifest


@dataclass
class ResumeState:
    """Everything needed to continue a run bit-for-bit."""

    global_step: int
    model: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    ema: dict[str, Any]
    rng: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        """Write the resume artifact (tensor/primitive payload only)."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": RESUME_SCHEMA,
                "global_step": self.global_step,
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "scaler": self.scaler,
                "ema": self.ema,
                "rng": self.rng,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ResumeState":
        """Read a resume artifact with ``weights_only=True``."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("schema") != RESUME_SCHEMA:
            raise ConfigError(
                f"{path}: unsupported resume schema {payload.get('schema')!r}; this build reads "
                f"{RESUME_SCHEMA!r}. Resume is an exact continuation, so a state written under another "
                "schema is a different run rather than an older one"
            )
        return cls(
            global_step=int(payload["global_step"]),
            model=payload["model"],
            optimizer=payload["optimizer"],
            scheduler=payload["scheduler"],
            scaler=payload["scaler"],
            ema=payload["ema"],
            rng=payload["rng"],
        )
