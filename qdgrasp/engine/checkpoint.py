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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn

from .. import __version__
from ..config.schema import ConfigError, ModelConfig, RobotConfig
from .compatibility import EmbodimentBinding

BUNDLE_SCHEMA_V1 = "qdgrasp/bundle/v1"
BUNDLE_SCHEMA_V2 = "qdgrasp/bundle/v2"
#: The schema this build writes and reads.  ``v1`` recorded one ambiguous
#: ``robot_config`` hash and gated a load on tensor shape; a bundle written
#: under it was produced by different semantics and is not loadable here.
BUNDLE_SCHEMA = BUNDLE_SCHEMA_V2
RESUME_SCHEMA = "qdgrasp/resume/v2"
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


def canonical_hash(payload: Any) -> str:
    """Stable SHA-256 for an artifact identity represented as JSON data."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    """Backward-compatible private spelling used by the bundle v1 code."""

    return canonical_hash(payload)


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
        """Hash of the profile the weights were *trained* on."""

        return str(self.manifest["hashes"]["training_robot_config"])

    @property
    def preprocess_hash(self) -> str:
        return str(self.manifest["hashes"]["preprocess"])


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

    # Code-level meanings a configuration cannot express.  A model that does not
    # declare any says so with an empty mapping rather than being absent, so the
    # gate below compares like with like.
    semantics = dict(model.semantics()) if hasattr(model, "semantics") else {}

    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "qdgrasp_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_config": model_config.to_document(),
        "training_robot_config": robot_config.to_document(),
        "preprocess": preprocess,
        "semantics": semantics,
        "data_manifest": data_manifest or {},
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in tensors.items()},
        "hashes": {
            "model_config": model_config.content_hash(),
            # The hand these weights were produced for.  What they are *run*
            # against is a runtime fact, and lives in an EmbodimentBinding.
            "training_robot_config": robot_config.content_hash(),
            "preprocess": _canonical_hash(preprocess),
            "semantics": _canonical_hash(semantics),
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
    missing = [
        key
        for key in ("preprocess", "model_config", "training_robot_config", "semantics")
        if key not in recorded
    ]
    if missing:
        raise ConfigError(f"{target}: bundle manifest records no {missing} hash")
    if _canonical_hash(manifest.get("preprocess", {})) != recorded["preprocess"]:
        raise ConfigError(f"{target}: the preprocess hash does not describe the preprocess document beside it")
    if _canonical_hash(manifest.get("semantics", {})) != recorded["semantics"]:
        raise ConfigError(f"{target}: the semantics hash does not describe the semantics document beside it")
    return manifest


def load_public_bundle(
    directory: str | Path,
    model: nn.Module,
    *,
    robot_config: RobotConfig | None = None,
    model_config: ModelConfig | None = None,
    preprocess: dict[str, Any] | None = None,
    binding: EmbodimentBinding | None = None,
) -> dict[str, Any]:
    """Verify a bundle semantically, then load the weights.

    Every check happens before ``load_state_dict``: a mismatch discovered after
    the module has been mutated leaves a model that is neither the one on disk
    nor the one that was constructed.

    Shapes are the last thing compared, not the first.  Two configurations can
    produce identically shaped tensors and mean different things -- a different
    number of flow steps, a preprocessing declared in millimetres -- and a
    loader that only fits tensors accepts both.
    """

    target = Path(directory)
    manifest = read_bundle_manifest(target)
    training_hash = manifest["hashes"]["training_robot_config"]

    if model_config is not None and model_config.content_hash() != manifest["hashes"]["model_config"]:
        raise ConfigError(
            f"{target}: model configuration hash mismatch; the bundle holds "
            f"{manifest['model_config'].get('name')!r} and its weights mean what that configuration says "
            "they mean"
        )
    if preprocess is not None and _canonical_hash(preprocess) != _canonical_hash(manifest["preprocess"]):
        raise ConfigError(
            f"{target}: preprocessing contract mismatch; the bundle declares {manifest['preprocess']!r}. "
            "Feeding inputs prepared another way produces confident predictions about a different scene"
        )
    declared = dict(model.semantics()) if hasattr(model, "semantics") else {}
    if declared != manifest.get("semantics", {}):
        raise ConfigError(
            f"{target}: architecture semantics mismatch; the bundle was produced under "
            f"{manifest.get('semantics', {})!r} and this build means {declared!r}. The tensors fit and the "
            "joints they describe do not"
        )
    if binding is not None:
        if binding.training_robot_hash != training_hash:
            raise ConfigError(
                f"{target}: the binding was made for training profile {binding.training_robot!r} and this "
                "bundle was produced for another one"
            )
        if robot_config is not None and binding.runtime_robot_hash != robot_config.content_hash():
            raise ConfigError(f"{target}: the binding does not name the runtime profile being loaded")
    elif robot_config is not None and robot_config.content_hash() != training_hash:
        raise ConfigError(
            f"{target}: robot profile hash mismatch; bundle was produced for "
            f"'{manifest['training_robot_config']['name']}' with joints "
            f"{manifest['training_robot_config']['joints']}. Running it on another hand needs an explicit "
            "EmbodimentBinding, not a relaxed comparison"
        )
    tensors = load_safetensors(str(target / WEIGHTS_FILE))
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if missing or unexpected:
        raise ConfigError(f"{target}: weight/state mismatch (missing={list(missing)}, unexpected={list(unexpected)})")
    return manifest


@dataclass
class ResumeState:
    """Everything needed to continue one *identified* run bit-for-bit.

    Resume is deliberately stricter than loading weights.  A caller must prove
    that the model, exact robot, dataset/protocol view and effective optimiser
    schedule are the same before any mutable training state is restored.
    """

    global_step: int
    qdgrasp_version: str
    model_config: dict[str, Any]
    model_config_hash: str
    robot_config: dict[str, Any]
    robot_config_hash: str
    data_manifest: dict[str, Any]
    data_manifest_hash: str
    protocol_hash: str | None
    dataset_view_hash: str | None
    effective_run_config: dict[str, Any]
    effective_run_config_hash: str
    optimizer_config: dict[str, Any]
    scheduler_config: dict[str, Any]
    weights_source: dict[str, str]
    model: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    stream: dict[str, Any]
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
                "qdgrasp_version": self.qdgrasp_version,
                "model_config": self.model_config,
                "model_config_hash": self.model_config_hash,
                "robot_config": self.robot_config,
                "robot_config_hash": self.robot_config_hash,
                "data_manifest": self.data_manifest,
                "data_manifest_hash": self.data_manifest_hash,
                "protocol_hash": self.protocol_hash,
                "dataset_view_hash": self.dataset_view_hash,
                "effective_run_config": self.effective_run_config,
                "effective_run_config_hash": self.effective_run_config_hash,
                "optimizer_config": self.optimizer_config,
                "scheduler_config": self.scheduler_config,
                "weights_source": self.weights_source,
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "stream": self.stream,
                "scaler": self.scaler,
                "ema": self.ema,
                "rng": self.rng,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> ResumeState:
        """Read a resume artifact with ``weights_only=True``."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ConfigError(f"{path}: resume artifact must contain a mapping")
        if payload.get("schema") != RESUME_SCHEMA:
            raise ConfigError(
                f"{path}: unsupported resume schema {payload.get('schema')!r}; this build reads "
                f"{RESUME_SCHEMA!r}. Resume is an exact continuation, so a state written under another "
                "schema is a different run rather than an older one"
            )
        required = {
            "global_step",
            "qdgrasp_version",
            "model_config",
            "model_config_hash",
            "robot_config",
            "robot_config_hash",
            "data_manifest",
            "data_manifest_hash",
            "protocol_hash",
            "dataset_view_hash",
            "effective_run_config",
            "effective_run_config_hash",
            "optimizer_config",
            "scheduler_config",
            "weights_source",
            "model",
            "optimizer",
            "scheduler",
            "stream",
            "scaler",
            "ema",
            "rng",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ConfigError(f"{path}: resume/v2 is missing required fields {missing}")

        hashed_documents = (
            ("model_config", "model_config_hash"),
            ("robot_config", "robot_config_hash"),
            ("data_manifest", "data_manifest_hash"),
            ("effective_run_config", "effective_run_config_hash"),
        )
        for document_key, hash_key in hashed_documents:
            document = payload[document_key]
            if not isinstance(document, dict):
                raise ConfigError(f"{path}: {document_key} must be a mapping")
            actual = canonical_hash(document)
            if actual != payload[hash_key]:
                raise ConfigError(
                    f"{path}: {document_key} hash mismatch (declared={payload[hash_key]!r}, actual={actual!r})"
                )
        return cls(
            global_step=int(payload["global_step"]),
            qdgrasp_version=str(payload["qdgrasp_version"]),
            model_config=payload["model_config"],
            model_config_hash=str(payload["model_config_hash"]),
            robot_config=payload["robot_config"],
            robot_config_hash=str(payload["robot_config_hash"]),
            data_manifest=payload["data_manifest"],
            data_manifest_hash=str(payload["data_manifest_hash"]),
            protocol_hash=None if payload["protocol_hash"] is None else str(payload["protocol_hash"]),
            dataset_view_hash=None if payload["dataset_view_hash"] is None else str(payload["dataset_view_hash"]),
            effective_run_config=payload["effective_run_config"],
            effective_run_config_hash=str(payload["effective_run_config_hash"]),
            optimizer_config=payload["optimizer_config"],
            scheduler_config=payload["scheduler_config"],
            weights_source=payload["weights_source"],
            model=payload["model"],
            optimizer=payload["optimizer"],
            scheduler=payload["scheduler"],
            stream=payload["stream"],
            scaler=payload["scaler"],
            ema=payload["ema"],
            rng=payload["rng"],
        )
