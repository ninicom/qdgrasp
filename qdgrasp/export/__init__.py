"""Export bundles for the formats QDGrasp v1 supports: PyTorch, TorchScript, ONNX."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from ..config.schema import ConfigError, ModelConfig, RobotConfig
from ..engine.checkpoint import sha256_file
from .torchscript import export_torchscript, verify_torchscript

SUPPORTED_FORMATS = ("torchscript", "onnx")
EXPORT_SCHEMA = "qdgrasp/export/v1"


@dataclass(frozen=True)
class ExportResult:
    """Artifact path plus the metadata sidecar written next to it."""

    path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def export_bundle(
    model: nn.Module,
    *,
    fmt: str,
    out_dir: str | Path,
    model_config: ModelConfig,
    robot_config: RobotConfig,
    preprocess: dict[str, Any],
    verify: bool = True,
) -> ExportResult:
    """Export ``model`` and write a sidecar carrying schema/robot/preprocess metadata."""

    if fmt not in SUPPORTED_FORMATS:
        raise ConfigError(f"unsupported export format '{fmt}'; QDGrasp v1 supports {SUPPORTED_FORMATS}")
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    example = model.example_inputs()

    if fmt == "torchscript":
        artifact = export_torchscript(model, target_dir / f"{model_config.name}.torchscript.pt", example)
        deviations = verify_torchscript(model, artifact, example) if verify else {}
    else:
        from .onnx import export_onnx, verify_onnx

        artifact = export_onnx(model, target_dir / f"{model_config.name}.onnx", example)
        deviations = verify_onnx(model, artifact, example) if verify else {}

    metadata = {
        "schema": EXPORT_SCHEMA,
        "format": fmt,
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "model_config": model_config.to_document(),
        "robot_config": robot_config.to_document(),
        "preprocess": preprocess,
        "joint_names": list(robot_config.joints),
        "outputs": ["translation", "rotation", "joints", "score"],
        "round_trip_max_abs_deviation": deviations,
        "hashes": {"model_config": model_config.content_hash(), "robot_config": robot_config.content_hash()},
    }
    metadata_path = artifact.with_suffix(artifact.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExportResult(path=artifact, metadata_path=metadata_path, metadata=metadata)


__all__ = ("EXPORT_SCHEMA", "ExportResult", "SUPPORTED_FORMATS", "export_bundle", "export_torchscript", "verify_torchscript")
