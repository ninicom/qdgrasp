"""Deterministic scene-record shards with positive-label admission."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from qdgrasp.config.schema import ConfigError

SceneRecordType = Literal["scene_state", "observation", "grasp"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DYNAMIC_STAGES = {"squeeze", "lift", "perturbation"}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _admit_positive_grasp(record: dict[str, Any]) -> None:
    if record.get("source_class") == "test_fixture_only":
        raise ConfigError("scene positive cannot use source_class=test_fixture_only")
    required_identity = ("scene_id", "target_object_id", "robot_profile", "candidate_id")
    missing_identity = [name for name in required_identity if not record.get(name)]
    if missing_identity:
        raise ConfigError(f"scene positive missing identity fields: {missing_identity}")
    if record.get("failure_reason") != "none" or record.get("label_stage") != "dynamic_valid":
        raise ConfigError("scene positive must have dynamic_valid label and failure_reason=none")
    if not bool(record.get("static_certificate", {}).get("passed")):
        raise ConfigError("scene positive missing passing static certificate")
    if not bool(record.get("swept_clearance_metrics", {}).get("passed")):
        raise ConfigError("scene positive missing passing swept-clearance evidence")

    evidence = record.get("dynamic_trajectory_evidence")
    if not isinstance(evidence, dict):
        raise ConfigError("scene positive missing dynamic trajectory evidence")
    stages = evidence.get("validated_stages")
    if not isinstance(stages, list) or not _DYNAMIC_STAGES.issubset(stages):
        raise ConfigError("scene positive missing squeeze/lift/perturbation evidence")
    loads = np.asarray(evidence.get("per_finger_loads", []), dtype=np.float64)
    if (
        loads.ndim != 2
        or loads.shape[0] == 0
        or loads.shape[1:] != (6,)
        or not np.all(np.isfinite(loads))
        or not np.any(np.abs(loads) > 0.0)
    ):
        raise ConfigError("scene positive missing finite measured per-finger loads")
    measured_lift = evidence.get("measured_target_lift")
    if not isinstance(measured_lift, (int, float)) or not math.isfinite(measured_lift):
        raise ConfigError("scene positive missing finite measured target lift")

    state_hashes = record.get("scene_state_hashes")
    required_hash_stages = ("initial", "squeeze", "lift", "perturbation")
    if not isinstance(state_hashes, dict) or any(
        not _is_sha256(state_hashes.get(stage)) for stage in required_hash_stages
    ):
        raise ConfigError("scene positive missing valid stage state hashes")
    identity_hashes = ("protocol_hash", "recipe_hash", "source_hash")
    if any(not _is_sha256(record.get(name)) for name in identity_hashes):
        raise ConfigError("scene positive missing valid protocol/recipe/source hashes")


def validate_scene_record(record: dict[str, Any], record_type: SceneRecordType) -> None:
    if not isinstance(record, dict) or record.get("record_type") != record_type:
        raise ConfigError(f"scene shard requires record_type={record_type}")
    if not record.get("scene_id"):
        raise ConfigError("scene record requires scene_id")
    if record_type == "scene_state":
        if not record.get("stage") or not _is_sha256(record.get("state_hash")):
            raise ConfigError("scene-state record requires stage and valid state_hash")
        if not _is_sha256(record.get("lineage_hash")):
            raise ConfigError("scene-state record requires valid lineage_hash")
    if record_type == "observation" and (
        not record.get("camera_id") or not _is_sha256(record.get("calibration_hash"))
    ):
        raise ConfigError("observation record requires camera_id and calibration_hash")
    if record_type == "grasp" and bool(record.get("dynamic_valid")):
        _admit_positive_grasp(record)


def write_scene_shard(
    records: list[dict[str, Any]], output_path: str | Path, *, record_type: SceneRecordType
) -> str:
    """Write canonical JSONL atomically and return its SHA-256."""
    canonical_lines: list[bytes] = []
    for record in records:
        normalized = _json_value(record)
        validate_scene_record(normalized, record_type)
        try:
            line = json.dumps(
                normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"scene record is not finite/canonical JSON: {exc}") from exc
        canonical_lines.append(line + b"\n")
    payload = b"".join(canonical_lines)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def read_scene_shard(
    shard_path: str | Path,
    *,
    record_type: SceneRecordType,
    expected_sha256: str,
    expected_records: int | None = None,
) -> list[dict[str, Any]]:
    """Verify and load one immutable scene JSONL shard."""
    path = Path(shard_path)
    if not path.is_file():
        raise ConfigError(f"scene shard not found: {path}")
    payload = path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_sha256:
        raise ConfigError(
            f"scene shard integrity mismatch for {path}: expected {expected_sha256}, got {actual_hash}"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            record = json.loads(line)
            validate_scene_record(record, record_type)
        except Exception as exc:
            raise ConfigError(f"invalid scene shard record {path}:{line_number}: {exc}") from exc
        records.append(record)
    if expected_records is not None and len(records) != expected_records:
        raise ConfigError(
            f"scene shard record count mismatch for {path}: expected {expected_records}, got {len(records)}"
        )
    return records
