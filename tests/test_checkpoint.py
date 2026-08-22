from __future__ import annotations

import json

import pytest
import torch

from qdgrasp.api import QDGrasp
from qdgrasp.config import ConfigError, RobotConfig, load_robot_config
from qdgrasp.engine.checkpoint import MANIFEST_FILE, WEIGHTS_FILE, load_public_bundle, read_bundle_manifest


def test_public_bundle_is_safetensors_and_json_only(tmp_path) -> None:
    grasper = QDGrasp()
    info = grasper.save_bundle(tmp_path / "bundle")
    files = sorted(item.name for item in info.directory.iterdir())
    assert files == [MANIFEST_FILE, WEIGHTS_FILE]
    header = (info.directory / WEIGHTS_FILE).read_bytes()
    assert b"__pickle__" not in header and b"cnumpy.core" not in header
    manifest = read_bundle_manifest(info.directory)
    assert manifest["hashes"]["model_config"] == grasper.model_hash
    assert manifest["preprocess"] == grasper.module.preprocess_schema()


def test_tampered_weights_are_rejected(tmp_path) -> None:
    info = QDGrasp().save_bundle(tmp_path / "bundle")
    weights = info.directory / WEIGHTS_FILE
    weights.write_bytes(weights.read_bytes() + b"\x00")
    with pytest.raises(ConfigError, match="weights hash mismatch"):
        read_bundle_manifest(info.directory)


def test_tampered_manifest_is_rejected(tmp_path) -> None:
    info = QDGrasp().save_bundle(tmp_path / "bundle")
    manifest_path = info.directory / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["preprocess"]["units"] = "centimeters"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigError, match="manifest hash mismatch"):
        read_bundle_manifest(info.directory)


def test_profile_mismatch_fails_before_inference(tmp_path) -> None:
    info = QDGrasp().save_bundle(tmp_path / "bundle")
    other = load_robot_config("dummy-hand.yaml")
    renamed = RobotConfig.model_validate(other.to_document() | {"name": "other-hand"})
    with pytest.raises(ConfigError, match="robot profile hash mismatch"):
        load_public_bundle(info.directory, QDGrasp().module, robot_config=renamed)


def test_bundle_round_trips_into_a_new_facade(tmp_path) -> None:
    source = QDGrasp()
    with torch.no_grad():
        source.module.score_head.bias.add_(0.25)
    info = source.save_bundle(tmp_path / "bundle")
    restored = QDGrasp.from_bundle(info.directory)
    assert torch.equal(restored.module.score_head.bias, source.module.score_head.bias)
    assert restored.robot_hash == source.robot_hash
