from __future__ import annotations

import json

import pytest
import torch

from qdgrasp.api import QDGrasp
from qdgrasp.config import ConfigError


def test_torchscript_round_trip_preserves_every_output(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    grasper = QDGrasp()
    result = grasper.export(fmt="torchscript", out_dir="runs/export")
    assert result.path.is_file()
    deviations = result.metadata["round_trip_max_abs_deviation"]
    assert set(deviations) == {"translation", "rotation", "joints", "score"}
    assert max(deviations.values()) <= 1e-5

    loaded = torch.jit.load(str(result.path))
    points = torch.randn(1, 96, 3)
    with torch.no_grad():
        expected = grasper.module(points)
        actual = loaded(points)
    for left, right in zip(expected, actual):
        assert torch.allclose(left, right, atol=1e-5)


def test_export_sidecar_carries_schema_and_robot_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    grasper = QDGrasp()
    result = grasper.export(fmt="torchscript", out_dir="runs/export")
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["schema"] == "qdgrasp/export/v1"
    assert metadata["joint_names"] == list(grasper.robot_config.joints)
    assert metadata["preprocess"]["layout"] == "[B, N, 3]"
    assert metadata["hashes"]["robot_config"] == grasper.robot_hash


def test_unsupported_format_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="unsupported export format"):
        QDGrasp().export(fmt="tensorrt", out_dir="runs/export")


def test_exported_graph_accepts_a_different_point_count(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    grasper = QDGrasp()
    result = grasper.export(fmt="torchscript", out_dir="runs/export")
    loaded = torch.jit.load(str(result.path))
    with torch.no_grad():
        translation, rotation, joints, score = loaded(torch.randn(1, 200, 3))
    assert translation.shape[0] == 1 and rotation.shape[-2:] == (3, 3)
    assert joints.shape[-1] == len(grasper.robot_config.joints)
    assert torch.isfinite(score).all()


def test_onnx_round_trip_when_the_export_extra_is_installed(tmp_path, monkeypatch) -> None:
    pytest.importorskip("onnxruntime", reason="ONNX verification needs the 'export' extra")
    monkeypatch.chdir(tmp_path)
    result = QDGrasp().export(fmt="onnx", out_dir="runs/export")
    assert result.path.is_file()
    deviations = result.metadata["round_trip_max_abs_deviation"]
    assert set(deviations) == {"translation", "rotation", "joints", "score"}
    assert max(deviations.values()) <= 1e-4
