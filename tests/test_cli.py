from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from qdgrasp.cli import app

RUNNER = CliRunner()


def test_env_reports_the_runtime() -> None:
    result = RUNNER.invoke(app, ["env"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["torch"]


def test_train_subcommand_writes_a_run_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = RUNNER.invoke(
        app,
        [
            "--quiet", "train",
            "--model", "qdgrasp-dummy-n.yaml",
            "--data", "dummy-tiny.yaml",
            "--robot", "dummy-hand.yaml",
            "--device", "cpu",
            "--max-steps", "3",
            "--run-name", "cli",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["global_step"] == 3
    assert (tmp_path / "runs/cli/bundle/weights.safetensors").is_file()


def test_unknown_flag_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = RUNNER.invoke(app, ["train", "--model", "qdgrasp-dummy-n.yaml", "--epochs", "3"])
    assert result.exit_code != 0


def test_key_value_grammar_is_not_accepted(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = RUNNER.invoke(app, ["train", "model=qdgrasp-dummy-n.yaml", "data=dummy-tiny.yaml"])
    assert result.exit_code != 0


def test_predict_and_export_subcommands(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    np.savez(tmp_path / "cloud.npz", points=np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32))
    predict = RUNNER.invoke(
        app,
        ["--quiet", "predict", "--model", "qdgrasp-dummy-n.yaml", "--robot", "dummy-hand.yaml",
         "--points", "cloud.npz", "--out", "runs/predict/grasps.npz"],
    )
    assert predict.exit_code == 0, predict.output
    assert (tmp_path / "runs/predict/grasps.npz").is_file()

    export = RUNNER.invoke(
        app,
        ["--quiet", "export", "--model", "qdgrasp-dummy-n.yaml", "--robot", "dummy-hand.yaml",
         "--format", "torchscript", "--out-dir", "runs/export"],
    )
    assert export.exit_code == 0, export.output
    assert json.loads(export.stdout)["format"] == "torchscript"
