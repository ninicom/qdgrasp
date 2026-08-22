from __future__ import annotations

import json

import pytest
import torch

from qdgrasp.api import QDGrasp
from qdgrasp.engine.callbacks import BaseCallback, LossHistory
from qdgrasp.engine.sampling import DeterministicBatchStream


def train(tmp_path, monkeypatch, **overrides):
    monkeypatch.chdir(tmp_path)
    history = LossHistory()
    grasper = QDGrasp()
    result = grasper.train("dummy-tiny.yaml", callbacks=[history], **overrides)
    return grasper, result, history


def test_cpu_train_smoke_writes_every_artifact(tmp_path, monkeypatch) -> None:
    _grasper, result, history = train(tmp_path, monkeypatch, max_steps=6, batch_size=4, run_name="smoke")
    assert result.global_step == 6
    assert len(history.history) == 6
    assert (tmp_path / "runs/smoke/results.json").is_file()
    assert (tmp_path / "runs/smoke/resume.pt").is_file()
    assert (tmp_path / "runs/smoke/bundle/bundle.json").is_file()
    payload = json.loads((tmp_path / "runs/smoke/results.json").read_text())
    assert payload["schema"] == "qdgrasp/results/v1"
    assert payload["runtime"]["effective"]["device"] == "cpu"
    assert set(payload["metrics"]) == {"loss", "translation_error", "joint_error"}


def test_same_seed_reproduces_the_loss_curve(tmp_path, monkeypatch) -> None:
    _a, _ra, first = train(tmp_path, monkeypatch, max_steps=5, run_name="a", seed=3)
    _b, _rb, second = train(tmp_path, monkeypatch, max_steps=5, run_name="b", seed=3)
    assert first.history == second.history


def test_different_seed_changes_the_loss_curve(tmp_path, monkeypatch) -> None:
    _a, _ra, first = train(tmp_path, monkeypatch, max_steps=5, run_name="a", seed=1)
    _b, _rb, second = train(tmp_path, monkeypatch, max_steps=5, run_name="b", seed=2)
    assert first.history != second.history


def test_resume_is_bit_exact_against_an_uninterrupted_run(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    continuous = LossHistory()
    QDGrasp().train("dummy-tiny.yaml", max_steps=8, run_name="full", callbacks=[continuous])

    first_half = LossHistory()
    QDGrasp().train("dummy-tiny.yaml", max_steps=8, stop_after_steps=4, run_name="part", callbacks=[first_half])
    assert len(first_half.history) == 4

    second_half = LossHistory()
    resumed = QDGrasp().train(
        "dummy-tiny.yaml", max_steps=8, resume="runs/part/resume.pt", run_name="part-2", callbacks=[second_half]
    )
    assert resumed.global_step == 8
    assert continuous.history == first_half.history + second_half.history


def test_missing_resume_artifact_is_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match="resume artifact not found"):
        QDGrasp().train("dummy-tiny.yaml", max_steps=1, resume="runs/nope/resume.pt")


def test_callbacks_observe_the_whole_lifecycle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    class Recorder(BaseCallback):
        def __init__(self) -> None:
            self.events: list[str] = []

        def on_train_start(self, state) -> None:
            self.events.append("start")

        def on_step_end(self, state) -> None:
            self.events.append("step")

        def on_validation_end(self, state) -> None:
            self.events.append("val")

        def on_train_end(self, state) -> None:
            self.events.append("end")

    recorder = Recorder()
    QDGrasp().train("dummy-tiny.yaml", max_steps=4, val_interval=2, run_name="cb", callbacks=[recorder])
    assert recorder.events[0] == "start"
    assert recorder.events[-1] == "end"
    assert recorder.events.count("step") == 4
    assert recorder.events.count("val") == 2


def test_validation_is_deterministic_and_finite(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    grasper = QDGrasp()
    first = grasper.val("dummy-tiny.yaml", batch_size=4)
    second = grasper.val("dummy-tiny.yaml", batch_size=4)
    assert first == second
    assert all(torch.isfinite(torch.tensor(value)) for value in first.values())


def test_ema_tracks_a_shadow_copy(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    grasper = QDGrasp()
    before = grasper.module.score_head.bias.detach().clone()
    grasper.train("dummy-tiny.yaml", max_steps=3, ema_decay=0.9, run_name="ema")
    assert not torch.equal(before, grasper.module.score_head.bias)


def test_batch_stream_position_survives_a_round_trip() -> None:
    stream = DeterministicBatchStream(10, 3, seed=5)
    [stream.next_indices() for _ in range(4)]
    state = stream.state_dict()
    expected = [stream.next_indices() for _ in range(3)]
    replay = DeterministicBatchStream(10, 3, seed=5)
    replay.load_state_dict(state)
    assert [replay.next_indices() for _ in range(3)] == expected


def test_predicted_joints_stay_inside_the_declared_limits(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    grasper = QDGrasp()
    grasper.train("dummy-tiny.yaml", max_steps=3, run_name="limits")
    results = grasper.predict(torch.randn(64, 3))
    lower = torch.tensor(grasper.robot_config.lower_limits)
    upper = torch.tensor(grasper.robot_config.upper_limits)
    assert torch.all(results.joint_values >= lower)
    assert torch.all(results.joint_values <= upper)
    assert torch.isfinite(results.joint_values).all()
