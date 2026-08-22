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
    assert set(payload["metrics"]) == {"loss", "translation_error", "rotation_error", "joint_error"}


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


def test_every_trainable_parameter_receives_a_gradient(tmp_path, monkeypatch) -> None:
    from qdgrasp.config import load_data_config
    from qdgrasp.dummy.data import build_dummy_points
    from qdgrasp.engine.sampling import collate_indices

    monkeypatch.chdir(tmp_path)
    grasper = QDGrasp()
    dataset = build_dummy_points(load_data_config("dummy-tiny.yaml"), grasper.robot_config, split="train")
    grasper.module.training_step(collate_indices(dataset, [0, 1])).backward()
    dead = [
        name
        for name, parameter in grasper.module.named_parameters()
        if parameter.requires_grad and (parameter.grad is None or not parameter.grad.any())
    ]
    assert dead == []


def test_two_stage_resume_advances_to_the_end_of_the_schedule(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    first = QDGrasp().train("dummy-tiny.yaml", max_steps=8, stop_after_steps=4, run_name="s1")
    assert first.global_step == 4
    second = QDGrasp().train(
        "dummy-tiny.yaml", max_steps=8, stop_after_steps=4, resume="runs/s1/resume.pt", run_name="s2"
    )
    assert second.global_step == 8


def test_val_reflects_the_trained_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    untrained = QDGrasp().val("dummy-tiny.yaml", batch_size=4)
    trained_model = QDGrasp()
    trained_model.train("dummy-tiny.yaml", max_steps=20, learning_rate=1e-2, run_name="trained")
    trained = trained_model.val("dummy-tiny.yaml", batch_size=4)
    assert trained["loss"] != untrained["loss"]
    assert trained["loss"] < untrained["loss"]


def test_align_optimizer_state_matches_each_parameter_device() -> None:
    from qdgrasp.engine.runner import STEP_STATE_KEY, align_optimizer_state

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    before = {
        id(parameter): {key: value.clone() for key, value in state.items() if isinstance(value, torch.Tensor)}
        for parameter, state in optimizer.state.items()
    }

    align_optimizer_state(optimizer)

    assert optimizer.state, "optimizer built no state to align"
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key != STEP_STATE_KEY:
                assert value.device == parameter.device
            assert torch.equal(value, before[id(parameter)][key])


def test_resumed_run_leaves_optimizer_state_on_the_runtime_device(tmp_path, monkeypatch) -> None:
    from qdgrasp.engine.runner import STEP_STATE_KEY, align_optimizer_state

    monkeypatch.chdir(tmp_path)
    QDGrasp().train("dummy-tiny.yaml", max_steps=8, stop_after_steps=4, run_name="dev1")

    seen: list[torch.optim.Optimizer] = []
    original = align_optimizer_state

    def spy(optimizer: torch.optim.Optimizer) -> None:
        original(optimizer)
        seen.append(optimizer)

    monkeypatch.setattr("qdgrasp.engine.runner.align_optimizer_state", spy)
    QDGrasp().train("dummy-tiny.yaml", max_steps=8, resume="runs/dev1/resume.pt", run_name="dev2")

    assert len(seen) == 1
    optimizer = seen[0]
    assert optimizer.state, "resume restored no optimizer state"
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != STEP_STATE_KEY:
                assert value.device == parameter.device
