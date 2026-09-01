from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file as load_safetensors
from torch import nn

from qdgrasp import QDGrasp
from qdgrasp.config import ConfigError, RunConfig, resolve_runtime
from qdgrasp.engine.callbacks import CallbackList, LossHistory
from qdgrasp.engine.checkpoint import ResumeState
from qdgrasp.engine.runner import Runner


class _StochasticTrainingModel(nn.Module):
    """Both train and validation draw from the global torch stream."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = torch.randn((), device=self.weight.device)
        return (self.weight - target).square()

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        draw = torch.randn(batch["points"].shape[0], device=self.weight.device)
        return {"loss": (self.weight * draw).mean()}


def _direct_runner(tmp_path: Path, name: str, *, val_interval: int) -> tuple[Runner, QDGrasp]:
    grasper = QDGrasp()
    config = RunConfig(
        max_steps=5,
        batch_size=2,
        val_interval=val_interval,
        project_dir="runs",
        run_name=name,
        seed=17,
    )
    return (
        Runner(
            run_config=config,
            runtime=resolve_runtime(config),
            model_config=grasper.model_config,
            robot_config=grasper.robot_config,
            callbacks=CallbackList([]),
        ),
        grasper,
    )


def test_validation_cadence_cannot_change_a_stochastic_training_trajectory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    train = [{"points": torch.zeros(3, 3)} for _ in range(6)]
    validation = [{"points": torch.zeros(3, 3)} for _ in range(5)]

    sparse, sparse_grasper = _direct_runner(tmp_path, "sparse", val_interval=0)
    frequent, frequent_grasper = _direct_runner(tmp_path, "frequent", val_interval=1)
    sparse_model = _StochasticTrainingModel()
    frequent_model = _StochasticTrainingModel()

    sparse_result = sparse.fit(sparse_model, train, validation, data_manifest={"dataset": "rng-test"})
    frequent_result = frequent.fit(frequent_model, train, validation, data_manifest={"dataset": "rng-test"})

    assert sparse_result.losses == frequent_result.losses
    assert torch.equal(sparse_model.weight, frequent_model.weight)
    assert sparse_grasper.model_config == frequent_grasper.model_config


def test_ema_is_used_for_final_metrics_and_the_public_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    grasper = QDGrasp()
    result = grasper.train(
        "dummy-tiny.yaml", max_steps=5, val_interval=2, ema_decay=0.9, run_name="ema-source"
    )
    payload = torch.load(result.artifacts["resume"], map_location="cpu", weights_only=True)
    published = load_safetensors(Path(result.artifacts["bundle"]) / "weights.safetensors")

    assert payload["ema"]["updates"] == 5
    assert payload["weights_source"] == {
        "resume_model": "live",
        "validation": "ema",
        "public_bundle": "ema",
    }
    assert all(torch.equal(published[key], value) for key, value in payload["ema"]["shadow"].items())

    published_model = QDGrasp(weights=result.artifacts["bundle"])
    assert result.metrics == published_model.val("dummy-tiny.yaml", batch_size=4)


def test_ema_resume_is_bit_exact_with_validation_between_training_steps(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    continuous_history = LossHistory()
    continuous = QDGrasp().train(
        "dummy-tiny.yaml",
        max_steps=8,
        val_interval=2,
        ema_decay=0.9,
        run_name="continuous",
        callbacks=[continuous_history],
    )

    first_history = LossHistory()
    first = QDGrasp().train(
        "dummy-tiny.yaml",
        max_steps=8,
        stop_after_steps=4,
        val_interval=2,
        ema_decay=0.9,
        run_name="first",
        callbacks=[first_history],
    )
    second_history = LossHistory()
    resumed = QDGrasp().train(
        "dummy-tiny.yaml",
        max_steps=8,
        val_interval=2,
        ema_decay=0.9,
        resume=first.artifacts["resume"],
        run_name="resumed",
        callbacks=[second_history],
    )

    assert continuous_history.history == first_history.history + second_history.history
    continuous_state = torch.load(continuous.artifacts["resume"], map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed.artifacts["resume"], map_location="cpu", weights_only=True)
    assert all(torch.equal(continuous_state["model"][key], resumed_state["model"][key]) for key in continuous_state["model"])
    assert all(
        torch.equal(continuous_state["ema"]["shadow"][key], resumed_state["ema"]["shadow"][key])
        for key in continuous_state["ema"]["shadow"]
    )
    assert continuous.metrics == resumed.metrics


def test_resume_identity_rejection_happens_before_model_mutation(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = QDGrasp().train("dummy-tiny.yaml", max_steps=4, run_name="source")
    target = QDGrasp()
    before = {key: value.detach().clone() for key, value in target.module.state_dict().items()}

    with pytest.raises(ConfigError, match="resume identity mismatch"):
        target.train(
            "dummy-tiny.yaml",
            max_steps=4,
            learning_rate=2e-3,
            resume=source.artifacts["resume"],
            run_name="wrong-learning-rate",
        )

    assert all(torch.equal(value, target.module.state_dict()[key]) for key, value in before.items())


def test_resume_v2_self_hashes_identity_documents_and_separates_stream_from_scaler(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = QDGrasp().train("dummy-tiny.yaml", max_steps=2, run_name="identity")
    path = Path(result.artifacts["resume"])
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["schema"] == "qdgrasp/resume/v2"
    assert payload["scaler"]["grad_scaler"] == {}
    assert "stream" in payload and "stream" not in payload["scaler"]

    payload["data_manifest"]["seed"] = -1
    tampered = tmp_path / "tampered-resume.pt"
    torch.save(payload, tampered)
    with pytest.raises(ConfigError, match="data_manifest hash mismatch"):
        ResumeState.load(tampered)
