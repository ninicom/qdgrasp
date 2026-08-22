"""Lightning Fabric runner: train, validate and produce the result bundle."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from lightning.fabric import Fabric
from torch import nn

from ..config.policy import EffectiveRuntime
from ..config.schema import ConfigError, ModelConfig, RobotConfig, RunConfig
from .callbacks import Callback, CallbackList, ProgressLogger
from .checkpoint import RESUME_FILE, BundleInfo, ResumeState, save_public_bundle
from .ema import ModelEma
from .sampling import DeterministicBatchStream, collate_indices, iterate_batches
from .seeding import RngSnapshot, capture_rng, restore_rng, seed_everything


LOGGER = logging.getLogger("qdgrasp.engine")
RESULTS_SCHEMA = "qdgrasp/results/v1"
RESULTS_FILE = "results.json"


STEP_STATE_KEY = "step"


def align_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Put every optimizer state tensor on the device of its own parameter.

    A resume artifact is read with ``map_location="cpu"``, and the model is only
    moved to the accelerator afterwards by ``Fabric.setup``, so restored moments
    would otherwise stay on the host and the first step after a resume would mix
    CPU state with accelerator gradients.  ``step`` is left alone: a
    non-capturable optimiser keeps it on the host, and matching a fresh run's
    layout exactly is what keeps resume bit-exact.
    """

    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if key != STEP_STATE_KEY and isinstance(value, torch.Tensor):
                state[key] = value.to(parameter.device)


@dataclass(frozen=True)
class RunResult:
    """Everything a run produced, mirrored on disk as ``results.json``."""

    run_dir: Path
    global_step: int
    final_loss: float
    metrics: dict[str, float]
    losses: tuple[tuple[int, float], ...]
    runtime: dict[str, Any]
    artifacts: dict[str, str]
    hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULTS_SCHEMA,
            "run_dir": self.run_dir.as_posix(),
            "global_step": self.global_step,
            "final_loss": self.final_loss,
            "metrics": self.metrics,
            "losses": [list(item) for item in self.losses],
            "runtime": self.runtime,
            "artifacts": self.artifacts,
            "hashes": self.hashes,
        }


@dataclass
class Runner:
    """Own the training/validation loop for one :class:`RunConfig`.

    Fabric provides device placement, precision and the backward call; the loop,
    the seeding, the checkpoint contract and the result bundle are QDGrasp's.
    """

    run_config: RunConfig
    runtime: EffectiveRuntime
    model_config: ModelConfig
    robot_config: RobotConfig
    callbacks: CallbackList = field(default_factory=lambda: CallbackList([ProgressLogger()]))

    def __post_init__(self) -> None:
        self.fabric = Fabric(
            accelerator=self.runtime.accelerator,
            devices=[self.runtime.device_index] if self.runtime.accelerator == "cuda" else 1,
            precision=self.runtime.precision,
        )
        self.fabric.launch()

    @property
    def run_dir(self) -> Path:
        return Path(self.run_config.project_dir) / self.run_config.run_name

    def _last_step(self, start_step: int) -> int:
        """Last step of this session: the schedule length, capped by the session budget."""

        budget = self.run_config.stop_after_steps
        if budget is None:
            return self.run_config.max_steps
        return min(self.run_config.max_steps, start_step + budget)

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: self.fabric.to_device(value) for key, value in batch.items()}

    def validate(self, model: nn.Module, dataset: Sequence[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Average every metric the model reports over ``dataset``."""

        model.eval()
        totals: dict[str, float] = {}
        batches = 0
        with torch.no_grad():
            for batch in iterate_batches(dataset, self.run_config.batch_size):
                metrics = model.validation_step(self._to_device(batch))
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                batches += 1
        model.train()
        return {key: value / max(batches, 1) for key, value in totals.items()}

    def fit(
        self,
        model: nn.Module,
        train_dataset: Sequence[dict[str, torch.Tensor]],
        val_dataset: Sequence[dict[str, torch.Tensor]] | None = None,
        *,
        preprocess: dict[str, Any] | None = None,
        data_manifest: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run ``max_steps`` optimisation steps and write the run artifacts."""

        seed_everything(self.runtime.seed, deterministic=self.runtime.deterministic)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.run_config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.run_config.max_steps)
        ema = ModelEma(model, self.run_config.ema_decay)
        stream = DeterministicBatchStream(len(train_dataset), self.run_config.batch_size, self.runtime.seed)
        start_step = 0

        if self.run_config.resume:
            start_step = self._restore(self.run_config.resume, model, optimizer, scheduler, ema, stream)
            LOGGER.info("resumed from %s at step %d", self.run_config.resume, start_step)

        fabric_model, fabric_optimizer = self.fabric.setup(model, optimizer)
        align_optimizer_state(optimizer)
        ema.to(self.runtime.device)
        history: list[tuple[int, float]] = []
        metrics: dict[str, float] = {}
        state: dict[str, Any] = {
            "runtime": self.runtime.to_dict(),
            "max_steps": self.run_config.max_steps,
            "global_step": start_step,
            "loss": float("nan"),
            "metrics": metrics,
        }
        self.callbacks.on_train_start(state)

        fabric_model.train()
        step = start_step
        for step in range(start_step + 1, self._last_step(start_step) + 1):
            batch = self._to_device(collate_indices(train_dataset, stream.next_indices()))
            fabric_optimizer.zero_grad(set_to_none=True)
            with self.fabric.autocast():
                loss = fabric_model.training_step(batch)
            self.fabric.backward(loss)
            if self.run_config.grad_clip > 0.0:
                self.fabric.clip_gradients(fabric_model, fabric_optimizer, max_norm=self.run_config.grad_clip)
            fabric_optimizer.step()
            scheduler.step()
            ema.update(model)

            history.append((step, float(loss.detach())))
            state["global_step"] = step
            state["loss"] = history[-1][1]
            self.callbacks.on_step_end(state)

            interval = self.run_config.val_interval
            if val_dataset is not None and interval and step % interval == 0:
                metrics = self.validate(fabric_model, val_dataset)
                state["metrics"] = metrics
                self.callbacks.on_validation_end(state)

        if val_dataset is not None and not metrics:
            metrics = self.validate(fabric_model, val_dataset)
            state["metrics"] = metrics
            self.callbacks.on_validation_end(state)

        self.callbacks.on_train_end(state)
        return self._write_artifacts(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            stream=stream,
            step=step,
            history=history,
            metrics=metrics,
            preprocess=preprocess or {},
            data_manifest=data_manifest or {},
        )

    def _restore(
        self,
        resume_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ema: ModelEma,
        stream: DeterministicBatchStream,
    ) -> int:
        path = Path(resume_path)
        if path.is_dir():
            path = path / RESUME_FILE
        if not path.is_file():
            raise ConfigError(f"resume artifact not found: {resume_path}")
        state = ResumeState.load(path)
        model.load_state_dict(state.model)
        optimizer.load_state_dict(state.optimizer)
        scheduler.load_state_dict(state.scheduler)
        # ``max_steps`` is the authoritative schedule length of the whole run, so a
        # session that resumes into a longer/shorter schedule must not inherit the
        # horizon recorded by the interrupted session.
        scheduler.T_max = self.run_config.max_steps
        if state.ema.get("shadow"):
            ema.load_state_dict(state.ema)
        stream.load_state_dict(state.scaler["stream"])
        restore_rng(RngSnapshot.from_payload(state.rng))
        return state.global_step

    def _write_artifacts(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ema: ModelEma,
        stream: DeterministicBatchStream,
        step: int,
        history: list[tuple[int, float]],
        metrics: dict[str, float],
        preprocess: dict[str, Any],
        data_manifest: dict[str, Any],
    ) -> RunResult:
        run_dir = self.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        bundle: BundleInfo = save_public_bundle(
            run_dir / "bundle",
            model=model,
            model_config=self.model_config,
            robot_config=self.robot_config,
            preprocess=preprocess,
            data_manifest=data_manifest,
        )
        resume = ResumeState(
            global_step=step,
            model={key: value.detach().cpu() for key, value in model.state_dict().items()},
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler={"amp": self.runtime.amp, "stream": stream.state_dict()},
            ema=ema.state_dict(),
            rng=capture_rng().to_payload(),
        )
        resume_path = resume.save(run_dir / RESUME_FILE)

        result = RunResult(
            run_dir=run_dir,
            global_step=step,
            final_loss=history[-1][1] if history else float("nan"),
            metrics=metrics,
            losses=tuple(history),
            runtime=self.runtime.to_dict(),
            artifacts={
                "bundle": bundle.directory.as_posix(),
                "resume": resume_path.as_posix(),
                "results": (run_dir / RESULTS_FILE).as_posix(),
            },
            hashes={
                "bundle": bundle.bundle_hash,
                "model_config": bundle.model_hash,
                "robot_config": bundle.robot_hash,
            },
        )
        payload = result.to_dict()
        payload["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (run_dir / RESULTS_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
