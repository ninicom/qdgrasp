"""Lightning Fabric runner: train, validate and produce the result bundle."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from lightning.fabric import Fabric
from torch import nn

from ..config.policy import EffectiveRuntime
from ..config.schema import ConfigError, ModelConfig, RobotConfig, RunConfig
from ..version import __version__
from .callbacks import CallbackList, ProgressLogger
from .checkpoint import RESUME_FILE, BundleInfo, ResumeState, canonical_hash, save_public_bundle
from .ema import ModelEma
from .sampling import DeterministicBatchStream, collate_indices, iterate_batches
from .seeding import RngSnapshot, capture_rng, isolated_rng, restore_rng, seed_everything

LOGGER = logging.getLogger("qdgrasp.engine")
RESULTS_SCHEMA = "qdgrasp/results/v1"
RESULTS_FILE = "results.json"


STEP_STATE_KEY = "step"
VALIDATION_SEED_SALT = 0x51444752
SESSION_ONLY_RUN_FIELDS = frozenset({"project_dir", "run_name", "resume", "stop_after_steps"})


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
        """Return deterministic sample-weighted metrics without changing caller state.

        ``validation_step`` reports a mean for its current batch.  Multiplying by
        the actual batch cardinality keeps a short final batch from receiving the
        same weight as a full one.
        """

        previous_mode = model.training
        totals: dict[str, float] = {}
        samples = 0
        validation_seed = (self.runtime.seed ^ VALIDATION_SEED_SALT) % (2**32)
        try:
            model.eval()
            with isolated_rng(validation_seed), torch.no_grad():
                for batch in iterate_batches(dataset, self.run_config.batch_size):
                    cardinality = self._batch_cardinality(batch)
                    metrics = model.validation_step(self._to_device(batch))
                    for key, value in metrics.items():
                        totals[key] = totals.get(key, 0.0) + float(value) * cardinality
                    samples += cardinality
        finally:
            model.train(previous_mode)
        return {key: value / max(samples, 1) for key, value in totals.items()}

    @staticmethod
    def _batch_cardinality(batch: dict[str, torch.Tensor]) -> int:
        """Infer the number of samples represented by one collated batch."""

        points = batch.get("points")
        if isinstance(points, torch.Tensor) and points.ndim:
            return int(points.shape[0])
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim:
                return int(value.shape[0])
        raise ConfigError("validation batch contains no batched tensor")

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

        preprocess = preprocess or {}
        data_manifest = data_manifest or {}
        seed_everything(self.runtime.seed, deterministic=self.runtime.deterministic)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.run_config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.run_config.max_steps)
        ema = ModelEma(model, self.run_config.ema_decay)
        stream = DeterministicBatchStream(len(train_dataset), self.run_config.batch_size, self.runtime.seed)
        start_step = 0

        if self.run_config.resume:
            start_step = self._restore(
                self.run_config.resume,
                model,
                optimizer,
                scheduler,
                ema,
                stream,
                data_manifest=data_manifest,
            )
            LOGGER.info("resumed from %s at step %d", self.run_config.resume, start_step)

        fabric_model, fabric_optimizer = self.fabric.setup(model, optimizer)
        align_optimizer_state(optimizer)
        ema.to(self.runtime.device)
        history: list[tuple[int, float]] = []
        metrics: dict[str, float] = {}
        last_validation_step: int | None = None
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
                with ema.average_parameters(model):
                    metrics = self.validate(fabric_model, val_dataset)
                last_validation_step = step
                state["metrics"] = metrics
                self.callbacks.on_validation_end(state)

        if val_dataset is not None and last_validation_step != step:
            with ema.average_parameters(model):
                metrics = self.validate(fabric_model, val_dataset)
            last_validation_step = step
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
            preprocess=preprocess,
            data_manifest=data_manifest,
        )

    def _effective_run_config(self) -> dict[str, Any]:
        """Full request plus the effective identity that must survive resume."""

        requested = self.run_config.to_document()
        continuation = {key: value for key, value in requested.items() if key not in SESSION_ONLY_RUN_FIELDS}
        return {
            "requested": requested,
            "effective_runtime": self.runtime.to_dict()["effective"],
            "continuation_identity": {
                "run": continuation,
                "runtime": self.runtime.to_dict()["effective"],
            },
        }

    @staticmethod
    def _optimizer_config(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Semantic optimiser settings, excluding mutable moments and step."""

        defaults = optimizer.defaults
        return {
            "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "learning_rate": float(defaults["lr"]),
            "betas": [float(value) for value in defaults["betas"]],
            "eps": float(defaults["eps"]),
            "weight_decay": float(defaults["weight_decay"]),
            "amsgrad": bool(defaults["amsgrad"]),
            "maximize": bool(defaults["maximize"]),
            "capturable": bool(defaults["capturable"]),
            "differentiable": bool(defaults["differentiable"]),
            "fused": defaults.get("fused"),
        }

    @staticmethod
    def _scheduler_config(scheduler: torch.optim.lr_scheduler.LRScheduler) -> dict[str, Any]:
        """Semantic schedule settings, excluding its mutable cursor."""

        return {
            "type": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
            "T_max": int(scheduler.T_max),
            "eta_min": float(scheduler.eta_min),
        }

    @staticmethod
    def _manifest_digests(data_manifest: dict[str, Any]) -> tuple[str | None, str | None]:
        """Read optional protocol/view identities without importing dataset code."""

        identities = data_manifest.get("identities", {})
        if not isinstance(identities, dict):
            identities = {}

        def first_string(*keys: str) -> str | None:
            for key in keys:
                value = data_manifest.get(key, identities.get(key))
                if isinstance(value, str) and value:
                    return value
            return None

        return first_string("protocol_hash"), first_string("dataset_view_hash", "view_hash")

    @property
    def _precision_plugin(self) -> Any:
        return self.fabric.strategy.precision

    def _scaler_state(self) -> dict[str, Any]:
        plugin = self._precision_plugin
        return {
            "plugin": f"{type(plugin).__module__}.{type(plugin).__qualname__}",
            "precision": self.runtime.precision,
            # For 16-mixed this is the real GradScaler state.  FP32 and bf16
            # correctly carry an empty mapping, not a boolean pretending to be it.
            "grad_scaler": plugin.state_dict(),
        }

    @staticmethod
    def _weights_source(ema: ModelEma) -> dict[str, str]:
        promoted = "ema" if ema.enabled else "live"
        return {"resume_model": "live", "validation": promoted, "public_bundle": promoted}

    def _validate_resume_identity(
        self,
        state: ResumeState,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ema: ModelEma,
        stream: DeterministicBatchStream,
        data_manifest: dict[str, Any],
    ) -> None:
        """Preflight every continuation identity before mutating live state."""

        mismatches: list[str] = []
        expected_model = self.model_config.to_document()
        expected_robot = self.robot_config.to_document()
        expected_data_hash = canonical_hash(data_manifest)
        expected_run = self._effective_run_config()
        protocol_hash, dataset_view_hash = self._manifest_digests(data_manifest)

        exact = (
            ("qdgrasp_version", state.qdgrasp_version, __version__),
            ("model_config", state.model_config, expected_model),
            ("model_config_hash", state.model_config_hash, self.model_config.content_hash()),
            ("robot_config", state.robot_config, expected_robot),
            ("robot_config_hash", state.robot_config_hash, self.robot_config.content_hash()),
            ("data_manifest_hash", state.data_manifest_hash, expected_data_hash),
            ("protocol_hash", state.protocol_hash, protocol_hash),
            ("dataset_view_hash", state.dataset_view_hash, dataset_view_hash),
            (
                "continuation_identity",
                state.effective_run_config.get("continuation_identity"),
                expected_run["continuation_identity"],
            ),
            ("optimizer_config", state.optimizer_config, self._optimizer_config(optimizer)),
            ("scheduler_config", state.scheduler_config, self._scheduler_config(scheduler)),
            ("weights_source", state.weights_source, self._weights_source(ema)),
        )
        for name, stored, expected in exact:
            if stored != expected:
                mismatches.append(f"{name}: stored={stored!r}, expected={expected!r}")

        if state.global_step < 0 or state.global_step > self.run_config.max_steps:
            mismatches.append(
                f"global_step: stored={state.global_step}, expected range=[0, {self.run_config.max_steps}]"
            )

        if not isinstance(state.stream, dict):
            mismatches.append("stream: stored state is not a mapping")
        else:
            if int(state.stream.get("dataset_size", -1)) != stream.dataset_size:
                mismatches.append(
                    f"stream.dataset_size: stored={state.stream.get('dataset_size')!r}, expected={stream.dataset_size}"
                )
            if int(state.stream.get("batch_size", -1)) != stream.batch_size:
                mismatches.append(
                    f"stream.batch_size: stored={state.stream.get('batch_size')!r}, expected={stream.batch_size}"
                )

        current_scaler = self._scaler_state()
        if not isinstance(state.scaler, dict):
            mismatches.append("scaler: stored state is not a mapping")
        else:
            for key in ("plugin", "precision"):
                if state.scaler.get(key) != current_scaler[key]:
                    mismatches.append(
                        f"scaler.{key}: stored={state.scaler.get(key)!r}, expected={current_scaler[key]!r}"
                    )
            if not isinstance(state.scaler.get("grad_scaler"), dict):
                mismatches.append("scaler.grad_scaler: stored state is not a mapping")

        if not isinstance(state.ema, dict):
            mismatches.append("ema: stored state is not a mapping")
        else:
            if float(state.ema.get("decay", -1.0)) != ema.decay:
                mismatches.append(f"ema.decay: stored={state.ema.get('decay')!r}, expected={ema.decay!r}")
            expected_updates = state.global_step if ema.enabled else 0
            if int(state.ema.get("updates", -1)) != expected_updates:
                mismatches.append(f"ema.updates: stored={state.ema.get('updates')!r}, expected={expected_updates}")
            shadow = state.ema.get("shadow")
            if not isinstance(shadow, dict) or bool(shadow) != ema.enabled:
                mismatches.append(
                    f"ema.shadow: stored={'present' if isinstance(shadow, dict) and shadow else 'empty'}, "
                    f"expected={'present' if ema.enabled else 'empty'}"
                )

        current_model = model.state_dict()
        if set(state.model) != set(current_model):
            missing = sorted(set(current_model) - set(state.model))
            unexpected = sorted(set(state.model) - set(current_model))
            mismatches.append(f"model state keys: missing={missing}, unexpected={unexpected}")
        else:
            for key, expected in current_model.items():
                stored = state.model[key]
                if not isinstance(stored, torch.Tensor):
                    mismatches.append(f"model.{key}: stored value is not a tensor")
                elif stored.shape != expected.shape or stored.dtype != expected.dtype:
                    mismatches.append(
                        f"model.{key}: stored shape/dtype={tuple(stored.shape)}/{stored.dtype}, "
                        f"expected={tuple(expected.shape)}/{expected.dtype}"
                    )

        last_epoch = state.scheduler.get("last_epoch") if isinstance(state.scheduler, dict) else None
        if last_epoch is None or int(last_epoch) != state.global_step:
            mismatches.append(f"scheduler.last_epoch: stored={last_epoch!r}, expected={state.global_step}")

        if mismatches:
            details = "\n  - ".join(mismatches)
            raise ConfigError(
                f"resume identity mismatch; use an explicit transfer/init path for another run:\n  - {details}"
            )

    def _restore(
        self,
        resume_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ema: ModelEma,
        stream: DeterministicBatchStream,
        *,
        data_manifest: dict[str, Any],
    ) -> int:
        path = Path(resume_path)
        if path.is_dir():
            path = path / RESUME_FILE
        if not path.is_file():
            raise ConfigError(f"resume artifact not found: {resume_path}")
        state = ResumeState.load(path)
        self._validate_resume_identity(state, model, optimizer, scheduler, ema, stream, data_manifest)

        model.load_state_dict(state.model)
        optimizer.load_state_dict(state.optimizer)
        scheduler.load_state_dict(state.scheduler)
        ema.load_state_dict(state.ema)
        stream.load_state_dict(state.stream)
        self._precision_plugin.load_state_dict(state.scaler["grad_scaler"])
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
        weights_source = self._weights_source(ema)
        with ema.average_parameters(model):
            bundle: BundleInfo = save_public_bundle(
                run_dir / "bundle",
                model=model,
                model_config=self.model_config,
                robot_config=self.robot_config,
                preprocess=preprocess,
                data_manifest=data_manifest,
            )

        effective_run_config = self._effective_run_config()
        protocol_hash, dataset_view_hash = self._manifest_digests(data_manifest)
        resume = ResumeState(
            global_step=step,
            qdgrasp_version=__version__,
            model_config=self.model_config.to_document(),
            model_config_hash=self.model_config.content_hash(),
            robot_config=self.robot_config.to_document(),
            robot_config_hash=self.robot_config.content_hash(),
            data_manifest=data_manifest,
            data_manifest_hash=canonical_hash(data_manifest),
            protocol_hash=protocol_hash,
            dataset_view_hash=dataset_view_hash,
            effective_run_config=effective_run_config,
            effective_run_config_hash=canonical_hash(effective_run_config),
            optimizer_config=self._optimizer_config(optimizer),
            scheduler_config=self._scheduler_config(scheduler),
            weights_source=weights_source,
            model={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            stream=stream.state_dict(),
            scaler=self._scaler_state(),
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
        payload["created_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        (run_dir / RESULTS_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
