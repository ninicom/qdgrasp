"""``QDGrasp``: the one public object for train, val, predict and export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from ..config import (
    ConfigError,
    DataConfig,
    ModelConfig,
    RobotConfig,
    RunConfig,
    get_dataset_builder,
    get_model_builder,
    load_data_config,
    load_model_config,
    load_robot_config,
    parse_document,
    resolve_runtime,
)
from ..engine.callbacks import Callback, CallbackList, ProgressLogger
from ..engine.checkpoint import BundleInfo, load_public_bundle, read_bundle_manifest, save_public_bundle
from ..engine.runner import RunResult, Runner
from ..engine.seeding import seed_everything
from ..export import ExportResult, export_bundle
from .protocols import GraspModel
from .results import GraspResults


LOGGER = logging.getLogger("qdgrasp.api")

DEFAULT_MODEL = "qdgrasp-dummy-n.yaml"
DEFAULT_ROBOT = "dummy-hand.yaml"


class QDGrasp:
    """Public façade over a model configuration and its robot profile.

    Args:
        model: Model YAML path or packaged preset name.
        robot: Robot profile YAML path or packaged preset name.
        weights: Optional public bundle directory to load weights from.
        seed: Seed applied before the module is built so weight initialisation
            is reproducible across processes.

    Example:
        >>> from qdgrasp import QDGrasp
        >>> grasper = QDGrasp("qdgrasp-dummy-n.yaml", robot="dummy-hand.yaml")
        >>> result = grasper.train(data="dummy-tiny.yaml", device="cpu", max_steps=5)
    """

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        robot: str | Path = DEFAULT_ROBOT,
        weights: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        self.model_config: ModelConfig = load_model_config(model)
        self.robot_config: RobotConfig = load_robot_config(robot)
        self.seed = seed_everything(seed, deterministic=False)
        builder = get_model_builder(self.model_config.type)
        self.module: nn.Module = builder(self.model_config, self.robot_config)
        if not isinstance(self.module, GraspModel):
            raise ConfigError(
                f"model type '{self.model_config.type}' does not satisfy the GraspModel protocol"
            )
        self.bundle_manifest: dict[str, Any] | None = None
        if weights is not None:
            self.load_weights(weights)

    def __repr__(self) -> str:
        parameters = sum(item.numel() for item in self.module.parameters())
        return (
            f"QDGrasp(model='{self.model_config.name}', type='{self.model_config.type}', "
            f"robot='{self.robot_config.name}', joints={len(self.robot_config.joints)}, parameters={parameters})"
        )

    @property
    def model_hash(self) -> str:
        """Content hash of the model configuration."""

        return self.model_config.content_hash()

    @property
    def robot_hash(self) -> str:
        """Content hash of the robot profile."""

        return self.robot_config.content_hash()

    def load_weights(self, directory: str | Path) -> dict[str, Any]:
        """Load a public bundle, rejecting a mismatched robot profile."""

        self.bundle_manifest = load_public_bundle(directory, self.module, robot_config=self.robot_config)
        return self.bundle_manifest

    def save_bundle(self, directory: str | Path, *, data_manifest: dict[str, Any] | None = None) -> BundleInfo:
        """Write the current weights as a public bundle."""

        return save_public_bundle(
            directory,
            model=self.module,
            model_config=self.model_config,
            robot_config=self.robot_config,
            preprocess=self.module.preprocess_schema(),
            data_manifest=data_manifest,
        )

    def _run_config(self, overrides: dict[str, Any]) -> RunConfig:
        return parse_document(overrides, RunConfig, origin="run configuration")

    def _datasets(self, data: str | Path, splits: Sequence[str]) -> tuple[DataConfig, dict[str, Any]]:
        data_config = load_data_config(data)
        builder = get_dataset_builder(data_config.type)
        return data_config, {split: builder(data_config, self.robot_config, split=split) for split in splits}

    def train(self, data: str | Path, *, callbacks: Sequence[Callback] | None = None, **overrides: Any) -> RunResult:
        """Train on ``data`` and return the run bundle.

        Unknown keyword arguments are rejected by the run schema; there is no
        silent alias layer.
        """

        run_config = self._run_config(overrides)
        runtime = resolve_runtime(run_config)
        data_config, datasets = self._datasets(data, ("train", "val"))
        runner = Runner(
            run_config=run_config,
            runtime=runtime,
            model_config=self.model_config,
            robot_config=self.robot_config,
            callbacks=CallbackList(list(callbacks) if callbacks is not None else [ProgressLogger()]),
        )
        manifest = datasets["train"].manifest() | {"data_config": data_config.to_document()}
        return runner.fit(
            self.module,
            datasets["train"],
            datasets["val"],
            preprocess=self.module.preprocess_schema(),
            data_manifest=manifest,
        )

    def val(self, data: str | Path, **overrides: Any) -> dict[str, float]:
        """Validate on the ``val`` split and return averaged metrics."""

        run_config = self._run_config(overrides)
        runtime = resolve_runtime(run_config)
        _data_config, datasets = self._datasets(data, ("val",))
        runner = Runner(
            run_config=run_config,
            runtime=runtime,
            model_config=self.model_config,
            robot_config=self.robot_config,
            callbacks=CallbackList([]),
        )
        module = runner.fabric.setup_module(self.module)
        return runner.validate(module, datasets["val"])

    def predict(self, points: torch.Tensor, *, device: str = "cpu") -> GraspResults:
        """Predict ranked grasps for one ``[N, 3]`` point cloud."""

        runtime = resolve_runtime(RunConfig(device=device))
        self.module.to(runtime.device)
        self.module.eval()
        tensor = torch.as_tensor(points, dtype=torch.float32, device=runtime.device)
        return self.module.predict_results(tensor)

    def export(self, *, fmt: str = "torchscript", out_dir: str | Path = "runs/export", verify: bool = True) -> ExportResult:
        """Export the current weights to TorchScript or ONNX with a metadata sidecar."""

        self.module.to("cpu")
        return export_bundle(
            self.module,
            fmt=fmt,
            out_dir=out_dir,
            model_config=self.model_config,
            robot_config=self.robot_config,
            preprocess=self.module.preprocess_schema(),
            verify=verify,
        )

    @classmethod
    def from_bundle(cls, directory: str | Path) -> "QDGrasp":
        """Rebuild a façade straight from a public bundle directory."""

        manifest = read_bundle_manifest(directory)
        instance = cls.__new__(cls)
        instance.model_config = parse_document(manifest["model_config"], ModelConfig, origin=str(directory))
        instance.robot_config = parse_document(manifest["robot_config"], RobotConfig, origin=str(directory))
        builder = get_model_builder(instance.model_config.type)
        instance.module = builder(instance.model_config, instance.robot_config)
        instance.bundle_manifest = load_public_bundle(directory, instance.module, robot_config=instance.robot_config)
        return instance


def load(directory: str | Path) -> QDGrasp:
    """Convenience wrapper around :meth:`QDGrasp.from_bundle`."""

    if not Path(directory).is_dir():
        raise ConfigError(f"bundle directory not found: {directory}")
    return QDGrasp.from_bundle(directory)
