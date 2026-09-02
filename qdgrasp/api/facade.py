"""``QDGrasp``: the one public object for train, val, predict and export."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..config import (
    ConfigError,
    ModelConfig,
    RobotConfig,
    RunConfig,
    get_dataset_builder,
    get_model_builder,
    load_data_config,
    load_model_config,
    load_robot_config,
    parse_document,
    parse_versioned_document,
    resolve_runtime,
)
from ..corrective import assert_public_training_allowed
from ..engine.callbacks import Callback, CallbackList, ProgressLogger
from ..engine.checkpoint import BundleInfo, load_public_bundle, read_bundle_manifest, save_public_bundle
from ..engine.compatibility import EmbodimentBinding, bind_embodiment
from ..engine.runner import Runner, RunResult
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
        #: Last corrective-gate decision, set whenever a dataset is opened.
        self.gate_report: Any | None = None
        #: Declared cross-embodiment binding, when weights run on another hand.
        self.embodiment_binding: EmbodimentBinding | None = None
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

    def load_weights(
        self,
        directory: str | Path,
        *,
        binding: EmbodimentBinding | None = None,
    ) -> dict[str, Any]:
        """Load a public bundle, rejecting one whose semantics differ from this model.

        The comparison covers the model configuration and the declared
        preprocessing as well as the robot profile: identically shaped weights
        produced under other settings are a different model, not this one.  Pass
        ``binding`` to run weights against a hand they were not trained on.
        """

        self.bundle_manifest = load_public_bundle(
            directory,
            self.module,
            robot_config=self.robot_config,
            model_config=self.model_config,
            preprocess=self.module.preprocess_schema(),
            binding=binding,
        )
        self.embodiment_binding = binding
        return self.bundle_manifest

    def bind_to(self, runtime_robot: str | Path, *, protocol: Any | None = None) -> EmbodimentBinding:
        """Declare that this model's weights may drive another hand.

        ``protocol`` is the locked protocol whose held-out embodiment permits the
        pairing; without one, only the identity binding exists.
        """

        return bind_embodiment(self.robot_config, load_robot_config(runtime_robot), protocol=protocol)

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

    def _datasets(self, data: str | Path, splits: Sequence[str], *, purpose: str) -> tuple[Any, dict[str, Any]]:
        data_config = load_data_config(data)
        # PLAN.md §9.3: a corpus that fails its own audit or positive gate stops
        # the public path here, before a model, an optimiser or a run directory
        # exists to make the attempt look like a run that merely went badly.
        self.gate_report = assert_public_training_allowed(data_config, purpose=purpose)
        builder_name = getattr(data_config, "type", None) or getattr(data_config, "schema_version", "dgn_open")
        if not isinstance(builder_name, str) or not builder_name:
            raise ConfigError("data configuration does not declare a dataset builder")
        builder = get_dataset_builder(builder_name)
        return data_config, {split: builder(data_config, self.robot_config, split=split) for split in splits}

    def train(self, data: str | Path, *, callbacks: Sequence[Callback] | None = None, **overrides: Any) -> RunResult:
        """Train on ``data`` and return the run bundle.

        Unknown keyword arguments are rejected by the run schema; there is no
        silent alias layer.
        """

        run_config = self._run_config(overrides)
        runtime = resolve_runtime(run_config)
        data_config, datasets = self._datasets(data, ("train", "val"), purpose="training")
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
        _data_config, datasets = self._datasets(data, ("val",), purpose="validation")
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
        # After a declared transfer the module's own profile is the runtime hand,
        # not the one the weights were produced for; the binding is the only
        # place that still knows both.
        binding = self.embodiment_binding
        return self.module.predict_results(
            tensor,
            training_robot_hash=binding.training_robot_hash if binding is not None else None,
            runtime_robot_hash=binding.runtime_robot_hash if binding is not None else None,
        )

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
    def from_bundle(cls, directory: str | Path) -> QDGrasp:
        """Rebuild a façade straight from a public bundle directory."""

        manifest = read_bundle_manifest(directory)
        instance = cls.__new__(cls)
        # Both documents are parsed against the model their own ``schema`` names.
        # Both active hands ship as ``robot/v2``, so a fixed ``robot/v1`` parse
        # made this path unable to rebuild any bundle they produced.
        instance.model_config = parse_versioned_document(manifest["model_config"], "model", origin=str(directory))
        instance.robot_config = parse_versioned_document(
            manifest["training_robot_config"], "robot", origin=str(directory)
        )
        builder = get_model_builder(instance.model_config.type)
        instance.module = builder(instance.model_config, instance.robot_config)
        instance.gate_report = None
        instance.embodiment_binding = None
        instance.bundle_manifest = load_public_bundle(
            directory,
            instance.module,
            robot_config=instance.robot_config,
            model_config=instance.model_config,
            preprocess=instance.module.preprocess_schema(),
        )
        return instance


def load(directory: str | Path) -> QDGrasp:
    """Convenience wrapper around :meth:`QDGrasp.from_bundle`."""

    if not Path(directory).is_dir():
        raise ConfigError(f"bundle directory not found: {directory}")
    return QDGrasp.from_bundle(directory)
