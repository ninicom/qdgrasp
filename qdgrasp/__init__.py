"""Public QDGrasp library surface."""

from __future__ import annotations

__version__ = "0.1.0a1"

from .api import GraspDataset, GraspModel, GraspResults, QDGrasp, load
from .config import ConfigError, DataConfig, ModelConfig, RobotConfig, RunConfig
from .dataset.loader import create_dgn_open_dataset  # noqa: F401  (registers the v2 dataset builder)
from .dataset.schema import DataConfigV2
from .robot import RobotConfigV2, RobotSpec
from .runtime import EnvironmentInfo, environment_info, require_cuda

__all__ = (
    "ConfigError",
    "DataConfig",
    "DataConfigV2",
    "EnvironmentInfo",
    "GraspDataset",
    "GraspModel",
    "GraspResults",
    "ModelConfig",
    "QDGrasp",
    "RobotConfig",
    "RobotConfigV2",
    "RobotSpec",
    "RunConfig",
    "__version__",
    "environment_info",
    "load",
    "require_cuda",
)
