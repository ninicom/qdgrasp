"""Declarative configuration: schemas, allowlist registry, loader and policy."""

from __future__ import annotations

from .loader import (
    dump_document,
    load_data_config,
    load_document,
    load_model_config,
    load_robot_config,
    load_run_config,
    parse_document,
    preset_names,
    read_yaml_mapping,
    resolve_document_path,
)
from .policy import Adjustment, EffectiveRuntime, resolve_runtime
from .registry import (
    RegistryError,
    get_dataset_builder,
    get_model_builder,
    register_dataset,
    register_model,
    registered_datasets,
    registered_models,
)
from .schema import (
    DATA_SCHEMA_V1,
    MODEL_SCHEMA_V1,
    ROBOT_SCHEMA_V1,
    RUN_SCHEMA_V1,
    SUPPORTED_SCHEMAS,
    ConfigError,
    DataConfig,
    ModelConfig,
    RobotConfig,
    RunConfig,
)

__all__ = (
    "Adjustment",
    "ConfigError",
    "DATA_SCHEMA_V1",
    "DataConfig",
    "EffectiveRuntime",
    "MODEL_SCHEMA_V1",
    "ModelConfig",
    "ROBOT_SCHEMA_V1",
    "RUN_SCHEMA_V1",
    "RegistryError",
    "RobotConfig",
    "RunConfig",
    "SUPPORTED_SCHEMAS",
    "dump_document",
    "get_dataset_builder",
    "get_model_builder",
    "load_data_config",
    "load_document",
    "load_model_config",
    "load_robot_config",
    "load_run_config",
    "parse_document",
    "preset_names",
    "read_yaml_mapping",
    "register_dataset",
    "register_model",
    "registered_datasets",
    "registered_models",
    "resolve_document_path",
    "resolve_runtime",
)
