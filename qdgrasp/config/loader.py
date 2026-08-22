"""YAML loading and preset resolution for QDGrasp documents."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from .schema import ConfigError, DataConfig, ModelConfig, RobotConfig, RunConfig, _Document


DocumentT = TypeVar("DocumentT", bound=_Document)

PRESET_PACKAGE = "qdgrasp.presets"


def preset_names() -> tuple[str, ...]:
    """Sorted file names of the YAML presets shipped inside the package."""

    return tuple(
        sorted(entry.name for entry in resources.files(PRESET_PACKAGE).iterdir() if entry.name.endswith(".yaml"))
    )


def resolve_document_path(reference: str | Path) -> Path:
    """Resolve ``reference`` as a relative/absolute file, else as a packaged preset."""

    candidate = Path(reference)
    if candidate.is_file():
        return candidate
    packaged = resources.files(PRESET_PACKAGE).joinpath(candidate.name)
    if candidate.parent in (Path(""), Path(".")) and packaged.is_file():
        return Path(str(packaged))
    raise ConfigError(f"configuration '{reference}' is neither a file nor a packaged preset {preset_names()}")


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Parse a YAML mapping with ``safe_load``; no tags, no object construction."""

    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if content is None:
        raise ConfigError(f"{path}: document is empty")
    if not isinstance(content, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(content).__name__}")
    return content


def parse_document(document: dict[str, Any], model: type[DocumentT], *, origin: str) -> DocumentT:
    """Validate a mapping against ``model`` and surface unknown keys as errors."""

    try:
        return model.model_validate(document)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}" for error in exc.errors()
        )
        raise ConfigError(f"{origin}: {details}") from exc


def load_document(reference: str | Path, model: type[DocumentT]) -> DocumentT:
    """Resolve, read and validate one YAML document into ``model``."""

    path = resolve_document_path(reference)
    return parse_document(read_yaml_mapping(path), model, origin=str(reference))


def load_model_config(reference: str | Path) -> ModelConfig:
    """Load a ``qdgrasp/model/v1`` document."""

    return load_document(reference, ModelConfig)


def load_robot_config(reference: str | Path) -> RobotConfig:
    """Load a ``qdgrasp/robot/v1`` document."""

    return load_document(reference, RobotConfig)


def load_data_config(reference: str | Path) -> DataConfig:
    """Load a ``qdgrasp/data/v1`` document."""

    return load_document(reference, DataConfig)


def load_run_config(reference: str | Path) -> RunConfig:
    """Load a ``qdgrasp/run/v1`` document."""

    return load_document(reference, RunConfig)


def dump_document(document: _Document) -> str:
    """Serialise a validated document back to deterministic YAML."""

    return yaml.safe_dump(document.to_document(), sort_keys=True, default_flow_style=False, allow_unicode=True)
