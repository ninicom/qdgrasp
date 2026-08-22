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


def _all_preset_files() -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    root = resources.files(PRESET_PACKAGE)
    for entry in root.iterdir():
        if entry.is_file() and entry.name.endswith(".yaml"):
            results.append((entry.name, Path(str(entry))))
        elif entry.is_dir():
            for child in entry.iterdir():
                if child.is_file() and child.name.endswith(".yaml"):
                    results.append((f"{entry.name}/{child.name}", Path(str(child))))
                    results.append((child.name, Path(str(child))))
    return results


def preset_names() -> tuple[str, ...]:
    """Sorted file names of the YAML presets shipped inside the package."""

    return tuple(sorted({name for name, _ in _all_preset_files()}))


def resolve_document_path(reference: str | Path) -> Path:
    """Resolve ``reference`` as a relative/absolute file, else as a packaged preset."""

    candidate = Path(reference)
    if candidate.is_file():
        return candidate
    preset_map = dict(_all_preset_files())
    ref_str = str(reference)
    if ref_str in preset_map:
        return preset_map[ref_str]
    if candidate.name in preset_map:
        return preset_map[candidate.name]
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


def load_robot_config(reference: str | Path) -> RobotConfig | Any:
    """Load a ``qdgrasp/robot/v1`` or ``qdgrasp/robot/v2`` document."""

    from ..robot.schema import ROBOT_SCHEMA_V2, RobotConfigV2

    path = resolve_document_path(reference)
    mapping = read_yaml_mapping(path)
    schema_ver = mapping.get("schema")
    if schema_ver == "qdgrasp/robot/v1":
        return parse_document(mapping, RobotConfig, origin=str(reference))
    if schema_ver == ROBOT_SCHEMA_V2:
        return parse_document(mapping, RobotConfigV2, origin=str(reference))
    raise ConfigError(
        f"{reference}: unsupported robot schema {schema_ver!r}; "
        f"supported: {sorted(SUPPORTED_SCHEMAS['robot'])}"
    )


def load_data_config(reference: str | Path) -> DataConfig:
    """Load a ``qdgrasp/data/v1`` document."""

    return load_document(reference, DataConfig)


def load_run_config(reference: str | Path) -> RunConfig:
    """Load a ``qdgrasp/run/v1`` document."""

    return load_document(reference, RunConfig)


def dump_document(document: _Document) -> str:
    """Serialise a validated document back to deterministic YAML."""

    return yaml.safe_dump(document.to_document(), sort_keys=True, default_flow_style=False, allow_unicode=True)
