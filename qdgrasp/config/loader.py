"""YAML loading and preset resolution for QDGrasp documents."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import yaml
from pydantic import ValidationError

from .registry import RegistryError, get_document_model
from .schema import ConfigError, DataConfig, ModelConfig, RobotConfig, RunConfig, _Document

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps config free of a robot import
    from ..robot.schema import RobotConfigV2


DocumentT = TypeVar("DocumentT", bound=_Document)

PRESET_PACKAGE = "qdgrasp.presets"


def _all_preset_files() -> list[tuple[str, Path]]:
    """Packaged presets, addressable by ``subdir/name.yaml`` and by bare name.

    The bare name is a convenience shortcut; the qualified name is always
    available and is what a caller should use when two presets could collide.
    """

    results: list[tuple[str, Path]] = []
    root = resources.files(PRESET_PACKAGE)
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if entry.is_file() and entry.name.endswith(".yaml"):
            results.append((entry.name, Path(str(entry))))
        elif entry.is_dir():
            for child in sorted(entry.iterdir(), key=lambda item: item.name):
                if child.is_file() and child.name.endswith(".yaml"):
                    results.append((f"{entry.name}/{child.name}", Path(str(child))))
                    results.append((child.name, Path(str(child))))
    return results


def _preset_index() -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Return the preset lookup plus every basename claimed by more than one file."""

    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for name, path in _all_preset_files():
        existing = index.get(name)
        if existing is not None and existing != path:
            duplicates.setdefault(name, [existing]).append(path)
            continue
        index[name] = path
    return index, duplicates


def preset_names() -> tuple[str, ...]:
    """Sorted file names of the YAML presets shipped inside the package."""

    return tuple(sorted({name for name, _ in _all_preset_files()}))


def resolve_document_path(reference: str | Path) -> Path:
    """Resolve ``reference`` as a relative/absolute file, else as a packaged preset."""

    candidate = Path(reference)
    if candidate.is_file():
        return candidate
    preset_map, duplicates = _preset_index()
    ref_str = str(reference)
    for key in (ref_str, candidate.name):
        if key in duplicates:
            # Silently picking one of them would make the choice depend on
            # directory iteration order.
            claimants = ", ".join(sorted(str(item) for item in duplicates[key]))
            raise ConfigError(
                f"preset name '{key}' is ambiguous between {claimants}; "
                "use the 'subdirectory/name.yaml' form"
            )
        if key in preset_map:
            return preset_map[key]
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


def load_versioned_document(reference: str | Path, kind: str) -> _Document:
    """Load a document and dispatch on its declared ``schema`` identifier.

    The mapping from schema identifier to model lives in the registry, so a later
    phase can add a version without the configuration layer importing it back.
    """

    path = resolve_document_path(reference)
    mapping = read_yaml_mapping(path)
    schema_id = mapping.get("schema")
    if not isinstance(schema_id, str):
        raise ConfigError(f"{reference}: document is missing a string 'schema' identifier")
    try:
        model = get_document_model(kind, schema_id)
    except RegistryError as exc:
        raise ConfigError(f"{reference}: {exc}") from exc
    return parse_document(mapping, model, origin=str(reference))


def load_robot_config(reference: str | Path) -> "RobotConfig | RobotConfigV2":
    """Load any registered robot profile document."""

    return cast("RobotConfig | RobotConfigV2", load_versioned_document(reference, "robot"))


def load_data_config(reference: str | Path) -> DataConfig:
    """Load a ``qdgrasp/data/v1`` document."""

    return load_document(reference, DataConfig)


def load_run_config(reference: str | Path) -> RunConfig:
    """Load a ``qdgrasp/run/v1`` document."""

    return load_document(reference, RunConfig)


def dump_document(document: _Document) -> str:
    """Serialise a validated document back to deterministic YAML."""

    return yaml.safe_dump(document.to_document(), sort_keys=True, default_flow_style=False, allow_unicode=True)
