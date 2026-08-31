"""Scene resolution: load one, or generate one, never quietly substitute (P3.5-06).

The rule is three lines and the whole point of the module:

    a valid scene reference   -> load, validate, compile
    a broken scene reference  -> fail
    no scene reference        -> generate a virtual drop scene

The middle line is the one that matters.  Falling back to a generated scene when
the requested one fails to load would silently change the problem being solved,
and every measurement downstream would be about a different world than the one
the caller asked for.  So a broken reference raises, and the caller decides.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import mujoco

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.builders.base import build_scene_mujoco_model
from qdgrasp.scenes.contracts import SceneSpec
from qdgrasp.scenes.serialize import load_scene_spec
from qdgrasp.scenes.virtual_drop import (
    DropObjectRequest,
    VirtualDropSceneSpec,
    build_virtual_drop_scene,
)


class SceneSource(str, Enum):
    """Where the resolved scene came from.  Always reported, never inferred."""

    LOADED = "loaded"
    GENERATED = "generated"


class SceneLoadError(RuntimeError):
    """A scene was requested by reference and could not be produced from it."""


@dataclasses.dataclass(frozen=True)
class ResolvedScene:
    """A compiled scene plus the provenance of how it was obtained."""

    spec: SceneSpec
    model: mujoco.MjModel
    source: SceneSource
    scene_ref: str | None
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def generated(self) -> bool:
        return self.source is SceneSource.GENERATED


def _load_scene_spec(scene_ref: str | Path) -> SceneSpec:
    """Load a canonical ``SceneSpec`` from a stored scene record."""

    path = Path(scene_ref)
    if not path.is_file():
        raise SceneLoadError(f"scene reference does not exist: {path}")
    try:
        return load_scene_spec(path)
    except Exception as error:
        raise SceneLoadError(f"scene reference {path} failed to load: {type(error).__name__}: {error}") from error


def resolve_scene(
    *,
    scene_ref: str | Path | None = None,
    scene_spec: SceneSpec | None = None,
    objects: Sequence[DropObjectRequest] = (),
    virtual_scene_config: VirtualDropSceneSpec | None = None,
    seed: int = 0,
    scene_id: str = "virtual-drop",
) -> ResolvedScene:
    """Resolve a scene from a reference, an in-memory spec, or generation.

    Exactly one of ``scene_ref`` and ``scene_spec`` may be given.  When neither
    is, ``objects`` and ``virtual_scene_config`` are used to generate one; a
    generation request with no objects is an error rather than an empty scene.
    """

    if scene_ref is not None and scene_spec is not None:
        raise ConfigError("scene_ref and scene_spec are mutually exclusive")

    if scene_ref is not None or scene_spec is not None:
        spec = scene_spec if scene_spec is not None else _load_scene_spec(scene_ref)  # type: ignore[arg-type]
        try:
            model = build_scene_mujoco_model(spec, include_objects=True, dynamic_objects=True)
        except Exception as error:
            raise SceneLoadError(
                f"scene {spec.scene_id!r} loaded but failed to compile: {type(error).__name__}: {error}. "
                "A generated scene is not a substitute for the one that was asked for."
            ) from error
        return ResolvedScene(
            spec=spec,
            model=model,
            source=SceneSource.LOADED,
            scene_ref=str(scene_ref) if scene_ref is not None else None,
            detail={"object_count": len(spec.objects), "support_count": len(spec.supports)},
        )

    if not objects:
        raise ConfigError(
            "no scene reference was given and no objects were supplied, so there is nothing to generate a scene from"
        )
    config = virtual_scene_config or VirtualDropSceneSpec()
    spec = build_virtual_drop_scene(config, objects, seed=seed, scene_id=scene_id)
    try:
        model = build_scene_mujoco_model(spec, include_objects=True, dynamic_objects=True)
    except Exception as error:
        raise SceneLoadError(
            f"generated scene {spec.scene_id!r} failed to compile: {type(error).__name__}: {error}"
        ) from error
    return ResolvedScene(
        spec=spec,
        model=model,
        source=SceneSource.GENERATED,
        scene_ref=None,
        detail={
            "environment": config.environment,
            "object_count": len(spec.objects),
            "virtual_scene_hash": config.content_hash(),
            "seed": seed,
        },
    )
