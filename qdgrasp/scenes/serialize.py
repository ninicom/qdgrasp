"""Canonical on-disk form for :class:`~qdgrasp.scenes.contracts.SceneSpec`.

P3.5 needs "load a scene if one is given" to mean something specific, and until
now a ``SceneSpec`` only ever existed in memory or inside an adapter.  This is
the round-trip: a JSON document with the transforms written out in full, no
pickling, and a content hash so two records can be compared without reading them.

Loading is strict.  A document with a wrong schema, a malformed transform or a
missing field raises rather than being patched up, because a scene that loaded
"mostly" is exactly the silent substitution the resolver exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.contracts import (
    CameraSpec,
    SceneObjectSpec,
    SceneSpec,
    SupportGeometrySpec,
)

SCENE_SPEC_SCHEMA_V1 = "qdgrasp/scene-spec/v1"


def _transform_to_list(transform: np.ndarray, label: str) -> list[list[float]]:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ConfigError(f"{label} transform must be a finite 4x4 matrix")
    return [[float(entry) for entry in row] for row in value]


def _transform_from_list(raw: Any, label: str) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ConfigError(f"{label} transform must be a finite 4x4 matrix")
    return value


def scene_spec_to_document(spec: SceneSpec) -> dict[str, Any]:
    """Render a scene spec as a plain JSON-compatible document."""

    return {
        "schema": SCENE_SPEC_SCHEMA_V1,
        "scene_id": spec.scene_id,
        "source_dataset": spec.source_dataset,
        "source_version": spec.source_version,
        "source_split": spec.source_split,
        "environment": spec.environment,
        "gravity": [float(value) for value in spec.gravity],
        "timestep": float(spec.timestep),
        "solver_profile": spec.solver_profile,
        "settle_seed": int(spec.settle_seed),
        "source_record_hash": spec.source_record_hash,
        "license_record": spec.license_record,
        "redistributable": bool(spec.redistributable),
        "objects": [
            {
                "object_id": item.object_id,
                "asset_ref": item.asset_ref,
                "T_world_object": _transform_to_list(item.T_world_object, f"object {item.object_id}"),
                "scale": float(item.scale),
                "mass": None if item.mass is None else float(item.mass),
                "friction": None if item.friction is None else [float(value) for value in item.friction],
            }
            for item in spec.objects
        ],
        "supports": [
            {
                "support_id": item.support_id,
                "geom_type": item.geom_type,
                "params": item.params,
                "T_world_support": _transform_to_list(item.T_world_support, f"support {item.support_id}"),
            }
            for item in spec.supports
        ],
        "cameras": [
            {
                "camera_id": item.camera_id,
                "intrinsics": [[float(entry) for entry in row] for row in np.asarray(item.intrinsics)],
                "distortion": None
                if item.distortion is None
                else [float(value) for value in np.asarray(item.distortion).reshape(-1)],
                "T_world_camera": _transform_to_list(item.T_world_camera, f"camera {item.camera_id}"),
            }
            for item in spec.cameras
        ],
    }


def scene_spec_from_document(document: dict[str, Any]) -> SceneSpec:
    """Rebuild a scene spec, refusing anything that is not exactly the schema."""

    if not isinstance(document, dict):
        raise ConfigError("scene document must be a mapping")
    if document.get("schema") != SCENE_SPEC_SCHEMA_V1:
        raise ConfigError(f"unsupported scene document schema: {document.get('schema')!r}")
    try:
        objects = [
            SceneObjectSpec(
                object_id=item["object_id"],
                asset_ref=item["asset_ref"],
                T_world_object=_transform_from_list(item["T_world_object"], f"object {item['object_id']}"),
                scale=float(item.get("scale", 1.0)),
                mass=None if item.get("mass") is None else float(item["mass"]),
                friction=None if item.get("friction") is None else tuple(float(v) for v in item["friction"]),
            )
            for item in document["objects"]
        ]
        supports = [
            SupportGeometrySpec(
                support_id=item["support_id"],
                geom_type=item["geom_type"],
                params=item["params"],
                T_world_support=_transform_from_list(item["T_world_support"], f"support {item['support_id']}"),
            )
            for item in document["supports"]
        ]
        cameras = [
            CameraSpec(
                camera_id=item["camera_id"],
                intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
                distortion=None if item.get("distortion") is None else np.asarray(item["distortion"], dtype=np.float64),
                T_world_camera=_transform_from_list(item["T_world_camera"], f"camera {item['camera_id']}"),
            )
            for item in document.get("cameras", [])
        ]
        return SceneSpec(
            scene_id=document["scene_id"],
            source_dataset=document["source_dataset"],
            source_version=document["source_version"],
            source_split=document["source_split"],
            environment=document["environment"],
            objects=objects,
            supports=supports,
            cameras=cameras,
            gravity=tuple(float(value) for value in document["gravity"]),  # type: ignore[arg-type]
            timestep=float(document["timestep"]),
            solver_profile=document.get("solver_profile", "default"),
            settle_seed=int(document.get("settle_seed", 0)),
            source_record_hash=document.get("source_record_hash"),
            license_record=document.get("license_record"),
            redistributable=bool(document.get("redistributable", False)),
        )
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"scene document is malformed: {type(error).__name__}: {error}") from error


def scene_spec_hash(spec: SceneSpec) -> str:
    payload = json.dumps(scene_spec_to_document(spec), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_scene_spec(path: str | Path, spec: SceneSpec) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(scene_spec_to_document(spec), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_scene_spec(path: str | Path) -> SceneSpec:
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigError(f"scene document not found: {resolved}")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"scene document {resolved} is not valid JSON: {error}") from error
    return scene_spec_from_document(document)
