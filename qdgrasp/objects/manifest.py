"""Serialization, hashing, and loading of procedural object assets and manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import trimesh

from ..config.schema import ConfigError
from .collision import validate_collision_representation
from .schema import ObjectManifestSpec, SubGeomSpec


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hexadecimal digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def export_mesh_deterministic_obj(mesh: trimesh.Trimesh) -> bytes:
    """Export a trimesh to Wavefront OBJ format with bit-exact reproducibility."""
    lines: list[str] = ["# QDGrasp Procedural Object Asset", "# License: CC0-1.0"]

    # Vertices formatted with fixed precision
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    # Faces (1-indexed)
    for f in mesh.faces:
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")

    return "\n".join(lines).encode("utf-8") + b"\n"


def create_object_asset(
    object_id: str,
    family: str,
    shape_type: str,
    mesh: trimesh.Trimesh,
    collision_geoms: Sequence[SubGeomSpec],
    params: dict[str, Any],
    mass: float,
    inertia: tuple[float, float, float],
    license: str = "CC0-1.0",
) -> tuple[bytes, ObjectManifestSpec]:
    """Create deterministic OBJ mesh bytes and complete validated manifest."""
    validate_collision_representation(mesh, collision_geoms)

    mesh_bytes = export_mesh_deterministic_obj(mesh)
    mesh_sha256 = sha256_bytes(mesh_bytes)

    bounds = mesh.bounds
    bounding_box = (
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[0][2]),
        float(bounds[1][0]),
        float(bounds[1][1]),
        float(bounds[1][2]),
    )

    manifest = ObjectManifestSpec(
        object_id=object_id,
        family=family,
        shape_type=shape_type,
        params=params,
        mesh_filename=f"{object_id}.obj",
        mesh_sha256=mesh_sha256,
        mass=float(mass),
        inertia=(float(inertia[0]), float(inertia[1]), float(inertia[2])),
        bounding_box=bounding_box,
        collision_geoms=tuple(collision_geoms),
        license=license,
    )
    return mesh_bytes, manifest


def save_object_asset(
    mesh_bytes: bytes,
    manifest: ObjectManifestSpec,
    output_dir: Path,
) -> Path:
    """Save the mesh OBJ and its JSON manifest into output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / manifest.mesh_filename
    manifest_path = output_dir / f"{manifest.object_id}.manifest.json"

    mesh_path.write_bytes(mesh_bytes)
    manifest_data = manifest.model_dump(by_alias=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def load_object_asset(manifest_path: Path) -> tuple[trimesh.Trimesh, ObjectManifestSpec]:
    """Load and cryptographically verify an object asset from disk."""
    if not manifest_path.is_file():
        raise ConfigError(f"object manifest not found: {manifest_path}")

    raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ObjectManifestSpec.model_validate(raw_data)

    mesh_path = manifest_path.parent / manifest.mesh_filename
    if not mesh_path.is_file():
        raise ConfigError(f"mesh file missing for manifest: {mesh_path}")

    mesh_bytes = mesh_path.read_bytes()
    actual_hash = sha256_bytes(mesh_bytes)
    if actual_hash != manifest.mesh_sha256:
        raise ConfigError(
            f"mesh integrity failure for {mesh_path}: "
            f"expected {manifest.mesh_sha256}, got {actual_hash}"
        )

    mesh = trimesh.load(mesh_path, file_type="obj", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ConfigError(f"failed to load Trimesh from {mesh_path}")

    return mesh, manifest
