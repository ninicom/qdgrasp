"""Procedural object generation, manifest tracking, and collision guards."""

from __future__ import annotations

from .collision import validate_collision_representation
from .generate import (
    generate_box,
    generate_capsule,
    generate_compound_convex,
    generate_cylinder,
    generate_sphere,
    generate_superquadric,
)
from .manifest import (
    create_object_asset,
    export_mesh_deterministic_obj,
    load_object_asset,
    save_object_asset,
    sha256_bytes,
)
from .schema import ObjectManifestSpec, SubGeomSpec

__all__ = (
    "ObjectManifestSpec",
    "SubGeomSpec",
    "create_object_asset",
    "export_mesh_deterministic_obj",
    "generate_box",
    "generate_capsule",
    "generate_compound_convex",
    "generate_cylinder",
    "generate_sphere",
    "generate_superquadric",
    "load_object_asset",
    "save_object_asset",
    "sha256_bytes",
    "validate_collision_representation",
)
