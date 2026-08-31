"""Procedural object generation, manifest tracking, and collision guards."""

from __future__ import annotations

from .coacd import (
    CoACDAlgorithmConfig,
    CoACDConfig,
    CoACDError,
    CoACDExecutionConfig,
    CoACDExecutionError,
    CoACDResult,
    CollisionAsset,
    CollisionValidationError,
    MeshPreprocessConfig,
    MeshRepairUnavailable,
    MeshValidationError,
    TooManyConvexPartsError,
    build_collision_asset,
    decompose_collision_mesh,
)
from .collision import validate_collision_representation
from .generate import (
    generate_box,
    generate_capsule,
    generate_compound_convex,
    generate_cylinder,
    generate_sphere,
    generate_superquadric,
)
from .ingest import (
    AssetIngestError,
    AssetIngestRequest,
    IngestErrorCode,
    IngestResult,
    IngestStatus,
    NormalizationConfig,
    PhysicsProperties,
    ingest_asset,
    normalize_mesh,
)
from .manifest import (
    create_object_asset,
    export_mesh_deterministic_obj,
    load_object_asset,
    save_object_asset,
    sha256_bytes,
)
from .manifest_v2 import (
    DecompositionRecord,
    ObjectAssetManifestV2,
    load_object_asset_manifest_v2,
    write_object_asset_manifest_v2,
)
from .schema import ObjectManifestSpec, SubGeomSpec

__all__ = (
    "AssetIngestError",
    "AssetIngestRequest",
    "CoACDAlgorithmConfig",
    "CoACDConfig",
    "CoACDError",
    "CoACDExecutionConfig",
    "CoACDExecutionError",
    "CoACDResult",
    "CollisionAsset",
    "CollisionValidationError",
    "DecompositionRecord",
    "IngestErrorCode",
    "IngestResult",
    "IngestStatus",
    "MeshPreprocessConfig",
    "MeshRepairUnavailable",
    "MeshValidationError",
    "NormalizationConfig",
    "ObjectAssetManifestV2",
    "ObjectManifestSpec",
    "PhysicsProperties",
    "SubGeomSpec",
    "TooManyConvexPartsError",
    "build_collision_asset",
    "create_object_asset",
    "decompose_collision_mesh",
    "export_mesh_deterministic_obj",
    "generate_box",
    "generate_capsule",
    "generate_compound_convex",
    "generate_cylinder",
    "generate_sphere",
    "generate_superquadric",
    "ingest_asset",
    "load_object_asset",
    "load_object_asset_manifest_v2",
    "normalize_mesh",
    "save_object_asset",
    "sha256_bytes",
    "validate_collision_representation",
    "write_object_asset_manifest_v2",
)
