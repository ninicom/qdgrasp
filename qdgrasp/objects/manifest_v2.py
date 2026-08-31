"""Immutable ``ObjectAssetManifestV2`` for ingested assets (P3.5-05).

V1 describes a procedural object the project generated and therefore already
knows everything about.  V2 describes an asset that came from *outside*: it has
to carry the input hash, the exact transforms that took it to metres, the
collision representation it ended up with, the tool and config that produced
that representation, and a license record that was supplied rather than assumed.

The manifest is written once and never edited.  Rewriting it under the same ID
is refused rather than silently overwritten, because a manifest that can change
is not evidence of anything.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from qdgrasp.objects.coacd import CoACDResult
from qdgrasp.objects.ingest import IngestResult, IngestStatus

OBJECT_ASSET_MANIFEST_V2 = "qdgrasp/object-asset-manifest/v2"


class ManifestImmutabilityError(RuntimeError):
    """Raised when a write would replace an existing manifest."""


@dataclasses.dataclass(frozen=True)
class DecompositionRecord:
    """Provenance of the collision representation, or its absence."""

    method: str
    profile: str | None = None
    tool_version: str | None = None
    config_hash: str | None = None
    part_sha256: tuple[str, ...] = ()
    piece_count: int = 0
    total_volume_m3: float = 0.0
    platform_tag: str | None = None
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_coacd(cls, result: CoACDResult) -> DecompositionRecord:
        return cls(
            method="coacd",
            profile=result.profile,
            tool_version=result.tool_version,
            config_hash=result.config_hash,
            part_sha256=tuple(result.part_sha256),
            piece_count=result.piece_count,
            total_volume_m3=result.total_volume_m3,
            platform_tag=result.platform_tag,
            warnings=tuple(result.warnings),
        )

    def to_document(self) -> dict[str, Any]:
        document = dataclasses.asdict(self)
        document["part_sha256"] = list(self.part_sha256)
        document["warnings"] = list(self.warnings)
        return document


@dataclasses.dataclass(frozen=True)
class ObjectAssetManifestV2:
    """One ingested asset, described completely enough to be re-derived."""

    object_id: str
    status: str
    license_record: str
    redistributable: bool
    input_sha256: str
    normalized_sha256: str
    scale_to_meters: float
    transforms: tuple[dict[str, Any], ...]
    triangle_count: int
    vertex_count: int
    extents_m: tuple[float, float, float]
    volume_m3: float
    is_convex: bool
    watertight: bool
    mass_properties: dict[str, Any]
    collision: dict[str, Any]
    decomposition: DecompositionRecord
    ingest_request_hash: str
    schema_version: str = OBJECT_ASSET_MANIFEST_V2

    @classmethod
    def from_ingest(
        cls,
        result: IngestResult,
        *,
        decomposition: DecompositionRecord | None = None,
    ) -> ObjectAssetManifestV2:
        geometry = result.geometry
        return cls(
            object_id=result.object_id,
            status=result.status.value,
            license_record=result.license_record,
            redistributable=result.redistributable,
            input_sha256=geometry.input_sha256,
            normalized_sha256=geometry.normalized_sha256,
            scale_to_meters=geometry.scale_to_meters,
            transforms=tuple(record.to_document() for record in geometry.transforms),
            triangle_count=geometry.triangle_count,
            vertex_count=geometry.vertex_count,
            extents_m=geometry.extents_m,
            volume_m3=geometry.volume_m3,
            is_convex=geometry.is_convex,
            watertight=geometry.watertight,
            mass_properties=result.mass_properties,
            collision=result.collision,
            decomposition=decomposition
            or DecompositionRecord(
                method=str(result.collision.get("source", "unknown")),
                piece_count=int(result.collision.get("part_count", 0)),
                total_volume_m3=float(result.collision.get("total_volume_m3", 0.0)),
            ),
            ingest_request_hash=result.request_hash,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "object_id": self.object_id,
            "status": self.status,
            "license_record": self.license_record,
            "redistributable": self.redistributable,
            "input_sha256": self.input_sha256,
            "normalized_sha256": self.normalized_sha256,
            "scale_to_meters": self.scale_to_meters,
            "transforms": list(self.transforms),
            "triangle_count": self.triangle_count,
            "vertex_count": self.vertex_count,
            "extents_m": list(self.extents_m),
            "volume_m3": self.volume_m3,
            "is_convex": self.is_convex,
            "watertight": self.watertight,
            "mass_properties": self.mass_properties,
            "collision": self.collision,
            "decomposition": self.decomposition.to_document(),
            "ingest_request_hash": self.ingest_request_hash,
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def is_dynamic_ready(self) -> bool:
        return self.status == IngestStatus.DYNAMIC_READY.value

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> ObjectAssetManifestV2:
        if document.get("schema") != OBJECT_ASSET_MANIFEST_V2:
            raise ValueError(f"unsupported object asset manifest schema: {document.get('schema')!r}")
        decomposition = dict(document["decomposition"])
        decomposition["part_sha256"] = tuple(decomposition.get("part_sha256", ()))
        decomposition["warnings"] = tuple(decomposition.get("warnings", ()))
        return cls(
            object_id=document["object_id"],
            status=document["status"],
            license_record=document["license_record"],
            redistributable=bool(document["redistributable"]),
            input_sha256=document["input_sha256"],
            normalized_sha256=document["normalized_sha256"],
            scale_to_meters=float(document["scale_to_meters"]),
            transforms=tuple(document["transforms"]),
            triangle_count=int(document["triangle_count"]),
            vertex_count=int(document["vertex_count"]),
            extents_m=tuple(float(value) for value in document["extents_m"]),  # type: ignore[arg-type]
            volume_m3=float(document["volume_m3"]),
            is_convex=bool(document["is_convex"]),
            watertight=bool(document["watertight"]),
            mass_properties=document["mass_properties"],
            collision=document["collision"],
            decomposition=DecompositionRecord(**decomposition),
            ingest_request_hash=document["ingest_request_hash"],
        )


def write_object_asset_manifest_v2(
    path: str | Path, manifest: ObjectAssetManifestV2, *, overwrite: bool = False
) -> Path:
    """Write the manifest, refusing to replace one that already exists."""

    target = Path(path)
    if target.exists() and not overwrite:
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing == manifest.to_document():
            return target
        raise ManifestImmutabilityError(
            f"{target} already holds a different manifest; write a new ID rather than editing this one"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_object_asset_manifest_v2(path: str | Path) -> ObjectAssetManifestV2:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"object asset manifest not found: {resolved}")
    return ObjectAssetManifestV2.from_document(json.loads(resolved.read_text(encoding="utf-8")))
