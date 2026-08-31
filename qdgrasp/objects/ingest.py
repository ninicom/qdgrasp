"""Raw mesh ingestion into simulation-ready object assets (P3.5-01/02).

The plan's rule for this module is short and it is the whole design: *no
repairing until it passes*.  Every transform is recorded as a derivative with
its own hash, the source bytes are never touched, and anything the pipeline
cannot establish -- units, mass, license -- is a refusal rather than a guess.

Three consequences show up throughout.

The raw bytes are hashed **before** anything reads them, so the identity of an
input never depends on a loader's interpretation of it.

Units are required.  A mesh whose file format does not declare a unit is not
"probably millimetres"; it is an ingest that fails until someone says.  The one
scale factor derived from the declared unit is applied exactly once and recorded,
because applying it twice is the classic silent failure of asset pipelines.

Mass and density are separable from geometry.  A mesh with neither cannot become
a dynamic body, but it is still a perfectly good ``geometry_ready`` asset, and
the pipeline says so instead of inventing a density.

The module deliberately lives beside the object manifest and collision code
rather than under ``qdgrasp/assets/``, which is a data directory the wheel audit
pins by exact path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh

INGEST_SCHEMA_V1 = "qdgrasp/asset-ingest/v1"

#: Mesh containers the ingest accepts.  Anything else is a refusal, not a guess.
SourceFormat = Literal["obj", "ply", "stl", "glb"]

#: Declared unit of the source geometry.  ``explicit_scale`` means the caller
#: supplies ``scale_to_meters`` directly.
Units = Literal["m", "cm", "mm", "explicit_scale"]

#: How the collision representation is obtained.
CollisionPolicy = Literal["existing", "convex_if_possible", "coacd"]

_UNIT_SCALE: dict[str, float] = {"m": 1.0, "cm": 0.01, "mm": 0.001}

_AXES: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


class IngestStatus(str, Enum):
    """How far an ingest got, stated positively rather than as a pass/fail bit."""

    #: Geometry is normalised and valid, but no mass or density was supplied, so
    #: the asset may not be spawned as a dynamic body.
    GEOMETRY_READY = "geometry_ready"
    #: Geometry plus mass properties: the asset can become a dynamic body.
    DYNAMIC_READY = "dynamic_ready"


class IngestErrorCode(str, Enum):
    """Every way this pipeline is allowed to say no."""

    SOURCE_AMBIGUOUS = "source_ambiguous"
    SOURCE_MISSING = "source_missing"
    PATH_ESCAPES_ROOT = "path_escapes_root"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNIT_UNDECLARED = "unit_undeclared"
    SCALE_CONFLICT = "scale_conflict"
    MESH_UNREADABLE = "mesh_unreadable"
    MESH_EMPTY = "mesh_empty"
    NON_FINITE_GEOMETRY = "non_finite_geometry"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    TRIANGLE_BUDGET_EXCEEDED = "triangle_budget_exceeded"
    BOUNDS_OUT_OF_RANGE = "bounds_out_of_range"
    DISCONNECTED_COMPONENTS = "disconnected_components"
    LICENSE_MISSING = "license_missing"
    MASS_PROPERTIES_INVALID = "mass_properties_invalid"
    COLLISION_UNAVAILABLE = "collision_unavailable"


class AssetIngestError(ValueError):
    """A typed refusal.  The code is part of the contract, not the message."""

    def __init__(self, code: IngestErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


@dataclasses.dataclass(frozen=True)
class PhysicsProperties:
    """Mass properties, all optional, none inferred silently."""

    mass: float | None = None
    density: float | None = None
    center_of_mass: tuple[float, float, float] | None = None
    inertia: tuple[float, float, float] | None = None
    friction: tuple[float, float, float] | None = None
    restitution: float | None = None

    def validate(self) -> None:
        for name in ("mass", "density", "restitution"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                if name == "restitution" and value == 0.0:
                    continue
                raise AssetIngestError(
                    IngestErrorCode.MASS_PROPERTIES_INVALID, f"{name} must be finite and positive, got {value!r}"
                )
        for name in ("center_of_mass", "inertia", "friction"):
            value = getattr(self, name)
            if value is not None and (len(value) != 3 or not np.all(np.isfinite(value))):
                raise AssetIngestError(
                    IngestErrorCode.MASS_PROPERTIES_INVALID, f"{name} must be three finite numbers, got {value!r}"
                )
        if self.inertia is not None and any(value <= 0.0 for value in self.inertia):
            raise AssetIngestError(IngestErrorCode.MASS_PROPERTIES_INVALID, "inertia diagonal must be positive")


@dataclasses.dataclass(frozen=True)
class NormalizationConfig:
    """The cleaning budget, hashed into the derivative record.

    These are limits, not repairs: exceeding one is a refusal.  The only edits
    the pipeline makes are merging exactly-duplicate vertices and dropping
    zero-area faces, both of which are recorded with counts.
    """

    merge_duplicate_vertices: bool = True
    drop_degenerate_faces: bool = True
    #: Faces below this area (m^2) count as degenerate.
    degenerate_face_area_m2: float = 1e-12
    max_triangles: int = 500_000
    #: Rejected outside this range, in metres, on the longest bounding-box edge.
    min_extent_m: float = 0.001
    max_extent_m: float = 2.0
    #: More separated components than this and the "object" is a scene.
    max_connected_components: int = 1

    def content_hash(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class AssetIngestRequest:
    """One ingest, fully specified.  Exactly one source, and units declared."""

    object_id: str
    license_record: str
    redistributable: bool
    source_format: SourceFormat
    units: Units
    manifest_ref: str | None = None
    local_mesh_path: str | Path | None = None
    mesh_bytes: bytes | None = None
    scale_to_meters: float | None = None
    up_axis: str = "+z"
    forward_axis: str = "+x"
    source_frame: str = "object"
    physics: PhysicsProperties = dataclasses.field(default_factory=PhysicsProperties)
    collision_policy: CollisionPolicy = "convex_if_possible"
    normalization: NormalizationConfig = dataclasses.field(default_factory=NormalizationConfig)

    def sources(self) -> list[str]:
        present = []
        if self.manifest_ref is not None:
            present.append("manifest_ref")
        if self.local_mesh_path is not None:
            present.append("local_mesh_path")
        if self.mesh_bytes is not None:
            present.append("mesh_bytes")
        return present

    def resolved_scale(self) -> float:
        """The single scale factor to metres, applied exactly once."""

        if self.units == "explicit_scale":
            if self.scale_to_meters is None:
                raise AssetIngestError(
                    IngestErrorCode.UNIT_UNDECLARED, "units='explicit_scale' requires scale_to_meters"
                )
            return float(self.scale_to_meters)
        declared = _UNIT_SCALE[self.units]
        if self.scale_to_meters is not None and not np.isclose(self.scale_to_meters, declared):
            raise AssetIngestError(
                IngestErrorCode.SCALE_CONFLICT,
                f"units={self.units!r} implies {declared} but scale_to_meters={self.scale_to_meters}",
            )
        return declared

    def validate(self) -> None:
        present = self.sources()
        if len(present) > 1:
            raise AssetIngestError(IngestErrorCode.SOURCE_AMBIGUOUS, f"more than one source given: {present}")
        if not present:
            raise AssetIngestError(IngestErrorCode.SOURCE_MISSING, "exactly one source is required")
        if self.source_format not in ("obj", "ply", "stl", "glb"):
            raise AssetIngestError(IngestErrorCode.UNSUPPORTED_FORMAT, f"source_format={self.source_format!r}")
        if self.units not in ("m", "cm", "mm", "explicit_scale"):
            raise AssetIngestError(IngestErrorCode.UNIT_UNDECLARED, f"units={self.units!r}")
        if not self.license_record.strip():
            raise AssetIngestError(IngestErrorCode.LICENSE_MISSING, "license_record must not be empty")
        for axis_name, axis in (("up_axis", self.up_axis), ("forward_axis", self.forward_axis)):
            if axis not in _AXES:
                raise AssetIngestError(
                    IngestErrorCode.UNSUPPORTED_FORMAT, f"{axis_name}={axis!r} is not one of {sorted(_AXES)}"
                )
        if _AXES[self.up_axis] @ _AXES[self.forward_axis] != 0.0:
            raise AssetIngestError(IngestErrorCode.UNSUPPORTED_FORMAT, "up_axis and forward_axis must be orthogonal")
        self.physics.validate()
        self.resolved_scale()

    def request_hash(self) -> str:
        """Hash of the request, with the raw bytes represented by their digest."""

        payload = {
            "schema": INGEST_SCHEMA_V1,
            "object_id": self.object_id,
            "license_record": self.license_record,
            "redistributable": self.redistributable,
            "source_format": self.source_format,
            "units": self.units,
            "manifest_ref": self.manifest_ref,
            "local_mesh_path": str(self.local_mesh_path) if self.local_mesh_path is not None else None,
            "mesh_bytes_sha256": hashlib.sha256(self.mesh_bytes).hexdigest() if self.mesh_bytes else None,
            "scale_to_meters": self.scale_to_meters,
            "up_axis": self.up_axis,
            "forward_axis": self.forward_axis,
            "source_frame": self.source_frame,
            "physics": dataclasses.asdict(self.physics),
            "collision_policy": self.collision_policy,
            "normalization": dataclasses.asdict(self.normalization),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class TransformRecord:
    """One reproducible edit, with what it changed and by how much."""

    name: str
    detail: dict[str, Any]

    def to_document(self) -> dict[str, Any]:
        return {"name": self.name, "detail": self.detail}


@dataclasses.dataclass
class NormalizedGeometry:
    """The visual mesh after normalisation, plus how it got there."""

    mesh: trimesh.Trimesh
    input_sha256: str
    normalized_sha256: str
    scale_to_meters: float
    transforms: list[TransformRecord]
    triangle_count: int
    vertex_count: int
    extents_m: tuple[float, float, float]
    volume_m3: float
    is_convex: bool
    watertight: bool

    def to_document(self) -> dict[str, Any]:
        return {
            "input_sha256": self.input_sha256,
            "normalized_sha256": self.normalized_sha256,
            "scale_to_meters": self.scale_to_meters,
            "transforms": [record.to_document() for record in self.transforms],
            "triangle_count": self.triangle_count,
            "vertex_count": self.vertex_count,
            "extents_m": list(self.extents_m),
            "volume_m3": self.volume_m3,
            "is_convex": self.is_convex,
            "watertight": self.watertight,
        }


def read_source_bytes(request: AssetIngestRequest, *, allowed_root: str | Path | None = None) -> bytes:
    """Return the raw source bytes, refusing anything outside ``allowed_root``.

    The sandbox check resolves symlinks before comparing, so a link pointing out
    of the root is caught rather than followed.
    """

    request.validate()
    if request.mesh_bytes is not None:
        return request.mesh_bytes
    if request.local_mesh_path is None:
        raise AssetIngestError(
            IngestErrorCode.SOURCE_MISSING, "manifest_ref ingestion requires the manifest loader, not raw bytes"
        )
    path = Path(request.local_mesh_path).resolve()
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        if not path.is_relative_to(root):
            raise AssetIngestError(IngestErrorCode.PATH_ESCAPES_ROOT, f"{path} is outside {root}")
    if not path.is_file():
        raise AssetIngestError(IngestErrorCode.SOURCE_MISSING, f"mesh file not found: {path}")
    return path.read_bytes()


def _load_mesh(raw: bytes, source_format: str) -> trimesh.Trimesh:
    import io

    try:
        loaded = trimesh.load(io.BytesIO(raw), file_type=source_format, force="mesh", process=False)
    except Exception as error:
        raise AssetIngestError(IngestErrorCode.MESH_UNREADABLE, f"{type(error).__name__}: {error}") from error
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise AssetIngestError(IngestErrorCode.MESH_EMPTY, "scene contains no geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise AssetIngestError(IngestErrorCode.MESH_UNREADABLE, f"loader returned {type(loaded).__name__}")
    if loaded.faces.shape[0] == 0 or loaded.vertices.shape[0] == 0:
        raise AssetIngestError(IngestErrorCode.MESH_EMPTY, "mesh has no faces or no vertices")
    return loaded


def _orientation_matrix(up_axis: str, forward_axis: str) -> np.ndarray:
    """Rotation taking the declared source axes onto ``+z`` up, ``+x`` forward."""

    up = _AXES[up_axis]
    forward = _AXES[forward_axis]
    left = np.cross(up, forward)
    source = np.stack([forward, left, up])
    return source  # rows are the source axes expressed in the target frame


def normalize_mesh(request: AssetIngestRequest, raw: bytes) -> NormalizedGeometry:
    """Hash, load, reorient, scale and clean a source mesh into metres."""

    request.validate()
    input_sha256 = hashlib.sha256(raw).hexdigest()
    mesh = _load_mesh(raw, request.source_format)
    transforms: list[TransformRecord] = []

    if mesh.faces.shape[1] != 3:
        raise AssetIngestError(IngestErrorCode.MESH_UNREADABLE, "faces are not triangles after load")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if not np.all(np.isfinite(vertices)):
        raise AssetIngestError(IngestErrorCode.NON_FINITE_GEOMETRY, "source vertices contain NaN or Inf")

    rotation = _orientation_matrix(request.up_axis, request.forward_axis)
    if not np.allclose(rotation, np.eye(3)):
        vertices = vertices @ rotation.T
        transforms.append(
            TransformRecord("reorient", {"up_axis": request.up_axis, "forward_axis": request.forward_axis})
        )

    scale = request.resolved_scale()
    if scale != 1.0:
        vertices = vertices * scale
        transforms.append(TransformRecord("scale_to_meters", {"factor": scale}))

    working = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)

    config = request.normalization
    if config.merge_duplicate_vertices:
        before = int(working.vertices.shape[0])
        working.merge_vertices()
        removed = before - int(working.vertices.shape[0])
        if removed:
            transforms.append(TransformRecord("merge_duplicate_vertices", {"removed": removed}))
    if config.drop_degenerate_faces:
        areas = working.area_faces
        keep = areas > config.degenerate_face_area_m2
        removed = int((~keep).sum())
        if removed:
            working.update_faces(keep)
            working.remove_unreferenced_vertices()
            transforms.append(
                TransformRecord(
                    "drop_degenerate_faces",
                    {"removed": removed, "area_threshold_m2": config.degenerate_face_area_m2},
                )
            )
    if working.faces.shape[0] == 0:
        raise AssetIngestError(IngestErrorCode.DEGENERATE_GEOMETRY, "every face was degenerate")

    triangles = int(working.faces.shape[0])
    if triangles > config.max_triangles:
        raise AssetIngestError(
            IngestErrorCode.TRIANGLE_BUDGET_EXCEEDED, f"{triangles} triangles exceeds {config.max_triangles}"
        )
    extents = tuple(float(value) for value in working.extents)
    longest = max(extents)
    if not np.isfinite(longest) or longest < config.min_extent_m or longest > config.max_extent_m:
        raise AssetIngestError(
            IngestErrorCode.BOUNDS_OUT_OF_RANGE,
            f"longest extent {longest:.6f} m outside [{config.min_extent_m}, {config.max_extent_m}]",
        )
    components = working.split(only_watertight=False)
    if len(components) > config.max_connected_components:
        raise AssetIngestError(
            IngestErrorCode.DISCONNECTED_COMPONENTS,
            f"{len(components)} components exceeds {config.max_connected_components}",
        )

    from qdgrasp.objects.manifest import export_mesh_deterministic_obj

    normalized_bytes = export_mesh_deterministic_obj(working)
    return NormalizedGeometry(
        mesh=working,
        input_sha256=input_sha256,
        normalized_sha256=hashlib.sha256(normalized_bytes).hexdigest(),
        scale_to_meters=scale,
        transforms=transforms,
        triangle_count=triangles,
        vertex_count=int(working.vertices.shape[0]),
        extents_m=extents,  # type: ignore[arg-type]
        volume_m3=float(abs(working.volume)),
        is_convex=bool(working.is_convex),
        watertight=bool(working.is_watertight),
    )


def derive_mass_properties(
    geometry: NormalizedGeometry, physics: PhysicsProperties
) -> tuple[IngestStatus, dict[str, Any]]:
    """Resolve mass and inertia, saying which numbers were derived and how.

    An asset with neither mass nor density stops at ``geometry_ready``.  That is
    a state, not a failure: the geometry is usable for collision queries and
    rendering, and only spawning it as a dynamic body is withheld.
    """

    physics.validate()
    if physics.mass is None and physics.density is None:
        return IngestStatus.GEOMETRY_READY, {
            "mass_kg": None,
            "density_kg_m3": None,
            "inertia_kg_m2": None,
            "center_of_mass_m": None,
            "derived": [],
            "reason": "neither mass nor density was supplied",
        }

    derived: list[str] = []
    volume = geometry.volume_m3
    if physics.mass is not None:
        mass = float(physics.mass)
        density = mass / volume if volume > 0.0 else None
        if density is not None:
            derived.append("density_from_mass_and_volume")
    else:
        if volume <= 0.0:
            raise AssetIngestError(
                IngestErrorCode.MASS_PROPERTIES_INVALID,
                "density was supplied but the mesh volume is not positive, so mass cannot be derived",
            )
        density = float(physics.density)  # type: ignore[arg-type]
        mass = density * volume
        derived.append("mass_from_density_and_volume")

    if physics.center_of_mass is not None:
        center = tuple(float(value) for value in physics.center_of_mass)
    else:
        center = tuple(float(value) for value in geometry.mesh.center_mass)
        derived.append("center_of_mass_from_geometry")

    if physics.inertia is not None:
        inertia = tuple(float(value) for value in physics.inertia)
    else:
        # trimesh reports the inertia tensor at unit density; scaling by the
        # resolved density keeps the two consistent by construction.
        tensor = np.asarray(geometry.mesh.moment_inertia, dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(tensor)
        if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
            raise AssetIngestError(
                IngestErrorCode.MASS_PROPERTIES_INVALID,
                "geometry does not yield a positive-definite inertia tensor",
            )
        scale = mass / max(volume, 1e-18)
        inertia = tuple(float(value * scale) for value in eigenvalues)
        derived.append("inertia_from_geometry_and_mass")

    return IngestStatus.DYNAMIC_READY, {
        "mass_kg": mass,
        "density_kg_m3": density,
        "inertia_kg_m2": list(inertia),
        "center_of_mass_m": list(center),
        "derived": derived,
        "reason": None,
    }


@dataclasses.dataclass
class IngestResult:
    """Everything one ingest produced, including what it refused to decide."""

    request_hash: str
    object_id: str
    status: IngestStatus
    geometry: NormalizedGeometry
    mass_properties: dict[str, Any]
    collision: dict[str, Any]
    license_record: str
    redistributable: bool

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": INGEST_SCHEMA_V1,
            "request_hash": self.request_hash,
            "object_id": self.object_id,
            "status": self.status.value,
            "geometry": self.geometry.to_document(),
            "mass_properties": self.mass_properties,
            "collision": self.collision,
            "license_record": self.license_record,
            "redistributable": self.redistributable,
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ingest_asset(
    request: AssetIngestRequest,
    *,
    allowed_root: str | Path | None = None,
    raw: bytes | None = None,
    collision_parts: Sequence[trimesh.Trimesh] | None = None,
) -> IngestResult:
    """Run the normalisation pipeline for one request.

    ``collision_parts`` lets a caller supply an already-computed decomposition
    (from :mod:`qdgrasp.objects.coacd`), keeping this function free of any
    dependency on a decomposition backend being installed.
    """

    request.validate()
    source = raw if raw is not None else read_source_bytes(request, allowed_root=allowed_root)
    geometry = normalize_mesh(request, source)
    status, mass_properties = derive_mass_properties(geometry, request.physics)
    collision = _resolve_collision(request, geometry, collision_parts)
    return IngestResult(
        request_hash=request.request_hash(),
        object_id=request.object_id,
        status=status,
        geometry=geometry,
        mass_properties=mass_properties,
        collision=collision,
        license_record=request.license_record,
        redistributable=request.redistributable,
    )


def _resolve_collision(
    request: AssetIngestRequest,
    geometry: NormalizedGeometry,
    collision_parts: Sequence[trimesh.Trimesh] | None,
) -> dict[str, Any]:
    """Pick the collision representation the policy asks for, or refuse."""

    if request.collision_policy == "existing":
        if collision_parts is None:
            raise AssetIngestError(
                IngestErrorCode.COLLISION_UNAVAILABLE, "collision_policy='existing' requires supplied parts"
            )
        parts = list(collision_parts)
        source = "supplied"
    elif request.collision_policy == "convex_if_possible":
        if geometry.is_convex:
            # A convex mesh is already its own collision hull; decomposing it
            # would be work that produces the same shape with a new hash.
            parts = [geometry.mesh]
            source = "source_is_convex"
        elif collision_parts is not None:
            parts = list(collision_parts)
            source = "supplied"
        else:
            parts = [geometry.mesh.convex_hull]
            source = "convex_hull"
    else:  # coacd
        if collision_parts is None:
            raise AssetIngestError(
                IngestErrorCode.COLLISION_UNAVAILABLE,
                "collision_policy='coacd' requires a decomposition; call decompose_collision_mesh first",
            )
        parts = list(collision_parts)
        source = "coacd"

    if not parts:
        raise AssetIngestError(IngestErrorCode.COLLISION_UNAVAILABLE, "collision representation is empty")
    summaries = []
    for index, part in enumerate(parts):
        vertices = np.asarray(part.vertices, dtype=np.float64)
        if vertices.size == 0 or not np.all(np.isfinite(vertices)):
            raise AssetIngestError(
                IngestErrorCode.NON_FINITE_GEOMETRY, f"collision part {index} has empty or non-finite vertices"
            )
        summaries.append(
            {
                "index": index,
                "vertices": int(vertices.shape[0]),
                "faces": int(np.asarray(part.faces).shape[0]),
                "volume_m3": float(abs(part.volume)),
                "is_convex": bool(part.is_convex),
            }
        )
    return {
        "policy": request.collision_policy,
        "source": source,
        "part_count": len(parts),
        "parts": summaries,
        "total_volume_m3": float(sum(item["volume_m3"] for item in summaries)),
    }
