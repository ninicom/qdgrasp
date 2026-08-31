"""Public Python API for convex collision decomposition (P3.5-03/04).

This is a library call, not a service.  It never opens a socket, never writes to
a dataset, and never picks a mass for you.  ``decompose_collision_mesh`` takes a
mesh and a fully-specified config and returns a typed result; where that result
gets written is the manifest layer's decision, not this module's.

Three rules from the plan shape the implementation.

**No ``**kwargs`` passthrough.**  Every official CoACD parameter is a typed field
with range or enum validation, and every one of them appears in the config hash.
The mapping onto upstream argument names is an explicit table, and it is checked
against the *installed* release's signature before the call: a parameter the
installed CoACD does not accept raises :class:`CoACDExecutionError` naming it,
rather than being dropped on the floor.  A silently ignored parameter is worse
than a missing one, because the config hash would still claim it was applied.

**Provenance is named, never "the old default".**  The Stage 0 artifacts were
produced at threshold 0.1, seed 0, simplify 5000; the DexGraspNet standard uses
0.4.  Those are two different profiles with two different names, and neither is
allowed to be described as the historical default.

**Mass is the caller's.**  A mesh with no mass and no density still decomposes
perfectly well -- it just does not become a dynamic asset, and
:class:`CollisionAsset` says which of the two it is.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import trimesh

COACD_CONFIG_SCHEMA_V1 = "qdgrasp/coacd-config/v1"

#: Profiles the config may be built from.  ``custom`` means the caller set the
#: fields themselves and takes responsibility for the combination.
CoACDProfile = Literal[
    "upstream_default_v1",
    "qdgrasp_metric_v1",
    "legacy_kaggle_stage0_0_1_v1",
    "legacy_dexgraspnet_standard_0_4_v1",
    "custom",
]

PreprocessMode = Literal["auto", "on", "off"]
ApproximationMode = Literal["ch", "box"]
RepairMode = Literal["none", "manifoldplus"]


class CoACDError(RuntimeError):
    """Base for every typed refusal this module raises."""


class MeshValidationError(CoACDError):
    """The input mesh is not something a decomposition can be run on."""


class MeshRepairUnavailable(CoACDError):
    """A repair mode was requested and its backend is not installed."""


class CoACDExecutionError(CoACDError):
    """The decomposition backend is absent, mismatched, or failed."""


class TooManyConvexPartsError(CoACDError):
    """The decomposition produced more parts than the execution budget allows."""


class CollisionValidationError(CoACDError):
    """The produced parts are not usable as a collision representation."""


def _hash_document(document: Any) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclasses.dataclass(frozen=True)
class MeshPreprocessConfig:
    """What happens to the mesh before CoACD sees it."""

    unit_scale_to_meters: float = 1.0
    triangulate: bool = True
    repair_mode: RepairMode = "none"
    manifold_depth: int = 8
    simplify_faces: int | None = None
    #: Legacy pipelines normalise the bounding-box diagonal to a fixed length so
    #: that a unitless threshold means the same thing across objects.  It is
    #: mutually exclusive with ``real_metric``.
    normalize_diagonal_m: float | None = None

    def validate(self) -> None:
        if not np.isfinite(self.unit_scale_to_meters) or self.unit_scale_to_meters <= 0.0:
            raise MeshValidationError(
                f"unit_scale_to_meters must be finite and positive, got {self.unit_scale_to_meters!r}"
            )
        if self.repair_mode not in ("none", "manifoldplus"):
            raise MeshValidationError(f"repair_mode={self.repair_mode!r}")
        if not 1 <= self.manifold_depth <= 16:
            raise MeshValidationError(f"manifold_depth must be within [1, 16], got {self.manifold_depth}")
        if self.simplify_faces is not None and self.simplify_faces < 4:
            raise MeshValidationError(f"simplify_faces must be at least 4, got {self.simplify_faces}")
        if self.normalize_diagonal_m is not None and (
            not np.isfinite(self.normalize_diagonal_m) or self.normalize_diagonal_m <= 0.0
        ):
            raise MeshValidationError(f"normalize_diagonal_m must be positive, got {self.normalize_diagonal_m!r}")


@dataclasses.dataclass(frozen=True)
class CoACDAlgorithmConfig:
    """Every official CoACD parameter, typed and validated.

    Defaults follow the upstream signature audited at CoACD ``1.0.13``; the
    profiles below override them by name rather than by memory.
    """

    threshold: float = 0.05
    max_convex_hull: int = -1
    preprocess_mode: PreprocessMode = "auto"
    preprocess_resolution: int = 50
    resolution: int = 2000
    mcts_nodes: int = 20
    mcts_iterations: int = 150
    mcts_max_depth: int = 3
    pca: bool = False
    merge: bool = True
    decimate: bool = False
    max_ch_vertex: int = 256
    extrude: bool = False
    extrude_margin: float = 0.01
    apx_mode: ApproximationMode = "ch"
    seed: int = 0
    #: Interprets ``threshold`` in metres instead of in normalised units.
    real_metric: bool = False

    def validate(self) -> None:
        if not 0.01 <= self.threshold <= 1.0:
            raise MeshValidationError(f"threshold must be within [0.01, 1.0], got {self.threshold}")
        if self.max_convex_hull < -1 or self.max_convex_hull == 0:
            raise MeshValidationError(f"max_convex_hull must be -1 or a positive count, got {self.max_convex_hull}")
        if self.preprocess_mode not in ("auto", "on", "off"):
            raise MeshValidationError(f"preprocess_mode={self.preprocess_mode!r}")
        if not 20 <= self.preprocess_resolution <= 100:
            raise MeshValidationError(
                f"preprocess_resolution must be within [20, 100], got {self.preprocess_resolution}"
            )
        if not 100 <= self.resolution <= 20000:
            raise MeshValidationError(f"resolution must be within [100, 20000], got {self.resolution}")
        if not 10 <= self.mcts_nodes <= 40:
            raise MeshValidationError(f"mcts_nodes must be within [10, 40], got {self.mcts_nodes}")
        if not 60 <= self.mcts_iterations <= 2000:
            raise MeshValidationError(f"mcts_iterations must be within [60, 2000], got {self.mcts_iterations}")
        if not 2 <= self.mcts_max_depth <= 7:
            raise MeshValidationError(f"mcts_max_depth must be within [2, 7], got {self.mcts_max_depth}")
        if not 4 <= self.max_ch_vertex <= 2048:
            raise MeshValidationError(f"max_ch_vertex must be within [4, 2048], got {self.max_ch_vertex}")
        if not np.isfinite(self.extrude_margin) or self.extrude_margin <= 0.0:
            raise MeshValidationError(f"extrude_margin must be positive, got {self.extrude_margin!r}")
        if self.apx_mode not in ("ch", "box"):
            raise MeshValidationError(f"apx_mode={self.apx_mode!r}")
        if self.seed < 0:
            raise MeshValidationError(f"seed must be non-negative, got {self.seed}")


@dataclasses.dataclass(frozen=True)
class CoACDExecutionConfig:
    """Wrapper-level budgets.  None of these are forwarded to upstream."""

    log_level: Literal["off", "info", "warn", "error"] = "warn"
    timeout_s: float | None = 600.0
    max_input_vertices: int = 500_000
    max_input_faces: int = 1_000_000
    max_output_parts: int = 256
    max_workers: int = 1

    def validate(self) -> None:
        if self.log_level not in ("off", "info", "warn", "error"):
            raise MeshValidationError(f"log_level={self.log_level!r}")
        if self.timeout_s is not None and (not np.isfinite(self.timeout_s) or self.timeout_s <= 0.0):
            raise MeshValidationError(f"timeout_s must be positive, got {self.timeout_s!r}")
        for name in ("max_input_vertices", "max_input_faces", "max_output_parts", "max_workers"):
            value = getattr(self, name)
            if value < 1:
                raise MeshValidationError(f"{name} must be at least 1, got {value}")


#: Named profiles.  Each is a complete statement of provenance; none of them is
#: "the default".
_PROFILES: dict[str, dict[str, Any]] = {
    "upstream_default_v1": {
        "preprocess": {},
        "algorithm": {},
    },
    # Threshold in metres, no diagonal normalisation: the combination QDGrasp
    # uses for assets that already carry a real scale.
    "qdgrasp_metric_v1": {
        "preprocess": {"normalize_diagonal_m": None},
        "algorithm": {"threshold": 0.05, "real_metric": True, "seed": 0},
    },
    # Reproduces the Stage 0 artifacts exactly: threshold 0.1, seed 0,
    # simplify 5000, bounding-box diagonal normalised to 2 m.
    "legacy_kaggle_stage0_0_1_v1": {
        "preprocess": {"simplify_faces": 5000, "normalize_diagonal_m": 2.0, "repair_mode": "manifoldplus"},
        "algorithm": {"threshold": 0.1, "seed": 0, "real_metric": False},
    },
    # The DexGraspNet standard setting.  It is *not* what produced the Stage 0
    # artifacts and must not be used to reproduce them.
    "legacy_dexgraspnet_standard_0_4_v1": {
        "preprocess": {"simplify_faces": 5000, "normalize_diagonal_m": 2.0, "repair_mode": "manifoldplus"},
        "algorithm": {"threshold": 0.4, "seed": 0, "real_metric": False},
    },
}


@dataclasses.dataclass(frozen=True)
class CoACDConfig:
    """A complete, hashable decomposition configuration."""

    profile: CoACDProfile = "upstream_default_v1"
    mesh_preprocess: MeshPreprocessConfig = dataclasses.field(default_factory=MeshPreprocessConfig)
    algorithm: CoACDAlgorithmConfig = dataclasses.field(default_factory=CoACDAlgorithmConfig)
    execution: CoACDExecutionConfig = dataclasses.field(default_factory=CoACDExecutionConfig)

    @classmethod
    def profile_config(cls, name: CoACDProfile, **overrides: Any) -> CoACDConfig:
        """Build a config from a named profile, optionally overriding fields.

        Any override makes the profile ``custom``: a config that is not exactly
        the named profile must not keep its name, or reproducing an artifact
        "with profile X" would stop meaning anything.
        """

        if name == "custom":
            raise MeshValidationError("'custom' is what a profile becomes when overridden, not one to request")
        if name not in _PROFILES:
            raise MeshValidationError(f"unknown profile {name!r}; known: {sorted(_PROFILES)}")
        spec = _PROFILES[name]
        config = cls(
            profile=name,
            mesh_preprocess=dataclasses.replace(MeshPreprocessConfig(), **spec["preprocess"]),
            algorithm=dataclasses.replace(CoACDAlgorithmConfig(), **spec["algorithm"]),
        )
        if overrides:
            config = dataclasses.replace(config, profile="custom", **overrides)
        config.validate()
        return config

    #: Kept as an alias because the plan writes ``CoACDConfig.profile(...)``;
    #: the field of the same name makes that spelling impossible as a method.
    @classmethod
    def from_profile(cls, name: CoACDProfile, **overrides: Any) -> CoACDConfig:
        return cls.profile_config(name, **overrides)

    def validate(self) -> None:
        self.mesh_preprocess.validate()
        self.algorithm.validate()
        self.execution.validate()
        if self.algorithm.real_metric and self.mesh_preprocess.normalize_diagonal_m is not None:
            raise MeshValidationError(
                "real_metric=True defines the threshold in metres and cannot be combined with "
                "normalize_diagonal_m, which rescales the mesh out of metres"
            )
        if self.profile not in (*_PROFILES, "custom"):
            raise MeshValidationError(f"unknown profile {self.profile!r}")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": COACD_CONFIG_SCHEMA_V1,
            "profile": self.profile,
            "mesh_preprocess": dataclasses.asdict(self.mesh_preprocess),
            "algorithm": dataclasses.asdict(self.algorithm),
            "execution": dataclasses.asdict(self.execution),
        }

    def content_hash(self) -> str:
        return _hash_document(self.to_document())

    def algorithm_hash(self) -> str:
        return _hash_document(dataclasses.asdict(self.algorithm))


#: Mapping from this module's algorithm fields onto upstream argument names.
#: Explicit so that a rename upstream is a loud failure rather than a parameter
#: that quietly stops being applied.
UPSTREAM_ARGUMENT_NAMES: dict[str, str] = {
    "threshold": "threshold",
    "max_convex_hull": "max_convex_hull",
    "preprocess_mode": "preprocess_mode",
    "preprocess_resolution": "preprocess_resolution",
    "resolution": "resolution",
    "mcts_nodes": "mcts_nodes",
    "mcts_iterations": "mcts_iterations",
    "mcts_max_depth": "mcts_max_depth",
    "pca": "pca",
    "merge": "merge",
    "decimate": "decimate",
    "max_ch_vertex": "max_ch_vertex",
    "extrude": "extrude",
    "extrude_margin": "extrude_margin",
    "apx_mode": "apx_mode",
    "seed": "seed",
    "real_metric": "real_metric",
}


class CoacdBackend(Protocol):
    """The one upstream entry point this module calls."""

    def __call__(self, mesh: Any, **kwargs: Any) -> Sequence[tuple[Any, Any]]: ...


def _load_backend() -> tuple[CoacdBackend, str, Callable[..., Any]]:
    """Import the installed CoACD, or refuse with a typed error."""

    try:
        import coacd  # type: ignore[import-not-found]
    except ImportError as error:
        raise CoACDExecutionError(
            "the CoACD backend is not installed; `decompose_collision_mesh` needs it to produce parts. "
            "Install the audited release into this environment, or use collision_policy='convex_if_possible'."
        ) from error
    version = str(getattr(coacd, "__version__", "unknown"))
    return coacd.run_coacd, version, coacd.Mesh


def _check_upstream_signature(backend: Callable[..., Any], requested: Sequence[str]) -> None:
    """Refuse to run if the installed release does not accept every parameter."""

    try:
        signature = inspect.signature(backend)
    except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
        return
    accepted = set(signature.parameters)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return
    missing = sorted(name for name in requested if name not in accepted)
    if missing:
        raise CoACDExecutionError(
            "the installed CoACD does not accept these parameters, so applying them is impossible and "
            f"ignoring them would make the config hash a lie: {missing}"
        )


@dataclasses.dataclass
class CoACDResult:
    """One decomposition, with everything needed to reproduce or audit it."""

    convex_parts: list[trimesh.Trimesh]
    prepared_mesh: trimesh.Trimesh
    input_sha256: str
    prepared_sha256: str
    part_sha256: list[str]
    source_to_metric_transform: list[list[float]]
    config_hash: str
    profile: str
    tool_version: str
    platform_tag: str
    piece_count: int
    total_volume_m3: float
    elapsed_s: float
    warnings: list[str]
    cache_hit: bool = False

    def to_document(self) -> dict[str, Any]:
        document = dataclasses.asdict(self)
        document.pop("convex_parts")
        document.pop("prepared_mesh")
        return document

    def content_hash(self) -> str:
        return _hash_document(self.to_document())


@dataclasses.dataclass
class CollisionAsset:
    """A decomposition plus, when the caller supplied one, its mass properties."""

    result: CoACDResult
    mass_kg: float | None
    density_kg_m3: float | None
    inertia_kg_m2: list[float] | None
    center_of_mass_m: list[float] | None
    #: ``dynamic_ready`` only when mass could be resolved from caller input.
    status: Literal["geometry_ready", "dynamic_ready"]

    def to_document(self) -> dict[str, Any]:
        return {
            "result": self.result.to_document(),
            "mass_kg": self.mass_kg,
            "density_kg_m3": self.density_kg_m3,
            "inertia_kg_m2": self.inertia_kg_m2,
            "center_of_mass_m": self.center_of_mass_m,
            "status": self.status,
        }


def _as_mesh(mesh_or_path: trimesh.Trimesh | str | Path | bytes) -> tuple[trimesh.Trimesh, bytes]:
    """Accept a mesh, a path or raw bytes, and return the mesh with its bytes."""

    if isinstance(mesh_or_path, trimesh.Trimesh):
        from qdgrasp.objects.manifest import export_mesh_deterministic_obj

        return mesh_or_path, export_mesh_deterministic_obj(mesh_or_path)
    if isinstance(mesh_or_path, bytes):
        raw = mesh_or_path
        import io

        loaded = trimesh.load(io.BytesIO(raw), file_type="obj", force="mesh", process=False)
    else:
        path = Path(mesh_or_path)
        if not path.is_file():
            raise MeshValidationError(f"mesh file not found: {path}")
        raw = path.read_bytes()
        loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshValidationError(f"input did not load as a single mesh, got {type(loaded).__name__}")
    return loaded, raw


def _prepare_mesh(mesh: trimesh.Trimesh, config: CoACDConfig) -> tuple[trimesh.Trimesh, np.ndarray, list[str]]:
    """Apply the preprocessing the config asks for, returning the transform."""

    warnings: list[str] = []
    preprocess = config.mesh_preprocess
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise MeshValidationError("mesh has no vertices or no faces")
    if not np.all(np.isfinite(vertices)):
        raise MeshValidationError("mesh vertices contain NaN or Inf")
    if vertices.shape[0] > config.execution.max_input_vertices:
        raise MeshValidationError(
            f"{vertices.shape[0]} vertices exceeds max_input_vertices={config.execution.max_input_vertices}"
        )
    if faces.shape[0] > config.execution.max_input_faces:
        raise MeshValidationError(f"{faces.shape[0]} faces exceeds max_input_faces={config.execution.max_input_faces}")

    transform = np.eye(4, dtype=np.float64)
    if preprocess.unit_scale_to_meters != 1.0:
        vertices = vertices * preprocess.unit_scale_to_meters
        transform[:3, :3] *= preprocess.unit_scale_to_meters

    if preprocess.repair_mode == "manifoldplus":
        raise MeshRepairUnavailable(
            "repair_mode='manifoldplus' needs the ManifoldPlus backend, which is not bundled. "
            "Repair the mesh before ingest, or use repair_mode='none'."
        )

    working = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if preprocess.simplify_faces is not None and working.faces.shape[0] > preprocess.simplify_faces:
        try:
            simplified = working.simplify_quadric_decimation(face_count=preprocess.simplify_faces)
            if simplified.faces.shape[0] > 0:
                working = simplified
            else:
                warnings.append("simplification produced an empty mesh and was skipped")
        except Exception as error:  # noqa: BLE001 - simplification is optional
            warnings.append(f"simplification unavailable: {type(error).__name__}")

    if preprocess.normalize_diagonal_m is not None:
        diagonal = float(np.linalg.norm(working.extents))
        if diagonal <= 0.0:
            raise MeshValidationError("mesh has a zero bounding-box diagonal")
        factor = preprocess.normalize_diagonal_m / diagonal
        working = trimesh.Trimesh(
            vertices=np.asarray(working.vertices, dtype=np.float64) * factor,
            faces=np.asarray(working.faces, dtype=np.int64),
            process=False,
        )
        transform[:3, :3] *= factor
    return working, transform, warnings


def _cache_key(
    input_bytes: bytes, transform: np.ndarray, config: CoACDConfig, tool_digest: str, platform_tag: str
) -> str:
    digest = hashlib.sha256()
    digest.update(hashlib.sha256(input_bytes).digest())
    digest.update(np.asarray(transform, dtype=np.float64).tobytes())
    digest.update(_hash_document(dataclasses.asdict(config.mesh_preprocess)).encode("utf-8"))
    digest.update(config.algorithm_hash().encode("utf-8"))
    digest.update(tool_digest.encode("utf-8"))
    digest.update(platform_tag.encode("utf-8"))
    return digest.hexdigest()


def decompose_collision_mesh(
    mesh_or_path: trimesh.Trimesh | str | Path | bytes,
    *,
    config: CoACDConfig,
    cache_dir: str | Path | None = None,
    backend: CoacdBackend | None = None,
    tool_version: str | None = None,
) -> CoACDResult:
    """Decompose a mesh into convex parts.  No network, no writes you did not ask for.

    ``cache_dir`` is a content-addressed local cache keyed by the input bytes,
    the source-to-metric transform, the preprocessing and algorithm configs, the
    tool build and the platform tag.  Nothing else is written anywhere.
    """

    config.validate()
    mesh, raw = _as_mesh(mesh_or_path)
    prepared, transform, warnings = _prepare_mesh(mesh, config)

    mesh_ctor: Callable[..., Any] | None = None
    if backend is None:
        backend, detected_version, mesh_ctor = _load_backend()
        tool_version = tool_version or detected_version
    tool_version = tool_version or "injected-backend"
    platform_tag = f"{platform.system()}-{platform.machine()}-py{platform.python_version()}"

    key = _cache_key(raw, transform, config, tool_version, platform_tag)
    cache_path = Path(cache_dir) / f"{key}.npz" if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        return _result_from_cache(cache_path, prepared, raw, transform, config, tool_version, platform_tag)

    kwargs = {UPSTREAM_ARGUMENT_NAMES[name]: value for name, value in dataclasses.asdict(config.algorithm).items()}
    _check_upstream_signature(backend, list(kwargs))

    payload: Any = prepared
    if mesh_ctor is not None:
        payload = mesh_ctor(np.asarray(prepared.vertices, dtype=np.float64), np.asarray(prepared.faces, dtype=np.int32))

    started = time.perf_counter()
    try:
        raw_parts = backend(payload, **kwargs)
    except CoACDError:
        raise
    except Exception as error:
        raise CoACDExecutionError(f"CoACD failed: {type(error).__name__}: {error}") from error
    elapsed = time.perf_counter() - started

    parts = _validate_parts(raw_parts, config)
    result = CoACDResult(
        convex_parts=parts,
        prepared_mesh=prepared,
        input_sha256=hashlib.sha256(raw).hexdigest(),
        prepared_sha256=_mesh_digest(prepared),
        part_sha256=[_mesh_digest(part) for part in parts],
        source_to_metric_transform=[[float(value) for value in row] for row in transform],
        config_hash=config.content_hash(),
        profile=config.profile,
        tool_version=tool_version,
        platform_tag=platform_tag,
        piece_count=len(parts),
        total_volume_m3=float(sum(abs(part.volume) for part in parts)),
        elapsed_s=elapsed,
        warnings=warnings,
    )
    if cache_path is not None:
        _write_cache(cache_path, result)
    return result


def _mesh_digest(mesh: trimesh.Trimesh) -> str:
    from qdgrasp.objects.manifest import export_mesh_deterministic_obj

    return hashlib.sha256(export_mesh_deterministic_obj(mesh)).hexdigest()


def _validate_parts(raw_parts: Sequence[Any], config: CoACDConfig) -> list[trimesh.Trimesh]:
    """Turn the backend's output into meshes, refusing anything unusable."""

    if raw_parts is None or len(raw_parts) == 0:
        raise CollisionValidationError(
            "the decomposition returned no parts; an empty or single-hull stand-in would hide the failure"
        )
    if len(raw_parts) > config.execution.max_output_parts:
        raise TooManyConvexPartsError(
            f"{len(raw_parts)} parts exceeds max_output_parts={config.execution.max_output_parts}"
        )
    parts: list[trimesh.Trimesh] = []
    for index, item in enumerate(raw_parts):
        if isinstance(item, trimesh.Trimesh):
            part = item
        else:
            try:
                vertices, faces = item
            except (TypeError, ValueError) as error:
                raise CollisionValidationError(f"part {index} is not a (vertices, faces) pair") from error
            part = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
        vertices = np.asarray(part.vertices, dtype=np.float64)
        if vertices.size == 0 or not np.all(np.isfinite(vertices)):
            raise CollisionValidationError(f"part {index} is empty or contains non-finite vertices")
        if abs(part.volume) <= 0.0:
            raise CollisionValidationError(f"part {index} has zero volume")
        parts.append(part)
    return parts


def _write_cache(path: Path, result: CoACDResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for index, part in enumerate(result.convex_parts):
        arrays[f"vertices_{index}"] = np.asarray(part.vertices, dtype=np.float64)
        arrays[f"faces_{index}"] = np.asarray(part.faces, dtype=np.int64)
    np.savez_compressed(
        path,
        metadata=np.frombuffer(json.dumps(result.to_document(), sort_keys=True).encode("utf-8"), dtype=np.uint8),
        **arrays,
    )


def _result_from_cache(
    path: Path,
    prepared: trimesh.Trimesh,
    raw: bytes,
    transform: np.ndarray,
    config: CoACDConfig,
    tool_version: str,
    platform_tag: str,
) -> CoACDResult:
    archive = np.load(path, allow_pickle=False)
    document = json.loads(bytes(archive["metadata"]).decode("utf-8"))
    parts = []
    index = 0
    while f"vertices_{index}" in archive:
        parts.append(
            trimesh.Trimesh(vertices=archive[f"vertices_{index}"], faces=archive[f"faces_{index}"], process=False)
        )
        index += 1
    return CoACDResult(
        convex_parts=parts,
        prepared_mesh=prepared,
        input_sha256=document["input_sha256"],
        prepared_sha256=document["prepared_sha256"],
        part_sha256=document["part_sha256"],
        source_to_metric_transform=document["source_to_metric_transform"],
        config_hash=document["config_hash"],
        profile=document["profile"],
        tool_version=tool_version,
        platform_tag=platform_tag,
        piece_count=document["piece_count"],
        total_volume_m3=document["total_volume_m3"],
        elapsed_s=document["elapsed_s"],
        warnings=list(document["warnings"]),
        cache_hit=True,
    )


def build_collision_asset(
    mesh_or_path: trimesh.Trimesh | str | Path | bytes,
    *,
    config: CoACDConfig,
    mass_kg: float | None = None,
    density_kg_m3: float | None = None,
    cache_dir: str | Path | None = None,
    backend: CoacdBackend | None = None,
) -> CollisionAsset:
    """Decompose, then attach mass properties **only** if the caller gave them.

    ``mass_kg`` and ``density_kg_m3`` are mutually exclusive, and neither has a
    default.  The old batch script's 0.2 kg is not a fallback for an arbitrary
    object; without one of the two, the asset stops at ``geometry_ready``.
    """

    if mass_kg is not None and density_kg_m3 is not None:
        raise MeshValidationError("mass_kg and density_kg_m3 are mutually exclusive")
    result = decompose_collision_mesh(mesh_or_path, config=config, cache_dir=cache_dir, backend=backend)
    volume = result.total_volume_m3
    if mass_kg is None and density_kg_m3 is None:
        return CollisionAsset(
            result=result,
            mass_kg=None,
            density_kg_m3=None,
            inertia_kg_m2=None,
            center_of_mass_m=None,
            status="geometry_ready",
        )
    if volume <= 0.0:
        raise CollisionValidationError("cannot derive mass properties from a decomposition with no volume")
    if mass_kg is not None:
        mass = float(mass_kg)
        density = mass / volume
    else:
        density = float(density_kg_m3)  # type: ignore[arg-type]
        mass = density * volume

    weighted_center = np.zeros(3, dtype=np.float64)
    inertia = np.zeros((3, 3), dtype=np.float64)
    for part in result.convex_parts:
        part_volume = abs(float(part.volume))
        part_mass = density * part_volume
        weighted_center += part_mass * np.asarray(part.center_mass, dtype=np.float64)
        inertia += density * np.asarray(part.moment_inertia, dtype=np.float64)
    center = weighted_center / mass if mass > 0.0 else weighted_center
    eigenvalues = np.linalg.eigvalsh(inertia)
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
        raise CollisionValidationError("the decomposition does not yield a positive-definite inertia tensor")
    return CollisionAsset(
        result=result,
        mass_kg=mass,
        density_kg_m3=density,
        inertia_kg_m2=[float(value) for value in eigenvalues],
        center_of_mass_m=[float(value) for value in center],
        status="dynamic_ready",
    )
