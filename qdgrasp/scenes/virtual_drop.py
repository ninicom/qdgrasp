"""Virtual drop scenes for objects that arrive without one (P3.5-07).

When a caller has objects but no scene, the system builds one: a finite support,
a bounded spawn region, and initial placements that do not overlap.  After that
the objects move only because gravity and contact move them -- nothing here ever
writes a settled pose, because a scene whose final state was written by the
generator is not evidence that the state is reachable.

Table is the baseline.  Tray and bin exist but are deliberately downstream of a
passing table gate: container walls catch objects that a table would let fall,
which is exactly the kind of help that hides a placement bug.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.contracts import SceneObjectSpec, SceneSpec, SupportGeometrySpec

VIRTUAL_DROP_SCHEMA_V1 = "qdgrasp/virtual-drop-scene/v1"

Environment = Literal["table", "tray", "bin"]
BoundaryPolicy = Literal["reject", "contain"]
RegionType = Literal["aabb", "obb"]

#: Every stochastic decision draws from its own stream, so changing the
#: randomisation of one aspect does not silently reshuffle the others.
SEED_STREAM_NAMES: tuple[str, ...] = (
    "asset",
    "layout",
    "orientation",
    "drop",
    "material",
    "observation",
)


@dataclasses.dataclass(frozen=True)
class SpawnRegion:
    """Bounded volume the generator may place objects in."""

    region_type: RegionType = "aabb"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_extents: tuple[float, float, float] = (0.12, 0.12, 0.0)
    #: Yaw of the region about ``+z``; only meaningful for ``obb``.
    yaw: float = 0.0

    def validate(self) -> None:
        if self.region_type not in ("aabb", "obb"):
            raise ConfigError(f"spawn region type must be aabb or obb, got {self.region_type!r}")
        extents = np.asarray(self.half_extents, dtype=np.float64)
        if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents < 0.0):
            raise ConfigError("spawn region half extents must be three finite non-negative numbers")
        if extents[0] <= 0.0 or extents[1] <= 0.0:
            raise ConfigError("spawn region must have positive extent in x and y")

    def sample_xy(self, rng: np.random.Generator) -> np.ndarray:
        offset = rng.uniform(-1.0, 1.0, size=2) * np.asarray(self.half_extents[:2], dtype=np.float64)
        if self.region_type == "obb" and self.yaw != 0.0:
            cos, sin = np.cos(self.yaw), np.sin(self.yaw)
            offset = np.array([cos * offset[0] - sin * offset[1], sin * offset[0] + cos * offset[1]])
        return np.asarray(self.center[:2], dtype=np.float64) + offset

    def contains_xy(self, point: Sequence[float], margin: float = 0.0) -> bool:
        local = np.asarray(point, dtype=np.float64)[:2] - np.asarray(self.center[:2], dtype=np.float64)
        if self.region_type == "obb" and self.yaw != 0.0:
            cos, sin = np.cos(-self.yaw), np.sin(-self.yaw)
            local = np.array([cos * local[0] - sin * local[1], sin * local[0] + cos * local[1]])
        limits = np.asarray(self.half_extents[:2], dtype=np.float64) + margin
        return bool(np.all(np.abs(local) <= limits))


@dataclasses.dataclass(frozen=True)
class SettleThresholds:
    """What "settled" means, pinned so two runs cannot disagree about it."""

    linear_velocity_mps: float = 0.002
    angular_velocity_radps: float = 0.05
    kinetic_energy_j: float = 1e-5
    pose_delta_m: float = 5e-4
    pose_delta_rad: float = 5e-3
    consecutive_steps: int = 25
    timeout_steps: int = 5000
    #: Interpenetration allowed in the *settled* state, and at placement time.
    max_penetration_m: float = 0.002
    #: Interpenetration allowed at any instant.  A dropped object compresses the
    #: contact on landing -- measured at 7.7 mm for a 4 cm box from 8 cm, decaying
    #: to 0.07 mm once at rest -- and that transient is how MuJoCo represents an
    #: impulse, not a defect.  This bound exists to catch the other thing:
    #: tunnelling and solver blow-up, which are an order of magnitude larger.
    max_transient_penetration_m: float = 0.02

    def validate(self) -> None:
        for name in (
            "linear_velocity_mps",
            "angular_velocity_radps",
            "kinetic_energy_j",
            "pose_delta_m",
            "pose_delta_rad",
            "max_penetration_m",
            "max_transient_penetration_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ConfigError(f"settle threshold {name} must be finite and positive, got {value!r}")
        if self.consecutive_steps < 1 or self.timeout_steps < self.consecutive_steps:
            raise ConfigError("timeout_steps must be at least consecutive_steps, and both must be positive")
        if self.max_transient_penetration_m < self.max_penetration_m:
            raise ConfigError(
                "max_transient_penetration_m must be at least max_penetration_m: a settled state that is "
                "allowed to penetrate more deeply than any instant is incoherent"
            )


@dataclasses.dataclass(frozen=True)
class DropObjectRequest:
    """One object to place, with what the generator needs to keep it apart."""

    object_id: str
    asset_ref: str
    scale: float = 1.0
    mass: float | None = None
    friction: tuple[float, float, float] | None = None
    #: Radius of a sphere enclosing the object, used for non-overlap sampling.
    #: Read from the manifest when omitted.
    bounding_radius_m: float | None = None


@dataclasses.dataclass(frozen=True)
class VirtualDropSceneSpec:
    """A generated scene: support, spawn region, drop policy and settle rule."""

    environment: Environment = "table"
    support_size_m: tuple[float, float, float] = (0.6, 0.6, 0.04)
    support_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    support_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    wall_height_m: float = 0.12
    wall_thickness_m: float = 0.01
    spawn_region: SpawnRegion = dataclasses.field(default_factory=SpawnRegion)
    drop_height_range_m: tuple[float, float] = (0.05, 0.12)
    object_count_range: tuple[int, int] = (1, 1)
    initial_clearance_m: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.002
    solver_profile: str = "default"
    boundary_policy: BoundaryPolicy = "reject"
    settle_thresholds: SettleThresholds = dataclasses.field(default_factory=SettleThresholds)
    upright_only: bool = False
    max_placement_attempts: int = 200

    def validate(self) -> None:
        if self.environment not in ("table", "tray", "bin"):
            raise ConfigError(f"environment must be table, tray or bin, got {self.environment!r}")
        size = np.asarray(self.support_size_m, dtype=np.float64)
        if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ConfigError("support_size_m must be three positive numbers")
        low, high = self.drop_height_range_m
        if not (np.isfinite(low) and np.isfinite(high)) or low < 0.0 or high < low:
            raise ConfigError(
                f"drop_height_range_m is not an ordered non-negative interval: {self.drop_height_range_m}"
            )
        count_low, count_high = self.object_count_range
        if count_low < 1 or count_high < count_low:
            raise ConfigError(f"object_count_range is not an ordered positive interval: {self.object_count_range}")
        if not np.isfinite(self.initial_clearance_m) or self.initial_clearance_m < 0.0:
            raise ConfigError("initial_clearance_m must be finite and non-negative")
        if not np.isfinite(self.timestep) or self.timestep <= 0.0:
            raise ConfigError("timestep must be finite and positive")
        if self.boundary_policy not in ("reject", "contain"):
            raise ConfigError(f"boundary_policy must be reject or contain, got {self.boundary_policy!r}")
        if self.max_placement_attempts < 1:
            raise ConfigError("max_placement_attempts must be at least 1")
        self.spawn_region.validate()
        self.settle_thresholds.validate()

    def to_document(self) -> dict[str, Any]:
        document = dataclasses.asdict(self)
        document["schema"] = VIRTUAL_DROP_SCHEMA_V1
        return document

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PlacementError(RuntimeError):
    """No non-overlapping placement was found within the attempt budget."""


def seed_streams(base_seed: int, scene_id: str) -> dict[str, np.random.Generator]:
    """One independent generator per named stream."""

    streams: dict[str, np.random.Generator] = {}
    for name in SEED_STREAM_NAMES:
        material = f"{base_seed}|{scene_id}|{name}".encode()
        derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        streams[name] = np.random.default_rng(derived)
    return streams


def _support_specs(spec: VirtualDropSceneSpec) -> list[SupportGeometrySpec]:
    """Support geometry for the requested environment, top surface at z = 0."""

    width, depth, thickness = spec.support_size_m
    origin = np.asarray(spec.support_pose, dtype=np.float64)
    top = np.eye(4)
    top[:3, 3] = origin + np.array([0.0, 0.0, -thickness / 2.0])
    supports = [
        SupportGeometrySpec(
            support_id="support_surface",
            geom_type="box",
            params={"size": [width, depth, thickness], "friction": list(spec.support_friction)},
            T_world_support=top,
        )
    ]
    if spec.environment == "table":
        return supports
    height = spec.wall_height_m
    thick = spec.wall_thickness_m
    for sign in (1.0, -1.0):
        pose = np.eye(4)
        pose[:3, 3] = origin + np.array([sign * (width / 2.0 + thick / 2.0), 0.0, height / 2.0])
        supports.append(
            SupportGeometrySpec(
                support_id=f"support_wall_x_{'pos' if sign > 0 else 'neg'}",
                geom_type="box",
                params={"size": [thick, depth + 2 * thick, height], "friction": list(spec.support_friction)},
                T_world_support=pose,
            )
        )
        pose = np.eye(4)
        pose[:3, 3] = origin + np.array([0.0, sign * (depth / 2.0 + thick / 2.0), height / 2.0])
        supports.append(
            SupportGeometrySpec(
                support_id=f"support_wall_y_{'pos' if sign > 0 else 'neg'}",
                geom_type="box",
                params={"size": [width + 2 * thick, thick, height], "friction": list(spec.support_friction)},
                T_world_support=pose,
            )
        )
    return supports


def _bounding_radius(request: DropObjectRequest) -> float:
    if request.bounding_radius_m is not None:
        return float(request.bounding_radius_m) * float(request.scale)
    from pathlib import Path

    from qdgrasp.objects.manifest import load_object_asset

    path = Path(request.asset_ref)
    manifest_path = path if path.name.endswith(".manifest.json") else path.with_name(f"{path.stem}.manifest.json")
    if not manifest_path.is_file():
        raise ConfigError(
            f"cannot determine a bounding radius for {request.object_id}: no manifest at {manifest_path} "
            "and no bounding_radius_m supplied"
        )
    _, manifest = load_object_asset(manifest_path)
    box = np.asarray(manifest.bounding_box, dtype=np.float64)
    extents = box[3:] - box[:3]
    return float(np.linalg.norm(extents) / 2.0 * request.scale)


def build_virtual_drop_scene(
    spec: VirtualDropSceneSpec,
    objects: Sequence[DropObjectRequest],
    *,
    seed: int,
    scene_id: str = "virtual-drop",
) -> SceneSpec:
    """Place objects above a generated support, without overlap.

    Placement is rejection sampling against the enclosing spheres, so a request
    that cannot fit raises rather than quietly dropping objects on top of each
    other and letting the solver sort it out.
    """

    spec.validate()
    if not objects:
        raise ConfigError("a virtual drop scene needs at least one object")
    count_low, count_high = spec.object_count_range
    if not count_low <= len(objects) <= count_high:
        raise ConfigError(f"{len(objects)} objects requested but object_count_range is {spec.object_count_range}")

    streams = seed_streams(seed, scene_id)
    layout, orientation, drop = streams["layout"], streams["orientation"], streams["drop"]

    placed: list[tuple[np.ndarray, float]] = []
    scene_objects: list[SceneObjectSpec] = []
    for request in objects:
        radius = _bounding_radius(request)
        for _ in range(spec.max_placement_attempts):
            xy = spec.spawn_region.sample_xy(layout)
            if not spec.spawn_region.contains_xy(xy):
                continue
            if any(
                float(np.linalg.norm(xy - other_xy)) < radius + other_radius + spec.initial_clearance_m
                for other_xy, other_radius in placed
            ):
                continue
            break
        else:
            raise PlacementError(
                f"no non-overlapping placement for {request.object_id} within "
                f"{spec.max_placement_attempts} attempts; enlarge the spawn region or drop the object count"
            )
        placed.append((xy, radius))

        height = float(drop.uniform(*spec.drop_height_range_m))
        transform = np.eye(4)
        if spec.upright_only:
            yaw = float(orientation.uniform(-np.pi, np.pi))
            transform[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        else:
            transform[:3, :3] = Rotation.random(random_state=int(orientation.integers(0, 2**31 - 1))).as_matrix()
        transform[:3, 3] = [xy[0], xy[1], radius + height]
        scene_objects.append(
            SceneObjectSpec(
                object_id=request.object_id,
                asset_ref=request.asset_ref,
                T_world_object=transform,
                scale=request.scale,
                mass=request.mass,
                friction=request.friction,
            )
        )

    return SceneSpec(
        scene_id=scene_id,
        source_dataset="qdgrasp-virtual-drop",
        source_version=VIRTUAL_DROP_SCHEMA_V1,
        source_split="generated",
        environment=spec.environment,
        objects=scene_objects,
        supports=_support_specs(spec),
        cameras=[],
        gravity=spec.gravity,
        timestep=spec.timestep,
        solver_profile=spec.solver_profile,
        settle_seed=seed,
        source_record_hash=spec.content_hash(),
        license_record="generated",
        redistributable=True,
    )
