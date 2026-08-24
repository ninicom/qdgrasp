"""Pinned hand-independent object matrix for P3.2.1-11."""

from __future__ import annotations

from dataclasses import dataclass

import trimesh

from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.generate import generate_compound_convex, generate_superquadric
from qdgrasp.objects.schema import SubGeomSpec


@dataclass(frozen=True)
class CanonicalFullFlowObject:
    name: str
    mesh: trimesh.Trimesh
    collision_geoms: tuple[SubGeomSpec, ...]
    mass: float = 0.005

    @property
    def object_pos(self) -> tuple[float, float, float]:
        return (0.0, 0.0, max(0.0, -float(self.mesh.bounds[0, 2])))


def _box() -> CanonicalFullFlowObject:
    edge = 0.050
    return CanonicalFullFlowObject(
        name="box_50mm",
        mesh=trimesh.creation.box(extents=(edge, edge, edge)),
        collision_geoms=(
            SubGeomSpec(type="box", size=(edge / 2, edge / 2, edge / 2)),
        ),
    )


def _cylinder() -> CanonicalFullFlowObject:
    radius, height = 0.025, 0.050
    return CanonicalFullFlowObject(
        name="cylinder_r25_h50",
        mesh=trimesh.creation.cylinder(radius=radius, height=height, sections=64),
        collision_geoms=(
            SubGeomSpec(type="cylinder", size=(radius, height / 2)),
        ),
    )


def _superquadric() -> CanonicalFullFlowObject:
    mesh, geoms, _, _, _ = generate_superquadric(
        get_generator(42, "p3.2.1", "canonical", "superquadric"),
        scale_range=(0.025, 0.025),
        shape_range=(0.8, 0.8),
        density=1.0,
    )
    return CanonicalFullFlowObject(
        name="superquadric_50mm_e08",
        mesh=mesh,
        collision_geoms=tuple(geoms),
    )


def _compound() -> CanonicalFullFlowObject:
    mesh, geoms, _, _, _ = generate_compound_convex(
        get_generator(42, "p3.2.1", "canonical", "compound"),
        shape_family="t_shape",
        scale_range=(0.025, 0.025),
        density=1.0,
    )
    # The source generator centers the stem at z=0.  The public object pose
    # property handles table placement identically for every hand.
    return CanonicalFullFlowObject(
        name="compound_t_25mm",
        mesh=mesh,
        collision_geoms=tuple(geoms),
    )


def build_canonical_full_flow_matrix() -> tuple[CanonicalFullFlowObject, ...]:
    """Return box/cylinder/superquadric/compound objects in pinned order."""
    return (_box(), _cylinder(), _superquadric(), _compound())
