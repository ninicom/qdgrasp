from __future__ import annotations

import pytest
import trimesh

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.collision import validate_collision_representation
from qdgrasp.objects.generate import generate_box, generate_compound_convex
from qdgrasp.objects.schema import SubGeomSpec


def test_collision_representation_guard_passes_valid_geoms() -> None:
    rng = get_generator(333, "valid_col")
    mesh, geoms, _, _, _ = generate_compound_convex(rng, shape_family="t_shape")
    validate_collision_representation(mesh, geoms)


def test_collision_guard_rejects_empty_geoms() -> None:
    mesh = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    with pytest.raises(ConfigError, match="no collision geometries"):
        validate_collision_representation(mesh, [])


def test_collision_guard_rejects_unbounded_mismatch() -> None:
    mesh = trimesh.creation.box(extents=(0.10, 0.10, 0.10))
    # Provide an undersized collision box (1cm vs 10cm)
    geoms = [SubGeomSpec(type="box", size=(0.005, 0.005, 0.005), pos=(0.0, 0.0, 0.0))]
    with pytest.raises(ConfigError, match="collision representation bounding mismatch"):
        validate_collision_representation(mesh, geoms, tolerance=0.01)
