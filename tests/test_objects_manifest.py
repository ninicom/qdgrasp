from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.generate import generate_box
from qdgrasp.objects.manifest import create_object_asset, load_object_asset, save_object_asset


def test_object_manifest_save_and_load_round_trip() -> None:
    rng = get_generator(555, "round_trip")
    mesh, geoms, params, mass, inertia = generate_box(rng)

    mesh_bytes, manifest = create_object_asset(
        object_id="box_test_001",
        family="primitive",
        shape_type="box",
        mesh=mesh,
        collision_geoms=geoms,
        params=params,
        mass=mass,
        inertia=inertia,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest_path = save_object_asset(mesh_bytes, manifest, out_dir)

        loaded_mesh, loaded_manifest = load_object_asset(manifest_path)
        assert loaded_manifest.object_id == "box_test_001"
        assert loaded_manifest.family == "primitive"
        assert loaded_manifest.mass == manifest.mass
        assert loaded_manifest.inertia == manifest.inertia
        assert loaded_manifest.mesh_sha256 == manifest.mesh_sha256
        assert len(loaded_mesh.vertices) == len(mesh.vertices)


def test_object_manifest_detects_mesh_corruption() -> None:
    rng = get_generator(666, "corruption")
    mesh, geoms, params, mass, inertia = generate_box(rng)

    mesh_bytes, manifest = create_object_asset(
        object_id="corrupt_box",
        family="primitive",
        shape_type="box",
        mesh=mesh,
        collision_geoms=geoms,
        params=params,
        mass=mass,
        inertia=inertia,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest_path = save_object_asset(mesh_bytes, manifest, out_dir)

        # Corrupt the mesh file on disk
        mesh_file = out_dir / manifest.mesh_filename
        mesh_file.write_bytes(b"# Corrupted OBJ content\n")

        with pytest.raises(ConfigError, match="mesh integrity failure"):
            load_object_asset(manifest_path)
