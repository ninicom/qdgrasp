"""P3.5-01/02/05: what the ingest pipeline refuses, and what it records.

The interesting assertions here are the negative ones.  An asset pipeline earns
its keep by declining to guess -- about units, about mass, about a license -- and
a test suite that only feeds it good meshes never checks the part that matters.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import trimesh

from qdgrasp.objects.ingest import (
    AssetIngestError,
    AssetIngestRequest,
    IngestErrorCode,
    IngestStatus,
    NormalizationConfig,
    PhysicsProperties,
    ingest_asset,
    normalize_mesh,
    read_source_bytes,
)
from qdgrasp.objects.manifest_v2 import (
    ManifestImmutabilityError,
    ObjectAssetManifestV2,
    load_object_asset_manifest_v2,
    write_object_asset_manifest_v2,
)


def _obj_bytes(mesh: trimesh.Trimesh) -> bytes:
    exported = mesh.export(file_type="obj")
    return exported.encode("utf-8") if isinstance(exported, str) else exported


@pytest.fixture(scope="module")
def box_bytes() -> bytes:
    return _obj_bytes(trimesh.creation.box(extents=(0.06, 0.04, 0.02)))


def _request(**overrides) -> AssetIngestRequest:
    base = {
        "object_id": "box",
        "license_record": "CC0-1.0",
        "redistributable": True,
        "source_format": "obj",
        "units": "m",
    }
    base.update(overrides)
    return AssetIngestRequest(**base)


# -- sources ---------------------------------------------------------------


def test_exactly_one_source_is_required(box_bytes: bytes) -> None:
    with pytest.raises(AssetIngestError) as missing:
        _request().validate()
    assert missing.value.code is IngestErrorCode.SOURCE_MISSING

    with pytest.raises(AssetIngestError) as ambiguous:
        _request(mesh_bytes=box_bytes, local_mesh_path="/tmp/x.obj").validate()
    assert ambiguous.value.code is IngestErrorCode.SOURCE_AMBIGUOUS


def test_a_path_outside_the_allowed_root_is_refused(tmp_path, box_bytes: bytes) -> None:
    inside = tmp_path / "root"
    inside.mkdir()
    outside = tmp_path / "elsewhere.obj"
    outside.write_bytes(box_bytes)
    with pytest.raises(AssetIngestError) as error:
        read_source_bytes(_request(local_mesh_path=outside), allowed_root=inside)
    assert error.value.code is IngestErrorCode.PATH_ESCAPES_ROOT


def test_a_missing_license_is_refused(box_bytes: bytes) -> None:
    with pytest.raises(AssetIngestError) as error:
        _request(mesh_bytes=box_bytes, license_record="   ").validate()
    assert error.value.code is IngestErrorCode.LICENSE_MISSING


# -- units -----------------------------------------------------------------


def test_the_unit_scale_is_applied_exactly_once() -> None:
    """A millimetre mesh and a metre mesh of the same object normalise identically."""

    metres = _obj_bytes(trimesh.creation.box(extents=(0.06, 0.04, 0.02)))
    millimetres = _obj_bytes(trimesh.creation.box(extents=(60.0, 40.0, 20.0)))
    from_m = normalize_mesh(_request(mesh_bytes=metres, units="m"), metres)
    from_mm = normalize_mesh(_request(mesh_bytes=millimetres, units="mm"), millimetres)
    assert from_m.normalized_sha256 == from_mm.normalized_sha256
    np.testing.assert_allclose(from_mm.extents_m, (0.06, 0.04, 0.02), atol=1e-9)
    assert from_mm.scale_to_meters == pytest.approx(0.001)


def test_a_scale_that_contradicts_the_declared_unit_is_refused(box_bytes: bytes) -> None:
    with pytest.raises(AssetIngestError) as error:
        _request(mesh_bytes=box_bytes, units="mm", scale_to_meters=1.0).validate()
    assert error.value.code is IngestErrorCode.SCALE_CONFLICT


def test_explicit_scale_requires_a_factor(box_bytes: bytes) -> None:
    with pytest.raises(AssetIngestError) as error:
        _request(mesh_bytes=box_bytes, units="explicit_scale").validate()
    assert error.value.code is IngestErrorCode.UNIT_UNDECLARED


# -- geometry budgets ------------------------------------------------------


def test_an_oversized_object_is_refused(box_bytes: bytes) -> None:
    huge = _obj_bytes(trimesh.creation.box(extents=(3.0, 3.0, 3.0)))
    with pytest.raises(AssetIngestError) as error:
        normalize_mesh(_request(mesh_bytes=huge), huge)
    assert error.value.code is IngestErrorCode.BOUNDS_OUT_OF_RANGE
    del box_bytes


def test_a_scene_of_separate_parts_is_not_an_object() -> None:
    left = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    right = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    right.apply_translation([0.2, 0.0, 0.0])
    raw = _obj_bytes(trimesh.util.concatenate([left, right]))
    with pytest.raises(AssetIngestError) as error:
        normalize_mesh(_request(mesh_bytes=raw), raw)
    assert error.value.code is IngestErrorCode.DISCONNECTED_COMPONENTS


def test_the_triangle_budget_is_enforced() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.apply_scale(0.05 / max(mesh.extents))
    raw = _obj_bytes(mesh)
    config = NormalizationConfig(max_triangles=8)
    with pytest.raises(AssetIngestError) as error:
        normalize_mesh(_request(mesh_bytes=raw, normalization=config), raw)
    assert error.value.code is IngestErrorCode.TRIANGLE_BUDGET_EXCEEDED


def test_the_input_hash_is_of_the_raw_bytes(box_bytes: bytes) -> None:
    import hashlib

    geometry = normalize_mesh(_request(mesh_bytes=box_bytes), box_bytes)
    assert geometry.input_sha256 == hashlib.sha256(box_bytes).hexdigest()
    assert geometry.normalized_sha256 != geometry.input_sha256


# -- mass ------------------------------------------------------------------


def test_no_mass_and_no_density_stops_at_geometry_ready(box_bytes: bytes) -> None:
    result = ingest_asset(_request(mesh_bytes=box_bytes))
    assert result.status is IngestStatus.GEOMETRY_READY
    assert result.mass_properties["mass_kg"] is None
    assert result.mass_properties["reason"]


def test_density_derives_mass_and_says_so(box_bytes: bytes) -> None:
    result = ingest_asset(_request(mesh_bytes=box_bytes, physics=PhysicsProperties(density=600.0)))
    assert result.status is IngestStatus.DYNAMIC_READY
    assert result.mass_properties["mass_kg"] == pytest.approx(600.0 * 0.06 * 0.04 * 0.02)
    assert "mass_from_density_and_volume" in result.mass_properties["derived"]
    assert "inertia_from_geometry_and_mass" in result.mass_properties["derived"]


def test_a_negative_mass_is_refused(box_bytes: bytes) -> None:
    with pytest.raises(AssetIngestError) as error:
        _request(mesh_bytes=box_bytes, physics=PhysicsProperties(mass=-1.0)).validate()
    assert error.value.code is IngestErrorCode.MASS_PROPERTIES_INVALID


# -- collision -------------------------------------------------------------


def test_a_convex_source_is_not_decomposed(box_bytes: bytes) -> None:
    result = ingest_asset(_request(mesh_bytes=box_bytes, collision_policy="convex_if_possible"))
    assert result.collision["source"] == "source_is_convex"
    assert result.collision["part_count"] == 1


def test_the_coacd_policy_requires_a_decomposition(box_bytes: bytes) -> None:
    with pytest.raises(AssetIngestError) as error:
        ingest_asset(_request(mesh_bytes=box_bytes, collision_policy="coacd"))
    assert error.value.code is IngestErrorCode.COLLISION_UNAVAILABLE


def test_supplied_collision_parts_are_recorded(box_bytes: bytes) -> None:
    parts = [trimesh.creation.box(extents=(0.03, 0.04, 0.02)), trimesh.creation.box(extents=(0.03, 0.04, 0.02))]
    result = ingest_asset(_request(mesh_bytes=box_bytes, collision_policy="coacd"), collision_parts=parts)
    assert result.collision["source"] == "coacd"
    assert result.collision["part_count"] == 2
    assert all(item["is_convex"] for item in result.collision["parts"])


# -- request identity ------------------------------------------------------


def test_the_request_hash_covers_every_field(box_bytes: bytes) -> None:
    base = _request(mesh_bytes=box_bytes, physics=PhysicsProperties(density=600.0))
    assert (
        base.request_hash() == _request(mesh_bytes=box_bytes, physics=PhysicsProperties(density=600.0)).request_hash()
    )
    changed = dataclasses.replace(base, physics=PhysicsProperties(density=601.0))
    assert changed.request_hash() != base.request_hash()
    rescaled = dataclasses.replace(base, normalization=NormalizationConfig(max_triangles=99))
    assert rescaled.request_hash() != base.request_hash()


# -- manifest v2 -----------------------------------------------------------


def test_manifest_round_trips_and_refuses_a_silent_rewrite(tmp_path, box_bytes: bytes) -> None:
    result = ingest_asset(_request(mesh_bytes=box_bytes, physics=PhysicsProperties(density=600.0)))
    manifest = ObjectAssetManifestV2.from_ingest(result)
    path = write_object_asset_manifest_v2(tmp_path / "box.manifest_v2.json", manifest)

    reloaded = load_object_asset_manifest_v2(path)
    assert reloaded == manifest
    assert reloaded.content_hash() == manifest.content_hash()
    assert reloaded.is_dynamic_ready

    # Writing the identical manifest again is a no-op, not an error.
    write_object_asset_manifest_v2(path, manifest)

    other = ingest_asset(_request(mesh_bytes=box_bytes, physics=PhysicsProperties(density=900.0)))
    with pytest.raises(ManifestImmutabilityError):
        write_object_asset_manifest_v2(path, ObjectAssetManifestV2.from_ingest(other))


def test_manifest_rejects_a_foreign_schema(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "qdgrasp/object-asset-manifest/v1"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_object_asset_manifest_v2(path)
