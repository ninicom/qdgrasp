"""P3.5-03/04: the CoACD façade's contract, exercised without the backend.

The decomposition binary is not installed here, and almost none of what the plan
asks for needs it: parameter validation, profile provenance, config hashing, the
refusal to pass an unsupported argument, the cache key, and the rule that mass
is the caller's are all properties of the wrapper.  A stub backend stands in for
the solver so those properties can be tested on their own.

What is *not* covered without the binary is output-class parity against the
Stage 0 artifacts.  The profile's parameters are pinned here; the parity of the
parts it produces needs both CoACD and ManifoldPlus installed, and the test says
so rather than pretending.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import trimesh

from qdgrasp.objects.coacd import (
    UPSTREAM_ARGUMENT_NAMES,
    CoACDAlgorithmConfig,
    CoACDConfig,
    CoACDExecutionConfig,
    CoACDExecutionError,
    CollisionValidationError,
    MeshPreprocessConfig,
    MeshRepairUnavailable,
    MeshValidationError,
    TooManyConvexPartsError,
    build_collision_asset,
    decompose_collision_mesh,
)


def _stub_backend(mesh, **kwargs):
    """Return two disjoint boxes, ignoring the parameters it was handed."""

    del mesh, kwargs
    left = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    right = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    right.apply_translation([0.03, 0.0, 0.0])
    return [(np.asarray(part.vertices), np.asarray(part.faces)) for part in (left, right)]


@pytest.fixture(scope="module")
def mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.06, 0.04, 0.02))


# -- profiles --------------------------------------------------------------


def test_stage0_and_dexgraspnet_profiles_are_distinct_and_pinned() -> None:
    """Neither profile may be described as "the old default"; both are named."""

    stage0 = CoACDConfig.from_profile("legacy_kaggle_stage0_0_1_v1")
    standard = CoACDConfig.from_profile("legacy_dexgraspnet_standard_0_4_v1")

    assert stage0.algorithm.threshold == pytest.approx(0.1)
    assert stage0.algorithm.seed == 0
    assert stage0.mesh_preprocess.simplify_faces == 5000
    assert stage0.mesh_preprocess.normalize_diagonal_m == pytest.approx(2.0)

    assert standard.algorithm.threshold == pytest.approx(0.4)
    assert standard.mesh_preprocess.normalize_diagonal_m == pytest.approx(2.0)

    assert stage0.content_hash() != standard.content_hash()


def test_the_metric_profile_uses_metres_and_no_diagonal_normalisation() -> None:
    config = CoACDConfig.from_profile("qdgrasp_metric_v1")
    assert config.algorithm.real_metric is True
    assert config.mesh_preprocess.normalize_diagonal_m is None


def test_overriding_a_profile_makes_it_custom() -> None:
    config = CoACDConfig.from_profile("upstream_default_v1", algorithm=CoACDAlgorithmConfig(threshold=0.2))
    assert config.profile == "custom"
    with pytest.raises(MeshValidationError):
        CoACDConfig.from_profile("custom")
    with pytest.raises(MeshValidationError):
        CoACDConfig.from_profile("no_such_profile_v9")


def test_real_metric_and_diagonal_normalisation_cannot_be_combined() -> None:
    config = CoACDConfig(
        mesh_preprocess=MeshPreprocessConfig(normalize_diagonal_m=2.0),
        algorithm=CoACDAlgorithmConfig(real_metric=True),
    )
    with pytest.raises(MeshValidationError, match="real_metric"):
        config.validate()


# -- parameters ------------------------------------------------------------


def test_every_official_parameter_is_typed_and_hashed() -> None:
    """No ``**kwargs`` passthrough: each field is mapped and each changes the hash."""

    fields = {field.name for field in dataclasses.fields(CoACDAlgorithmConfig)}
    assert fields == set(UPSTREAM_ARGUMENT_NAMES)

    base = CoACDConfig()
    for name in fields:
        current = getattr(base.algorithm, name)
        if isinstance(current, bool):
            changed = not current
        elif isinstance(current, int):
            changed = current + 1 if name != "max_convex_hull" else 4
        elif isinstance(current, float):
            changed = current + 0.01
        else:
            changed = "box" if current == "ch" else "on"
        mutated = dataclasses.replace(base, algorithm=dataclasses.replace(base.algorithm, **{name: changed}))
        assert mutated.algorithm_hash() != base.algorithm_hash(), f"{name} is missing from the config hash"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", 5.0),
        ("max_convex_hull", 0),
        ("preprocess_mode", "sometimes"),
        ("preprocess_resolution", 5),
        ("resolution", 1),
        ("mcts_nodes", 1),
        ("mcts_iterations", 1),
        ("mcts_max_depth", 99),
        ("max_ch_vertex", 1),
        ("extrude_margin", 0.0),
        ("apx_mode", "sphere"),
        ("seed", -1),
    ],
)
def test_out_of_range_and_invalid_enums_are_refused(field: str, value: object) -> None:
    algorithm = dataclasses.replace(CoACDAlgorithmConfig(), **{field: value})
    with pytest.raises(MeshValidationError):
        algorithm.validate()


def test_an_unsupported_upstream_parameter_is_refused_not_ignored(mesh: trimesh.Trimesh) -> None:
    """A silently dropped parameter would leave the config hash claiming it applied."""

    def narrow_backend(payload, threshold=0.05, seed=0):
        del payload, threshold, seed
        return _stub_backend(None)

    with pytest.raises(CoACDExecutionError, match="does not accept"):
        decompose_collision_mesh(mesh, config=CoACDConfig(), backend=narrow_backend, tool_version="narrow")


# -- execution -------------------------------------------------------------


def test_decomposition_records_its_provenance(mesh: trimesh.Trimesh) -> None:
    config = CoACDConfig.from_profile("qdgrasp_metric_v1")
    result = decompose_collision_mesh(mesh, config=config, backend=_stub_backend, tool_version="stub-1")
    assert result.piece_count == 2
    assert len(result.part_sha256) == 2
    assert result.profile == "qdgrasp_metric_v1"
    assert result.config_hash == config.content_hash()
    assert result.tool_version == "stub-1"
    assert result.total_volume_m3 > 0.0
    assert not result.cache_hit


def test_the_cache_is_content_addressed(tmp_path, mesh: trimesh.Trimesh) -> None:
    config = CoACDConfig.from_profile("qdgrasp_metric_v1")
    first = decompose_collision_mesh(
        mesh, config=config, cache_dir=tmp_path, backend=_stub_backend, tool_version="stub-1"
    )
    second = decompose_collision_mesh(
        mesh, config=config, cache_dir=tmp_path, backend=_stub_backend, tool_version="stub-1"
    )
    assert not first.cache_hit and second.cache_hit
    assert first.part_sha256 == second.part_sha256

    # A different tool build is a different cache entry, not the same one.
    third = decompose_collision_mesh(
        mesh, config=config, cache_dir=tmp_path, backend=_stub_backend, tool_version="stub-2"
    )
    assert not third.cache_hit


def test_an_empty_decomposition_is_an_error_not_a_single_hull(mesh: trimesh.Trimesh) -> None:
    with pytest.raises(CollisionValidationError):
        decompose_collision_mesh(mesh, config=CoACDConfig(), backend=lambda *_a, **_k: [], tool_version="stub")


def test_too_many_parts_fails_closed(mesh: trimesh.Trimesh) -> None:
    def many(payload, **kwargs):
        del payload, kwargs
        return _stub_backend(None) * 5

    config = CoACDConfig(execution=CoACDExecutionConfig(max_output_parts=3))
    with pytest.raises(TooManyConvexPartsError):
        decompose_collision_mesh(mesh, config=config, backend=many, tool_version="stub")


def test_input_budgets_fail_closed(mesh: trimesh.Trimesh) -> None:
    config = CoACDConfig(execution=CoACDExecutionConfig(max_input_faces=2))
    with pytest.raises(MeshValidationError, match="max_input_faces"):
        decompose_collision_mesh(mesh, config=config, backend=_stub_backend, tool_version="stub")


def test_manifoldplus_repair_is_declared_unavailable_rather_than_skipped(mesh: trimesh.Trimesh) -> None:
    config = CoACDConfig.from_profile("legacy_kaggle_stage0_0_1_v1")
    with pytest.raises(MeshRepairUnavailable):
        decompose_collision_mesh(mesh, config=config, backend=_stub_backend, tool_version="stub")


def test_a_missing_backend_is_a_typed_error(mesh: trimesh.Trimesh) -> None:
    try:
        import coacd  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        with pytest.raises(CoACDExecutionError, match="not installed"):
            decompose_collision_mesh(mesh, config=CoACDConfig())
    else:  # pragma: no cover - only where the wheel is installed
        pytest.skip("CoACD is installed; the missing-backend path cannot be exercised")


# -- mass ------------------------------------------------------------------


def test_mass_and_density_are_mutually_exclusive(mesh: trimesh.Trimesh) -> None:
    with pytest.raises(MeshValidationError, match="mutually exclusive"):
        build_collision_asset(mesh, config=CoACDConfig(), mass_kg=0.2, density_kg_m3=600.0, backend=_stub_backend)


def test_without_a_mass_the_asset_stops_at_geometry_ready(mesh: trimesh.Trimesh) -> None:
    """The old batch script's 0.2 kg default is not a fallback for any object."""

    asset = build_collision_asset(mesh, config=CoACDConfig(), backend=_stub_backend)
    assert asset.status == "geometry_ready"
    assert asset.mass_kg is None and asset.inertia_kg_m2 is None


def test_a_supplied_mass_yields_a_dynamic_asset(mesh: trimesh.Trimesh) -> None:
    asset = build_collision_asset(mesh, config=CoACDConfig(), mass_kg=0.2, backend=_stub_backend)
    assert asset.status == "dynamic_ready"
    assert asset.mass_kg == pytest.approx(0.2)
    assert asset.inertia_kg_m2 is not None and all(value > 0.0 for value in asset.inertia_kg_m2)
    assert asset.density_kg_m3 == pytest.approx(0.2 / asset.result.total_volume_m3)
