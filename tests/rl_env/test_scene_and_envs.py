"""P3.5-06/07/08/10/11: resolution, drop, settle and the environments.

The load-vs-generate distinction gets the most attention here, because it is the
one whose failure mode is silent: a broken scene that quietly becomes a generated
one changes the problem being measured without changing anything visible in the
result.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from qdgrasp.config.schema import ConfigError
from qdgrasp.objects.generate import generate_box
from qdgrasp.objects.manifest import create_object_asset, save_object_asset
from qdgrasp.rl.envs import (
    ACTIVE_ROBOT_PROFILES,
    DexAcquireConfig,
    DexAcquireEnv,
    DexAcquireSceneEnv,
    ObjectSettleConfig,
    ObjectSettleEnv,
    build_hand_scene_model,
)
from qdgrasp.rl.tasks import random_policy_probe, run_scripted_episode
from qdgrasp.scenes.resolver import SceneLoadError, SceneSource, resolve_scene
from qdgrasp.scenes.serialize import load_scene_spec, scene_spec_hash, write_scene_spec
from qdgrasp.scenes.settle import SettleOutcome, certify_settle, classify_settle_failure, replay_snapshot
from qdgrasp.scenes.virtual_drop import (
    DropObjectRequest,
    PlacementError,
    SettleThresholds,
    SpawnRegion,
    VirtualDropSceneSpec,
    build_virtual_drop_scene,
)


@pytest.fixture(scope="module")
def assets(tmp_path_factory) -> dict[str, str]:
    directory = tmp_path_factory.mktemp("assets")
    rng = np.random.default_rng(0)
    refs: dict[str, str] = {}
    for name in ("target", "neighbour"):
        mesh, geoms, params, mass, inertia = generate_box(rng, size_range=(0.028, 0.034), density=650.0)
        mesh_bytes, manifest = create_object_asset(name, "primitive", "box", mesh, geoms, params, mass, inertia)
        refs[name] = str(save_object_asset(mesh_bytes, manifest, directory))
    return refs


def _objects(assets: dict[str, str], *names: str) -> tuple[DropObjectRequest, ...]:
    return tuple(DropObjectRequest(object_id=name, asset_ref=assets[name]) for name in names)


@pytest.fixture(scope="module")
def scene_config() -> VirtualDropSceneSpec:
    return VirtualDropSceneSpec(
        spawn_region=SpawnRegion(half_extents=(0.05, 0.05, 0.0)),
        drop_height_range_m=(0.02, 0.04),
        object_count_range=(1, 2),
    )


# -- resolution ------------------------------------------------------------


def test_no_scene_reference_generates_one(assets, scene_config) -> None:
    resolved = resolve_scene(objects=_objects(assets, "target"), virtual_scene_config=scene_config, seed=1)
    assert resolved.source is SceneSource.GENERATED
    assert resolved.generated
    assert resolved.detail["environment"] == "table"


def test_a_valid_reference_is_loaded_not_regenerated(tmp_path, assets, scene_config) -> None:
    generated = resolve_scene(objects=_objects(assets, "target"), virtual_scene_config=scene_config, seed=2)
    path = write_scene_spec(tmp_path / "scene.json", generated.spec)
    loaded = resolve_scene(scene_ref=path)
    assert loaded.source is SceneSource.LOADED
    assert scene_spec_hash(loaded.spec) == scene_spec_hash(generated.spec)


def test_a_broken_reference_fails_instead_of_falling_back(tmp_path, assets, scene_config) -> None:
    """The silent-substitution failure this whole module exists to prevent."""

    broken = tmp_path / "broken.json"
    broken.write_text('{"schema": "not-a-qdgrasp-scene"}', encoding="utf-8")
    with pytest.raises(SceneLoadError):
        resolve_scene(
            scene_ref=broken,
            objects=_objects(assets, "target"),
            virtual_scene_config=scene_config,
        )


def test_generation_without_objects_is_an_error() -> None:
    with pytest.raises(ConfigError):
        resolve_scene()


def test_scene_documents_round_trip(tmp_path, assets, scene_config) -> None:
    resolved = resolve_scene(objects=_objects(assets, "target", "neighbour"), virtual_scene_config=scene_config, seed=3)
    path = write_scene_spec(tmp_path / "s.json", resolved.spec)
    assert scene_spec_hash(load_scene_spec(path)) == scene_spec_hash(resolved.spec)


# -- drop placement --------------------------------------------------------


def test_the_same_seed_places_objects_identically(assets, scene_config) -> None:
    left = build_virtual_drop_scene(scene_config, _objects(assets, "target", "neighbour"), seed=9)
    right = build_virtual_drop_scene(scene_config, _objects(assets, "target", "neighbour"), seed=9)
    other = build_virtual_drop_scene(scene_config, _objects(assets, "target", "neighbour"), seed=10)
    assert scene_spec_hash(left) == scene_spec_hash(right)
    assert scene_spec_hash(left) != scene_spec_hash(other)


def test_placements_do_not_overlap(assets, scene_config) -> None:
    spec = build_virtual_drop_scene(scene_config, _objects(assets, "target", "neighbour"), seed=11)
    positions = np.stack([item.T_world_object[:3, 3] for item in spec.objects])
    separation = float(np.linalg.norm(positions[0][:2] - positions[1][:2]))
    assert separation > scene_config.initial_clearance_m


def test_an_impossible_placement_raises_rather_than_stacking(assets) -> None:
    """Two objects in a spawn region smaller than one of them cannot both fit."""

    cramped = VirtualDropSceneSpec(
        spawn_region=SpawnRegion(half_extents=(0.001, 0.001, 0.0)),
        object_count_range=(1, 2),
        max_placement_attempts=25,
    )
    with pytest.raises(PlacementError):
        build_virtual_drop_scene(cramped, _objects(assets, "target", "neighbour"), seed=1)


def test_objects_start_above_the_support(assets, scene_config) -> None:
    spec = build_virtual_drop_scene(scene_config, _objects(assets, "target"), seed=12)
    assert spec.objects[0].T_world_object[2, 3] > scene_config.drop_height_range_m[0]


# -- settle ----------------------------------------------------------------


def test_a_dropped_scene_settles_and_snapshots(assets, scene_config) -> None:
    resolved = resolve_scene(objects=_objects(assets, "target", "neighbour"), virtual_scene_config=scene_config, seed=7)
    data = mujoco.MjData(resolved.model)
    snapshot = certify_settle(
        resolved.spec,
        resolved.model,
        data,
        scene_config.settle_thresholds,
        spawn_region=scene_config.spawn_region,
    )
    assert snapshot.outcome is SettleOutcome.SETTLED
    assert snapshot.settled
    assert len(snapshot.objects) == 2
    assert snapshot.telemetry["max_penetration_m"] >= 0.0
    assert snapshot.content_hash() == snapshot.content_hash()

    from qdgrasp.scenes.settle import SceneSnapshot

    assert SceneSnapshot.from_document(snapshot.to_document()) == snapshot


def test_replay_restores_the_snapshot_without_warm_state(assets, scene_config) -> None:
    resolved = resolve_scene(objects=_objects(assets, "target"), virtual_scene_config=scene_config, seed=8)
    data = mujoco.MjData(resolved.model)
    snapshot = certify_settle(resolved.spec, resolved.model, data, scene_config.settle_thresholds)
    replayed = replay_snapshot(resolved.model, snapshot)
    assert replayed is not data
    for state in snapshot.objects:
        body = mujoco.mj_name2id(resolved.model, mujoco.mjtObj.mjOBJ_BODY, state.object_id)
        np.testing.assert_allclose(np.array(replayed.xpos[body]), state.position_m, atol=1e-9)


def test_a_timeout_is_reported_as_a_timeout(assets, scene_config) -> None:
    resolved = resolve_scene(objects=_objects(assets, "target"), virtual_scene_config=scene_config, seed=13)
    impatient = SettleThresholds(consecutive_steps=1, timeout_steps=2)
    snapshot = certify_settle(resolved.spec, resolved.model, mujoco.MjData(resolved.model), impatient)
    assert snapshot.outcome is SettleOutcome.SETTLE_TIMEOUT


def test_failure_classes_have_a_fixed_precedence() -> None:
    """A diverging solve trips several conditions; the cause outranks the effect."""

    assert (
        classify_settle_failure({SettleOutcome.EXCESSIVE_PENETRATION, SettleOutcome.NON_FINITE_STATE})
        is SettleOutcome.NON_FINITE_STATE
    )
    assert (
        classify_settle_failure({SettleOutcome.SETTLE_TIMEOUT, SettleOutcome.FELL_OFF_SUPPORT})
        is SettleOutcome.FELL_OFF_SUPPORT
    )
    with pytest.raises(ValueError):
        classify_settle_failure(set())


# -- object settle environment ---------------------------------------------


def test_the_settle_environment_resets_deterministically(assets, scene_config) -> None:
    config = ObjectSettleConfig(objects=_objects(assets, "target"), virtual_scene=scene_config, max_steps=30)
    first, first_info = ObjectSettleEnv(config).reset(seed=4)
    second, second_info = ObjectSettleEnv(config).reset(seed=4)
    third, _ = ObjectSettleEnv(config).reset(seed=5)
    np.testing.assert_array_equal(first, second)
    assert first_info["scene_signature"] == second_info["scene_signature"]
    assert not np.array_equal(first, third)


def test_the_settle_environment_reaches_rest(assets, scene_config) -> None:
    config = ObjectSettleConfig(objects=_objects(assets, "target"), virtual_scene=scene_config, max_steps=60)
    env = ObjectSettleEnv(config)
    observation, _ = env.reset(seed=3)
    assert observation.shape == (env.observation_space().shape[0],)
    while True:
        observation, _reward, terminated, truncated, info = env.step(np.zeros(1))
        if terminated or truncated:
            break
    assert terminated and info["settled"]
    assert env.certify().outcome is SettleOutcome.SETTLED


# -- acquisition environments ----------------------------------------------


@pytest.mark.parametrize("profile", ACTIVE_ROBOT_PROFILES)
def test_both_active_hands_compile_into_a_scene(profile: str, assets, scene_config) -> None:
    resolved = resolve_scene(objects=_objects(assets, "target"), virtual_scene_config=scene_config, seed=6)
    model, robot = build_hand_scene_model(resolved.spec, profile)
    assert model.nu == len(robot.actuated_joint_names)


def test_shadow_is_refused_while_paused(assets, scene_config) -> None:
    resolved = resolve_scene(objects=_objects(assets, "target"), virtual_scene_config=scene_config, seed=6)
    with pytest.raises(ConfigError, match="paused_by_ADR-0008"):
        build_hand_scene_model(resolved.spec, "shadow_hand.yaml")


def _acquire_config(profile: str, assets, scene_config, **overrides) -> DexAcquireConfig:
    base = {
        "robot_profile": profile,
        "objects": _objects(assets, "target"),
        "target_object_id": "target",
        "virtual_scene": scene_config,
        "max_steps": 40,
        "settle_steps": 400,
    }
    base.update(overrides)
    return DexAcquireConfig(**base)


@pytest.mark.parametrize("profile", ACTIVE_ROBOT_PROFILES)
def test_the_episode_does_not_start_in_contact(profile: str, assets, scene_config) -> None:
    """An episode that begins touching the target measures the placement."""

    env = DexAcquireEnv(_acquire_config(profile, assets, scene_config))
    env.reset(seed=21)
    contact = env._read_contacts()
    assert contact["links_in_contact"] == 0
    assert contact["penetration"] == 0.0


@pytest.mark.parametrize("profile", ACTIVE_ROBOT_PROFILES)
def test_a_random_policy_stays_finite_and_scores_nothing(profile: str, assets, scene_config) -> None:
    env = DexAcquireEnv(_acquire_config(profile, assets, scene_config))
    probe = random_policy_probe(env, seed=21, steps=40)
    assert probe["observations_finite"]
    assert probe["successes"] == 0


@pytest.mark.parametrize("profile", ACTIVE_ROBOT_PROFILES)
def test_the_scripted_fixture_reaches_its_expected_class(profile: str, assets, scene_config) -> None:
    """Pinned so a regression in asset, scene, settle or control is visible.

    The class is ``horizon`` for both active hands: the fixture runs the whole
    episode without tripping a safety barrier or losing the target.  It does not
    complete an acquire, and that is recorded rather than tuned away.
    """

    from qdgrasp.rl.tasks import ScriptedAcquireSpec

    spec = ScriptedAcquireSpec()
    env = DexAcquireEnv(_acquire_config(profile, assets, scene_config, max_steps=spec.total_steps))
    result = run_scripted_episode(env, seed=21, spec=spec)
    assert result["observations_finite"]
    assert result["terminal_reason"] == "horizon"
    assert result["descend_ended_at"] is not None, "the descent must end on contact, not on a step count"


def test_an_out_of_range_action_is_clipped_not_passed_through(assets, scene_config) -> None:
    env = DexAcquireEnv(_acquire_config("leap_hand.yaml", assets, scene_config, max_steps=5))
    env.reset(seed=21)
    dimension = env.action_space().shape[0]
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(np.full(dimension, 50.0))
        commanded = np.asarray(env._data.ctrl[list(env._indices.actuator_ids)])
        assert np.all(commanded >= env._joint_lower - 1e-9)
        assert np.all(commanded <= env._joint_upper + 1e-9)
        if terminated or truncated:
            break


def test_a_wrong_action_dimension_is_rejected(assets, scene_config) -> None:
    env = DexAcquireEnv(_acquire_config("leap_hand.yaml", assets, scene_config, max_steps=5))
    env.reset(seed=21)
    with pytest.raises(ValueError):
        env.step(np.zeros(3))


def test_the_reward_total_equals_its_logged_terms(assets, scene_config) -> None:
    env = DexAcquireEnv(_acquire_config("leap_hand.yaml", assets, scene_config, max_steps=5))
    env.reset(seed=21)
    _, reward, _, _, info = env.step(np.zeros(env.action_space().shape[0]))
    terms = info["reward_terms"]
    assert reward == pytest.approx(terms["total"])
    assert terms["total"] == pytest.approx(sum(v for k, v in terms.items() if k != "total"))


def test_the_clutter_environment_requires_a_non_target(assets, scene_config) -> None:
    with pytest.raises(ValueError, match="clutter"):
        DexAcquireSceneEnv(_acquire_config("leap_hand.yaml", assets, scene_config))


def test_the_clutter_environment_accounts_for_neighbours(assets, scene_config) -> None:
    config = _acquire_config(
        "leap_hand.yaml",
        assets,
        scene_config,
        objects=_objects(assets, "target", "neighbour"),
        max_steps=5,
    )
    env = DexAcquireSceneEnv(config)
    _, info = env.reset(seed=15)
    assert info["non_target_object_ids"] == ["neighbour"]
    _, _, _, _, step_info = env.step(np.zeros(env.action_space().shape[0]))
    assert "neighbour" in step_info["non_target_displacement_m"]
