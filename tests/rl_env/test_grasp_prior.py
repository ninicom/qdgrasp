"""P3.5-11/17: the fitted pinch prior, and the artifact it makes possible.

The assertion that matters is the first one: both active hands acquire the
target through the environment's ordinary bounded action.  Without it the
environments are only demonstrably *safe*, and "an RL environment nothing can
succeed in" is not readiness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qdgrasp.objects.generate import generate_box
from qdgrasp.objects.manifest import create_object_asset, save_object_asset
from qdgrasp.rl.envs import ACTIVE_ROBOT_PROFILES, DexAcquireConfig, DexAcquireEnv
from qdgrasp.rl.tasks.grasp_prior import (
    PINCH_POSTURES,
    GraspPriorSpec,
    build_pinch_prior,
    prior_frame_to_world,
    run_prior_episode,
    target_pinch_frame,
)
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.scenes.virtual_drop import DropObjectRequest, SpawnRegion, VirtualDropSceneSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def target_ref(tmp_path_factory) -> str:
    directory = tmp_path_factory.mktemp("prior-assets")
    rng = np.random.default_rng(0)
    mesh, geoms, params, mass, inertia = generate_box(rng, size_range=(0.028, 0.034), density=650.0)
    mesh_bytes, manifest = create_object_asset("target", "primitive", "box", mesh, geoms, params, mass, inertia)
    return str(save_object_asset(mesh_bytes, manifest, directory))


@pytest.fixture(scope="module")
def prior_cache() -> dict:
    return {}


def _config(profile: str, target_ref: str, spec: GraspPriorSpec) -> DexAcquireConfig:
    return DexAcquireConfig(
        robot_profile=profile,
        objects=(DropObjectRequest(object_id="target", asset_ref=target_ref),),
        target_object_id="target",
        virtual_scene=VirtualDropSceneSpec(
            spawn_region=SpawnRegion(half_extents=(0.03, 0.03, 0.0)), drop_height_range_m=(0.02, 0.04)
        ),
        max_steps=spec.total_steps,
        settle_steps=400,
    )


def test_both_active_hands_have_a_pinned_pinch_posture() -> None:
    assert set(PINCH_POSTURES) == set(ACTIVE_ROBOT_PROFILES)
    for profile, posture in PINCH_POSTURES.items():
        robot = RobotSpec.from_config(profile, sample_anchors=False)
        assert len(posture) == len(robot.actuated_joint_names)


@pytest.mark.parametrize("profile", ACTIVE_ROBOT_PROFILES)
def test_the_prior_opens_wider_than_it_squeezes(profile: str) -> None:
    """Open must clear the target and squeeze must press into it, or there is no grip."""

    import torch

    robot = RobotSpec.from_config(profile, sample_anchors=False)
    prior = build_pinch_prior(robot, profile, half_width=0.015)
    palm_pos = torch.as_tensor(prior.palm_offset[None], dtype=torch.float32)
    palm_rot = torch.as_tensor(prior.palm_rotation[None], dtype=torch.float32)

    def aperture(q: np.ndarray) -> float:
        tips = robot.fingertip_positions(palm_pos, palm_rot, torch.as_tensor(q[None], dtype=torch.float32))[0].numpy()
        return float(tips[0][0] - tips[3][0]) / 2.0

    assert aperture(prior.open_q) > 0.015 > aperture(prior.squeeze_q)
    assert prior.contact_residual_m < 0.005


def test_the_pinch_axis_is_the_narrowest_horizontal_one() -> None:
    rotation = np.eye(3)
    axis, half_width = target_pinch_frame(rotation, np.array([0.03, 0.01, 0.05]))
    assert half_width == pytest.approx(0.01)
    np.testing.assert_allclose(np.abs(axis), [0.0, 1.0, 0.0], atol=1e-9)

    # The vertical axis is never chosen, even when it is the narrowest.
    _axis, half = target_pinch_frame(rotation, np.array([0.03, 0.04, 0.002]))
    assert half in (pytest.approx(0.03), pytest.approx(0.04))


def test_the_prior_frame_puts_the_pinch_on_x_and_up_on_z() -> None:
    frame = prior_frame_to_world(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(frame[:, 0], [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(frame[:, 2], [0.0, 0.0, 1.0], atol=1e-9)
    assert np.linalg.det(frame) == pytest.approx(1.0)


@pytest.mark.parametrize("profile", ACTIVE_ROBOT_PROFILES)
def test_the_prior_acquires_the_target_through_the_bounded_action(
    profile: str, target_ref: str, prior_cache: dict
) -> None:
    """The environments admit an acquire, for both hands, with no safety violation."""

    spec = GraspPriorSpec()
    result = run_prior_episode(
        DexAcquireEnv(_config(profile, target_ref, spec)), seed=21, spec=spec, prior_cache=prior_cache
    )
    assert result["success"], f"{profile}: {result['terminal_reason']} lift={result['max_lift_m']:.4f}"
    assert result["terminal_reason"] == "success"
    assert result["max_lift_m"] >= 0.05
    assert result["links_in_contact"] >= 2
    assert result["observations_finite"]


def test_a_target_wider_than_the_aperture_is_not_acquired(tmp_path, prior_cache: dict) -> None:
    """The negative class: the predicate must be capable of refusing."""

    rng = np.random.default_rng(3)
    mesh, geoms, params, mass, inertia = generate_box(rng, size_range=(0.075, 0.080), density=650.0)
    mesh_bytes, manifest = create_object_asset("too_wide", "primitive", "box", mesh, geoms, params, mass, inertia)
    ref = str(save_object_asset(mesh_bytes, manifest, tmp_path))
    spec = GraspPriorSpec()
    config = DexAcquireConfig(
        robot_profile="leap_hand.yaml",
        objects=(DropObjectRequest(object_id="too_wide", asset_ref=ref),),
        target_object_id="too_wide",
        virtual_scene=VirtualDropSceneSpec(
            spawn_region=SpawnRegion(half_extents=(0.02, 0.02, 0.0)), drop_height_range_m=(0.02, 0.04)
        ),
        max_steps=spec.total_steps,
        settle_steps=400,
    )
    result = run_prior_episode(DexAcquireEnv(config), seed=31, spec=spec, prior_cache=prior_cache)
    assert not result["success"]


def test_the_tiny_artifact_is_complete_and_hashed() -> None:
    root = PROJECT_ROOT / "datasets/qdgrasp-rl-env-tiny"
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    summary = manifest["summary"]
    cases = {item["case"] for item in manifest["cases"]}

    assert {
        "object_only",
        "generated_scene_positive",
        "loaded_scene_positive",
        "negative_out_of_aperture",
        "random_policy",
    } <= cases
    assert summary["positive_successes"] == summary["positive_cases"] > 0
    assert summary["negatives_behaved"] == summary["negative_cases"] > 0
    assert summary["randoms_behaved"] == summary["random_cases"] > 0
    assert manifest["release_class"] == "experimental_non_release"
    assert manifest["shadow_hand"] == "paused_by_ADR-0008"

    hashes = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert hashes["artifacts"], "the artifact manifest must carry raw evidence hashes"
    for entry in hashes["artifacts"]:
        assert len(entry["sha256"]) == 64
        assert (root / entry["path"]).is_file()
