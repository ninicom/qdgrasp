import pytest

from qdgrasp.dataset.pipeline.validators.mujoco_rollout import RolloutSceneObject
from qdgrasp.dataset.pipeline.validators.scene_rollout import validate_scene_grasp_rollout
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.scenes.release_recipes import build_release_grasp_recipe


@pytest.mark.parametrize("profile", ["leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"])
def test_release_recipe_is_deterministic_and_passes_measured_scene_rollout(profile):
    recipe = build_release_grasp_recipe(profile)
    repeated = build_release_grasp_recipe(profile)
    assert recipe.recipe_hash == repeated.recipe_hash
    assert recipe.protocol_hash == repeated.protocol_hash
    obstacle = RolloutSceneObject(
        object_id="obstacle",
        collision_geoms=(
            SubGeomSpec(
                type="box",
                size=(0.015, 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
        pos=(0.3, 0.0, 0.02),
        mass=0.02,
    )
    validation = validate_scene_grasp_rollout(
        recipe.hand_xml_path,
        recipe.target_geoms,
        recipe.robot_spec.fingertip_links,
        target_object_id="target",
        non_target_objects=[obstacle],
        protocol_hash=recipe.protocol_hash,
        recipe_hash=recipe.recipe_hash,
        source_hash="d" * 64,
        rollout_kwargs=recipe.rollout_kwargs,
    )
    assert validation.passed, validation.trajectory_metrics
    assert validation.failure_stage == "none"
    assert validation.trajectory_metrics["swept_clearance_passed"] == 1.0
    assert validation.trajectory_metrics["non_target_motion"]["obstacle"]["impulse"] == 0.0
