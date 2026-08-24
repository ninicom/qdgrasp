import pytest

from qdgrasp.dataset.pipeline.generated_reachable import (
    build_generated_reachable_object,
    generated_reachable_rng,
)
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset


@pytest.mark.parametrize("profile", ["leap_hand", "wonik_allegro", "shadow_hand"])
def test_generated_reachable_contains_geometry_but_no_grasp_oracle(profile):
    fixture = build_generated_reachable_object(profile)

    assert fixture.mesh.is_volume
    assert fixture.mass == 0.005
    assert fixture.object_pos == (0.0, 0.0, 0.0)
    assert len(fixture.collision_geoms) == 2
    forbidden = {
        "q",
        "qpos",
        "joint_targets",
        "palm_pos",
        "palm_rot",
        "contacts",
        "contact_points",
    }
    assert forbidden.isdisjoint(vars(fixture))


def test_generated_reachable_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unsupported generated-reachable profile"):
        build_generated_reachable_object("unknown_hand")


@pytest.mark.parametrize("profile", ["leap_hand", "wonik_allegro", "shadow_hand"])
def test_pipeline_discovers_full_generated_reachable_positive(profile):
    fixture = build_generated_reachable_object(profile)
    spec = RobotSpec.from_config(f"{profile}.yaml", sample_anchors=False)

    outcomes, accounting = run_pipeline_chunk(
        recipe_id="region_opposition_v1",
        spec=spec,
        mesh=fixture.mesh,
        collision_geoms=fixture.collision_geoms,
        hand_xml_path=resolve_robot_asset(spec.config.source_asset),
        rng=generated_reachable_rng(profile),
        num_candidates=fixture.candidate_budget,
        object_mass=fixture.mass,
        object_pos=fixture.object_pos,
        run_dynamic=True,
    )

    positives = [outcome for outcome in outcomes if outcome.dynamic_valid]
    assert accounting["accepted"] >= 1
    assert positives
    assert all(outcome.failure_stage == "none" for outcome in positives)
