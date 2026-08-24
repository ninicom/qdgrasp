import numpy as np
import pytest
import trimesh

from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset


@pytest.fixture
def test_box_mesh():
    return trimesh.creation.box(extents=[0.05, 0.05, 0.05])


@pytest.fixture
def test_collision_geoms():
    return [SubGeomSpec(type="box", size=(0.025, 0.025, 0.025), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))]


@pytest.mark.parametrize("recipe_id", ["surface_fixed_v1", "region_opposition_v1", "wrench_guided_v1"])
def test_orchestrator_recipes(recipe_id, test_box_mesh, test_collision_geoms):
    spec = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    xml_path = resolve_robot_asset(spec.config.source_asset)
    rng = np.random.default_rng(42)

    outcomes, reasons = run_pipeline_chunk(
        recipe_id=recipe_id,
        spec=spec,
        mesh=test_box_mesh,
        collision_geoms=test_collision_geoms,
        hand_xml_path=xml_path,
        rng=rng,
        num_candidates=2,
        run_dynamic=False,  # Test static flow fast
    )

    assert len(outcomes) == 2
    assert sum(reasons.values()) == 2
    for outcome in outcomes:
        assert isinstance(outcome.proposal_valid, bool)
        assert isinstance(outcome.ik_valid, bool)
        assert isinstance(outcome.failure_stage, str)
        assert isinstance(outcome.failure_reason, str)
