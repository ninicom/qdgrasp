import pytest
import numpy as np

from qdgrasp.scenes.contracts import SceneObservation
from qdgrasp.scenes.target_selection import (
    select_target_uniform_visible,
    select_target_difficulty_weighted,
    select_target_declutter_ordered,
    get_target_selector
)

@pytest.fixture
def mock_obs():
    return SceneObservation(
        scene_id="test",
        camera_id="cam",
        frame_id="0",
        timestamp=0.0,
        T_world_camera=np.eye(4),
        calibration_hash="hash",
        visibility_by_object={
            "obj1": 0.9,
            "obj2": 0.5,
            "obj3": 0.01  # Below threshold
        }
    )

def test_uniform_visible(mock_obs):
    rng = np.random.default_rng(42)
    selected = select_target_uniform_visible(mock_obs, rng, min_visibility=0.05)
    assert selected in ["obj1", "obj2"]

def test_difficulty_weighted(mock_obs):
    rng = np.random.default_rng(42)
    selected = select_target_difficulty_weighted(mock_obs, rng, min_visibility=0.05)
    assert selected in ["obj1", "obj2"]

def test_declutter_ordered(mock_obs):
    selected = select_target_declutter_ordered(mock_obs, min_visibility=0.05)
    # obj1 has highest visibility (0.9)
    assert selected == "obj1"

def test_no_valid_targets():
    obs = SceneObservation(
        scene_id="test",
        camera_id="cam",
        frame_id="0",
        timestamp=0.0,
        T_world_camera=np.eye(4),
        calibration_hash="hash",
        visibility_by_object={"obj1": 0.01}
    )
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="No objects meet"):
        select_target_uniform_visible(obs, rng, min_visibility=0.05)
