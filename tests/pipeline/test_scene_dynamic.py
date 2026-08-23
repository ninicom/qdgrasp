import pytest
import numpy as np
from qdgrasp.dataset.pipeline.validators.scene_dynamic import SceneDynamicValidator

def test_target_not_lifted():
    validator = SceneDynamicValidator()
    res = validator.validate("target", {}, {}, target_lifted=False)
    assert not res.passed
    assert res.failure_stage == "target_not_lifted"

def test_non_target_disturbed():
    validator = SceneDynamicValidator(displacement_threshold=0.05)

    init_state = {
        "target": {"pos": np.array([0, 0, 0])},
        "obstacle": {"pos": np.array([0, 1, 0])}
    }

    # Obstacle moved by 0.1 on y-axis
    final_state = {
        "target": {"pos": np.array([0, 0, 1])},
        "obstacle": {"pos": np.array([0, 1.1, 0])}
    }

    res = validator.validate("target", init_state, final_state, target_lifted=True)
    assert not res.passed
    assert res.failure_stage == "non_target_disturbed"
    assert res.trajectory_metrics["disturbed_object"] == "obstacle"

def test_validation_passed():
    validator = SceneDynamicValidator(displacement_threshold=0.05)

    init_state = {
        "target": {"pos": np.array([0, 0, 0])},
        "obstacle": {"pos": np.array([0, 1, 0])}
    }

    # Obstacle moved by 0.01 (within threshold)
    final_state = {
        "target": {"pos": np.array([0, 0, 1])},
        "obstacle": {"pos": np.array([0, 1.01, 0])}
    }

    res = validator.validate("target", init_state, final_state, target_lifted=True)
    assert res.passed
    assert res.failure_stage == "none"
