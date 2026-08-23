import pytest
import numpy as np
import mujoco

from qdgrasp.scenes.clearance import check_approach_clearance, ClearanceError

def test_empty_approach_path():
    model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
    data = mujoco.MjData(model)

    with pytest.raises(ClearanceError, match="approach_blocked"):
        check_approach_clearance(
            model=model,
            data=data,
            target_object_id="obj1",
            approach_path=np.array([]),
            hand_geom_ids=[]
        )

def test_mock_clearance_success():
    # Uses the mock implementation which always returns True if path not empty
    model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
    data = mujoco.MjData(model)

    path = np.array([np.eye(4)])
    result = check_approach_clearance(
        model=model,
        data=data,
        target_object_id="obj1",
        approach_path=path,
        hand_geom_ids=[]
    )

    assert result is True
