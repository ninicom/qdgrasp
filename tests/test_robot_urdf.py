from __future__ import annotations

from pathlib import Path

import pytest

from qdgrasp.config import ConfigError
from qdgrasp.robot.urdf import parse_urdf


def test_parse_leap_urdf() -> None:
    path = Path(".references/robot-assets/leap-hand-sim/assets/leap_hand/robot.urdf")
    if not path.is_file():
        pytest.skip("LEAP URDF reference not present")
    model = parse_urdf(path)
    assert model.name in ("leap_hand", "onshape")
    assert len(model.links) == 17
    assert len(model.movable_joints) == 16
    assert len(model.topological_links()) == 17

    model.validate_semantic_links(
        palm_link="palm_lower",
        fingertip_links=("fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"),
    )


def test_parse_allegro_urdf() -> None:
    path = Path(
        ".references/robot-assets/wonik-allegro-ros2/src/allegro_hand_controllers/urdf/allegro_hand_description_right_A.urdf"
    )
    if not path.is_file():
        pytest.skip("Allegro URDF reference not present")
    model = parse_urdf(path)
    assert len(model.links) == 22
    assert len(model.movable_joints) == 16


def test_parse_dex_shadow_urdf() -> None:
    path = Path(".references/robot-assets/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf")
    if not path.is_file():
        pytest.skip("dex-urdf Shadow URDF reference not present")
    model = parse_urdf(path)
    assert len(model.links) == 33
    assert len(model.movable_joints) == 24


def test_semantic_link_validation_rejects_missing_palm() -> None:
    path = Path(".references/robot-assets/leap-hand-sim/assets/leap_hand/robot.urdf")
    if not path.is_file():
        pytest.skip("LEAP URDF reference not present")
    model = parse_urdf(path)
    with pytest.raises(ConfigError, match="declared palm_link 'imaginary_palm' does not exist"):
        model.validate_semantic_links(palm_link="imaginary_palm")
