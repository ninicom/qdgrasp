from __future__ import annotations

from pathlib import Path
import pytest

from qdgrasp.config import ConfigError
from qdgrasp.robot.mjcf import parse_mjcf


def test_parse_leap_mjcf() -> None:
    path = Path(".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml")
    if not path.is_file():
        pytest.skip("LEAP MJCF reference not present")
    model = parse_mjcf(path)
    assert model.nq == 16
    assert model.nu == 16
    assert model.nbody == 18
    assert "palm" in model.bodies
    model.validate_semantic_bodies(
        palm_body="palm",
        fingertip_bodies=("if_ds", "mf_ds", "rf_ds", "th_ds"),
    )


def test_parse_allegro_mjcf() -> None:
    path = Path(".references/robot-assets/mujoco-menagerie/wonik_allegro/right_hand.xml")
    if not path.is_file():
        pytest.skip("Allegro MJCF reference not present")
    model = parse_mjcf(path)
    assert model.nq == 16
    assert model.nu == 16
    assert model.nbody == 22
    assert "palm" in model.bodies
    model.validate_semantic_bodies(
        palm_body="palm",
        fingertip_bodies=("ff_tip", "mf_tip", "rf_tip", "th_tip"),
    )


def test_parse_shadow_mjcf_coupling() -> None:
    path = Path(".references/robot-assets/mujoco-menagerie/shadow_hand/right_hand.xml")
    if not path.is_file():
        pytest.skip("Shadow MJCF reference not present")
    model = parse_mjcf(path)
    assert model.nq == 24
    assert model.nu == 20
    assert model.nbody == 26
    assert "rh_palm" in model.bodies
    model.validate_semantic_bodies(
        palm_body="rh_palm",
        fingertip_bodies=("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"),
    )


def test_declared_mimic_must_match_the_asset_tendon() -> None:
    """The profile's coupling ratio is checked against the MJCF's own tendon."""

    from qdgrasp.config import ConfigError, load_robot_config
    from qdgrasp.robot.schema import RobotConfigV2
    from qdgrasp.robot.spec import RobotSpec

    document = load_robot_config("shadow_hand.yaml").to_document()
    assert document["mimic_joints"]["rh_FFJ1"]["multiplier"] == 1.0
    document["mimic_joints"]["rh_FFJ1"]["multiplier"] = 0.5

    with pytest.raises(ConfigError, match="contradicts tendon"):
        RobotSpec.from_config(RobotConfigV2.model_validate(document), sample_anchors=False)


def test_shadow_tendons_are_extracted_with_named_coefficients() -> None:
    model = parse_mjcf(".references/robot-assets/mujoco-menagerie/shadow_hand/right_hand.xml")
    assert set(model.tendons) == {"rh_FFJ0", "rh_MFJ0", "rh_RFJ0", "rh_LFJ0"}
    coefficients = dict(model.tendons["rh_FFJ0"].joint_coefficients)
    assert coefficients == {"rh_FFJ2": 1.0, "rh_FFJ1": 1.0}
