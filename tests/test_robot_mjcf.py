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
