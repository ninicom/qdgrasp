from __future__ import annotations

from pathlib import Path
import pytest

from qdgrasp.sim.fixtures import evaluate_grasp_fixture
from qdgrasp.sim.mujoco import MujocoSim


def test_mujoco_sim_load_and_step() -> None:
    xml_path = Path(".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml")
    if not xml_path.is_file():
        pytest.skip("LEAP MJCF reference not present")

    sim = MujocoSim(xml_path)
    assert sim.nq == 16
    assert sim.nu == 16
    assert sim.nbody == 18

    sim.step(10)
    pos = sim.get_body_pos("palm")
    assert pos.shape == (3,)


def test_grasp_fixture_deterministic_repeatability() -> None:
    xml_path = Path(".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml")
    if not xml_path.is_file():
        pytest.skip("LEAP MJCF reference not present")

    res1 = evaluate_grasp_fixture(xml_path, seed=42)
    res2 = evaluate_grasp_fixture(xml_path, seed=42)

    assert res1.metrics == res2.metrics
    assert res1.success == res2.success
    assert res1.stable_lift == res2.stable_lift
