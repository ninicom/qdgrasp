from __future__ import annotations

from pathlib import Path

import pytest

from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.sim.labeling import evaluate_grasp_physics


def test_evaluate_grasp_physics_deterministic() -> None:
    xml_path = Path(".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml")
    if not xml_path.is_file():
        pytest.skip("LEAP MJCF reference not present")

    geoms = [SubGeomSpec(type="box", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.0))]

    res1 = evaluate_grasp_physics(
        hand_xml_path=xml_path,
        collision_geoms=geoms,
        palm_pos=(0.0, 0.0, 0.1),
        object_pos=(0.0, 0.0, 0.05),
        seed=42,
    )
    res2 = evaluate_grasp_physics(
        hand_xml_path=xml_path,
        collision_geoms=geoms,
        palm_pos=(0.0, 0.0, 0.1),
        object_pos=(0.0, 0.0, 0.05),
        seed=42,
    )

    assert res1.metrics == res2.metrics
    assert res1.success == res2.success
    assert res1.stable_lift == res2.stable_lift
    assert res1.contact_count == res2.contact_count
