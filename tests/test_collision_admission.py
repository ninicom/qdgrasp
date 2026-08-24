"""Exact compiled-scene collision admission oracles for P3.2.1-07."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qdgrasp.dataset.pipeline.validators.collision_admission import (
    admit_mujoco_collision_pose,
)
from qdgrasp.objects.schema import SubGeomSpec


HAND_XML = """
<mujoco model="collision_admission_fixture">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="palm">
      <geom name="palm_collision" type="sphere" size="0.01"/>
      <body name="tip0" pos="0 0 -0.035">
        <geom name="tip0_collision" type="sphere" size="0.005"/>
      </body>
      <body name="tip1" pos="0.08 0 -0.035">
        <geom name="tip1_collision" type="sphere" size="0.005"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _xml(tmp_path: Path) -> Path:
    path = tmp_path / "hand.xml"
    path.write_text(HAND_XML, encoding="utf-8")
    return path


def _admit(tmp_path: Path, *, palm_pos, active=(True, False)):
    return admit_mujoco_collision_pose(
        _xml(tmp_path),
        [SubGeomSpec(type="box", size=(0.025, 0.025, 0.025))],
        palm_body_name="palm",
        fingertip_body_names=("tip0", "tip1"),
        active_fingers=np.asarray(active, dtype=bool),
        palm_pos=np.asarray(palm_pos, dtype=np.float64),
        palm_rot=np.eye(3),
        joint_targets={},
        object_pos=(0.0, 0.0, 0.025),
        object_mass=0.1,
    )


def test_active_fingertip_object_contact_is_the_only_allowed_contact(tmp_path: Path) -> None:
    result = _admit(tmp_path, palm_pos=(0.0, 0.0, 0.09))
    assert result.passed
    assert result.reason == "passed"
    assert any(
        "tip0_collision" in (pair["geom1"], pair["geom2"])
        and any(str(name).startswith("object_subgeom_") for name in (pair["geom1"], pair["geom2"]))
        for pair in result.contact_pairs
    )
    assert result.min_hand_floor_clearance > 0.0


def test_same_contact_is_rejected_when_fingertip_is_inactive(tmp_path: Path) -> None:
    result = _admit(
        tmp_path,
        palm_pos=(0.0, 0.0, 0.09),
        active=(False, True),
    )
    assert not result.passed
    assert result.reason.startswith("forbidden_object_contact")


def test_palm_object_penetration_is_rejected_with_geom_evidence(tmp_path: Path) -> None:
    result = _admit(tmp_path, palm_pos=(0.0, 0.0, 0.055))
    assert not result.passed
    assert result.reason.startswith("forbidden_object_contact")
    assert result.max_penetration > 0.0
    assert any(
        "palm_collision" in (pair["geom1"], pair["geom2"])
        for pair in result.contact_pairs
    )


def test_hand_floor_penetration_is_rejected_independently_of_object(tmp_path: Path) -> None:
    result = _admit(tmp_path, palm_pos=(0.2, 0.0, 0.005))
    assert not result.passed
    assert result.reason == "hand_floor_contact"
    assert result.min_hand_floor_clearance < 0.0
