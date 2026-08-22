from __future__ import annotations

import pytest
import yaml

from qdgrasp.config import ConfigError, dump_document, load_robot_config, parse_document
from qdgrasp.config.schema import ROBOT_SCHEMA_V1
from qdgrasp.robot.provenance import get_profile_provenance, validate_profile_for_release
from qdgrasp.robot.schema import ROBOT_SCHEMA_V2, RobotConfigV2


def test_robot_v1_still_loads_dummy_hand() -> None:
    cfg = load_robot_config("dummy-hand.yaml")
    assert cfg.schema_version == ROBOT_SCHEMA_V1
    assert cfg.name == "dummy-hand"
    assert len(cfg.joints) == 4


def test_robot_v2_presets_round_trip() -> None:
    presets = ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml")
    for name in presets:
        cfg = load_robot_config(name)
        assert isinstance(cfg, RobotConfigV2)
        assert cfg.schema_version == ROBOT_SCHEMA_V2
        assert len(cfg.joints) in (16, 20)

        dumped = dump_document(cfg)
        reparsed = parse_document(yaml.safe_load(dumped), RobotConfigV2, origin=name)
        assert reparsed == cfg
        assert reparsed.content_hash() == cfg.content_hash()


def test_robot_v2_rejects_unknown_key() -> None:
    with pytest.raises(ConfigError, match="extra_field"):
        parse_document(
            {
                "schema": "qdgrasp/robot/v2",
                "name": "bad",
                "source_asset": "dummy.xml",
                "palm_link": "palm",
                "joints": ["j1"],
                "joint_limits": {"j1": [0.0, 1.0]},
                "extra_field": 123,
            },
            RobotConfigV2,
            origin="test",
        )


def test_robot_v2_rejects_duplicate_joints() -> None:
    with pytest.raises(ConfigError, match="unique"):
        parse_document(
            {
                "schema": "qdgrasp/robot/v2",
                "name": "bad",
                "source_asset": "dummy.xml",
                "palm_link": "palm",
                "joints": ["j1", "j1"],
                "joint_limits": {"j1": [0.0, 1.0]},
            },
            RobotConfigV2,
            origin="test",
        )


def test_robot_v2_rejects_infinite_limits() -> None:
    with pytest.raises(ConfigError, match="finite limits"):
        parse_document(
            {
                "schema": "qdgrasp/robot/v2",
                "name": "bad",
                "source_asset": "dummy.xml",
                "palm_link": "palm",
                "joints": ["j1"],
                "joint_limits": {"j1": [0.0, float("inf")]},
            },
            RobotConfigV2,
            origin="test",
        )


def test_provenance_enforcement_and_release_blocked() -> None:
    cfg = load_robot_config("leap_hand.yaml")
    assert isinstance(cfg, RobotConfigV2)
    validate_profile_for_release(cfg)
    summary = get_profile_provenance(cfg)
    assert summary["profile_name"] == "leap_hand"
    assert summary["release_blocked"] is False

    blocked = RobotConfigV2(
        schema="qdgrasp/robot/v2",
        name="barrett_test",
        format="urdf",
        source_asset="barrett.urdf",
        palm_link="base",
        joints=("j1",),
        joint_limits={"j1": (0.0, 1.0)},
        release_blocked=True,
        provenance={"restriction_reason": "research_only"},
    )
    with pytest.raises(ConfigError, match="release_blocked=True"):
        validate_profile_for_release(blocked)
