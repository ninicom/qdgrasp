import pytest

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.release import generate_scene_tiny, release_blueprints


def test_release_index_has_required_environment_tier_and_split_isolation():
    blueprints = release_blueprints()
    assert len(blueprints) == 12
    assert {item.environment for item in blueprints} == {"table", "bin", "shelf"}
    assert {item.clutter_tier for item in blueprints} == {"single", "sparse", "dense"}
    assert len({item.scene_id for item in blueprints}) == 12
    assert len({item.template_id for item in blueprints}) == 12
    assert {item.robot_profile for item in blueprints if item.robot_profile} == {
        "leap_hand.yaml",
        "wonik_allegro.yaml",
        "shadow_hand.yaml",
    }


def test_release_dry_run_is_bounded_and_does_not_write(tmp_path):
    summary = generate_scene_tiny(
        tmp_path,
        scene_limit=1,
        frame_limit=1,
        worker_count=1,
        dry_run=True,
    )
    assert summary["scene_count"] == 1
    assert summary["worker_count"] == 1
    assert not summary["full_root_scan"]
    assert not summary["source_copy"]
    assert not list(tmp_path.iterdir())

    with pytest.raises(ConfigError, match="worker_count=1"):
        generate_scene_tiny(tmp_path, worker_count=2, dry_run=True)
    with pytest.raises(ConfigError, match="scene_limit"):
        generate_scene_tiny(tmp_path, scene_limit=13, dry_run=True)
    with pytest.raises(ConfigError, match="frame_limit"):
        generate_scene_tiny(tmp_path, frame_limit=3, dry_run=True)
