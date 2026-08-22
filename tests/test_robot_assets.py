from __future__ import annotations

from pathlib import Path

import pytest

from qdgrasp.config import ConfigError
from qdgrasp.robot.assets import ROBOT_ASSET_ROOT_ENV, resolve_robot_asset


def test_asset_uri_rejects_parent_traversal() -> None:
    with pytest.raises(ConfigError, match="invalid robot asset URI"):
        resolve_robot_asset("asset://mujoco-menagerie/../outside.xml")


def test_asset_uri_rejects_absolute_path() -> None:
    with pytest.raises(ConfigError, match="invalid robot asset URI"):
        resolve_robot_asset("asset:///tmp/outside.xml")


def test_explicit_asset_root_precedes_source_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    relative = Path("mujoco-menagerie/leap_hand/right_hand.xml")
    expected = tmp_path / relative
    expected.parent.mkdir(parents=True)
    expected.write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setenv(ROBOT_ASSET_ROOT_ENV, str(tmp_path))

    assert resolve_robot_asset(f"asset://{relative.as_posix()}") == expected.resolve()


def test_source_fallback_does_not_depend_on_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROBOT_ASSET_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = resolve_robot_asset("asset://mujoco-menagerie/leap_hand/right_hand.xml")

    assert resolved.name == "right_hand.xml"
    assert ".references/robot-assets" in resolved.as_posix()
