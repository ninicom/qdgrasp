"""Resolve robot assets that deliberately live outside the QDGrasp wheel."""

from __future__ import annotations

import os
from pathlib import Path

from ..config.schema import ConfigError

ROBOT_ASSET_URI_PREFIX = "asset://"
ROBOT_ASSET_ROOT_ENV = "QDGRASP_ROBOT_ASSETS_ROOT"


def resolve_robot_asset(reference: str | Path) -> Path:
    """Resolve a filesystem path or an ``asset://`` robot asset URI.

    Robot meshes and MJCFs are not redistributed in the main wheel.  Installed
    users therefore provide the checked-out asset root explicitly, while a
    source checkout keeps the conventional ``.references/robot-assets``
    fallback for development and CI.
    """

    value = str(reference)
    direct = Path(value)
    if direct.is_file():
        return direct.resolve()
    if not value.startswith(ROBOT_ASSET_URI_PREFIX):
        raise ConfigError(
            f"robot asset '{value}' was not found. Use an existing file path or an "
            f"{ROBOT_ASSET_URI_PREFIX} URI with {ROBOT_ASSET_ROOT_ENV} set."
        )

    relative = Path(value.removeprefix(ROBOT_ASSET_URI_PREFIX))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ConfigError(f"invalid robot asset URI '{value}'")

    roots: list[Path] = []
    configured_root = os.environ.get(ROBOT_ASSET_ROOT_ENV)
    if configured_root:
        roots.append(Path(configured_root))
    # Source checkouts have a stable fallback relative to this module.  Do not
    # consult cwd: the same profile must resolve identically after chdir.
    roots.append(Path(__file__).resolve().parents[2] / ".references" / "robot-assets")
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()

    hint = (
        f"Set {ROBOT_ASSET_ROOT_ENV} to the directory containing '{relative.parts[0]}' "
        "from the pinned robot-assets checkout."
    )
    raise ConfigError(f"robot asset URI '{value}' could not be resolved. {hint}")
