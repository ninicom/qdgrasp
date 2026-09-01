#!/usr/bin/env python3
"""Build and inspect the wheel from outside the source tree.

This protects package-data globs and verifies that an installed distribution can
discover the nested robot presets while robot meshes remain an explicit external
asset dependency.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MEMBERS = {
    "qdgrasp/assets/derived/allegro_hand_description_right_A.normalized.urdf",
    "qdgrasp/assets/derived/allegro_hand_description_right_A.normalized.urdf.manifest.json",
    "qdgrasp/presets/robots/leap_hand.yaml",
    "qdgrasp/presets/robots/wonik_allegro.yaml",
    "qdgrasp/presets/robots/shadow_hand.yaml",
}
FORBIDDEN_PREFIXES = ("qdgrasp/data/", "qdgrasp/nn/")
FORBIDDEN_MEMBERS = {"qdgrasp/models/data.py"}
ASSET_ROOT = ROOT / ".references" / "robot-assets"


def build_wheel(output_dir: Path) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build the wheel check")
    # Build from a fresh, minimal source tree.  Setuptools' reusable ``build/``
    # directory can retain packages that were removed from discovery and then
    # silently copy them into a later wheel, defeating a package quarantine.
    source_root = output_dir.parent / "source"
    source_root.mkdir(parents=True)
    for filename in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
        shutil.copy2(ROOT / filename, source_root / filename)
    shutil.copytree(
        ROOT / "qdgrasp",
        source_root / "qdgrasp",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(output_dir)],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"wheel build failed: {result.stderr.strip() or result.stdout.strip()}")
    wheels = sorted(output_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel, found {wheels}")
    return wheels[0]


def verify_members(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    missing = sorted(REQUIRED_MEMBERS - members)
    if missing:
        raise RuntimeError(f"wheel is missing required nested package data: {missing}")
    forbidden = sorted(member for member in members if member.startswith(FORBIDDEN_PREFIXES))
    forbidden.extend(sorted(FORBIDDEN_MEMBERS & members))
    if forbidden:
        raise RuntimeError(f"wheel contains quarantined legacy namespaces: {forbidden[:20]}")


def verify_installed_profile(wheel: Path, install_root: Path, cwd: Path) -> None:
    cwd.mkdir()
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install the isolated wheel target")
    install = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(install_root),
            str(wheel),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if install.returncode:
        raise RuntimeError(f"could not install wheel into isolated target: {install.stderr.strip()}")
    code = """
from qdgrasp.config import preset_names
from qdgrasp.robot import RobotSpec

required = {'robots/leap_hand.yaml', 'robots/wonik_allegro.yaml', 'robots/shadow_hand.yaml'}
missing = required - set(preset_names())
if missing:
    raise RuntimeError(f'nested presets missing after wheel install: {sorted(missing)}')
spec = RobotSpec.from_config('robots/leap_hand.yaml', sample_anchors=False)
if spec.config.name != 'leap_hand':
    raise RuntimeError(f'wrong profile loaded: {spec.config.name}')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(install_root)
    environment["QDGRASP_ROBOT_ASSETS_ROOT"] = str(ASSET_ROOT)
    check = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=environment, check=False)
    if check.returncode:
        raise RuntimeError("installed wheel cannot discover and load the LEAP robot profile")


def main() -> int:
    if not ASSET_ROOT.is_dir():
        print(f"Wheel gate: FAIL\n- robot assets missing at {ASSET_ROOT}")
        return 1
    try:
        with tempfile.TemporaryDirectory(prefix="qdgrasp-wheel-") as directory:
            temp = Path(directory)
            wheel = build_wheel(temp / "dist")
            verify_members(wheel)
            verify_installed_profile(wheel, temp / "site", temp / "outside-source")
    except Exception as exc:  # noqa: BLE001 - print one fail-closed gate result
        print(f"Wheel gate: FAIL\n- {exc}")
        return 1
    print("Wheel gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
