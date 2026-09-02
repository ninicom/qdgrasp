from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "qdgrasp"
PHASE1_MODULES = (
    PACKAGE_ROOT / "__init__.py",
    PACKAGE_ROOT / "cli.py",
    PACKAGE_ROOT / "geometry.py",
    *sorted((PACKAGE_ROOT / "api").glob("*.py")),
    *sorted((PACKAGE_ROOT / "config").glob("*.py")),
    *sorted((PACKAGE_ROOT / "dummy").glob("*.py")),
    *sorted((PACKAGE_ROOT / "export").glob("*.py")),
    PACKAGE_ROOT / "engine" / "callbacks.py",
    PACKAGE_ROOT / "engine" / "checkpoint.py",
    PACKAGE_ROOT / "engine" / "ema.py",
    PACKAGE_ROOT / "engine" / "runner.py",
    PACKAGE_ROOT / "engine" / "sampling.py",
    PACKAGE_ROOT / "engine" / "seeding.py",
)
FORBIDDEN_BASE_IMPORTS = ("ultralytics", "cv2", "MinkowskiEngine", "spconv", "open3d", "pytorch3d", "isaacgym")


def test_import_does_not_change_the_working_directory() -> None:
    script = "import os; before = os.getcwd(); import qdgrasp; print(before == os.getcwd())"
    output = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=os.getcwd(), check=False)
    assert output.returncode == 0, output.stderr
    assert output.stdout.strip() == "True"


def test_base_import_pulls_no_unapproved_dependency() -> None:
    script = (
        "import sys, json; import qdgrasp; "
        f"print(json.dumps([name for name in {FORBIDDEN_BASE_IMPORTS!r} if name in sys.modules]))"
    )
    output = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert output.returncode == 0, output.stderr
    assert output.stdout.strip() == "[]"


@pytest.mark.parametrize("path", PHASE1_MODULES, ids=lambda path: path.name)
def test_no_hard_coded_cuda_calls(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert not re.search(r"\.cuda\(\)", source), f"{path} hard-codes a CUDA transfer"


@pytest.mark.parametrize("path", PHASE1_MODULES, ids=lambda path: path.name)
def test_no_absolute_developer_paths(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert "/home/" not in source and "/media/" not in source, f"{path} embeds an absolute developer path"


@pytest.mark.parametrize("path", PHASE1_MODULES, ids=lambda path: path.name)
def test_no_dynamic_configuration_execution(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    # Method calls (``model.eval()``) and backticked prose are not dynamic execution.
    for pattern in (r"(?<![.\w`])eval\(", r"(?<![.\w`])exec\(", r"(?<![.`])globals\(\)", r"(?<![.`])__import__\("):
        assert not re.search(pattern, source), f"{path} matches forbidden pattern {pattern}"


def test_public_surface_is_importable_from_the_root() -> None:
    import qdgrasp

    for name in ("QDGrasp", "GraspResults", "RunConfig", "require_cuda", "__version__"):
        assert hasattr(qdgrasp, name)
