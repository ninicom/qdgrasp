"""Every notebook must set the environment variable the code actually reads.

This exists because it did not. Three notebooks -- the MVP one, the P3.5 GPU
readiness one and the P4 CUDA gate -- exported ``QDGRASP_ROBOT_ASSETS`` while
``qdgrasp/robot/assets.py`` reads ``QDGRASP_ROBOT_ASSETS_ROOT``. Nothing on the
development machine noticed, because the variable is already set here. On a
fresh Kaggle runtime the first call that touches a robot profile raises, which
means the two GPU runs this project has been waiting on would both have failed
after the install cell -- and the failure would have looked like a code problem
rather than a one-word typo.

A notebook is code that runs somewhere nobody can attach a debugger to. What it
exports is worth a test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from qdgrasp.robot.assets import ROBOT_ASSET_ROOT_ENV

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
BUILDERS = sorted((REPO_ROOT / "scripts").glob("build_*notebook.py"))

#: Any assignment to an environment variable whose name starts this way.
PATTERN = re.compile(r"""os\.environ\[\s*["'](QDGRASP_ROBOT_ASSETS[A-Z_]*)["']\s*\]\s*=""")


def _notebook_sources(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_a_notebook_sets_the_variable_the_code_reads(path: Path) -> None:
    names = set(PATTERN.findall(_notebook_sources(path)))
    wrong = sorted(names - {ROBOT_ASSET_ROOT_ENV})
    assert not wrong, f"{path.name} exports {wrong}; qdgrasp reads {ROBOT_ASSET_ROOT_ENV}"


@pytest.mark.parametrize("path", BUILDERS, ids=lambda path: path.name)
def test_a_notebook_builder_sets_the_variable_the_code_reads(path: Path) -> None:
    names = set(PATTERN.findall(path.read_text(encoding="utf-8")))
    wrong = sorted(names - {ROBOT_ASSET_ROOT_ENV})
    assert not wrong, f"{path.name} emits {wrong}; qdgrasp reads {ROBOT_ASSET_ROOT_ENV}"


def test_the_gpu_notebooks_do_set_it_at_all() -> None:
    """A notebook that touches a robot profile and sets nothing fails the same way."""

    for name in ("phase4_cuda_gate.ipynb", "phase3_5_rl_readiness.ipynb"):
        source = _notebook_sources(REPO_ROOT / "notebooks" / name)
        assert ROBOT_ASSET_ROOT_ENV in source, f"{name} never points qdgrasp at the robot assets"
