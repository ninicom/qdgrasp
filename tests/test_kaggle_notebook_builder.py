from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from scripts import build_kaggle_notebook

REPO_ROOT = Path(__file__).resolve().parents[1]


def _builder_source() -> str:
    return (REPO_ROOT / "scripts" / "build_kaggle_notebook.py").read_text(encoding="utf-8")


def test_notebook_revision_is_an_immutable_git_hash():
    revision = build_kaggle_notebook.KAGGLE_CODE_REVISION
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)
    subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=REPO_ROOT, check=True)
    renderer = subprocess.check_output(
        ["git", "show", f"{revision}:scripts/render_4view_rollout.py"], cwd=REPO_ROOT, text=True
    )
    assert '_assert_evidence_equal("dynamic_trajectory_evidence"' in renderer


def test_notebook_reports_only_measured_phase33_grasp_evidence():
    source = _builder_source()
    assert "Phase 3.4" not in source
    assert "qdgrasp-dummy-n.yaml" not in source
    assert "Phase 3 GPU Training" not in source
    assert "measured release-control evidence" in source
    assert 'item["actual_outcome"] == "PASS"' in source
    assert 'result["actual_outcome"] == "PASS"' in source
    assert '"--profile", "release", "--dataset-root", dataset_root' in source
    assert 'dataset_root="/tmp/qdgrasp_repo/datasets/qdgrasp-scene-tiny"' in source


def test_notebook_checkout_and_install_use_the_same_pinned_revision():
    source = _builder_source()
    tree = ast.parse(source)
    assert "@feature/phase3-data-layer" not in source
    assert 'f"git+{{REPO_URL}}@{{CODE_REVISION}}"' in source
    assert "assert checked_out == CODE_REVISION" in source
    assert "assert assets_revision == MENAGERIE_REVISION" in source
    assert tree is not None


def test_generated_phase1_and_phase3_notebooks_are_equivalent_and_pinned():
    phase1 = REPO_ROOT / "kaggle-phase1" / "qdgrasp-phase-1-cuda-framework-gate.ipynb"
    phase3 = REPO_ROOT / "kaggle-phase3" / "qdgrasp-phase-3-cuda-gate.ipynb"
    assert phase1.read_bytes() == phase3.read_bytes()
    notebook = json.loads(phase3.read_text(encoding="utf-8"))
    rendered_source = "".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert build_kaggle_notebook.KAGGLE_CODE_REVISION in rendered_source
    assert build_kaggle_notebook.MENAGERIE_REVISION in rendered_source
    assert "feature/phase3-data-layer" not in rendered_source
