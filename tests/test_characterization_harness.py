"""Guards for the Phase 3.2.1 characterization harness (P3.2.1-01).

The frozen corpus under `evidence/phase3_2_1/baseline/` is the control case for
every later remediation claim, so both the harness and the corpus need to stay
honest: evidence fields must be present, cells must be reproducible, and a
moved failure signature must be reported rather than silently absorbed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "characterize_pipeline_failures.py"
BASELINE_PATH = REPO_ROOT / "evidence" / "phase3_2_1" / "baseline" / "corpus.json"


def _load_harness():
    spec = importlib.util.spec_from_file_location("characterize_pipeline_failures", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.is_file():
        pytest.skip(f"frozen baseline corpus not present at {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_baseline_corpus_reproduces_the_reported_ik_failure(baseline: dict) -> None:
    """3 hands x 3 recipes x 2 candidates, all stopping at `IK: max_iter`."""
    summary = baseline["summary"]
    assert summary["cells"] == 9
    assert summary["total_candidates"] == 18
    assert summary["failure_signature"] == {"ik:max_iter": 18}
    assert summary["candidates_reaching_dynamic_rollout"] == 0
    assert summary["pipeline_generated_positives"] == 0


def test_baseline_corpus_carries_the_required_provenance(baseline: dict) -> None:
    """Section 6 of the plan pins what an experiment record has to contain."""
    provenance = baseline["provenance"]
    for key in (
        "git_commit",
        "git_worktree_dirty",
        "source_hashes",
        "robot_profile_hashes",
        "object_hashes",
        "environment",
        "seed",
        "candidate_budget",
        "run_dynamic",
    ):
        assert key in provenance, f"provenance missing {key}"
    assert set(provenance["robot_profile_hashes"]) == {
        "leap_hand",
        "wonik_allegro",
        "shadow_hand",
    }
    assert provenance["source_hashes"], "no pipeline source hashes recorded"


def test_every_candidate_records_stage_and_residual_telemetry(baseline: dict) -> None:
    for cell in baseline["cells_detail"]:
        assert set(cell["stage_accounting"]), "cell has no stage accounting"
        for record in cell["candidates"]:
            assert record["failure_stage"]
            assert "failure_reason" in record
            kin = record.get("kinematics")
            assert kin is not None, "IK-stage candidate without kinematic telemetry"
            assert kin["position_residuals"], "no per-finger position residuals recorded"
            assert kin["normal_residuals"], "no per-finger normal residuals recorded"
            assert kin["reason"]


def test_a_cell_is_reproducible_from_the_same_seed(harness) -> None:
    """Same seed and budget must give byte-identical stage evidence."""
    kwargs = dict(
        hand="leap_hand",
        recipe_id="surface_fixed_v1",
        object_name="box_50mm",
        object_spec=harness.OBJECT_BUILDERS["box_50mm"](),
        seed=42,
        candidates=2,
        run_dynamic=False,
    )
    first = harness.run_cell(**kwargs)
    second = harness.run_cell(**kwargs)
    assert first["failure_signature"] == second["failure_signature"]
    assert first["stage_accounting"] == second["stage_accounting"]
    assert first["candidates"] == second["candidates"]


def test_current_harness_records_solver_progress_metrics(harness) -> None:
    cell = harness.run_cell(
        hand="leap_hand",
        recipe_id="surface_fixed_v1",
        object_name="box_50mm",
        object_spec=harness.OBJECT_BUILDERS["box_50mm"](),
        seed=42,
        candidates=4,
        run_dynamic=False,
    )
    characterized = next(
        candidate for candidate in cell["candidates"] if "kinematics" in candidate
    )
    metrics = characterized["kinematics"]["solver_metrics"]
    assert metrics is not None
    assert {
        "initial_cost",
        "final_cost",
        "accepted_steps",
        "rejected_steps",
        "jacobian_rank",
        "jacobian_condition",
        "final_damping",
        "finite",
    } <= set(metrics)


def test_current_harness_reports_active_and_dense_residual_maxima(harness) -> None:
    cell = harness.run_cell(
        hand="leap_hand",
        recipe_id="region_opposition_v1",
        object_name="box_50mm",
        object_spec=harness.OBJECT_BUILDERS["box_50mm"](),
        seed=42,
        candidates=1,
        run_dynamic=False,
    )
    record = cell["candidates"][0]
    active = record["proposal"]["active_fingers"]
    position = record["kinematics"]["position_residuals"]
    normal = record["kinematics"]["normal_residuals"]
    active_indices = [index for index, enabled in enumerate(active) if enabled]

    assert record["kinematics"]["active_max_position_residual"] == pytest.approx(
        max(position[index] for index in active_indices)
    )
    assert record["kinematics"]["active_max_normal_residual"] == pytest.approx(
        max(normal[index] for index in active_indices)
    )
    assert record["kinematics"]["max_position_residual"] == pytest.approx(max(position))
    assert record["kinematics"]["max_normal_residual"] == pytest.approx(max(normal))


def test_comparison_reports_a_moved_failure_signature(harness) -> None:
    def corpus(signature: dict[str, int]) -> dict:
        return {
            "summary": {"failure_signature": signature},
            "provenance": {
                "seed": 42,
                "candidate_budget": 2,
                "source_hashes": {"a.py": "hash-a"},
            },
            "cells_detail": [
                {
                    "hand": "leap_hand",
                    "recipe_id": "surface_fixed_v1",
                    "object": "box_50mm",
                    "failure_signature": signature,
                }
            ],
        }

    unchanged = harness.compare_to_baseline(
        corpus({"ik:max_iter": 2}), corpus({"ik:max_iter": 2})
    )
    assert unchanged["changed_cells"] == []

    moved = harness.compare_to_baseline(
        corpus({"ik:max_iter": 2}), corpus({"collision:palm_penetration": 2})
    )
    assert len(moved["changed_cells"]) == 1
    assert moved["changed_cells"][0]["baseline_signature"] == {"ik:max_iter": 2}
    assert moved["changed_cells"][0]["current_signature"] == {
        "collision:palm_penetration": 2
    }
    assert moved["seed_matches"] and moved["budget_matches"]
