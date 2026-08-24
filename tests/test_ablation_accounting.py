from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ablate_recipes.py"
SPEC = importlib.util.spec_from_file_location("ablate_recipes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stage_rates_use_conditional_denominators() -> None:
    reasons = {
        "proposal_rejected": 2,
        "ik_rejected": 2,
        "collision_rejected": 1,
        "static_force_rejected": 1,
        "dynamic_rejected": 1,
        "dynamic_skipped": 0,
        "accepted": 1,
    }
    rates = MODULE._stage_rates(8, reasons)
    assert rates == {
        "proposal_yield": pytest.approx(6 / 8),
        "ik_convergence_rate": pytest.approx(4 / 6),
        "collision_pass_rate": pytest.approx(3 / 4),
        "static_pass_rate": pytest.approx(2 / 3),
        "dynamic_pass_rate": pytest.approx(1 / 2),
    }


def test_ablation_rejects_unbounded_candidate_count() -> None:
    with pytest.raises(ValueError, match="must be in"):
        MODULE.run_ablation(num_candidates_per_obj=5)


def test_positive_control_matrix_is_bounded_to_84_candidates() -> None:
    assert sum(MODULE.POSITIVE_CONTROL_BUDGETS.values()) * len(MODULE.REQUIRED_RECIPES) == 84
    assert 84 <= MODULE.MAX_TOTAL_CANDIDATES


def test_ablation_rejects_unknown_matrix() -> None:
    with pytest.raises(ValueError, match="matrix must be"):
        MODULE.run_ablation(matrix="unknown")


def _recipe_result(
    *, accepted: int, accepted_cells: int, accepted_hands: list[str], contacts: int
) -> dict[str, object]:
    return {
        "accepted_hands": accepted_hands,
        "accepted_cell_count": accepted_cells,
        "distinct_accepted_contact_count": contacts,
        "reasons": {"accepted": accepted},
        "stage_counts": {
            "static_passed": accepted + 1,
            "collision_passed": accepted + 2,
            "ik_passed": accepted + 3,
            "proposal_passed": accepted + 4,
        },
    }


def test_selection_requires_dynamic_evidence_for_all_three_hands() -> None:
    weak = _recipe_result(
        accepted=2,
        accepted_cells=2,
        accepted_hands=["leap_hand", "wonik_allegro"],
        contacts=2,
    )
    results = {name: weak for name in MODULE.REQUIRED_RECIPES}
    decision = MODULE._selection_decision(results, run_dynamic=True)
    assert decision["status"] == "inconclusive"
    assert decision["reason"] == "no_recipe_has_three_hand_dynamic_evidence"


def test_selection_uses_pre_registered_score_and_rejects_ties() -> None:
    hands = list(MODULE.REQUIRED_HANDS)
    results = {
        "surface_fixed_v1": _recipe_result(accepted=3, accepted_cells=3, accepted_hands=hands, contacts=3),
        "region_opposition_v1": _recipe_result(accepted=4, accepted_cells=4, accepted_hands=hands, contacts=4),
        "wrench_guided_v1": _recipe_result(accepted=3, accepted_cells=3, accepted_hands=hands, contacts=3),
    }
    selected = MODULE._selection_decision(results, run_dynamic=True)
    assert selected["status"] == "selected"
    assert selected["winner"] == "region_opposition_v1"

    results["wrench_guided_v1"] = results["region_opposition_v1"]
    tied = MODULE._selection_decision(results, run_dynamic=True)
    assert tied["status"] == "inconclusive"
    assert tied["reason"] == "selection_score_tie"


def test_selection_is_inconclusive_when_dynamic_is_disabled() -> None:
    decision = MODULE._selection_decision({}, run_dynamic=False)
    assert decision["status"] == "inconclusive"
    assert decision["reason"] == "dynamic_disabled"
