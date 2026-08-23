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
