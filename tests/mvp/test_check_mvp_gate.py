"""The MVP closure gate must fail on a doctored artifact, not just pass on a good one.

A checker that only ever runs against a passing tree proves nothing about what
it would do with a failing one, so each test here writes a synthetic artifact
set, confirms it passes, then breaks exactly one thing and confirms the gate
notices.

This lives beside the package tests rather than in ``scripts/tests`` because
``check_mvp`` recomputes the scope and evaluation-manifest hashes through the
real schema -- that is the whole point of it -- so unlike the stdlib-only
checkers it needs the installed package to run.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_mvp import run_checks


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tier(name: str, episodes: int, successes: int, *, passed: bool = True, **overrides: object) -> dict[str, object]:
    tier = {
        "tier": name,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "wilson_lower": 0.9,
        "wilson_upper": 1.0,
        "invalid_state": 0,
        "safety_violation": 0,
        "checkpoint_reload_mismatch": 0,
        "failure_buckets": {},
        "min_success_rate": 0.7,
        "min_wilson_lower_bound": None,
        "passed": passed,
        "ledger_path": f"runs/mvp/evaluation/candidate/tier-{name.lower()}.jsonl",
    }
    tier.update(overrides)
    return tier


class MvpClosureGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temporary, True)
        self.root = self.temporary / "repo"
        self.runs = self.root / "runs" / "mvp"
        (self.root / "configs").mkdir(parents=True)
        shutil.copytree(PROJECT_ROOT / "configs" / "mvp", self.root / "configs" / "mvp")

        sys.path.insert(0, str(self.root))
        from qdgrasp.mvp.config import load_mvp_scope

        scope = load_mvp_scope(self.root / "configs/mvp/dexacquire-mvp-v0.yaml")
        self.scope = scope

        checkpoint = self.runs / "policy" / "ppo.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"weights")
        import hashlib

        digest = hashlib.sha256(b"weights").hexdigest()

        _write(
            self.runs / "demonstrations/index.json",
            {
                "scope_hash": scope.content_hash(),
                "splits": {
                    "train": {"episodes_accepted": 390, "search_rescued": 24, "non_zero_residual_fraction": 0.08},
                    "dev": {"episodes_accepted": 118, "search_rescued": 5, "non_zero_residual_fraction": 0.07},
                },
            },
        )
        for split in ("train", "dev"):
            (self.runs / "demonstrations" / split).mkdir(parents=True, exist_ok=True)
            (self.runs / "demonstrations" / split / "ledger.jsonl").write_text("{}\n", encoding="utf-8")

        _write(
            self.runs / "policy/training-report.json",
            {
                "fingerprint": {"scope_hash": scope.content_hash()},
                "candidate": str(checkpoint),
                "bc": {
                    "reload_parity": True,
                    "dev": {"success_rate": 0.93, "safety_violation": 0},
                    "checkpoint": str(self.runs / "policy" / "bc.pt"),
                },
                "ppo": {
                    "dev": {"success_rate": 0.95, "safety_violation": 0},
                    "promoted": True,
                },
            },
        )
        _write(
            self.runs / "evaluation/controller_prior.json",
            {"tiers": [_tier("A", 100, 100), _tier("B", 300, 270), _tier("C", 200, 180)]},
        )
        _write(
            self.runs / "evaluation/ppo.json",
            {
                "fingerprint": {"eval_manifest_hash": scope.eval_manifest_hash()},
                "checkpoint_sha256": digest,
                "tiers": [_tier("A", 100, 100), _tier("B", 300, 280), _tier("C", 200, 180)],
            },
        )
        (self.root / "notebooks").mkdir(parents=True, exist_ok=True)
        (self.root / "notebooks/mvp_grasp_policy.ipynb").write_text("{}", encoding="utf-8")
        card = self.root / "docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md"
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text("release_class: experimental_non_release\n## Giới hạn\n", encoding="utf-8")

    def failures(self) -> list[str]:
        return [check.name for check in run_checks(self.root, self.runs) if not check.passed]

    def test_a_complete_artifact_set_passes(self) -> None:
        self.assertEqual(self.failures(), [])

    def test_a_failed_tier_fails_the_gate(self) -> None:
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        report["tiers"][1]["passed"] = False
        _write(self.runs / "evaluation/ppo.json", report)
        self.assertIn("tier_b_gate", self.failures())

    def test_a_safety_violation_fails_the_gate(self) -> None:
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        report["tiers"][2]["safety_violation"] = 1
        _write(self.runs / "evaluation/ppo.json", report)
        self.assertIn("tier_c_zero_invalid_and_safe", self.failures())

    def test_evaluating_a_different_checkpoint_fails_the_gate(self) -> None:
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        report["checkpoint_sha256"] = "0" * 64
        _write(self.runs / "evaluation/ppo.json", report)
        self.assertIn("evaluated_checkpoint_is_the_candidate", self.failures())

    def test_a_stale_eval_manifest_fails_the_gate(self) -> None:
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        report["fingerprint"]["eval_manifest_hash"] = "0" * 64
        _write(self.runs / "evaluation/ppo.json", report)
        self.assertIn("evaluation_matches_locked_manifest", self.failures())

    def test_an_expert_that_never_beats_the_prior_fails_the_gate(self) -> None:
        index = json.loads((self.runs / "demonstrations/index.json").read_text(encoding="utf-8"))
        index["splits"]["train"]["search_rescued"] = 0
        _write(self.runs / "demonstrations/index.json", index)
        self.assertIn("expert_improves_on_the_prior", self.failures())

    def test_a_bc_baseline_below_the_floor_fails_the_gate(self) -> None:
        report = json.loads((self.runs / "policy/training-report.json").read_text(encoding="utf-8"))
        report["bc"]["dev"]["success_rate"] = 0.5
        _write(self.runs / "policy/training-report.json", report)
        self.assertIn("bc_dev_success", self.failures())

    def test_promoting_a_regressed_ppo_fails_the_gate(self) -> None:
        report = json.loads((self.runs / "policy/training-report.json").read_text(encoding="utf-8"))
        report["ppo"]["dev"]["success_rate"] = 0.5
        report["ppo"]["promoted"] = True
        _write(self.runs / "policy/training-report.json", report)
        self.assertIn("ppo_promotion_rule_respected", self.failures())

    def test_a_weak_controller_prior_fails_the_gate(self) -> None:
        _write(
            self.runs / "evaluation/controller_prior.json",
            {"tiers": [_tier("A", 100, 80), _tier("B", 300, 270), _tier("C", 200, 180)]},
        )
        self.assertIn("controller_prior_canonical_floor", self.failures())

    def test_a_missing_model_card_fails_the_gate(self) -> None:
        (self.root / "docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md").unlink()
        self.assertIn("model_card_present", self.failures())


if __name__ == "__main__":
    unittest.main()
