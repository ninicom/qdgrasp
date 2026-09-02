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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qdgrasp.mvp.contracts import TRAINING_REPORT_SCHEMA
from qdgrasp.mvp.env import OBSERVATION_DIMENSION, environment_fingerprint
from qdgrasp.mvp.evaluate import EVAL_REPORT_SCHEMA, wilson_lower_bound, wilson_upper_bound
from qdgrasp.mvp.expert import DEMONSTRATION_INDEX_SCHEMA, DemonstrationSet
from qdgrasp.mvp.policy import (
    ACTION_DISTRIBUTION,
    ResidualActorCritic,
    RunningNormalizer,
    load_checkpoint,
    save_checkpoint,
)
from qdgrasp.mvp.prior import PinchPriorTable
from scripts.check_mvp import run_checks


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tier(name: str, episodes: int, successes: int, *, passed: bool = True, **overrides: object) -> dict[str, object]:
    minimums = {"A": (0.95, None), "B": (0.85, 0.8), "C": (0.7, None)}
    minimum_rate, minimum_wilson = minimums[name]
    tier = {
        "tier": name,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "wilson_lower": wilson_lower_bound(successes, episodes),
        "wilson_upper": wilson_upper_bound(successes, episodes),
        "invalid_state": 0,
        "safety_violation": 0,
        "checkpoint_reload_mismatch": 0,
        "failure_buckets": {"timeout": episodes - successes} if successes < episodes else {},
        "min_success_rate": minimum_rate,
        "min_wilson_lower_bound": minimum_wilson,
        "passed": passed,
        "ledger_path": f"runs/mvp/evaluation/ppo/tier-{name.lower()}.jsonl",
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

        prior = PinchPriorTable.load(self.root / "configs/mvp/leap-pinch-prior-v0.json")
        fingerprint = environment_fingerprint(scope, prior)
        split_summaries = {}
        for split, seed in (("train", 1), ("dev", 2)):
            demonstrations = DemonstrationSet(
                observations=np.zeros((2, OBSERVATION_DIMENSION), dtype=np.float64),
                actions=np.ones((2, scope.action.dimension), dtype=np.float64) * 0.1,
                episode_index=np.zeros(2, dtype=np.int64),
                variant_ids=(f"{split}-variant",),
                seeds=(seed,),
                ledger=[
                    {
                        "seed": seed,
                        "variant_id": f"{split}-variant",
                        "accepted": True,
                        "prior_candidate_succeeded": False,
                    }
                ],
            )
            demonstrations.save(self.runs / "demonstrations" / split)
            split_summaries[split] = demonstrations.summary()
        train_content_hash = split_summaries["train"]["content_hash"]
        bc_training_config = {"schema": "test/mvp-bc-config/v1", "epochs": 1}
        ppo_training_config = {"schema": "test/mvp-ppo-config/v1", "iterations": 1}
        normalizer = RunningNormalizer(dimension=OBSERVATION_DIMENSION)
        bc_checkpoint = self.runs / "policy" / "bc.pt"
        checkpoint = self.runs / "policy" / "ppo.pt"
        save_checkpoint(
            bc_checkpoint,
            ResidualActorCritic(hidden=(8,)),
            normalizer,
            fingerprint=fingerprint,
            stage="bc",
            metadata={
                "dataset_content_hash": train_content_hash,
                "training_config": bc_training_config,
            },
        )
        save_checkpoint(
            checkpoint,
            ResidualActorCritic(hidden=(8,)),
            normalizer,
            fingerprint=fingerprint,
            stage="ppo",
            metadata={
                "parent": str(bc_checkpoint),
                "dataset_content_hash": train_content_hash,
                "training_config": ppo_training_config,
            },
        )
        candidate_payload = load_checkpoint(checkpoint)
        import hashlib

        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

        _write(
            self.runs / "demonstrations/index.json",
            {
                "schema": DEMONSTRATION_INDEX_SCHEMA,
                "scope_hash": scope.content_hash(),
                "prior_hash": prior.content_hash(),
                "fingerprint": fingerprint,
                "splits": split_summaries,
            },
        )

        _write(
            self.runs / "policy/training-report.json",
            {
                "schema": TRAINING_REPORT_SCHEMA,
                "action_distribution": ACTION_DISTRIBUTION,
                "fingerprint": fingerprint,
                "demonstrations": split_summaries["train"],
                "lineage": candidate_payload["lineage"],
                "candidate": str(checkpoint),
                "bc": {
                    "training_config": bc_training_config,
                    "reload_parity": True,
                    "dev": {"success_rate": 0.93, "safety_violation": 0},
                    "checkpoint": str(bc_checkpoint),
                },
                "ppo": {
                    "training_config": ppo_training_config,
                    "dev": {"success_rate": 0.95, "safety_violation": 0},
                    "promoted": True,
                    "checkpoint": str(checkpoint),
                },
            },
        )
        _write(
            self.runs / "evaluation/controller_prior.json",
            {
                "schema": EVAL_REPORT_SCHEMA,
                "fingerprint": fingerprint,
                "checkpoint_fingerprint": {
                    "stored": None,
                    "effective": fingerprint,
                    "verdict": "not_applicable",
                },
                "tiers": [_tier("A", 100, 100), _tier("B", 300, 270), _tier("C", 200, 180)],
            },
        )
        _write(
            self.runs / "evaluation/ppo.json",
            {
                "schema": EVAL_REPORT_SCHEMA,
                "candidate": "ppo",
                "checkpoint": str(checkpoint),
                "fingerprint": fingerprint,
                "checkpoint_fingerprint": {
                    "stored": fingerprint,
                    "effective": fingerprint,
                    "verdict": "match",
                },
                "checkpoint_sha256": digest,
                "tiers": [_tier("A", 100, 100), _tier("B", 300, 280), _tier("C", 200, 180)],
                "all_tiers_passed": True,
            },
        )
        for tier, episodes in (("a", 100), ("b", 300), ("c", 200)):
            ledger = self.runs / "evaluation" / "ppo" / f"tier-{tier}.jsonl"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text("{}\n" * episodes, encoding="utf-8")
        (self.root / "notebooks").mkdir(parents=True, exist_ok=True)
        (self.root / "notebooks/mvp_grasp_policy.ipynb").write_text("{}", encoding="utf-8")
        card = self.root / "docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md"
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text("release_class: experimental_non_release\n## Giới hạn\n", encoding="utf-8")

    def failures(self) -> list[str]:
        return [check.name for check in run_checks(self.root, self.runs) if not check.passed]

    @staticmethod
    def _stored_arrays(archive: Path) -> dict[str, np.ndarray]:
        with np.load(archive, allow_pickle=False) as stored:
            return {name: stored[name].copy() for name in stored.files}

    def test_a_complete_artifact_set_passes(self) -> None:
        self.assertEqual(self.failures(), [])

    def test_a_v0_training_report_cannot_reuse_v1_checkpoint_results(self) -> None:
        report = json.loads((self.runs / "policy/training-report.json").read_text(encoding="utf-8"))
        report["schema"] = "qdgrasp/mvp-training-report/v0"
        _write(self.runs / "policy/training-report.json", report)
        self.assertIn("training_report_schema", self.failures())

    def test_a_checkpoint_outside_the_evidence_root_is_not_replaced_by_its_basename(self) -> None:
        outside = self.root / "outside" / "ppo.pt"
        outside.parent.mkdir(parents=True)
        shutil.copy2(self.runs / "policy/ppo.pt", outside)
        report = json.loads((self.runs / "policy/training-report.json").read_text(encoding="utf-8"))
        report["candidate"] = str(outside)
        _write(self.runs / "policy/training-report.json", report)
        self.assertIn("candidate_checkpoint_is_contained", self.failures())

    def test_an_evaluation_without_checkpoint_fingerprint_verdict_fails(self) -> None:
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        report.pop("checkpoint_fingerprint")
        _write(self.runs / "evaluation/ppo.json", report)
        self.assertIn("checkpoint_fingerprint_verdict", self.failures())

    def test_a_doctored_tier_summary_is_recomputed(self) -> None:
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        report["tiers"][1]["success_rate"] = 1.0
        _write(self.runs / "evaluation/ppo.json", report)
        self.assertIn("tier_b_contract", self.failures())

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

    def test_a_doctored_demonstration_array_fails_the_gate(self) -> None:
        archive = self.runs / "demonstrations/train/demonstrations.npz"
        pristine = self._stored_arrays(archive)
        for name in ("observations", "actions", "episode_index"):
            with self.subTest(array=name):
                arrays = {key: value.copy() for key, value in pristine.items()}
                arrays[name].reshape(-1)[0] += 1
                np.savez_compressed(archive, **arrays)
                self.assertIn("train_raw_arrays_match_manifest", self.failures())
        np.savez_compressed(archive, **pristine)
        self.assertEqual(self.failures(), [])

    def test_a_demonstration_archive_with_the_wrong_array_set_fails_the_gate(self) -> None:
        archive = self.runs / "demonstrations/train/demonstrations.npz"
        pristine = self._stored_arrays(archive)
        doctored = {
            "extra": {**pristine, "rewards": np.zeros(2, dtype=np.float64)},
            "missing": {name: value for name, value in pristine.items() if name != "episode_index"},
        }
        for label, arrays in doctored.items():
            with self.subTest(archive=label):
                np.savez_compressed(archive, **arrays)
                self.assertIn("train_raw_arrays_match_manifest", self.failures())
        np.savez_compressed(archive, **pristine)
        self.assertEqual(self.failures(), [])

    def test_a_pickled_demonstration_archive_fails_the_gate(self) -> None:
        archive = self.runs / "demonstrations/train/demonstrations.npz"
        arrays = self._stored_arrays(archive)
        arrays["actions"] = np.array([{"residual": 0.1}, None], dtype=object)
        np.savez_compressed(archive, **arrays)
        self.assertIn("train_raw_arrays_match_manifest", self.failures())

    def test_a_doctored_demonstration_ledger_fails_the_gate(self) -> None:
        ledger = self.runs / "demonstrations/train/ledger.jsonl"
        pristine = ledger.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in pristine.splitlines() if line.strip()]
        for field, value in (("seed", 99), ("variant_id", "forged-variant"), ("accepted", False)):
            with self.subTest(field=field):
                ledger.write_text(
                    "".join(
                        json.dumps({**row, field: value}, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in rows
                    ),
                    encoding="utf-8",
                )
                failures = self.failures()
                self.assertIn("train_ledger_content_hash", failures)
                self.assertIn("train_raw_arrays_match_manifest", failures)
        ledger.write_text(pristine, encoding="utf-8")
        self.assertEqual(self.failures(), [])

    def test_a_doctored_demonstration_index_identity_fails_the_gate(self) -> None:
        path = self.runs / "demonstrations/index.json"
        pristine = json.loads(path.read_text(encoding="utf-8"))
        doctored = (
            ("schema", "qdgrasp/mvp-demonstration-index/v0", "demonstration_index_schema"),
            ("scope_hash", "0" * 64, "demonstrations_match_locked_scope"),
            ("prior_hash", "0" * 64, "demonstrations_match_prior"),
            ("fingerprint", {"scope_hash": "0" * 64}, "demonstrations_match_environment_fingerprint"),
        )
        for field, value, check in doctored:
            with self.subTest(field=field):
                _write(path, {**pristine, field: value})
                self.assertIn(check, self.failures())
        _write(path, pristine)
        self.assertEqual(self.failures(), [])

    def test_doctored_checkpoint_training_config_fails_to_load(self) -> None:
        import torch

        checkpoint = self.runs / "policy/bc.pt"
        payload = torch.load(checkpoint, weights_only=True)
        payload["metadata"]["training_config"]["epochs"] = 2
        torch.save(payload, checkpoint)
        self.assertIn("bc_checkpoint_loads", self.failures())

    def test_a_doctored_checkpoint_dataset_hash_fails_to_load(self) -> None:
        import torch

        checkpoint = self.runs / "policy/bc.pt"
        payload = torch.load(checkpoint, weights_only=True)
        payload["metadata"]["dataset_content_hash"] = "0" * 64
        torch.save(payload, checkpoint)
        self.assertIn("bc_checkpoint_loads", self.failures())

    def test_a_training_report_that_cites_other_demonstrations_fails_the_gate(self) -> None:
        report = json.loads((self.runs / "policy/training-report.json").read_text(encoding="utf-8"))
        report["demonstrations"]["content_hash"] = "0" * 64
        _write(self.runs / "policy/training-report.json", report)
        self.assertIn("bc_dataset_lineage_matches_demonstrations", self.failures())

    def test_a_ppo_parent_that_is_not_the_reported_bc_checkpoint_fails_the_gate(self) -> None:
        # Byte-identical, so only the declared parent path moves: the gate must
        # bind the PPO parent to the retained rollback artifact, not to any
        # checkpoint that happens to hash the same.
        shadow = self.runs / "policy" / "rollback" / "bc.pt"
        shadow.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.runs / "policy/bc.pt", shadow)
        report = json.loads((self.runs / "policy/training-report.json").read_text(encoding="utf-8"))
        report["bc"]["checkpoint"] = str(shadow)
        _write(self.runs / "policy/training-report.json", report)
        self.assertIn("ppo_parent_checkpoint_lineage", self.failures())

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


def test_committed_v0_evidence_is_rejected_by_the_current_contract() -> None:
    failures = {
        check.name
        for check in run_checks(PROJECT_ROOT, PROJECT_ROOT / "evidence/mvp/round-3")
        if not check.passed
    }
    assert "training_report_schema" in failures
    assert "bc_checkpoint_loads" in failures
    assert "evaluation_report_schema" in failures


if __name__ == "__main__":
    unittest.main()
