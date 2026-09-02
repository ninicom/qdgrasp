"""The release gate has to fail on the artifact set that nearly passes.

``ROADMAP-MVP-RELEASE-001`` §2.2 and §2.3 are the specification under test.  The
fixture writes a complete, self-consistent v1 artifact set that reaches a
release verdict, and then each test breaks exactly one thing the release
contract cares about and confirms the gate names it.

The interesting failures are the ones the experimental gate cannot see: a
candidate that clears every absolute floor while losing paired successes to the
controller prior, a Tier D improvement that survives switching the learned
residual off, a residual that has collapsed to zero, and a tier summary that
disagrees with the ledger it was supposedly computed from.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.contracts import (
    ABLATION_REPORT_SCHEMA,
    CHALLENGE_DOMAIN_SCHEMA,
    CONTRIBUTION_REPORT_SCHEMA,
    TRAINING_REPORT_SCHEMA,
)
from qdgrasp.mvp.env import OBSERVATION_DIMENSION, environment_fingerprint
from qdgrasp.mvp.evaluate import (
    EVAL_REPORT_SCHEMA,
    paired_uplift,
    wilson_lower_bound,
    wilson_upper_bound,
)
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

#: Successes per tier for each arm.  The candidate is a strict superset of the
#: prior's successes on every tier, which is what "no paired regression" means
#: and what makes the Tier D interval sit above zero.
SUCCESSES = {
    "controller_prior": {"A": 100, "B": 280, "C": 150, "D": 180},
    "bc": {"A": 100, "B": 283, "C": 158, "D": 195},
    "candidate": {"A": 100, "B": 285, "C": 160, "D": 210},
}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ledger_rows(seeds: tuple[int, ...], successes: int) -> list[dict[str, Any]]:
    return [
        {
            "setup": {"seed": int(seed), "split": "eval", "variant_id": "cuboid_w120_d15", "randomized": True},
            "success": index < successes,
            "failure_bucket": "" if index < successes else "timeout",
            "invalid_state": False,
            "safety_violation": False,
        }
        for index, seed in enumerate(seeds)
    ]


class MvpReleaseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temporary, True)
        self.root = self.temporary / "repo"
        self.runs = self.root / "runs" / "mvp"
        self.root.mkdir(parents=True)
        shutil.copytree(PROJECT_ROOT / "configs" / "mvp", self.root / "configs" / "mvp")

        scope = load_mvp_scope(self.root / "configs/mvp/dexacquire-mvp-v1.yaml")
        self.scope = scope
        self.criteria = scope.release
        assert self.criteria is not None and scope.challenge is not None
        prior = PinchPriorTable.load(self.root / "configs/mvp/leap-pinch-prior-v0.json")
        fingerprint = environment_fingerprint(scope, prior)

        # -- the challenge domain Tier D is drawn from ---------------------
        _write(
            self.root / scope.challenge.domain_document,
            {
                "schema": CHALLENGE_DOMAIN_SCHEMA,
                "scope_hash": scope.content_hash(),
                "axes": {"friction_slide": [0.05, 0.35], "density": [1800.0, 2400.0]},
            },
        )
        self.domain_path = self.root / scope.challenge.domain_document

        # -- demonstrations ------------------------------------------------
        split_summaries = {}
        for split, seed in (("train", 1), ("dev", 2)):
            demonstrations = DemonstrationSet(
                observations=np.zeros((2, OBSERVATION_DIMENSION), dtype=np.float64),
                actions=np.full((2, scope.action.dimension), 0.1, dtype=np.float64),
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

        # -- checkpoints ---------------------------------------------------
        bc_config = {"schema": "test/mvp-bc-config/v1", "epochs": 1}
        ppo_config = {"schema": "test/mvp-ppo-config/v1", "iterations": 1}
        normalizer = RunningNormalizer(dimension=OBSERVATION_DIMENSION)
        self.bc_checkpoint = self.runs / "policy" / "bc.pt"
        self.candidate_checkpoint = self.runs / "policy" / "ppo.pt"
        save_checkpoint(
            self.bc_checkpoint,
            ResidualActorCritic(hidden=(8,)),
            normalizer,
            fingerprint=fingerprint,
            stage="bc",
            metadata={"dataset_content_hash": train_content_hash, "training_config": bc_config},
        )
        save_checkpoint(
            self.candidate_checkpoint,
            ResidualActorCritic(hidden=(8,)),
            normalizer,
            fingerprint=fingerprint,
            stage="ppo",
            metadata={
                "parent": str(self.bc_checkpoint),
                "dataset_content_hash": train_content_hash,
                "training_config": ppo_config,
            },
        )
        candidate_payload = load_checkpoint(self.candidate_checkpoint)
        self.candidate_sha256 = hashlib.sha256(self.candidate_checkpoint.read_bytes()).hexdigest()

        _write(
            self.runs / "policy/training-report.json",
            {
                "schema": TRAINING_REPORT_SCHEMA,
                "action_distribution": ACTION_DISTRIBUTION,
                "fingerprint": fingerprint,
                "demonstrations": split_summaries["train"],
                "lineage": candidate_payload["lineage"],
                "candidate": str(self.candidate_checkpoint),
                "candidate_selection": {
                    "evidence": self.criteria.candidate_evidence,
                    "ppo_promotion": self.criteria.ppo_promotion,
                    "locked_seeds_read": False,
                },
                "bc": {
                    "training_config": bc_config,
                    "reload_parity": True,
                    "dev": {"success_rate": 0.93, "safety_violation": 0},
                    "checkpoint": str(self.bc_checkpoint),
                },
                "ppo": {
                    "training_config": ppo_config,
                    "dev": {"success_rate": 0.95, "safety_violation": 0},
                    "promoted": True,
                    "checkpoint": str(self.candidate_checkpoint),
                },
            },
        )

        # -- the three evaluated arms, each with real ledgers --------------
        self.outcomes: dict[str, dict[str, list[bool]]] = {}
        for arm, per_tier in SUCCESSES.items():
            self._write_arm(arm, per_tier, fingerprint, candidate_payload)

        # -- ablation: with the residual off the candidate is the prior ----
        self._write_arm("ablation_disabled", SUCCESSES["controller_prior"], fingerprint, candidate_payload)
        _write(
            self.runs / "evaluation/ablation.json",
            {
                "schema": ABLATION_REPORT_SCHEMA,
                "tier": self.criteria.contribution_tier,
                "candidate_sha256": self.candidate_sha256,
                "residual_disabled": True,
                "paired_vs_prior": self._paired("controller_prior", "ablation_disabled", "D"),
                "residual_statistics": {"mean_magnitude": 0.14, "saturation_rate": 0.03},
            },
        )

        self._write_contribution()

        (self.root / "notebooks").mkdir(parents=True, exist_ok=True)
        (self.root / "notebooks/mvp_grasp_policy.ipynb").write_text("{}", encoding="utf-8")
        card = self.root / "docs/reports/MVP-GRASP-POLICY-MODEL-CARD-V1.md"
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text("release_class: release_candidate\n## Giới hạn\n", encoding="utf-8")

    # -- fixture helpers ---------------------------------------------------

    def _write_arm(
        self,
        arm: str,
        per_tier: dict[str, int],
        fingerprint: dict[str, str],
        candidate_payload: dict[str, Any],
    ) -> None:
        is_prior = arm == "controller_prior"
        checkpoint = None if is_prior else str(self.bc_checkpoint if arm == "bc" else self.candidate_checkpoint)
        tiers = []
        self.outcomes[arm] = {}
        for spec in sorted(self.scope.eval_tiers, key=lambda item: item.tier):
            seeds = self.scope.locked_seeds(spec.tier)
            rows = _ledger_rows(seeds, per_tier[spec.tier])
            ledger = self.runs / "evaluation" / arm / f"tier-{spec.tier.lower()}.jsonl"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.outcomes[arm][spec.tier] = [row["success"] for row in rows]
            tiers.append(self._tier_document(spec.tier, per_tier[spec.tier], str(ledger)))
        report = {
            "schema": EVAL_REPORT_SCHEMA,
            "candidate": arm,
            "checkpoint": checkpoint,
            "fingerprint": fingerprint,
            "checkpoint_fingerprint": {
                "stored": None if is_prior else candidate_payload["fingerprint"],
                "effective": fingerprint,
                "verdict": "not_applicable" if is_prior else "match",
            },
            "tiers": tiers,
            "all_tiers_passed": all(tier["passed"] for tier in tiers),
        }
        if not is_prior:
            report["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        name = "ppo" if arm == "candidate" else arm
        _write(self.runs / f"evaluation/{name}.json", report)

    def _tier_document(self, tier: str, successes: int, ledger_path: str) -> dict[str, Any]:
        spec = self.scope.tier(tier)
        episodes = spec.episodes
        rate = successes / episodes
        lower = wilson_lower_bound(successes, episodes)
        return {
            "tier": tier,
            "episodes": episodes,
            "successes": successes,
            "success_rate": rate,
            "wilson_lower": lower,
            "wilson_upper": wilson_upper_bound(successes, episodes),
            "invalid_state": 0,
            "safety_violation": 0,
            "checkpoint_reload_mismatch": 0,
            "failure_buckets": {"timeout": episodes - successes} if successes < episodes else {},
            "min_success_rate": spec.min_success_rate,
            "min_wilson_lower_bound": spec.min_wilson_lower_bound,
            "passed": (
                (spec.min_success_rate is None or rate >= spec.min_success_rate)
                and (spec.min_wilson_lower_bound is None or lower >= spec.min_wilson_lower_bound)
            ),
            "ledger_path": ledger_path,
        }

    def _paired(self, prior_arm: str, candidate_arm: str, tier: str) -> dict[str, Any]:
        assert self.criteria is not None
        return paired_uplift(
            self.outcomes[prior_arm][tier],
            self.outcomes[candidate_arm][tier],
            resamples=self.criteria.paired_resamples,
            seed=self.criteria.paired_seed,
            confidence=self.criteria.paired_confidence,
        )

    def _write_contribution(self) -> None:
        _write(
            self.runs / "contribution.json",
            {
                "schema": CONTRIBUTION_REPORT_SCHEMA,
                "scope_hash": self.scope.content_hash(),
                "eval_manifest_hash": self.scope.eval_manifest_hash(),
                "challenge_domain_sha256": hashlib.sha256(self.domain_path.read_bytes()).hexdigest(),
                "candidate_sha256": self.candidate_sha256,
                "paired": {
                    spec.tier: self._paired("controller_prior", "candidate", spec.tier)
                    for spec in self.scope.eval_tiers
                },
            },
        )

    def failures(self, mode: str = "release") -> list[str]:
        return [check.name for check in run_checks(self.root, self.runs, mode=mode) if not check.passed]

    def _rewrite_ledger(self, arm: str, tier: str, mutate) -> None:
        path = self.runs / "evaluation" / arm / f"tier-{tier.lower()}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mutate(rows)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    # -- the artifact set that does reach a release verdict ----------------

    def test_a_complete_release_artifact_set_passes(self) -> None:
        self.assertEqual(self.failures(), [])

    def test_the_same_tree_is_not_a_v0_experimental_artifact_set(self) -> None:
        # The two gates read different scope documents, so an artifact set is
        # admissible to exactly one of them.  Under the experimental gate these
        # v1 artifacts fail on identity rather than on merit -- the fingerprints
        # name a world that scope v0 does not describe.
        failures = self.failures(mode="experimental")
        for name in (
            "fingerprint_matches_scope",
            "candidate_checkpoint_matches_scope",
            "evaluation_matches_locked_manifest",
            "evaluation_tiers_complete",
        ):
            self.assertIn(name, failures)

    # -- §2.2: the challenge tier ------------------------------------------

    def test_a_tier_d_uplift_below_the_gate_fails(self) -> None:
        # Candidate rescues only six of the prior's failures: 2 pp, under the
        # five the contract demands.
        self._rewrite_ledger("candidate", "D", lambda rows: [
            row.update({"success": index < 186, "failure_bucket": "" if index < 186 else "timeout"})
            for index, row in enumerate(rows)
        ])
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        for tier in report["tiers"]:
            if tier["tier"] == "D":
                tier.update({"successes": 186, "success_rate": 186 / 300, "failure_buckets": {"timeout": 114}})
        _write(self.runs / "evaluation/ppo.json", report)
        self.outcomes["candidate"]["D"] = [index < 186 for index in range(300)]
        self._write_contribution()
        self.assertIn("tier_d_uplift_gate", self.failures())

    def test_a_candidate_that_loses_paired_successes_on_a_regression_tier_fails(self) -> None:
        # Tier B still clears its absolute floor: 279/300 is 93%, well above the
        # 85% gate.  What it does not do is beat the prior's 280.
        self._rewrite_ledger("candidate", "B", lambda rows: [
            row.update({"success": index < 279, "failure_bucket": "" if index < 279 else "timeout"})
            for index, row in enumerate(rows)
        ])
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        for tier in report["tiers"]:
            if tier["tier"] == "B":
                tier.update(
                    {
                        "successes": 279,
                        "success_rate": 279 / 300,
                        "wilson_lower": wilson_lower_bound(279, 300),
                        "wilson_upper": wilson_upper_bound(279, 300),
                        "failure_buckets": {"timeout": 21},
                    }
                )
        _write(self.runs / "evaluation/ppo.json", report)
        self.outcomes["candidate"]["B"] = [index < 279 for index in range(300)]
        self._write_contribution()
        failures = self.failures()
        self.assertIn("tier_b_no_paired_regression", failures)
        self.assertNotIn("tier_b_gate", failures)

    def test_arms_that_did_not_run_the_same_seeds_cannot_be_compared(self) -> None:
        self._rewrite_ledger("candidate", "C", lambda rows: rows[0]["setup"].update({"seed": 1}))
        self.assertIn("tier_c_arms_share_the_locked_seeds", self.failures())

    # -- §2.2: recomputation and the safety budget -------------------------

    def test_a_tier_summary_that_disagrees_with_its_ledger_fails(self) -> None:
        report = json.loads((self.runs / "evaluation/ppo.json").read_text(encoding="utf-8"))
        for tier in report["tiers"]:
            if tier["tier"] == "C":
                tier["failure_buckets"] = {"drop": 40}
        _write(self.runs / "evaluation/ppo.json", report)
        self.assertIn("tier_c_recomputed_from_raw_ledger", self.failures())

    def test_one_safety_violation_in_a_raw_ledger_fails(self) -> None:
        self._rewrite_ledger("candidate", "A", lambda rows: rows[0].update({"safety_violation": True}))
        failures = self.failures()
        self.assertIn("tier_a_within_safety_budget", failures)
        self.assertIn("tier_a_recomputed_from_raw_ledger", failures)

    def test_one_invalid_state_in_a_raw_ledger_fails(self) -> None:
        self._rewrite_ledger("candidate", "D", lambda rows: rows[0].update({"invalid_state": True}))
        self.assertIn("tier_d_within_safety_budget", self.failures())

    def test_a_doctored_paired_interval_is_recomputed(self) -> None:
        contribution = json.loads((self.runs / "contribution.json").read_text(encoding="utf-8"))
        contribution["paired"]["D"]["ci_lower_pp"] = 9.9
        _write(self.runs / "contribution.json", contribution)
        self.assertIn("tier_d_paired_comparison_recomputes", self.failures())

    # -- §2.3: the ablation ------------------------------------------------

    def test_an_improvement_that_survives_disabling_the_residual_fails(self) -> None:
        ablation = json.loads((self.runs / "evaluation/ablation.json").read_text(encoding="utf-8"))
        ablation["paired_vs_prior"] = self._paired("controller_prior", "candidate", "D")
        _write(self.runs / "evaluation/ablation.json", ablation)
        self.assertIn("disabling_the_residual_removes_the_uplift", self.failures())

    def test_a_residual_that_collapsed_to_zero_fails(self) -> None:
        ablation = json.loads((self.runs / "evaluation/ablation.json").read_text(encoding="utf-8"))
        ablation["residual_statistics"]["mean_magnitude"] = 0.0001
        _write(self.runs / "evaluation/ablation.json", ablation)
        self.assertIn("residual_has_not_degenerated", self.failures())

    def test_a_saturated_residual_fails(self) -> None:
        ablation = json.loads((self.runs / "evaluation/ablation.json").read_text(encoding="utf-8"))
        ablation["residual_statistics"]["saturation_rate"] = 0.9
        _write(self.runs / "evaluation/ablation.json", ablation)
        self.assertIn("residual_has_not_degenerated", self.failures())

    def test_a_missing_ablation_run_fails(self) -> None:
        (self.runs / "evaluation/ablation.json").unlink()
        failures = self.failures()
        self.assertIn("ablation_report_present", failures)
        self.assertIn("disabling_the_residual_removes_the_uplift", failures)

    def test_an_ablation_of_a_different_checkpoint_fails(self) -> None:
        ablation = json.loads((self.runs / "evaluation/ablation.json").read_text(encoding="utf-8"))
        ablation["candidate_sha256"] = "0" * 64
        _write(self.runs / "evaluation/ablation.json", ablation)
        self.assertIn("ablation_report_contract", self.failures())

    # -- §2.3: promotion and selection -------------------------------------

    def test_a_promoted_ppo_that_is_worse_than_bc_on_a_regression_tier_fails(self) -> None:
        report = json.loads((self.runs / "evaluation/bc.json").read_text(encoding="utf-8"))
        _write(self.runs / "evaluation/bc.json", report)
        self._rewrite_ledger("bc", "C", lambda rows: [
            row.update({"success": index < 170, "failure_bucket": "" if index < 170 else "timeout"})
            for index, row in enumerate(rows)
        ])
        self.assertIn("ppo_is_at_least_bc_on_every_regression_tier", self.failures())

    def test_a_missing_bc_rollback_evaluation_fails(self) -> None:
        (self.runs / "evaluation/bc.json").unlink()
        self.assertIn("bc_rollback_evaluated_on_the_locked_tiers", self.failures())

    def test_a_candidate_chosen_after_reading_the_locked_seeds_fails(self) -> None:
        report = json.loads((self.runs / "policy/training-report.json").read_text(encoding="utf-8"))
        report["candidate_selection"]["locked_seeds_read"] = True
        _write(self.runs / "policy/training-report.json", report)
        self.assertIn("candidate_selected_on_development_evidence_only", self.failures())

    def test_a_candidate_with_no_selection_record_fails(self) -> None:
        report = json.loads((self.runs / "policy/training-report.json").read_text(encoding="utf-8"))
        del report["candidate_selection"]
        _write(self.runs / "policy/training-report.json", report)
        self.assertIn("candidate_selected_on_development_evidence_only", self.failures())

    # -- §2.1: identity ----------------------------------------------------

    def test_a_challenge_domain_that_moves_an_unauthorised_axis_fails(self) -> None:
        domain = json.loads(self.domain_path.read_text(encoding="utf-8"))
        domain["axes"]["point_cloud_noise"] = [0.0, 1.0]
        _write(self.domain_path, domain)
        self.assertIn("challenge_domain_contract", self.failures())

    def test_a_missing_challenge_domain_fails(self) -> None:
        self.domain_path.unlink()
        failures = self.failures()
        self.assertIn("challenge_domain_present", failures)
        self.assertIn("contribution_report_contract", failures)

    def test_a_contribution_report_bound_to_another_scope_fails(self) -> None:
        contribution = json.loads((self.runs / "contribution.json").read_text(encoding="utf-8"))
        contribution["scope_hash"] = "0" * 64
        _write(self.runs / "contribution.json", contribution)
        self.assertIn("contribution_report_contract", self.failures())

    def test_a_missing_contribution_report_fails(self) -> None:
        (self.runs / "contribution.json").unlink()
        failures = self.failures()
        self.assertIn("contribution_report_present", failures)
        self.assertIn("tier_d_paired_comparison_recomputes", failures)

    def test_a_release_model_card_that_still_says_experimental_fails(self) -> None:
        card = self.root / "docs/reports/MVP-GRASP-POLICY-MODEL-CARD-V1.md"
        card.write_text(
            "release_class: release_candidate\nrelease_class: experimental_non_release\n## Giới hạn\n",
            encoding="utf-8",
        )
        self.assertIn("model_card_declares_release_candidate", self.failures())


def test_the_release_gate_refuses_the_published_v0_evidence() -> None:
    """The three published rounds are experimental, and must stay unreachable."""

    failures = {
        check.name
        for check in run_checks(PROJECT_ROOT, PROJECT_ROOT / "evidence/mvp/round-3", mode="release")
        if not check.passed
    }
    assert "release_contract_present" not in failures, "the release contract itself must load"
    assert "contribution_report_present" in failures
    assert "ablation_report_present" in failures
    assert "challenge_domain_present" in failures


if __name__ == "__main__":
    unittest.main()
