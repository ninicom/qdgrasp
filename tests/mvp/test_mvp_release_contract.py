"""The release contract has to be a contract, not a document that parses.

``ROADMAP-MVP-RELEASE-001`` §5 MR-02 requires every release criterion to be
frozen and tested *before* the run it judges exists.  So the tests here are
about what the schema refuses: a v0 scope that promotes itself to a release
class, a v1 scope missing the contribution tier, a challenge tier with no
uplift gate, a contribution tier that is also its own control.

The first test in the file is the one that protects everything already
published: scope v0's content hash is pinned to a literal.  Three rounds of
evidence, the committed evaluation manifest and every checkpoint fingerprint
are bound to that number, and adding fields to the shared model must not move
it.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qdgrasp.mvp.challenge import (
    ChallengeDomain,
    heldout_development_seeds,
    load_challenge_domain,
)
from qdgrasp.mvp.challenge import (
    challenge_development_seeds as development_seeds,
)
from qdgrasp.mvp.config import (
    EXPERIMENTAL_RELEASE_CLASS,
    RELEASE_CANDIDATE_CLASS,
    MvpScopeConfig,
    load_mvp_scope,
)
from qdgrasp.mvp.evaluate import evaluate_tier, paired_uplift

SCOPE_V0 = PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v0.yaml"
SCOPE_V1 = PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v1.yaml"
SCOPE_V2 = PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v2.yaml"

#: Pinned, not computed.  Everything published under v0 refers to these.
V0_SCOPE_HASH = "a897e36084c9ab11cbf8046e446ef318c48b7463d8b2db4af4d4ca0593109b8b"
V0_EVAL_MANIFEST_HASH = "ac57bf61fb1e4294f840d29fea4f26865f542164eb08ee6b6f39915875840e3a"


def _document(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# -- v0 stays exactly where it was ----------------------------------------


def test_scope_v0_identity_is_unchanged_by_the_release_schema() -> None:
    scope = load_mvp_scope(SCOPE_V0)
    assert scope.content_hash() == V0_SCOPE_HASH
    assert scope.eval_manifest_hash() == V0_EVAL_MANIFEST_HASH
    assert scope.eval_manifest()["schema"] == "qdgrasp/mvp-eval-manifest/v0"
    # The v1 fields must not appear in a v0 document at all: serialized as
    # nulls they would still be a different document.
    assert "challenge" not in scope.to_document()
    assert "release" not in scope.to_document()
    assert set(scope.to_document()["eval_tiers"][0]) == {
        "tier",
        "episodes",
        "membership",
        "randomized",
        "min_success_rate",
        "min_wilson_lower_bound",
    }


def test_committed_v0_eval_manifest_still_matches_its_scope() -> None:
    stored = json.loads((PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v0.eval-manifest.json").read_text("utf-8"))
    assert stored == load_mvp_scope(SCOPE_V0).eval_manifest()


# -- v1 is a release contract ---------------------------------------------


def test_scope_v1_declares_a_machine_readable_release_class() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    assert scope.release_class == RELEASE_CANDIDATE_CLASS
    assert scope.is_release_candidate is True
    assert load_mvp_scope(SCOPE_V0).is_release_candidate is False


def test_scope_v1_carries_the_challenge_tier_and_its_criteria() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    assert sorted(spec.tier for spec in scope.eval_tiers) == ["A", "B", "C", "D"]
    assert scope.release is not None and scope.challenge is not None
    assert scope.release.contribution_tier == "D"
    assert scope.release.regression_tiers == ("A", "B", "C")
    tier_d = scope.tier("D")
    assert tier_d.challenge_domain is True
    assert tier_d.min_success_rate is None
    assert tier_d.min_uplift_pp == 5.0
    assert tier_d.min_paired_ci_lower == 0.0
    assert tier_d.episodes == 300


def test_committed_v1_eval_manifest_matches_its_scope() -> None:
    stored = json.loads((PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v1.eval-manifest.json").read_text("utf-8"))
    scope = load_mvp_scope(SCOPE_V1)
    assert stored == scope.eval_manifest()
    assert stored["schema"] == "qdgrasp/mvp-eval-manifest/v1"
    assert stored["release_class"] == RELEASE_CANDIDATE_CLASS
    challenge = next(entry for entry in stored["tiers"] if entry["tier"] == "D")
    assert challenge["challenge_domain"] is True
    assert challenge["min_uplift_pp"] == 5.0


# -- seeds: nothing the candidate saw may judge it ------------------------


def test_every_locked_tier_is_disjoint_from_every_other_and_from_development() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    locked = {spec.tier: set(scope.locked_seeds(spec.tier)) for spec in scope.eval_tiers}
    for left, right in itertools.combinations(sorted(locked), 2):
        assert not locked[left] & locked[right], f"tiers {left} and {right} share seeds"
    development = {scope.episode_seed("train", index) for index in range(2000)}
    development |= {scope.episode_seed("dev", index) for index in range(2000)}
    for tier, seeds in locked.items():
        assert not seeds & development, f"tier {tier} shares seeds with train/dev"
    assert len(locked["D"]) == 300


def test_the_challenge_calibration_seeds_never_reach_the_tier_they_calibrated() -> None:
    """MR-03's whole point: nothing explored may judge what the exploration chose."""

    scope = load_mvp_scope(SCOPE_V1)
    development = set(development_seeds(scope, 3000))
    assert len(development) == 3000, "the development root must not collide with itself"
    for spec in scope.eval_tiers:
        assert not development & set(scope.locked_seeds(spec.tier)), f"tier {spec.tier} saw calibration seeds"
    training = {scope.episode_seed("train", index) for index in range(3000)}
    training |= {scope.episode_seed("dev", index) for index in range(3000)}
    assert not development & training
    assert scope.challenge is not None
    assert scope.challenge.development_seed_root != scope.seed_root


def test_the_v1_seed_root_does_not_reuse_the_v0_seeds() -> None:
    v0 = load_mvp_scope(SCOPE_V0)
    v1 = load_mvp_scope(SCOPE_V1)
    for tier in ("A", "B", "C"):
        assert not set(v0.locked_seeds(tier)) & set(v1.locked_seeds(tier))


# -- what the schema must refuse ------------------------------------------


def _reject(document: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MvpScopeConfig.model_validate(document)


def test_a_v0_scope_cannot_promote_itself_to_a_release_class() -> None:
    document = _document(SCOPE_V0)
    document["release_class"] = RELEASE_CANDIDATE_CLASS
    _reject(document, "scope v0 must declare release_class")


def test_a_v0_scope_cannot_carry_a_release_contract() -> None:
    document = _document(SCOPE_V0)
    document["challenge"] = _document(SCOPE_V1)["challenge"]
    _reject(document, "scope v0 cannot carry a challenge or release contract")


def test_a_v0_scope_cannot_add_a_challenge_tier() -> None:
    document = _document(SCOPE_V0)
    document["eval_tiers"].append(_document(SCOPE_V1)["eval_tiers"][3])
    _reject(document, "scope v0 eval_tiers must define exactly tiers A, B and C")


def test_a_v1_scope_without_the_release_contract_is_refused() -> None:
    document = _document(SCOPE_V1)
    del document["release"]
    _reject(document, "scope v1 requires both a challenge and a release contract")


def test_a_v1_scope_without_the_challenge_tier_is_refused() -> None:
    document = _document(SCOPE_V1)
    document["eval_tiers"] = [tier for tier in document["eval_tiers"] if tier["tier"] != "D"]
    _reject(document, "scope v1 eval_tiers must define exactly tiers A, B, C and D")


def test_a_v1_scope_that_is_not_a_release_candidate_is_refused() -> None:
    document = _document(SCOPE_V1)
    document["release_class"] = EXPERIMENTAL_RELEASE_CLASS
    _reject(document, "scope v1 must declare release_class")


def test_a_challenge_tier_without_an_uplift_gate_is_refused() -> None:
    document = _document(SCOPE_V1)
    tier = next(entry for entry in document["eval_tiers"] if entry["tier"] == "D")
    del tier["min_uplift_pp"]
    del tier["min_paired_ci_lower"]
    tier["min_success_rate"] = 0.5
    _reject(document, "is a challenge tier without an uplift gate")


def test_a_tier_with_no_gate_at_all_is_refused() -> None:
    document = _document(SCOPE_V1)
    tier = next(entry for entry in document["eval_tiers"] if entry["tier"] == "A")
    del tier["min_success_rate"]
    _reject(document, "declares no gate at all")


def test_an_uplift_gate_without_its_paired_interval_is_refused() -> None:
    document = _document(SCOPE_V1)
    tier = next(entry for entry in document["eval_tiers"] if entry["tier"] == "D")
    del tier["min_paired_ci_lower"]
    _reject(document, "must declare an uplift gate and its paired CI floor together")


def test_the_contribution_tier_cannot_also_be_its_own_control() -> None:
    document = _document(SCOPE_V1)
    document["release"]["regression_tiers"] = ["A", "B", "C", "D"]
    _reject(document, "cannot be both the contribution tier and a regression tier")


def test_a_tier_no_release_rule_classifies_is_refused() -> None:
    document = _document(SCOPE_V1)
    document["release"]["regression_tiers"] = ["A", "B"]
    _reject(document, "release criteria must classify every tier")


def test_a_challenge_domain_outside_the_declared_axes_is_refused() -> None:
    document = _document(SCOPE_V1)
    document["challenge"]["axes"] = ["half_width", "point_cloud_density"]
    with pytest.raises(ValueError, match="challenge"):
        MvpScopeConfig.model_validate(document)


def test_an_inverted_prior_success_band_is_refused() -> None:
    document = _document(SCOPE_V1)
    document["challenge"]["prior_success_band"] = [0.85, 0.40]
    _reject(document, "prior success band is not an ordered interval")


def test_an_unknown_release_class_is_refused() -> None:
    document = _document(SCOPE_V1)
    document["release_class"] = "production"
    with pytest.raises(ValueError, match="release_class"):
        MvpScopeConfig.model_validate(document)


# -- the challenge tier cannot be run from the base domain ----------------


def test_the_challenge_tier_refuses_to_run_without_its_locked_domain() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    with pytest.raises(NotImplementedError, match="locked challenge domain"):
        evaluate_tier("D", scope, workers=1)


# -- the paired estimator -------------------------------------------------


def test_identical_arms_have_no_uplift_and_an_interval_on_zero() -> None:
    outcomes = [True] * 200 + [False] * 100
    comparison = paired_uplift(outcomes, outcomes, resamples=2000, seed=11)
    assert comparison["uplift_pp"] == 0.0
    assert comparison["ci_lower_pp"] == 0.0 == comparison["ci_upper_pp"]
    assert comparison["candidate_only_successes"] == comparison["prior_only_successes"] == 0


def test_a_real_uplift_is_recovered_with_an_interval_above_zero() -> None:
    prior = [True] * 200 + [False] * 100
    candidate = [True] * 230 + [False] * 70
    comparison = paired_uplift(prior, candidate, resamples=20000, seed=20260902, confidence=0.95)
    assert comparison["uplift_pp"] == pytest.approx(10.0)
    assert comparison["ci_lower_pp"] > 0.0
    assert comparison["candidate_only_successes"] == 30
    assert comparison["prior_only_successes"] == 0


def test_the_paired_interval_is_reproducible_from_its_recorded_seed() -> None:
    prior = [True, False] * 150
    candidate = [True] * 160 + [False] * 140
    first = paired_uplift(prior, candidate, resamples=5000, seed=7)
    assert first == paired_uplift(prior, candidate, resamples=first["resamples"], seed=first["seed"])
    assert first["seed"] == 7 and first["resamples"] == 5000


def test_the_paired_estimator_refuses_unmatched_arms() -> None:
    with pytest.raises(ValueError, match="matched arms"):
        paired_uplift([True, False], [True], resamples=10, seed=0)
    with pytest.raises(ValueError, match="at least one episode"):
        paired_uplift([], [], resamples=10, seed=0)
    with pytest.raises(ValueError, match="positive resample count"):
        paired_uplift([True], [True], resamples=0, seed=0)
    with pytest.raises(ValueError, match=r"confidence must lie"):
        paired_uplift([True], [True], resamples=10, seed=0, confidence=0.5)


# -- the locked challenge domain ------------------------------------------

CHALLENGE_DOMAIN = PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v1.challenge.json"


def _domain_document() -> dict:
    return json.loads(CHALLENGE_DOMAIN.read_text(encoding="utf-8"))


def test_the_locked_challenge_domain_narrows_the_locked_scope() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    domain = load_challenge_domain(CHALLENGE_DOMAIN, scope)
    assert domain.scope_hash == scope.content_hash()
    assert set(domain.axes) <= set(scope.challenge.axes)
    for axis, (low, high) in domain.axes.items():
        if axis in ("density", "friction_slide"):
            base_low, base_high = getattr(scope.randomization, axis)
            assert base_low <= low <= high <= base_high
    assert [variant.variant_id for variant in domain.variants(scope)] == [
        "cuboid_w180_d12",
        "cuboid_w210_d15",
        "cuboid_w240_d18",
    ]


def test_the_locked_domain_is_the_one_that_was_calibrated() -> None:
    """The document's hash is what binds a Tier D result to a measured domain."""

    report = json.loads((PROJECT_ROOT / "evidence/mvp/release-v1/challenge-development/c1.json").read_text("utf-8"))
    domain = load_challenge_domain(CHALLENGE_DOMAIN, load_mvp_scope(SCOPE_V1))
    assert report["domain_hash"] == domain.content_hash()
    assert report["verdict"] == "admissible"
    assert report["safety_violation"] == 0 and report["invalid_state"] == 0
    assert report["failures"] >= report["min_prior_failures"]
    low, high = report["required_band"]
    assert low <= report["prior_success_rate"] <= high
    # Only real grasp failures may be counted: a domain whose difficulty came
    # from simulator errors or a violated safety budget is not a hard domain.
    assert set(report["failure_buckets"]) <= {"drop", "contact_loss", "approach_miss", "timeout"}


def test_a_challenge_domain_may_not_widen_any_axis() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    document = _domain_document()
    document["axes"]["friction_slide"] = [0.01, 0.55]
    with pytest.raises(ValueError, match="reaches outside the locked scope"):
        ChallengeDomain.model_validate(document).validate_against(scope)


def test_a_challenge_domain_may_not_move_an_unauthorised_axis() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    document = _domain_document()
    document["axes"]["drop_height"] = [0.002, 0.008]
    # Refused by the axis type before the scope is even consulted: the permitted
    # set is a closed literal, so an unlisted axis is not a value at all.
    with pytest.raises(ValueError, match="literal_error"):
        ChallengeDomain.model_validate(document).validate_against(scope)


def test_a_challenge_domain_bound_to_another_scope_is_refused() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    document = _domain_document()
    document["scope_hash"] = "0" * 64
    with pytest.raises(ValueError, match="bound to another scope"):
        ChallengeDomain.model_validate(document).validate_against(scope)


def test_a_challenge_domain_that_selects_no_variant_is_refused() -> None:
    scope = load_mvp_scope(SCOPE_V1)
    document = _domain_document()
    document["axes"]["half_width"] = [0.0001, 0.0002]
    with pytest.raises(ValueError, match="select no object variant"):
        ChallengeDomain.model_validate(document).validate_against(scope)


def test_an_empty_challenge_domain_is_refused() -> None:
    document = _domain_document()
    document["axes"] = {}
    with pytest.raises(ValueError, match="must narrow at least one axis"):
        ChallengeDomain.model_validate(document)


def test_the_challenge_domain_only_moves_the_challenge_split() -> None:
    """Supplying a domain must not move the ground under tiers A, B and C."""

    from qdgrasp.mvp.env import DexAcquireMvpEnv
    from qdgrasp.mvp.prior import PinchPriorTable

    scope = load_mvp_scope(SCOPE_V1)
    prior = PinchPriorTable.load(PROJECT_ROOT / "configs/mvp/leap-pinch-prior-v0.json")
    domain = load_challenge_domain(CHALLENGE_DOMAIN, scope)
    plain = DexAcquireMvpEnv(scope, prior)
    challenged = DexAcquireMvpEnv(scope, prior, challenge=domain)
    for split in ("train", "dev", "eval_a", "eval_b", "eval_c"):
        seed = scope.episode_seed(split, 3)
        assert plain.sample_setup(seed, split) == challenged.sample_setup(seed, split)
    seed = scope.episode_seed("eval_d", 3)
    assert plain.sample_setup(seed, "eval_d") != challenged.sample_setup(seed, "eval_d")


def test_every_challenge_episode_stays_inside_the_locked_domain() -> None:
    from qdgrasp.mvp.env import DexAcquireMvpEnv
    from qdgrasp.mvp.prior import PinchPriorTable

    scope = load_mvp_scope(SCOPE_V1)
    prior = PinchPriorTable.load(PROJECT_ROOT / "configs/mvp/leap-pinch-prior-v0.json")
    domain = load_challenge_domain(CHALLENGE_DOMAIN, scope)
    env = DexAcquireMvpEnv(scope, prior, challenge=domain)
    allowed = {variant.variant_id for variant in domain.variants(scope)}
    for seed in scope.locked_seeds("D"):
        setup = env.sample_setup(seed, "eval_d")
        assert setup.variant_id in allowed
        assert domain.axes["friction_slide"][0] <= setup.friction_slide <= domain.axes["friction_slide"][1]
        assert domain.axes["density"][0] <= setup.density <= domain.axes["density"][1]


# -- attempt three runs on fresh seeds and unchanged gates ----------------


def test_scope_v2_changes_the_seeds_and_nothing_that_judges_the_candidate() -> None:
    """A second attempt may move the test set.  It may not move the test.

    Every threshold in v2 is copied from v1, and this is the test that says
    so: if a later attempt ever quietly relaxes the uplift gate, the paired
    estimator's seed, the ablation bounds or the safety budget, it fails here
    rather than passing quietly with a number that looks better.
    """

    v1 = load_mvp_scope(SCOPE_V1)
    v2 = load_mvp_scope(SCOPE_V2)

    for tier in ("A", "B", "C", "D"):
        assert v1.tier(tier).model_dump() == v2.tier(tier).model_dump(), tier
    assert v1.release is not None and v2.release is not None
    assert v1.release.model_dump() == v2.release.model_dump()
    assert v1.randomization.model_dump() == v2.randomization.model_dump()
    assert v1.success.model_dump() == v2.success.model_dump()
    assert v1.controller.model_dump() == v2.controller.model_dump()
    assert v1.action.model_dump() == v2.action.model_dump()
    assert v1.reward.model_dump() == v2.reward.model_dump()
    assert [variant.model_dump() for variant in v1.objects] == [variant.model_dump() for variant in v2.objects]

    # What may differ: identity and the seed roots derived from it.
    document_v1, document_v2 = v1.to_document(), v2.to_document()
    assert sorted(key for key in document_v1 if document_v1[key] != document_v2[key]) == [
        "artifact_id",
        "challenge",
        "environment_id",
        "mvp_id",
        "seed_root",
    ]
    assert v1.challenge is not None and v2.challenge is not None
    assert v1.challenge.axes == v2.challenge.axes
    assert v1.challenge.prior_success_band == v2.challenge.prior_success_band
    assert v1.challenge.min_prior_failures == v2.challenge.min_prior_failures


def test_no_locked_seed_is_reused_between_attempts() -> None:
    v1 = load_mvp_scope(SCOPE_V1)
    v2 = load_mvp_scope(SCOPE_V2)
    for tier in ("A", "B", "C", "D"):
        assert not set(v1.locked_seeds(tier)) & set(v2.locked_seeds(tier)), tier
    # And the development roots of one attempt may not reach the other's tiers.
    development = set(development_seeds(v2, 2000)) | set(heldout_development_seeds(v2, 2000))
    development |= {v2.episode_seed("train", index) for index in range(3000)}
    development |= {v2.episode_seed("dev", index) for index in range(3000)}
    for scope in (v1, v2):
        for tier in ("A", "B", "C", "D"):
            assert not development & set(scope.locked_seeds(tier)), (scope.mvp_id, tier)


def test_the_v2_challenge_domain_is_the_v1_domain_rebound() -> None:
    v2 = load_mvp_scope(SCOPE_V2)
    domain_v1 = json.loads((PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v1.challenge.json").read_text("utf-8"))
    domain_v2 = load_challenge_domain(PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v2.challenge.json", v2)
    assert domain_v2.scope_hash == v2.content_hash()
    assert domain_v2.axes == {key: tuple(value) for key, value in domain_v1["axes"].items()}
    assert domain_v2.configuration_id == domain_v1["configuration_id"]


def test_the_committed_v2_eval_manifest_matches_its_scope() -> None:
    stored = json.loads((PROJECT_ROOT / "configs/mvp/dexacquire-mvp-v2.eval-manifest.json").read_text("utf-8"))
    assert stored == load_mvp_scope(SCOPE_V2).eval_manifest()
