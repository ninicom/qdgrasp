"""MVP-00: the locked scope round-trips and the eval manifest is immutable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qdgrasp.mvp.config import DEFAULT_SCOPE_PATH, MvpScopeConfig, load_mvp_scope

MANIFEST_PATH = Path("configs/mvp/dexacquire-mvp-v0.eval-manifest.json")

#: Pinned at MVP-00 lock time.  A change here is a new MVP version and a
#: revision record, never an edit.
LOCKED_SCOPE_HASH = "a897e36084c9ab11cbf8046e446ef318c48b7463d8b2db4af4d4ca0593109b8b"
LOCKED_EVAL_MANIFEST_HASH = "ac57bf61fb1e4294f840d29fea4f26865f542164eb08ee6b6f39915875840e3a"


@pytest.fixture(scope="module")
def scope() -> MvpScopeConfig:
    return load_mvp_scope()


def test_scope_document_round_trips(scope: MvpScopeConfig) -> None:
    reparsed = MvpScopeConfig.model_validate(scope.to_document())
    assert reparsed == scope
    assert reparsed.content_hash() == scope.content_hash()


def test_scope_hash_is_locked(scope: MvpScopeConfig) -> None:
    assert scope.content_hash() == LOCKED_SCOPE_HASH
    assert scope.eval_manifest_hash() == LOCKED_EVAL_MANIFEST_HASH


def test_unknown_keys_are_rejected() -> None:
    document = yaml.safe_load(DEFAULT_SCOPE_PATH.read_text(encoding="utf-8"))
    document["tuning_knob"] = 1
    with pytest.raises(ValidationError):
        MvpScopeConfig.model_validate(document)


def test_eval_manifest_artifact_matches_scope(scope: MvpScopeConfig) -> None:
    assert MANIFEST_PATH.is_file(), "the locked eval manifest artifact is missing"
    stored = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert stored == scope.eval_manifest()


def test_tier_sizes_match_the_plan(scope: MvpScopeConfig) -> None:
    assert (scope.tier("A").episodes, scope.tier("A").min_success_rate) == (100, 0.95)
    assert (scope.tier("B").episodes, scope.tier("B").min_success_rate) == (300, 0.85)
    assert scope.tier("B").min_wilson_lower_bound == 0.80
    assert (scope.tier("C").episodes, scope.tier("C").min_success_rate) == (200, 0.70)


def test_held_out_variants_never_reach_a_training_split(scope: MvpScopeConfig) -> None:
    held_out = {variant.variant_id for variant in scope.heldout_variants}
    for split in ("train", "dev", "eval_a", "eval_b"):
        reachable = {variant.variant_id for variant in scope.variants_for_split(split)}  # type: ignore[arg-type]
        assert not reachable & held_out, f"held-out variant leaked into split {split!r}"
    assert {variant.variant_id for variant in scope.variants_for_split("eval_c")} == held_out


def test_seeds_are_deterministic_and_split_disjoint(scope: MvpScopeConfig) -> None:
    assert scope.episode_seed("train", 7) == scope.episode_seed("train", 7)
    assert scope.episode_seed("train", 7) != scope.episode_seed("dev", 7)
    seeds = {split: set(scope.locked_seeds(tier)) for split, tier in (("a", "A"), ("b", "B"), ("c", "C"))}
    assert len(seeds["a"]) == 100 and len(seeds["b"]) == 300 and len(seeds["c"]) == 200
    assert not seeds["a"] & seeds["b"]
    assert not seeds["b"] & seeds["c"]
    assert not seeds["a"] & seeds["c"]
    train = {scope.episode_seed("train", index) for index in range(400)}
    assert not train & (seeds["a"] | seeds["b"] | seeds["c"])


def test_retain_phase_can_satisfy_the_retain_duration(scope: MvpScopeConfig) -> None:
    assert scope.episode.retain_steps >= scope.success.retain_duration_s * scope.episode.control_hz


def test_object_family_has_six_train_and_four_held_out_sizes(scope: MvpScopeConfig) -> None:
    assert len(scope.train_variants) == 6
    assert len(scope.heldout_variants) == 4
    widths = sorted(variant.half_width for variant in scope.train_variants)
    held = sorted(variant.half_width for variant in scope.heldout_variants)
    assert held[0] < widths[0], "no held-out size below the fitted range"
    assert held[-1] > widths[-1], "no held-out size above the fitted range"
