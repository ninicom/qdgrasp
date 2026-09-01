"""MVP-02..05: the expert, the checkpoint contract and the tier arithmetic.

These tests do not train anything real.  They pin the properties that decide
whether a trained number means what the report says it means: that the expert
only ever admits measured successes, that a checkpoint reproduces itself after a
reload, that the episode split never leaks a frame across a boundary, and that a
tier with a safety violation in it cannot pass however good its success rate is.
"""

from __future__ import annotations

import numpy as np
import pytest

from qdgrasp.mvp.bc import BehaviorCloningSpec, _episode_split, train_behavior_cloning
from qdgrasp.mvp.config import MvpScopeConfig, load_mvp_scope
from qdgrasp.mvp.env import DexAcquireMvpEnv, EpisodeResult, EpisodeSetup, environment_fingerprint
from qdgrasp.mvp.evaluate import _aggregate, wilson_lower_bound, wilson_upper_bound
from qdgrasp.mvp.expert import DemonstrationSet, ExpertSearchSpec, search_expert_episode
from qdgrasp.mvp.policy import (
    MvpPolicy,
    ResidualActorCritic,
    RunningNormalizer,
    checkpoint_reload_matches,
    load_policy,
    save_checkpoint,
)
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable


@pytest.fixture(scope="module")
def scope() -> MvpScopeConfig:
    return load_mvp_scope()


@pytest.fixture(scope="module")
def prior() -> PinchPriorTable:
    return PinchPriorTable.load(DEFAULT_PRIOR_PATH)


# -- MVP-02 ---------------------------------------------------------------


def test_expert_candidate_zero_is_the_prior(scope: MvpScopeConfig) -> None:
    rng = np.random.default_rng(0)
    candidates = ExpertSearchSpec(candidates=6).sample(rng, scope.action.dimension)
    assert candidates.shape == (6, 2, scope.action.dimension)
    np.testing.assert_array_equal(candidates[0], np.zeros_like(candidates[0]))
    assert np.all(np.abs(candidates) <= 1.0)


def test_expert_only_admits_measured_successes(scope: MvpScopeConfig, prior: PinchPriorTable) -> None:
    env = DexAcquireMvpEnv(scope, prior)
    episode, row = search_expert_episode(env, scope.episode_seed("train", 1), "train", ExpertSearchSpec(candidates=4))
    assert row["candidates"] == 4
    assert sum(row["candidate_outcomes"].values()) == 4
    if episode is None:
        assert not row["accepted"]
    else:
        assert row["accepted"] and episode.result.success
        assert episode.observations.shape[0] == episode.actions.shape[0]
        assert episode.score[0] == 1.0


def test_demonstration_summary_counts_every_attempt() -> None:
    ledger = [
        {"seed": 1, "variant_id": "a", "accepted": True, "prior_candidate_succeeded": True},
        {"seed": 2, "variant_id": "a", "accepted": True, "prior_candidate_succeeded": False},
        {"seed": 3, "variant_id": "b", "accepted": False, "prior_candidate_succeeded": False},
    ]
    dataset = DemonstrationSet(
        observations=np.zeros((4, 3)),
        actions=np.array([[0.0], [0.1], [0.0], [0.2]]),
        episode_index=np.array([0, 0, 1, 1]),
        variant_ids=("a", "a"),
        seeds=(1, 2),
        ledger=ledger,
    )
    summary = dataset.summary()
    assert summary["episodes_attempted"] == 3
    assert summary["episodes_accepted"] == 2
    assert summary["search_rescued"] == 1
    assert summary["acceptance_rate"] == pytest.approx(2 / 3)


# -- MVP-03 ---------------------------------------------------------------


def test_normalizer_matches_numpy_statistics() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(5.0, 2.0, size=(500, 4))
    normalizer = RunningNormalizer(dimension=4)
    for start in range(0, 500, 64):
        normalizer.update(data[start : start + 64])
    np.testing.assert_allclose(normalizer.mean, data.mean(axis=0), atol=1e-3)
    np.testing.assert_allclose(normalizer.std, data.std(axis=0), atol=1e-2)


def test_episode_split_never_cuts_an_episode() -> None:
    dataset = DemonstrationSet(
        observations=np.zeros((30, 2)),
        actions=np.zeros((30, 1)),
        episode_index=np.repeat(np.arange(6), 5),
        variant_ids=tuple("abcdef"),
        seeds=tuple(range(6)),
        ledger=[],
    )
    train_mask, validation_mask = _episode_split(dataset, 0.34, seed=0)
    assert not np.any(train_mask & validation_mask)
    for episode in range(6):
        frames = dataset.episode_index == episode
        assert train_mask[frames].all() or validation_mask[frames].all()


def test_behavior_cloning_learns_a_non_zero_mapping() -> None:
    """A cloned policy that always outputs zero would pass a loss check and fail here."""

    rng = np.random.default_rng(1)
    observations = rng.normal(size=(600, 4))
    # Target depends on the observation, so a constant predictor cannot fit it.
    actions = np.tanh(np.stack([observations[:, 0], -observations[:, 1]], axis=1))
    dataset = DemonstrationSet(
        observations=observations,
        actions=actions,
        episode_index=np.repeat(np.arange(60), 10),
        variant_ids=tuple(f"v{index}" for index in range(60)),
        seeds=tuple(range(60)),
        ledger=[],
    )
    network, normalizer, metrics = train_behavior_cloning(
        dataset, BehaviorCloningSpec(epochs=40, batch_size=128, hidden=(64, 64))
    )
    assert metrics["final_train_loss"] < 0.05
    assert metrics["mean_predicted_action_magnitude"] > 0.2
    assert network.observation_dim == 4 and network.action_dim == 2
    assert isinstance(normalizer, RunningNormalizer)


def test_checkpoint_round_trips_and_refuses_a_foreign_world(tmp_path, scope, prior) -> None:
    network = ResidualActorCritic(observation_dim=8, action_dim=2, hidden=(16,))
    with __import__("torch").no_grad():
        for parameter in network.actor.parameters():
            parameter.add_(0.1)
    normalizer = RunningNormalizer(dimension=8)
    normalizer.update(np.random.default_rng(0).normal(size=(64, 8)))
    fingerprint = environment_fingerprint(scope, prior)
    path = save_checkpoint(tmp_path / "candidate.pt", network, normalizer, fingerprint=fingerprint, stage="bc")

    observations = np.random.default_rng(2).normal(size=(16, 8))
    in_memory = MvpPolicy(network, normalizer)
    actions = np.stack([in_memory(row) for row in observations])
    assert checkpoint_reload_matches(path, observations, actions)

    load_policy(path, fingerprint=fingerprint)
    with pytest.raises(ValueError):
        load_policy(path, fingerprint={**fingerprint, "scope_hash": "0" * 64})


def test_reload_mismatch_is_detectable(tmp_path, scope, prior) -> None:
    network = ResidualActorCritic(observation_dim=8, action_dim=2, hidden=(16,))
    normalizer = RunningNormalizer(dimension=8)
    path = save_checkpoint(
        tmp_path / "candidate.pt", network, normalizer, fingerprint=environment_fingerprint(scope, prior), stage="bc"
    )
    observations = np.zeros((4, 8))
    assert not checkpoint_reload_matches(path, observations, np.full((4, 2), 0.9))


# -- MVP-05 ---------------------------------------------------------------


def test_wilson_bounds_bracket_the_point_estimate() -> None:
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(85, 100) < 0.85 < wilson_upper_bound(85, 100)
    assert wilson_lower_bound(100, 100) > 0.95
    # More samples at the same rate tighten the interval.
    assert wilson_lower_bound(850, 1000) > wilson_lower_bound(85, 100)


def _result(success: bool, *, safety: bool = False, invalid: bool = False, bucket: str = "none") -> EpisodeResult:
    setup = EpisodeSetup(
        seed=0,
        split="eval_a",
        variant_id="v",
        randomized=False,
        position=(0.0, 0.0),
        yaw=0.0,
        density=500.0,
        friction_slide=1.0,
        drop_height=0.0,
        mass=0.02,
    )
    return EpisodeResult(
        setup=setup,
        success=success,
        failure_bucket=bucket if not success else "none",
        steps=105,
        max_lift_m=0.06,
        terminal_lift_m=0.06,
        hold_steps=30,
        terminal_contact_groups=2,
        max_penetration_m=0.0,
        max_contact_force_n=1.0,
        total_contact_impulse_ns=0.1,
        max_step_impulse_ns=0.01,
        support_assisted_terminal=False,
        support_assisted_in_retain=False,
        invalid_state=invalid,
        safety_violation=safety,
        reward_total=0.0,
        reward_components={},
    )


def test_a_tier_with_a_safety_violation_cannot_pass(scope: MvpScopeConfig) -> None:
    perfect = [_result(True) for _ in range(scope.tier("A").episodes)]
    assert _aggregate("A", scope, perfect).passed

    tainted = list(perfect)
    tainted[0] = _result(True, safety=True)
    report = _aggregate("A", scope, tainted)
    assert report.success_rate == 1.0
    assert not report.passed, "a perfect success rate must not launder a safety violation"

    broken = list(perfect)
    broken[0] = _result(True, invalid=True)
    assert not _aggregate("A", scope, broken).passed


def test_a_short_tier_cannot_pass(scope: MvpScopeConfig) -> None:
    short = [_result(True) for _ in range(scope.tier("B").episodes - 1)]
    assert not _aggregate("B", scope, short).passed


def test_tier_b_needs_the_wilson_bound_as_well_as_the_rate(scope: MvpScopeConfig) -> None:
    episodes = scope.tier("B").episodes
    results = [_result(index < 258, bucket="timeout") for index in range(episodes)]
    report = _aggregate("B", scope, results)
    assert report.success_rate == pytest.approx(258 / episodes)
    assert report.success_rate >= scope.tier("B").min_success_rate
    assert report.passed == (report.wilson_lower >= scope.tier("B").min_wilson_lower_bound)


def test_reload_probe_survives_the_save_load_boundary(tmp_path, scope, prior) -> None:
    """The probe recorded at save time must still be reproducible after a reload."""

    from qdgrasp.mvp.policy import load_checkpoint, verify_reload_probe

    network = ResidualActorCritic(observation_dim=8, action_dim=2, hidden=(16,))
    normalizer = RunningNormalizer(dimension=8)
    normalizer.update(np.random.default_rng(4).normal(size=(64, 8)))
    path = save_checkpoint(
        tmp_path / "probe.pt", network, normalizer, fingerprint=environment_fingerprint(scope, prior), stage="bc"
    )
    assert verify_reload_probe(path)

    # Corrupting the weights without rewriting the probe must be detected: that
    # is exactly the failure `checkpoint_reload_mismatch` is meant to catch.  The
    # content lineage now catches it one step earlier than the probe does, and
    # refuses the checkpoint outright rather than reporting a mismatch about an
    # artifact it already knows was rewritten.
    import torch

    payload = load_checkpoint(path)
    payload["state_dict"]["actor.0.bias"] = payload["state_dict"]["actor.0.bias"] + 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="weight lineage mismatch"):
        verify_reload_probe(path)


def test_worker_pool_survives_a_torch_trained_parent(scope: MvpScopeConfig) -> None:
    """Episode workers must run after the parent has trained a torch model.

    A forked child inherits the parent's OpenMP and allocator locks in whatever
    state they were in at fork time, and a parent fresh from a backward pass
    holds them.  The symptom is not a crash but a hang -- workers idling at zero
    CPU while the parent waits forever -- so the regression is worth a real
    subprocess round trip rather than an assertion about the start method.
    """

    import torch

    from qdgrasp.mvp.evaluate import run_episodes

    network = torch.nn.Linear(32, 32)
    for _ in range(5):
        network(torch.randn(64, 32)).sum().backward()

    jobs = [(scope.episode_seed("dev", index), "dev", True, None) for index in range(4)]
    results = run_episodes(jobs, workers=2)
    assert len(results) == 4
