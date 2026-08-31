"""MVP-01: the environment resets deterministically and cannot be fooled.

The second half of this file is the one that matters.  An MVP is easiest to
fool by writing a success predicate that something other than a grasp can
satisfy, so the predicate is attacked directly: with the closing schedule
disabled the hand never touches the target, and no amount of flailing may
produce a success.
"""

from __future__ import annotations

import numpy as np
import pytest

from qdgrasp.mvp.config import MvpScopeConfig, load_mvp_scope
from qdgrasp.mvp.env import (
    OBSERVATION_DIMENSION,
    OBSERVATION_FIELDS,
    DexAcquireMvpEnv,
    environment_fingerprint,
)
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable


@pytest.fixture(scope="module")
def scope() -> MvpScopeConfig:
    return load_mvp_scope()


@pytest.fixture(scope="module")
def prior() -> PinchPriorTable:
    return PinchPriorTable.load(DEFAULT_PRIOR_PATH)


@pytest.fixture(scope="module")
def env(scope: MvpScopeConfig, prior: PinchPriorTable) -> DexAcquireMvpEnv:
    return DexAcquireMvpEnv(scope, prior)


def test_observation_layout_is_declared_and_finite(env: DexAcquireMvpEnv) -> None:
    observation = env.reset(3, "train")
    assert observation.shape == (OBSERVATION_DIMENSION,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert sum(size for _, size, _ in OBSERVATION_FIELDS) == OBSERVATION_DIMENSION
    assert all(description for _, _, description in OBSERVATION_FIELDS)


def test_reset_is_deterministic(env: DexAcquireMvpEnv) -> None:
    first = env.reset(17, "train")
    second = env.reset(17, "train")
    np.testing.assert_array_equal(first, second)


def test_identical_action_sequences_replay_identically(env: DexAcquireMvpEnv) -> None:
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.0, 1.0, size=(12, 8))

    def rollout() -> list[np.ndarray]:
        env.reset(23, "train")
        return [env.step(action)[0] for action in actions]

    for left, right in zip(rollout(), rollout()):
        np.testing.assert_array_equal(left, right)


def test_prior_alone_acquires_the_canonical_fixture(env: DexAcquireMvpEnv, scope: MvpScopeConfig) -> None:
    for variant in scope.train_variants:
        result = env.run_episode(0, "eval_a", variant_id=variant.variant_id, randomized=False)
        assert result.success, f"{variant.variant_id}: {result.failure_bucket}"
        assert not result.invalid_state and not result.safety_violation


def test_actions_stay_inside_the_declared_bounds(env: DexAcquireMvpEnv, scope: MvpScopeConfig) -> None:
    """An out-of-range action is clipped, never passed through to the actuators."""

    env.reset(5, "train")
    lower = env._joint_lower
    upper = env._joint_upper
    for _ in range(scope.episode.max_steps):
        if env.done:
            break
        env.step(np.full(8, 50.0))
        commanded = np.asarray(env._data.ctrl[list(env._index.actuator_ids)])
        assert np.all(commanded >= lower - 1e-9)
        assert np.all(commanded <= upper + 1e-9)


def test_wrong_action_dimension_is_rejected(env: DexAcquireMvpEnv) -> None:
    env.reset(5, "train")
    with pytest.raises(ValueError):
        env.step(np.zeros(4))


def test_stepping_a_finished_episode_is_an_error(env: DexAcquireMvpEnv) -> None:
    env.reset(5, "train")
    while not env.done:
        env.step(np.zeros(8))
    with pytest.raises(RuntimeError):
        env.step(np.zeros(8))


def test_random_policy_without_a_closing_hand_never_succeeds(
    scope: MvpScopeConfig, prior: PinchPriorTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the closing schedule removed, no action sequence may score a success.

    This is the false-positive probe.  The hand is held at its approach spread
    for the whole episode, so it can brush the target but never enclose it; if
    the predicate could be satisfied anyway, the predicate would be measuring
    something other than a grasp.
    """

    env = DexAcquireMvpEnv(scope, prior)
    original = DexAcquireMvpEnv.prior_command

    def open_handed(self: DexAcquireMvpEnv, step_index: int):  # type: ignore[no-untyped-def]
        palm, rotation, _, phase = original(self, step_index)
        return palm, rotation, self.scope.controller.approach_closure, phase

    monkeypatch.setattr(DexAcquireMvpEnv, "prior_command", open_handed)
    rng = np.random.default_rng(7)
    successes = 0
    for index in range(20):
        env.reset(scope.episode_seed("dev", index), "dev")
        while not env.done:
            env.step(rng.uniform(-1.0, 1.0, size=8))
        assert env.result is not None
        successes += int(env.result.success)
    assert successes == 0


def test_reward_never_decides_the_verdict(env: DexAcquireMvpEnv) -> None:
    """A failing episode may still bank positive shaping, and stays a failure."""

    result = env.run_episode(4, "dev")
    assert set(result.reward_components) == set(env.scope.reward.model_dump())
    assert isinstance(result.success, bool)
    assert (result.failure_bucket == "none") == result.success


def test_fingerprint_pins_every_hash_a_checkpoint_needs(scope: MvpScopeConfig, prior: PinchPriorTable) -> None:
    fingerprint = environment_fingerprint(scope, prior)
    assert fingerprint["environment_id"] == "QDGrasp-DexAcquire-MVP-v0"
    assert fingerprint["scope_hash"] == scope.content_hash()
    assert fingerprint["prior_hash"] == prior.content_hash()
    assert len(fingerprint["observation_schema_hash"]) == 64
