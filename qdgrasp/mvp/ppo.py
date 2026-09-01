"""MVP-04: residual PPO on top of the behaviour-cloned actor.

Clipped surrogate objective, GAE, a bounded Gaussian actor -- nothing exotic.
Two choices are worth naming.

*The safety budget is not negotiable.*  PPO acts through the same bounded
residual every other caller uses, so it cannot widen the workspace, exceed a
joint limit or spend more force than the budget allows.  An episode that trips
the budget terminates and is counted as a failure, which is the only pressure
the optimiser gets on the subject.

*Rollouts are collected by whole episodes.*  The verdict is an episode-level
predicate, so a truncated fragment would teach the critic about a state the
predicate never scores.  Episodes are cheap here (about a third of a second), so
the tidier accounting costs nothing worth having.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qdgrasp.mvp.config import MvpScopeConfig, load_mvp_scope
from qdgrasp.mvp.env import DexAcquireMvpEnv
from qdgrasp.mvp.policy import (
    LOG_STD_BOUNDS,
    ResidualActorCritic,
    RunningNormalizer,
    build_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable


@dataclasses.dataclass(frozen=True)
class PpoSpec:
    """Hyperparameters.  They live in the config hash, not in a notebook cell."""

    iterations: int = 40
    episodes_per_iteration: int = 64
    epochs: int = 6
    minibatch_size: int = 512
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.002
    max_grad_norm: float = 0.5
    #: Initial exploration scale, in unit-action space.  Round one left this at
    #: the network default of -1.0 (sigma 0.37); every stochastic rollout then
    #: jittered the palm target by millimetres at random every control step and
    #: the measured training success rate was 0.000 for all forty iterations, so
    #: PPO had no gradient to follow.  Exploration must perturb the prior, not
    #: drown it.
    init_log_std: float = -2.5
    seed: int = 0

    def to_document(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# -- rollout workers ------------------------------------------------------
#
# Spawned rather than forked: by the time PPO starts the parent has already
# trained a torch model, and a forked child inherits its OpenMP and allocator
# locks in a locked state, then hangs on its first allocation.
_SPAWN = multiprocessing.get_context("spawn")

_WORKER: dict[str, Any] = {}


def _worker_init(scope_path: str | None, prior_path: str) -> None:
    torch.set_num_threads(1)
    scope = load_mvp_scope(scope_path)
    prior = PinchPriorTable.load(prior_path)
    _WORKER["env"] = DexAcquireMvpEnv(scope, prior)
    _WORKER["version"] = None


def _worker_rollout(job: tuple[int, str, str, int]) -> dict[str, Any]:
    seed, split, checkpoint_path, version = job
    env: DexAcquireMvpEnv = _WORKER["env"]
    if _WORKER.get("version") != version:
        payload = load_checkpoint(checkpoint_path)
        network, normalizer = build_from_checkpoint(payload)
        _WORKER["network"] = network.eval()
        _WORKER["normalizer"] = normalizer
        _WORKER["version"] = version
    network: ResidualActorCritic = _WORKER["network"]
    normalizer: RunningNormalizer = _WORKER["normalizer"]
    std = np.exp(np.clip(network.log_std.detach().numpy(), *LOG_STD_BOUNDS))
    rng = np.random.default_rng(seed ^ (version * 0x9E3779B1))

    observations: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    rewards: list[float] = []
    observation = env.reset(seed, split)  # type: ignore[arg-type]
    while not env.done:
        normalised = torch.as_tensor(normalizer(observation), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            latent_mean = network.latent_mean(normalised)[0].numpy().astype(np.float64)
        # Sample in unconstrained space and apply the same tanh bijection whose
        # corrected density ``network.evaluate`` computes.  The stored action is
        # therefore exactly the bounded action the environment executes.
        action = np.tanh(latent_mean + std * rng.standard_normal(latent_mean.shape[0]))
        observations.append(np.asarray(observation, dtype=np.float32))
        executed_actions.append(action.astype(np.float32))
        observation, reward, _, _ = env.step(action)
        rewards.append(float(reward))
    result = env.result
    assert result is not None
    return {
        "observations": np.stack(observations),
        "actions": np.stack(executed_actions),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "success": bool(result.success),
        "failure_bucket": result.failure_bucket,
        "safety_violation": bool(result.safety_violation),
        "invalid_state": bool(result.invalid_state),
        "return": float(result.reward_total),
    }


def _collect(jobs: Sequence[tuple[int, str, str, int]], pool: ProcessPoolExecutor | None) -> list[dict[str, Any]]:
    """Roll out a batch of episodes, reusing one pool for the whole run.

    The pool outlives the iteration on purpose: a worker holds compiled MuJoCo
    models for every object variant it has seen, and rebuilding that per
    iteration would pay the compile cost forty times over for nothing.
    """

    if pool is None:
        return [_worker_rollout(job) for job in jobs]
    return list(pool.map(_worker_rollout, jobs, chunksize=2))


def _gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for one finished episode (no bootstrap)."""

    advantages = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * lam * running
        advantages[index] = running
    return advantages, advantages + values


def train_residual_ppo(
    network: ResidualActorCritic,
    normalizer: RunningNormalizer,
    scope: MvpScopeConfig,
    *,
    spec: PpoSpec | None = None,
    scope_path: str | None = None,
    prior_path: str = str(DEFAULT_PRIOR_PATH),
    fingerprint: dict[str, str] | None = None,
    workers: int | None = None,
    output_dir: str | Path | None = None,
    seed_offset: int = 0,
) -> tuple[ResidualActorCritic, dict[str, Any]]:
    """Fine-tune the actor in simulation and return it with its learning curve."""

    settings = spec or PpoSpec()
    torch.manual_seed(settings.seed)
    with torch.no_grad():
        network.log_std.fill_(settings.init_log_std)
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    _WORKER["scope_path"] = scope_path
    _WORKER["prior_path"] = prior_path
    if worker_count <= 1:
        _worker_init(scope_path, prior_path)

    pool = (
        None
        if worker_count <= 1
        else ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=_SPAWN,
            initializer=_worker_init,
            initargs=(scope_path, prior_path),
        )
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
    staging = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="qdgrasp-mvp-ppo-"))
    staging.mkdir(parents=True, exist_ok=True)
    live_path = staging / "ppo-live.pt"

    curve: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_success = -1.0
    episode_cursor = seed_offset

    try:
        for iteration in range(settings.iterations):
            save_checkpoint(
                live_path,
                network,
                normalizer,
                fingerprint=fingerprint or {},
                stage="ppo_live",
                metadata={"iteration": iteration},
            )
            jobs = [
                (scope.episode_seed("train", episode_cursor + index), "train", str(live_path), iteration + 1)
                for index in range(settings.episodes_per_iteration)
            ]
            episode_cursor += settings.episodes_per_iteration
            episodes = _collect(jobs, pool)

            observations = torch.as_tensor(
                normalizer(np.concatenate([episode["observations"] for episode in episodes])),
                dtype=torch.float32,
            )
            actions = torch.as_tensor(np.concatenate([episode["actions"] for episode in episodes]), dtype=torch.float32)
            with torch.no_grad():
                old_log_prob, _, values = network.evaluate(observations, actions)
            values_np = values.numpy().astype(np.float64)

            advantages_parts: list[np.ndarray] = []
            returns_parts: list[np.ndarray] = []
            cursor = 0
            for episode in episodes:
                length = len(episode["rewards"])
                episode_values = values_np[cursor : cursor + length]
                advantage, episode_return = _gae(
                    episode["rewards"].astype(np.float64), episode_values, settings.gamma, settings.gae_lambda
                )
                advantages_parts.append(advantage)
                returns_parts.append(episode_return)
                cursor += length
            advantages = torch.as_tensor(np.concatenate(advantages_parts), dtype=torch.float32)
            returns = torch.as_tensor(np.concatenate(returns_parts), dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            generator = torch.Generator().manual_seed(settings.seed + iteration)
            policy_losses: list[float] = []
            value_losses: list[float] = []
            for _ in range(settings.epochs):
                order = torch.randperm(observations.shape[0], generator=generator)
                for start in range(0, order.shape[0], settings.minibatch_size):
                    batch = order[start : start + settings.minibatch_size]
                    log_prob, entropy, value = network.evaluate(observations[batch], actions[batch])
                    ratio = torch.exp(log_prob - old_log_prob[batch])
                    surrogate = torch.min(
                        ratio * advantages[batch],
                        torch.clamp(ratio, 1.0 - settings.clip_ratio, 1.0 + settings.clip_ratio) * advantages[batch],
                    )
                    policy_loss = -surrogate.mean()
                    value_loss = torch.nn.functional.mse_loss(value, returns[batch])
                    loss = (
                        policy_loss
                        + settings.value_coefficient * value_loss
                        - settings.entropy_coefficient * entropy.mean()
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), settings.max_grad_norm)
                    optimizer.step()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))

            success_rate = float(np.mean([episode["success"] for episode in episodes]))
            row = {
                "iteration": iteration,
                "episodes": len(episodes),
                "train_success_rate": success_rate,
                "mean_return": float(np.mean([episode["return"] for episode in episodes])),
                "safety_violations": int(sum(episode["safety_violation"] for episode in episodes)),
                "invalid_states": int(sum(episode["invalid_state"] for episode in episodes)),
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else float("nan"),
                "value_loss": float(np.mean(value_losses)) if value_losses else float("nan"),
            }
            curve.append(row)
            if success_rate > best_success:
                best_success = success_rate
                best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}

    finally:
        if pool is not None:
            pool.shutdown()

    metrics = {
        "spec": settings.to_document(),
        "curve": curve,
        "best_train_success_rate": best_success,
        "episodes_consumed": episode_cursor - seed_offset,
        "staging_dir": str(staging),
    }
    if best_state is not None:
        network.load_state_dict(best_state)
    return network, metrics
