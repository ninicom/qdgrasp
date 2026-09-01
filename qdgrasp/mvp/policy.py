"""Actor, critic, observation normalizer and the checkpoint that carries them.

``ROADMAP-MVP-001`` §3.2 fixes the shape: a small MLP, two hidden layers of 256
units by default, reading normalised state and emitting a bounded Gaussian
residual.  §5 fixes the discipline around it: normalization statistics are fit
on the train split only and travel with the checkpoint, and every checkpoint
carries the commit and the environment/config/normalizer/dataset hashes.

A checkpoint that cannot say which world it was trained in is not a checkpoint,
so :func:`load_policy` refuses one whose fingerprint disagrees with the
environment it is being loaded against.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from qdgrasp.mvp.env import OBSERVATION_DIMENSION

POLICY_SCHEMA_V0 = "qdgrasp/mvp-policy/v0"

#: Bounds on the learned log standard deviation.  The floor keeps PPO's
#: exploration from collapsing to a delta; the ceiling keeps a bounded residual
#: from turning into uniform noise.
LOG_STD_BOUNDS = (-4.0, 0.5)


@dataclasses.dataclass
class RunningNormalizer:
    """Welford mean/variance over observations, fit on the train split only."""

    dimension: int
    count: float = 1e-4
    mean: np.ndarray = dataclasses.field(default=None)  # type: ignore[assignment]
    m2: np.ndarray = dataclasses.field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mean is None:
            self.mean = np.zeros(self.dimension, dtype=np.float64)
        if self.m2 is None:
            self.m2 = np.ones(self.dimension, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        values = np.asarray(batch, dtype=np.float64).reshape(-1, self.dimension)
        if values.shape[0] == 0:
            return
        batch_count = float(values.shape[0])
        batch_mean = values.mean(axis=0)
        batch_m2 = values.var(axis=0) * batch_count
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.m2 = self.m2 + batch_m2 + delta**2 * self.count * batch_count / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.m2 / max(self.count, 1e-6), 1e-8))

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(observation, dtype=np.float64) - self.mean) / self.std, -10.0, 10.0)

    def to_document(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "count": self.count,
            "mean": [float(value) for value in self.mean],
            "m2": [float(value) for value in self.m2],
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> RunningNormalizer:
        return cls(
            dimension=int(document["dimension"]),
            count=float(document["count"]),
            mean=np.asarray(document["mean"], dtype=np.float64),
            m2=np.asarray(document["m2"], dtype=np.float64),
        )


class ResidualActorCritic(nn.Module):
    """Bounded Gaussian actor and a value head, over normalised observations."""

    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIMENSION,
        action_dim: int = 8,
        hidden: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden = tuple(hidden)
        self.actor = _mlp(observation_dim, self.hidden, action_dim)
        self.critic = _mlp(observation_dim, self.hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        # The last actor layer starts near zero so an untrained policy is the
        # controller prior rather than a random shove.
        final = self.actor[-1]
        assert isinstance(final, nn.Linear)
        nn.init.uniform_(final.weight, -1e-3, 1e-3)
        nn.init.zeros_(final.bias)

    def mean_action(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observation))

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_action(observation)
        log_std = self.log_std.clamp(*LOG_STD_BOUNDS)
        return torch.distributions.Normal(mean, log_std.exp())

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    def evaluate(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        log_prob = distribution.log_prob(action).sum(-1)
        entropy = distribution.entropy().sum(-1)
        return log_prob, entropy, self.value(observation)


def _mlp(inputs: int, hidden: tuple[int, ...], outputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = inputs
    for size in hidden:
        layers.append(nn.Linear(width, size))
        layers.append(nn.Tanh())
        width = size
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


class MvpPolicy:
    """A checkpoint made callable: normalise, forward, clip, hand back an action."""

    def __init__(self, network: ResidualActorCritic, normalizer: RunningNormalizer) -> None:
        self.network = network.eval()
        self.normalizer = normalizer

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normalised = torch.as_tensor(self.normalizer(observation), dtype=torch.float32)
            action = self.network.mean_action(normalised.unsqueeze(0))[0]
        return np.clip(action.numpy().astype(np.float64), -1.0, 1.0)


#: Seed of the reload probe.  Fixed so a probe recorded by one process can be
#: recomputed by any other, on any machine, without shipping the seed too.
_PROBE_SEED = 0xC0FFEE
_PROBE_SAMPLES = 16


def _reload_probe(network: ResidualActorCritic, normalizer: RunningNormalizer) -> dict[str, Any]:
    """Record what this network answers, so a reload can be checked against it.

    Comparing a reloaded checkpoint against a policy that was itself just loaded
    from the same file proves nothing.  The probe is written by the process that
    *trained* the weights, so verifying it later is a real comparison across a
    save/load boundary rather than a tautology.
    """

    rng = np.random.default_rng(_PROBE_SEED)
    observations = rng.normal(size=(_PROBE_SAMPLES, network.observation_dim))
    policy = MvpPolicy(network, normalizer)
    actions = np.stack([policy(row) for row in observations])
    return {"observations": observations.tolist(), "actions": actions.tolist()}


def verify_reload_probe(path: str | Path, atol: float = 1e-6) -> bool:
    """Does the checkpoint on disk still answer what it answered when saved?"""

    payload = load_checkpoint(path)
    probe = payload.get("reload_probe")
    if not probe:
        return False
    network, normalizer = build_from_checkpoint(payload)
    policy = MvpPolicy(network, normalizer)
    reproduced = np.stack([policy(np.asarray(row)) for row in probe["observations"]])
    return bool(np.allclose(reproduced, np.asarray(probe["actions"]), atol=atol))


def save_checkpoint(
    path: str | Path,
    network: ResidualActorCritic,
    normalizer: RunningNormalizer,
    *,
    fingerprint: dict[str, str],
    stage: str,
    metadata: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> Path:
    """Write a self-describing checkpoint bundle."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": POLICY_SCHEMA_V0,
        "stage": stage,
        "fingerprint": dict(fingerprint),
        "architecture": {
            "observation_dim": network.observation_dim,
            "action_dim": network.action_dim,
            "hidden": list(network.hidden),
        },
        "normalizer": normalizer.to_document(),
        "metadata": dict(metadata or {}),
        "state_dict": network.state_dict(),
        "optimizer_state": optimizer_state,
        "reload_probe": _reload_probe(network, normalizer),
    }
    torch.save(payload, target)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != POLICY_SCHEMA_V0:
        raise ValueError(
            f"unsupported policy checkpoint schema: {payload.get('schema')!r}; this build reads "
            f"{POLICY_SCHEMA_V0!r}. A checkpoint written under another schema describes a different "
            "action contract and is not the same policy"
        )
    return payload


def build_from_checkpoint(payload: dict[str, Any]) -> tuple[ResidualActorCritic, RunningNormalizer]:
    architecture = payload["architecture"]
    network = ResidualActorCritic(
        observation_dim=int(architecture["observation_dim"]),
        action_dim=int(architecture["action_dim"]),
        hidden=tuple(int(size) for size in architecture["hidden"]),
    )
    network.load_state_dict(payload["state_dict"])
    return network, RunningNormalizer.from_document(payload["normalizer"])


def load_policy(path: str | Path, *, fingerprint: dict[str, str] | None = None) -> MvpPolicy:
    """Load a checkpoint, refusing one trained against a different world."""

    payload = load_checkpoint(path)
    if fingerprint is not None:
        stored = payload.get("fingerprint", {})
        mismatched = {key: (stored.get(key), value) for key, value in fingerprint.items() if stored.get(key) != value}
        if mismatched:
            raise ValueError(f"checkpoint fingerprint does not match this environment: {mismatched}")
    network, normalizer = build_from_checkpoint(payload)
    return MvpPolicy(network, normalizer)


def checkpoint_reload_matches(path: str | Path, observations: np.ndarray, actions: np.ndarray) -> bool:
    """Does a reloaded checkpoint reproduce the actions it produced in memory?

    ``ROADMAP-MVP-001`` §7 counts ``checkpoint_reload_mismatch`` as a gate of its
    own, so the comparison is exact-to-tolerance rather than a shrug.
    """

    policy = load_policy(path)
    reproduced = np.stack([policy(observation) for observation in np.asarray(observations)])
    return bool(np.allclose(reproduced, np.asarray(actions), atol=1e-6))


def write_json(path: str | Path, document: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
