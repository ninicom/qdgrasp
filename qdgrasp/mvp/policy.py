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
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from qdgrasp.mvp.env import OBSERVATION_DIMENSION

POLICY_SCHEMA_V0 = "qdgrasp/mvp-policy/v0"
POLICY_SCHEMA_V1 = "qdgrasp/mvp-policy/v1"
POLICY_SCHEMA = POLICY_SCHEMA_V1
ACTION_DISTRIBUTION = "tanh-squashed-normal/v1"

#: Bounds on the learned log standard deviation.  The floor keeps PPO's
#: exploration from collapsing to a delta; the ceiling keeps a bounded residual
#: from turning into uniform noise.
LOG_STD_BOUNDS = (-4.0, 0.5)
_EMPTY_SAMPLE_SHAPE = torch.Size()


@dataclasses.dataclass
class RunningNormalizer:
    """Welford mean/variance over observations, fit on the train split only."""

    dimension: int
    count: float = 1e-4
    mean: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0, dtype=np.float64))
    m2: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __post_init__(self) -> None:
        if self.mean.size == 0:
            self.mean = np.zeros(self.dimension, dtype=np.float64)
        if self.m2.size == 0:
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
        return torch.tanh(self.latent_mean(observation))

    def latent_mean(self, observation: torch.Tensor) -> torch.Tensor:
        """Mean in unconstrained action space, before the tanh bijection."""

        return self.actor(observation)

    def distribution(self, observation: torch.Tensor) -> SquashedNormal:
        mean = self.latent_mean(observation)
        log_std = self.log_std.clamp(*LOG_STD_BOUNDS)
        return SquashedNormal(mean, log_std.exp())

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    def evaluate(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        log_prob = distribution.log_prob(action).sum(-1)
        entropy = distribution.entropy().sum(-1)
        return log_prob, entropy, self.value(observation)


class SquashedNormal:
    """A Normal in latent space mapped bijectively into ``(-1, 1)``.

    The environment executes the returned value and PPO evaluates that same
    value with the tanh change-of-variables correction.  This removes the old
    raw-Normal/clip mismatch where PPO assigned probability to an action the
    environment never executed.
    """

    _EPS = 1e-6

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        self.base = torch.distributions.Normal(loc, scale)

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self.base.mean)

    def sample(self, sample_shape: torch.Size = _EMPTY_SAMPLE_SHAPE) -> torch.Tensor:
        return torch.tanh(self.base.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size = _EMPTY_SAMPLE_SHAPE) -> torch.Tensor:
        return torch.tanh(self.base.rsample(sample_shape))

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        bounded = action.clamp(-1.0 + self._EPS, 1.0 - self._EPS)
        latent = torch.atanh(bounded)
        correction = torch.log(1.0 - bounded.square() + self._EPS)
        return self.base.log_prob(latent) - correction

    def entropy(self) -> torch.Tensor:
        # The transformed distribution has no closed-form entropy.  The latent
        # entropy is a stable exploration proxy and does not change which action
        # PPO evaluates; the policy ratio uses the exact corrected log density.
        return self.base.entropy()


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


def _canonical_hash(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one checkpoint file without loading or executing its payload."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _state_dict_hash(state_dict: dict[str, Any]) -> str:
    """Content digest over tensor names, shapes, dtypes and bytes."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
    normalizer_document = normalizer.to_document()
    state_dict = network.state_dict()
    metadata_document = dict(metadata or {})
    training_config = metadata_document.get("training_config")
    dataset_content_hash = metadata_document.get("dataset_content_hash")
    parent = metadata_document.get("parent")
    parent_checkpoint_hash: str | None = None
    if stage in {"bc", "ppo"}:
        if not _is_sha256(dataset_content_hash):
            raise ValueError(f"{stage} checkpoint requires a SHA-256 dataset_content_hash")
        if not isinstance(training_config, dict) or not training_config:
            raise ValueError(f"{stage} checkpoint requires a non-empty training_config mapping")
    if stage == "bc" and parent is not None:
        raise ValueError("bc checkpoint cannot declare a parent checkpoint")
    if stage == "ppo":
        if not isinstance(parent, str) or not parent:
            raise ValueError("ppo checkpoint requires a parent checkpoint path")
        try:
            parent_checkpoint_hash = file_sha256(parent)
        except OSError as error:
            raise ValueError(f"ppo parent checkpoint is unreadable: {parent!r}") from error
        declared_parent_hash = metadata_document.get("parent_checkpoint_hash")
        if declared_parent_hash is not None and declared_parent_hash != parent_checkpoint_hash:
            raise ValueError(
                "ppo parent checkpoint hash does not match the declared parent_checkpoint_hash"
            )
        metadata_document["parent_checkpoint_hash"] = parent_checkpoint_hash
    lineage_body = {
        "parent": parent,
        "parent_checkpoint_hash": parent_checkpoint_hash,
        "fingerprint_hash": _canonical_hash(dict(fingerprint)),
        "normalizer_hash": _canonical_hash(normalizer_document),
        "weights_hash": _state_dict_hash(state_dict),
        "dataset_content_hash": dataset_content_hash,
        "training_config_hash": _canonical_hash(training_config) if isinstance(training_config, dict) else None,
    }
    payload = {
        "schema": POLICY_SCHEMA,
        "stage": stage,
        "fingerprint": dict(fingerprint),
        "architecture": {
            "observation_dim": network.observation_dim,
            "action_dim": network.action_dim,
            "hidden": list(network.hidden),
            "action_distribution": ACTION_DISTRIBUTION,
        },
        "normalizer": normalizer_document,
        "metadata": metadata_document,
        "lineage": {**lineage_body, "lineage_hash": _canonical_hash(lineage_body)},
        "state_dict": state_dict,
        "optimizer_state": optimizer_state,
        "reload_probe": _reload_probe(network, normalizer),
    }
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"invalid policy checkpoint payload in {path}: expected a mapping")
    if payload.get("schema") != POLICY_SCHEMA:
        raise ValueError(
            f"unsupported policy checkpoint schema: {payload.get('schema')!r}; this build reads "
            f"{POLICY_SCHEMA!r}. A checkpoint written under another schema describes a different "
            "action contract and is not the same policy"
        )
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("action_distribution") != ACTION_DISTRIBUTION:
        raise ValueError(
            f"{path}: checkpoint does not declare action distribution {ACTION_DISTRIBUTION!r}"
        )
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise TypeError(f"{path}: checkpoint is missing content lineage")
    if lineage.get("fingerprint_hash") != _canonical_hash(payload.get("fingerprint", {})):
        raise ValueError(f"{path}: checkpoint fingerprint lineage mismatch")
    if lineage.get("normalizer_hash") != _canonical_hash(payload.get("normalizer", {})):
        raise ValueError(f"{path}: checkpoint normalizer lineage mismatch")
    if lineage.get("weights_hash") != _state_dict_hash(payload.get("state_dict", {})):
        raise ValueError(f"{path}: checkpoint weight lineage mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{path}: checkpoint metadata must be a mapping")
    training_config = metadata.get("training_config")
    expected_config_hash = _canonical_hash(training_config) if isinstance(training_config, dict) else None
    if lineage.get("training_config_hash") != expected_config_hash:
        raise ValueError(f"{path}: checkpoint training-config lineage mismatch")
    if lineage.get("dataset_content_hash") != metadata.get("dataset_content_hash"):
        raise ValueError(f"{path}: checkpoint dataset lineage mismatch")
    if lineage.get("parent") != metadata.get("parent"):
        raise ValueError(f"{path}: checkpoint parent lineage mismatch")
    if lineage.get("parent_checkpoint_hash") != metadata.get("parent_checkpoint_hash"):
        raise ValueError(f"{path}: checkpoint parent-hash lineage mismatch")
    lineage_body = {
        key: lineage.get(key)
        for key in (
            "parent",
            "parent_checkpoint_hash",
            "fingerprint_hash",
            "normalizer_hash",
            "weights_hash",
            "dataset_content_hash",
            "training_config_hash",
        )
    }
    if lineage.get("lineage_hash") != _canonical_hash(lineage_body):
        raise ValueError(f"{path}: checkpoint aggregate lineage mismatch")
    stage = payload.get("stage")
    if stage in {"bc", "ppo"}:
        if not _is_sha256(lineage.get("dataset_content_hash")):
            raise ValueError(f"{path}: {stage} checkpoint has no demonstration content lineage")
        if not _is_sha256(lineage.get("training_config_hash")):
            raise ValueError(f"{path}: {stage} checkpoint has no training-config lineage")
    if stage == "bc" and (lineage.get("parent") is not None or lineage.get("parent_checkpoint_hash") is not None):
        raise ValueError(f"{path}: bc checkpoint unexpectedly declares a parent")
    if stage == "ppo":
        if not isinstance(lineage.get("parent"), str) or not lineage["parent"]:
            raise ValueError(f"{path}: ppo checkpoint has no parent path")
        if not _is_sha256(lineage.get("parent_checkpoint_hash")):
            raise ValueError(f"{path}: ppo checkpoint has no parent content hash")
        try:
            actual_parent_hash = file_sha256(lineage["parent"])
        except OSError as error:
            raise ValueError(f"{path}: ppo parent checkpoint is unreadable: {lineage['parent']!r}") from error
        if actual_parent_hash != lineage["parent_checkpoint_hash"]:
            raise ValueError(f"{path}: ppo parent checkpoint content changed")
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
