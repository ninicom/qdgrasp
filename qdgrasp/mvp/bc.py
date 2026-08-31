"""MVP-03: behaviour cloning of the expert residual.

Short and unglamorous by design.  The expert search has already done the hard
part; this fits a small network to what it chose, on the train split only, and
writes a checkpoint that can be reloaded and re-measured.

Two things are worth stating.

The normalizer is fitted here, on train observations alone, and saved inside the
checkpoint.  A normalizer refitted at evaluation time would quietly let held-out
statistics into a policy that is supposed never to have seen them.

The previous action is dropped out of a fraction of training samples.  The plan
puts the previous action in the observation (§3.3), and a demonstration whose
residual is constant within a segment makes ``a_t = a_{t-1}`` the cheapest
function that fits -- which is a shortcut, not a policy, and at rollout time it
degenerates into a random walk anchored to nothing.  Zeroing that block on a
fraction of samples means the copy cannot be relied on, so the rest of the
observation has to carry the prediction.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from torch import nn

from qdgrasp.mvp.env import OBSERVATION_FIELDS
from qdgrasp.mvp.expert import DemonstrationSet
from qdgrasp.mvp.policy import ResidualActorCritic, RunningNormalizer


def previous_action_slice() -> slice:
    """Where the previous action sits in the observation vector."""

    offset = 0
    for name, size, _ in OBSERVATION_FIELDS:
        if name == "previous_action":
            return slice(offset, offset + size)
        offset += size
    raise KeyError("the observation schema has no previous_action block")


@dataclasses.dataclass(frozen=True)
class BehaviorCloningSpec:
    """Hyperparameters, all of them in the config hash rather than a notebook."""

    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    hidden: tuple[int, ...] = (256, 256)
    #: Fraction of training samples whose ``previous_action`` block is zeroed.
    previous_action_dropout: float = 0.5
    seed: int = 0

    def to_document(self) -> dict[str, Any]:
        document = dataclasses.asdict(self)
        document["hidden"] = list(self.hidden)
        return document


def _episode_split(demonstrations: DemonstrationSet, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split by episode, never by frame (``ROADMAP-MVP-001`` §5)."""

    episodes = np.unique(demonstrations.episode_index)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    holdout = max(1, round(len(shuffled) * fraction))
    validation_episodes = set(shuffled[:holdout].tolist())
    mask = np.array([index in validation_episodes for index in demonstrations.episode_index])
    return ~mask, mask


def train_behavior_cloning(
    demonstrations: DemonstrationSet,
    spec: BehaviorCloningSpec | None = None,
) -> tuple[ResidualActorCritic, RunningNormalizer, dict[str, Any]]:
    """Fit the actor to the expert residuals and report the learning curve."""

    settings = spec or BehaviorCloningSpec()
    torch.manual_seed(settings.seed)
    train_mask, validation_mask = _episode_split(demonstrations, settings.validation_fraction, settings.seed)

    observations = demonstrations.observations.astype(np.float64)
    actions = demonstrations.actions.astype(np.float64)
    normalizer = RunningNormalizer(dimension=observations.shape[1])
    normalizer.update(observations[train_mask])

    features = torch.as_tensor(normalizer(observations), dtype=torch.float32)
    targets = torch.as_tensor(actions, dtype=torch.float32)
    train_features, train_targets = features[train_mask], targets[train_mask]
    validation_features, validation_targets = features[validation_mask], targets[validation_mask]

    network = ResidualActorCritic(
        observation_dim=observations.shape[1],
        action_dim=actions.shape[1],
        hidden=settings.hidden,
    )
    optimizer = torch.optim.AdamW(
        network.actor.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(settings.seed)
    copy_block = previous_action_slice()
    curve: list[dict[str, float]] = []
    for epoch in range(settings.epochs):
        network.train()
        order = torch.randperm(train_features.shape[0], generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, order.shape[0], settings.batch_size):
            batch = order[start : start + settings.batch_size]
            features_batch = train_features[batch]
            if settings.previous_action_dropout > 0.0:
                dropped = torch.rand(features_batch.shape[0], 1, generator=generator) < settings.previous_action_dropout
                features_batch = features_batch.clone()
                features_batch[:, copy_block] = torch.where(
                    dropped, torch.zeros_like(features_batch[:, copy_block]), features_batch[:, copy_block]
                )
            optimizer.zero_grad()
            loss = loss_fn(network.mean_action(features_batch), train_targets[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        network.eval()
        with torch.no_grad():
            validation_loss = (
                float(loss_fn(network.mean_action(validation_features), validation_targets).item())
                if validation_features.shape[0]
                else float("nan")
            )
        curve.append(
            {
                "epoch": epoch,
                "train_loss": total / max(batches, 1),
                "validation_loss": validation_loss,
            }
        )

    network.eval()
    with torch.no_grad():
        predicted = network.mean_action(features).numpy()
    metrics = {
        "spec": settings.to_document(),
        "transitions": int(observations.shape[0]),
        "train_transitions": int(train_mask.sum()),
        "validation_transitions": int(validation_mask.sum()),
        "episodes": demonstrations.episodes,
        "final_train_loss": curve[-1]["train_loss"] if curve else float("nan"),
        "final_validation_loss": curve[-1]["validation_loss"] if curve else float("nan"),
        # A cloned policy that outputs zero everywhere would have a small loss
        # and no behaviour; report the magnitude so that failure is visible.
        "mean_predicted_action_magnitude": float(np.mean(np.abs(predicted))),
        "mean_expert_action_magnitude": float(np.mean(np.abs(actions))),
        "curve": curve,
    }
    return network, normalizer, metrics
