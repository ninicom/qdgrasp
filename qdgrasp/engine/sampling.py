"""Deterministic, resumable batch ordering.

A dedicated permutation stream keeps the sample order reproducible and lets the
resume artifact restore the exact position inside an epoch, which a shuffling
``DataLoader`` iterator cannot express once it has been consumed.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import default_collate


class BatchIdentityError(ValueError):
    """Samples cannot be evaluated by one robot-bound model graph."""


class DeterministicBatchStream:
    """Endless stream of index batches over ``dataset_size`` samples."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int) -> None:
        if dataset_size < 1 or batch_size < 1:
            raise ValueError("dataset_size and batch_size must be >= 1")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.position = 0
        self.permutation = torch.randperm(dataset_size, generator=self.generator)

    def next_indices(self) -> list[int]:
        """Return the next batch of sample indices, reshuffling at epoch end."""

        indices: list[int] = []
        while len(indices) < self.batch_size:
            if self.position >= self.dataset_size:
                self.epoch += 1
                self.position = 0
                self.permutation = torch.randperm(self.dataset_size, generator=self.generator)
            take = min(self.batch_size - len(indices), self.dataset_size - self.position)
            indices.extend(int(value) for value in self.permutation[self.position : self.position + take])
            self.position += take
        return indices

    def state_dict(self) -> dict[str, Any]:
        """Serializable stream position."""

        return {
            "epoch": self.epoch,
            "position": self.position,
            "permutation": self.permutation.clone(),
            "generator": self.generator.get_state(),
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a position written by :meth:`state_dict`."""

        if int(state["dataset_size"]) != self.dataset_size or int(state["batch_size"]) != self.batch_size:
            raise ValueError("resume state was recorded for a different dataset/batch size")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.permutation = state["permutation"].clone()
        self.generator.set_state(state["generator"].to(torch.uint8))


def collate_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Canonical collation with an exact robot/profile/joint-order contract.

    Generic datasets without a ``robot_name`` remain supported.  Once a sample
    declares a robot, however, every item must declare the same robot, profile
    hash and ordered joint names; shape equality alone is not identity (LEAP and
    Allegro both currently expose sixteen actuated joints).
    """

    if not samples:
        raise BatchIdentityError("cannot collate an empty batch")
    names = [sample.get("robot_name") for sample in samples]
    declares_robot = [name is not None for name in names]
    if any(declares_robot):
        if not all(declares_robot):
            raise BatchIdentityError("a batch mixes robot-bound and identity-free samples")
        unique_names = {str(name) for name in names}
        if len(unique_names) != 1:
            raise BatchIdentityError(
                f"a batch may not mix robots {sorted(unique_names)}; use one robot graph per batch"
            )

        missing: dict[str, list[int]] = {}
        for field in ("robot_profile_hash", "joint_names"):
            absent = [index for index, sample in enumerate(samples) if field not in sample]
            if absent:
                missing[field] = absent
        if missing:
            raise BatchIdentityError(
                f"robot-bound samples are missing semantic identity fields {missing}; "
                "joint tensor width is not a substitute for joint order"
            )

        profile_hashes = {str(sample["robot_profile_hash"]) for sample in samples}
        joint_orders = {tuple(str(name) for name in sample["joint_names"]) for sample in samples}
        if len(profile_hashes) != 1 or len(joint_orders) != 1:
            raise BatchIdentityError(
                "samples name one robot but disagree on its profile hash or ordered joints"
            )

        tensor_fields = [
            {key: value for key, value in sample.items() if key not in {"robot_name", "robot_profile_hash", "joint_names"}}
            for sample in samples
        ]
        batch = default_collate(tensor_fields)
        batch["robot_name"] = next(iter(unique_names))
        batch["robot_profile_hash"] = next(iter(profile_hashes))
        batch["joint_names"] = next(iter(joint_orders))
        return batch

    return default_collate(list(samples))


def collate_indices(dataset: Sequence[dict[str, Any]], indices: Sequence[int]) -> dict[str, Any]:
    """Collate sample indices through :func:`collate_samples`."""

    return collate_samples([dataset[index] for index in indices])


def iterate_batches(dataset: Sequence[dict[str, Any]], batch_size: int) -> Iterator[dict[str, Any]]:
    """Yield in-order batches over the whole dataset (used for validation)."""

    for start in range(0, len(dataset), batch_size):
        yield collate_indices(dataset, range(start, min(start + batch_size, len(dataset))))


__all__ = [
    "BatchIdentityError",
    "DeterministicBatchStream",
    "collate_indices",
    "collate_samples",
    "iterate_batches",
]
