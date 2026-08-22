"""Deterministic, resumable batch ordering.

A dedicated permutation stream keeps the sample order reproducible and lets the
resume artifact restore the exact position inside an epoch, which a shuffling
``DataLoader`` iterator cannot express once it has been consumed.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import default_collate


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


def collate_indices(dataset: Sequence[dict[str, torch.Tensor]], indices: Sequence[int]) -> dict[str, torch.Tensor]:
    """Collate the given sample indices into one batch dictionary."""

    return default_collate([dataset[index] for index in indices])


def iterate_batches(dataset: Sequence[dict[str, torch.Tensor]], batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
    """Yield in-order batches over the whole dataset (used for validation)."""

    for start in range(0, len(dataset), batch_size):
        yield collate_indices(dataset, range(start, min(start + batch_size, len(dataset))))
