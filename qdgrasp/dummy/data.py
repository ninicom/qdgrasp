"""Deterministic synthetic dataset for Phase 1 lifecycle tests.

Samples are generated from the sample index alone, so a shard is reproducible
without storing bytes and identical on CPU and CUDA hosts.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from ..config.registry import register_dataset
from ..config.schema import ConfigError, DataConfig, RobotConfig


class DummyPointDataset(Dataset):
    """Point clouds around a random centre, with matching palm/joint targets."""

    def __init__(self, *, samples: int, num_points: int, seed: int, robot_config: RobotConfig, split: str) -> None:
        if samples < 1 or num_points < 1:
            raise ConfigError("dummy dataset needs samples >= 1 and num_points >= 1")
        self.samples = samples
        self.num_points = num_points
        self.seed = seed
        self.split = split
        self.robot_config = robot_config
        self.lower = torch.tensor(robot_config.lower_limits, dtype=torch.float32)
        self.upper = torch.tensor(robot_config.upper_limits, dtype=torch.float32)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        centre = torch.rand(3, generator=generator) * 0.2 - 0.1
        points = centre + 0.05 * torch.randn(self.num_points, 3, generator=generator)
        fraction = torch.rand(len(self.lower), generator=generator)
        joints = self.lower + (self.upper - self.lower) * fraction
        return {
            "points": points.to(torch.float32),
            "target_translation": centre.to(torch.float32),
            "target_joints": joints.to(torch.float32),
        }

    def manifest(self) -> dict[str, Any]:
        """Immutable description of what this split generates."""

        return {
            "generator": "dummy_points",
            "split": self.split,
            "samples": self.samples,
            "num_points": self.num_points,
            "seed": self.seed,
            "robot": self.robot_config.name,
            "joints": list(self.robot_config.joints),
        }


@register_dataset("dummy_points")
def build_dummy_points(
    data_config: DataConfig, robot_config: RobotConfig, *, split: str
) -> DummyPointDataset:
    """Registry entry point for ``type: dummy_points``."""

    params = dict(data_config.params)
    unknown = sorted(set(params) - {"train_samples", "val_samples", "num_points", "seed"})
    if unknown:
        raise ConfigError(f"dataset '{data_config.name}': unknown params {unknown}")
    if split not in ("train", "val"):
        raise ConfigError(f"dataset '{data_config.name}': unknown split '{split}'")
    samples = int(params.get("train_samples", 32) if split == "train" else params.get("val_samples", 8))
    return DummyPointDataset(
        samples=samples,
        num_points=int(params.get("num_points", 128)),
        seed=int(params.get("seed", 0)) + (0 if split == "train" else 7919),
        robot_config=robot_config,
        split=split,
    )
