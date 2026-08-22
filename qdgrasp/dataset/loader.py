"""Cross-embodiment dataset loader and registration with training framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..config.registry import register_dataset
from ..config.schema import ConfigError
from .batch import GraspBatch
from .manifest import load_dataset_manifest
from .schema import DataConfigV2
from .shards import read_shard_file


class DgnOpenDataset(Dataset):
    """PyTorch Dataset loading cross-embodiment grasp samples from verified shards."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        robot_name: str | None = None,
        point_count: int = 1024,
        manifest_file: str = "dataset_manifest.json",
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.split = split
        self.robot_name = robot_name
        self.point_count = point_count

        manifest_path = self.root / manifest_file
        self.manifest_spec = load_dataset_manifest(manifest_path)

        self.samples: list[dict[str, Any]] = []
        for shard_meta in self.manifest_spec.shards:
            if shard_meta.split == split:
                if self.robot_name is not None and shard_meta.robot_name != self.robot_name:
                    continue
                shard_p = self.root / shard_meta.filename
                shard_samples = read_shard_file(shard_p, expected_sha256=shard_meta.sha256)
                self.samples.extend(shard_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        pts = item["points"]
        if pts.shape[0] > self.point_count:
            pts = pts[: self.point_count]
        elif pts.shape[0] < self.point_count:
            pad = torch.zeros((self.point_count - pts.shape[0], 3), dtype=pts.dtype)
            pts = torch.cat([pts, pad], dim=0)

        return {
            "points": pts,
            "palm_pos": item["palm_pos"],
            "palm_rot": item["palm_rot"],
            "joint_angles": item["joint_angles"],
            "fingertip_positions": item["fingertip_positions"],
            "success": item["success"],
            "quality": item["quality"],
            "object_id": item["object_id"],
            "robot_name": item["robot_name"],
            # Phase 1 legacy aliases
            "target_translation": item["palm_pos"],
            "target_rotation": item["palm_rot"],
            "target_joints": item["joint_angles"],
        }

    def manifest(self) -> dict[str, Any]:
        """Return dataset provenance metadata for training run manifest."""
        return {
            "dataset_id": self.manifest_spec.dataset_id,
            "generator_version": self.manifest_spec.generator_version,
            "seed": self.manifest_spec.seed,
            "split": self.split,
            "robot_name": self.robot_name or "all",
            "samples": len(self.samples),
            "license": self.manifest_spec.license,
        }


def create_dgn_open_dataset(config: Any, *args: Any, split: str = "train", **kwargs: Any) -> Dataset:
    """Builder callback for registered dataset configuration."""
    if isinstance(config, DataConfigV2):
        root = config.dataset_root
        p_count = config.point_count
        m_file = config.manifest_file
    elif isinstance(config, dict):
        root = config.get("dataset_root", "datasets/dgn-open-tiny")
        p_count = config.get("point_count", 1024)
        m_file = config.get("manifest_file", "dataset_manifest.json")
    else:
        root = getattr(config, "dataset_root", "datasets/dgn-open-tiny")
        p_count = getattr(config, "point_count", 1024)
        m_file = getattr(config, "manifest_file", "dataset_manifest.json")

    robot_name = None
    if len(args) > 0 and hasattr(args[0], "name"):
        robot_name = getattr(args[0], "name")
    elif "robot_config" in kwargs and hasattr(kwargs["robot_config"], "name"):
        robot_name = kwargs["robot_config"].name
    elif "robot_name" in kwargs:
        robot_name = kwargs["robot_name"]

    return DgnOpenDataset(
        dataset_root=root,
        split=split,
        robot_name=robot_name,
        point_count=p_count,
        manifest_file=m_file,
    )


# Register builders with framework
register_dataset("dgn_open")(create_dgn_open_dataset)
register_dataset("dgn-open")(create_dgn_open_dataset)
register_dataset("qdgrasp/data/v2")(create_dgn_open_dataset)
