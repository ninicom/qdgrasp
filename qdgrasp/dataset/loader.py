"""Cross-embodiment dataset loader and registration with training framework."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..config.registry import register_dataset
from ..config.schema import ConfigError
from ..models.protocol import ProtocolDatasetView, load_protocol
from .artifact import DatasetArtifact
from .rng import derive_seed
from .schema import DataConfigV2


class DgnOpenDataset(Dataset):
    """PyTorch Dataset loading cross-embodiment grasp samples from verified shards."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        robot_name: str | None = None,
        point_count: int = 1024,
        manifest_file: str = "dataset_manifest.json",
        seed: int = 0,
        allowed_robot_names: Sequence[str] | None = None,
        robot_config: Any | None = None,
        protocol_file: str | Path | None = None,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.split = split
        self.robot_name = robot_name
        self.point_count = point_count
        self.seed = int(seed)

        self.artifact = DatasetArtifact.open_verified(self.root, manifest_file=manifest_file)
        self.manifest_spec = self.artifact.manifest
        allowed = (
            {str(name).removesuffix(".yaml") for name in allowed_robot_names}
            if allowed_robot_names is not None
            else set(self.artifact.robot_contracts)
        )
        unknown_allowed = sorted(allowed - set(self.artifact.robot_contracts))
        if unknown_allowed:
            raise ConfigError(f"data config permits robots absent from the artifact: {unknown_allowed}")
        if robot_name is not None and robot_name not in allowed:
            raise ConfigError(f"robot {robot_name!r} is outside the data config allowlist {sorted(allowed)}")

        if robot_config is not None:
            configured_name = str(getattr(robot_config, "name", ""))
            if robot_name is None or configured_name != robot_name:
                raise ConfigError(
                    f"dataset robot {robot_name!r} does not match bound robot profile {configured_name!r}"
                )
            expected_hash = self.manifest_spec.robot_profile_hashes[robot_name]
            actual_hash = robot_config.content_hash()
            if actual_hash != expected_hash:
                raise ConfigError(
                    f"bound robot profile hash mismatch for {robot_name}: expected {expected_hash}, got {actual_hash}"
                )
            expected_joints = self.artifact.robot_contracts[robot_name].joint_names
            if tuple(getattr(robot_config, "joints", ())) != expected_joints:
                raise ConfigError(f"bound robot joint order does not match artifact contract for {robot_name}")

        physical = self.artifact.samples(split=split, robot_name=robot_name)
        self.protocol_view: ProtocolDatasetView | None = None
        if protocol_file is None:
            self.samples = list(physical)
        else:
            # A protocol names an exact (split, robot, object_id) matrix, so it
            # can only be materialised for one hand at a time.  Training "on the
            # protocol" while reading every hand's shards is the thing this
            # refuses.
            if robot_name is None:
                raise ConfigError(
                    "a protocol view is defined per hand; open the dataset with a robot profile or drop "
                    "protocol_file to read the physical split"
                )
            protocol = load_protocol(protocol_file)
            self.protocol_view = ProtocolDatasetView(
                physical,
                protocol=protocol,
                split=split,
                robot=robot_name,
                manifest=self.manifest_spec.model_dump(by_alias=True),
                dataset_root=self.root,
                dataset_manifest_hash=self.artifact.manifest_hash,
            )
            self.samples = list(self.protocol_view)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        pts = item["points"]
        point_mask = torch.ones(self.point_count, dtype=torch.bool)
        if pts.shape[0] > self.point_count:
            generator = torch.Generator().manual_seed(
                derive_seed(
                    self.seed,
                    self.manifest_spec.dataset_id,
                    self.split,
                    item["robot_name"],
                    item["object_id"],
                    index,
                )
            )
            indices = torch.randperm(pts.shape[0], generator=generator)[: self.point_count]
            pts = pts[indices]
        elif pts.shape[0] < self.point_count:
            valid_points = int(pts.shape[0])
            pad = torch.zeros((self.point_count - pts.shape[0], 3), dtype=pts.dtype)
            pts = torch.cat([pts, pad], dim=0)
            point_mask[valid_points:] = False

        sample_robot = str(item["robot_name"])
        robot_contract = self.artifact.robot_contracts[sample_robot]

        return {
            "points": pts,
            "point_mask": point_mask,
            "palm_pos": item["palm_pos"],
            "palm_rot": item["palm_rot"],
            "joint_angles": item["joint_angles"],
            "fingertip_positions": item["fingertip_positions"],
            "success": item["success"],
            "quality": item["quality"],
            "object_id": item["object_id"],
            "robot_name": sample_robot,
            "robot_profile_hash": robot_contract.profile_hash,
            "joint_names": robot_contract.joint_names,
            "proposal_valid": item["proposal_valid"],
            "ik_valid": item["ik_valid"],
            "collision_valid": item["collision_valid"],
            "static_force_valid": item["static_force_valid"],
            "dynamic_valid": item["dynamic_valid"],
            "kinematics_valid": item["kinematics_valid"],
            "pose_target_valid": item["pose_target_valid"],
            "joint_target_valid": item["joint_target_valid"],
            "fk_target_valid": item["fk_target_valid"],
            # Phase 1 legacy aliases
            "target_translation": item["palm_pos"],
            "target_rotation": item["palm_rot"],
            "target_joints": item["joint_angles"],
        }

    def manifest(self) -> dict[str, Any]:
        """Return dataset provenance metadata for training run manifest."""
        document: dict[str, Any] = {
            "dataset_id": self.manifest_spec.dataset_id,
            "generator_version": self.manifest_spec.generator_version,
            "seed": self.manifest_spec.seed,
            "split": self.split,
            "robot_name": self.robot_name or "all",
            "samples": len(self.samples),
            "license": self.manifest_spec.license,
            "dataset_manifest_hash": self.artifact.manifest_hash,
            "robot_profile_hash": (
                self.manifest_spec.robot_profile_hashes[self.robot_name] if self.robot_name is not None else "mixed"
            ),
        }
        if self.protocol_view is not None:
            document["protocol_view"] = self.protocol_view.manifest()
        return document


def create_dgn_open_dataset(config: Any, *args: Any, split: str = "train", **kwargs: Any) -> Dataset:
    """Builder callback for registered dataset configuration."""
    if isinstance(config, DataConfigV2):
        root = config.dataset_root
        p_count = config.point_count
        m_file = config.manifest_file
        seed = config.seed
        allowed_profiles = config.robot_profiles
        protocol_file = config.protocol_file
    elif isinstance(config, dict):
        root = config.get("dataset_root", "datasets/dgn-open-tiny")
        p_count = config.get("point_count", 1024)
        m_file = config.get("manifest_file", "dataset_manifest.json")
        seed = config.get("seed", 0)
        allowed_profiles = config.get("robot_profiles")
        protocol_file = config.get("protocol_file")
    else:
        root = getattr(config, "dataset_root", "datasets/dgn-open-tiny")
        p_count = getattr(config, "point_count", 1024)
        m_file = getattr(config, "manifest_file", "dataset_manifest.json")
        seed = getattr(config, "seed", 0)
        allowed_profiles = getattr(config, "robot_profiles", None)
        protocol_file = getattr(config, "protocol_file", None)

    robot_name = None
    robot_config = None
    if len(args) > 0 and hasattr(args[0], "name"):
        robot_config = args[0]
        robot_name = robot_config.name
    elif "robot_config" in kwargs and hasattr(kwargs["robot_config"], "name"):
        robot_config = kwargs["robot_config"]
        robot_name = robot_config.name
    elif "robot_name" in kwargs:
        robot_name = kwargs["robot_name"]

    return DgnOpenDataset(
        dataset_root=root,
        split=split,
        robot_name=robot_name,
        point_count=p_count,
        manifest_file=m_file,
        seed=seed,
        allowed_robot_names=allowed_profiles,
        robot_config=robot_config,
        protocol_file=protocol_file,
    )


# Register builders with framework
register_dataset("dgn_open")(create_dgn_open_dataset)
register_dataset("dgn-open")(create_dgn_open_dataset)
register_dataset("qdgrasp/data/v2")(create_dgn_open_dataset)
