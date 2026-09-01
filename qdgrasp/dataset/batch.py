"""Batch container representing a batch of multi-finger grasp samples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GraspBatch:
    """Strongly-typed batch container for cross-embodiment grasp learning."""

    points: torch.Tensor  # [B, N, 3] Object point clouds
    palm_pos: torch.Tensor  # [B, 3] World palm translation
    palm_rot: torch.Tensor  # [B, 3, 3] World palm rotation
    joint_angles: torch.Tensor  # [B, J] Robot actuated joint angles
    fingertip_positions: torch.Tensor  # [B, K, 3] Fingertip positions in world
    success: torch.Tensor  # [B] Binary grasp success (1.0 or 0.0)
    quality: torch.Tensor  # [B] Continuous lift height / quality score
    kinematics_valid: torch.Tensor  # [B] Complete pose-and-joint target is available
    pose_target_valid: torch.Tensor  # [B] Palm translation/rotation target is available
    joint_target_valid: torch.Tensor  # [B] Joint target is available
    fk_target_valid: torch.Tensor  # [B] Fingertip target is available
    object_ids: list[str]
    robot_names: list[str]

    @property
    def batch_size(self) -> int:
        return self.points.shape[0]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> GraspBatch:
        """Move all batch tensors to target device."""
        return GraspBatch(
            points=self.points.to(device=device, non_blocking=non_blocking),
            palm_pos=self.palm_pos.to(device=device, non_blocking=non_blocking),
            palm_rot=self.palm_rot.to(device=device, non_blocking=non_blocking),
            joint_angles=self.joint_angles.to(device=device, non_blocking=non_blocking),
            fingertip_positions=self.fingertip_positions.to(device=device, non_blocking=non_blocking),
            success=self.success.to(device=device, non_blocking=non_blocking),
            quality=self.quality.to(device=device, non_blocking=non_blocking),
            kinematics_valid=self.kinematics_valid.to(device=device, non_blocking=non_blocking),
            pose_target_valid=self.pose_target_valid.to(device=device, non_blocking=non_blocking),
            joint_target_valid=self.joint_target_valid.to(device=device, non_blocking=non_blocking),
            fk_target_valid=self.fk_target_valid.to(device=device, non_blocking=non_blocking),
            object_ids=list(self.object_ids),
            robot_names=list(self.robot_names),
        )

    def pin_memory(self) -> GraspBatch:
        """Pin CPU memory for asynchronous GPU transfer if accelerator is available."""
        if not torch.cuda.is_available():
            return self
        return GraspBatch(
            points=self.points.pin_memory(),
            palm_pos=self.palm_pos.pin_memory(),
            palm_rot=self.palm_rot.pin_memory(),
            joint_angles=self.joint_angles.pin_memory(),
            fingertip_positions=self.fingertip_positions.pin_memory(),
            success=self.success.pin_memory(),
            quality=self.quality.pin_memory(),
            kinematics_valid=self.kinematics_valid.pin_memory(),
            pose_target_valid=self.pose_target_valid.pin_memory(),
            joint_target_valid=self.joint_target_valid.pin_memory(),
            fk_target_valid=self.fk_target_valid.pin_memory(),
            object_ids=list(self.object_ids),
            robot_names=list(self.robot_names),
        )

    @classmethod
    def collate(cls, items: Sequence[dict[str, Any]]) -> GraspBatch:
        """Collate individual sample dictionaries into a batched container."""
        points = torch.stack([item["points"] for item in items], dim=0)
        palm_pos = torch.stack([item["palm_pos"] for item in items], dim=0)
        palm_rot = torch.stack([item["palm_rot"] for item in items], dim=0)
        joint_angles = torch.stack([item["joint_angles"] for item in items], dim=0)
        fingertip_positions = torch.stack([item["fingertip_positions"] for item in items], dim=0)
        success = torch.stack([item["success"] for item in items], dim=0)
        quality = torch.stack([item["quality"] for item in items], dim=0)
        kinematics_valid = torch.stack(
            [torch.as_tensor(item["kinematics_valid"], dtype=torch.bool) for item in items], dim=0
        )
        pose_target_valid = torch.stack(
            [torch.as_tensor(item["pose_target_valid"], dtype=torch.bool) for item in items], dim=0
        )
        joint_target_valid = torch.stack(
            [torch.as_tensor(item["joint_target_valid"], dtype=torch.bool) for item in items], dim=0
        )
        fk_target_valid = torch.stack(
            [torch.as_tensor(item["fk_target_valid"], dtype=torch.bool) for item in items], dim=0
        )

        object_ids = [str(item["object_id"]) for item in items]
        robot_names = [str(item["robot_name"]) for item in items]

        return cls(
            points=points,
            palm_pos=palm_pos,
            palm_rot=palm_rot,
            joint_angles=joint_angles,
            fingertip_positions=fingertip_positions,
            success=success,
            quality=quality,
            kinematics_valid=kinematics_valid,
            pose_target_valid=pose_target_valid,
            joint_target_valid=joint_target_valid,
            fk_target_valid=fk_target_valid,
            object_ids=object_ids,
            robot_names=robot_names,
        )
