"""Reference dummy grasp model used to exercise the Phase 1 lifecycle.

The architecture is deliberately trivial -- a pooled point MLP with the four
QDGrasp output heads.  Its job is to make ``train``/``val``/``predict``/
``export`` runnable and testable before the real model lands in Phase 4.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..api.results import GraspResults
from ..config.registry import register_model
from ..config.schema import ConfigError, ModelConfig, RobotConfig
from ..geometry import rot6d_to_matrix


class DummyGraspModel(nn.Module):
    """Pooled point encoder with palm, rotation, named-joint and score heads."""

    def __init__(self, model_config: ModelConfig, robot_config: RobotConfig) -> None:
        super().__init__()
        params = dict(model_config.params)
        unknown = sorted(set(params) - {"hidden", "grasps"})
        if unknown:
            raise ConfigError(f"model '{model_config.name}': unknown params {unknown}")
        self.model_config = model_config
        self.robot_config = robot_config
        self.hidden = int(params.get("hidden", 64))
        self.grasps = int(params.get("grasps", 4))
        if self.hidden < 1 or self.grasps < 1:
            raise ConfigError(f"model '{model_config.name}': hidden and grasps must be >= 1")
        self.joint_names = robot_config.joints
        joints = len(self.joint_names)

        self.encoder = nn.Sequential(
            nn.Linear(3, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )
        self.translation_head = nn.Linear(self.hidden, self.grasps * 3)
        self.rotation_head = nn.Linear(self.hidden, self.grasps * 6)
        self.joint_head = nn.Linear(self.hidden, self.grasps * joints)
        self.score_head = nn.Linear(self.hidden, self.grasps)
        self.register_buffer("joint_lower", torch.tensor(robot_config.lower_limits, dtype=torch.float32))
        self.register_buffer("joint_upper", torch.tensor(robot_config.upper_limits, dtype=torch.float32))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map ``[B, N, 3]`` points onto ``(translation, rotation, joints, score)``.

        Returns a tuple rather than a mapping so the same callable can be traced
        for TorchScript and ONNX without a wrapper.
        """

        if not torch.jit.is_tracing() and (points.dim() != 3 or points.shape[-1] != 3):
            raise ValueError(f"expected points of shape [B, N, 3], got {tuple(points.shape)}")
        batch = points.shape[0]
        features = self.encoder(points).mean(dim=1)
        centroid = points.mean(dim=1, keepdim=True)
        translation = self.translation_head(features).view(batch, self.grasps, 3) + centroid
        rotation = rot6d_to_matrix(self.rotation_head(features).view(batch, self.grasps, 6))
        span = self.joint_upper - self.joint_lower
        joints = self.joint_lower + span * torch.sigmoid(self.joint_head(features).view(batch, self.grasps, -1))
        score = torch.sigmoid(self.score_head(features))
        return translation, rotation, joints, score

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean-squared error against the dummy translation/joint targets."""

        translation, _rotation, joints, score = self(batch["points"])
        target_translation = batch["target_translation"].unsqueeze(1)
        target_joints = batch["target_joints"].unsqueeze(1)
        translation_loss = torch.nn.functional.mse_loss(translation, target_translation.expand_as(translation))
        joint_loss = torch.nn.functional.mse_loss(joints, target_joints.expand_as(joints))
        score_loss = torch.nn.functional.mse_loss(score, torch.ones_like(score))
        return translation_loss + joint_loss + 0.1 * score_loss

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Report loss plus translation/joint errors for the best-scoring grasp."""

        translation, _rotation, joints, score = self(batch["points"])
        best = score.argmax(dim=1)
        index = torch.arange(translation.shape[0], device=translation.device)
        translation_error = (translation[index, best] - batch["target_translation"]).norm(dim=-1).mean()
        joint_error = (joints[index, best] - batch["target_joints"]).abs().mean()
        return {
            "loss": self.training_step(batch),
            "translation_error": translation_error,
            "joint_error": joint_error,
        }

    @torch.no_grad()
    def predict_results(self, points: torch.Tensor) -> GraspResults:
        """Rank the grasps produced for a single ``[N, 3]`` point cloud."""

        if points.dim() == 2:
            points = points.unsqueeze(0)
        if points.shape[0] != 1:
            raise ValueError("predict_results handles one point cloud at a time")
        translation, rotation, joints, score = self(points)
        order = torch.argsort(score[0], descending=True)
        return GraspResults(
            translation=translation[0, order],
            rotation=rotation[0, order],
            joint_names=self.joint_names,
            joint_values=joints[0, order],
            score=score[0, order],
            seed_points=points[0].mean(dim=0, keepdim=True).expand(len(order), 3).contiguous(),
            frame=self.robot_config.frame,
            model_hash=self.model_config.content_hash(),
            robot_hash=self.robot_config.content_hash(),
        )

    def preprocess_schema(self) -> dict[str, Any]:
        """Input contract recorded in the public bundle."""

        return {
            "input": "points",
            "layout": "[B, N, 3]",
            "dtype": "float32",
            "units": "meters",
            "frame": self.robot_config.frame,
            "normalization": "none",
        }

    def example_inputs(self) -> tuple[torch.Tensor, ...]:
        """A deterministic traced-export sample."""

        generator = torch.Generator().manual_seed(0)
        return (torch.randn(1, 64, 3, generator=generator),)


@register_model("dummy_grasp")
def build_dummy_grasp(model_config: ModelConfig, robot_config: RobotConfig) -> DummyGraspModel:
    """Registry entry point for ``type: dummy_grasp``."""

    return DummyGraspModel(model_config, robot_config)
