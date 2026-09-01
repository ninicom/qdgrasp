"""Protocols the runner, exporter and façade depend on.

Any model that satisfies :class:`GraspModel` can be trained, validated,
exported and served by QDGrasp without subclassing a framework base class.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from .results import GraspResults


@runtime_checkable
class GraspModel(Protocol):
    """Minimal contract between a grasp model and the QDGrasp engine."""

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the scalar loss for one batch."""

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return scalar metrics for one batch."""

    def predict_results(
        self,
        points: torch.Tensor,
        *,
        training_robot_hash: str | None = None,
        runtime_robot_hash: str | None = None,
    ) -> GraspResults:
        """Convert one point cloud ``[N, 3]`` into ranked grasps."""

    def preprocess_schema(self) -> dict[str, Any]:
        """Describe the input contract stored in the public bundle."""

    def example_inputs(self) -> tuple[torch.Tensor, ...]:
        """Representative inputs used for tracing and export round-trips."""


@runtime_checkable
class GraspDataset(Protocol):
    """Dataset contract: map-style access to batched tensor dictionaries."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: ...
