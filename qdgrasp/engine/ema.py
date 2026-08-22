"""Exponential moving average of model parameters."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ModelEma:
    """Track a shadow copy of the float parameters/buffers of ``model``.

    Args:
        model: Module whose float tensors are averaged.
        decay: Averaging factor in ``[0.0, 1.0)``; ``0.0`` disables averaging.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0.0, 1.0)")
        self.decay = decay
        self.enabled = decay > 0.0
        self.shadow: dict[str, torch.Tensor] = {}
        if self.enabled:
            self.shadow = {key: value.detach().clone() for key, value in self._float_state(model).items()}

    @staticmethod
    def _float_state(model: nn.Module) -> dict[str, torch.Tensor]:
        return {key: value for key, value in model.state_dict().items() if value.is_floating_point()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the current model tensors into the shadow copy."""

        if not self.enabled:
            return
        for key, value in self._float_state(model).items():
            shadow = self.shadow[key]
            shadow.mul_(self.decay).add_(value.detach().to(shadow.device), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        """Serializable EMA state."""

        return {"decay": self.decay, "shadow": {key: value.cpu() for key, value in self.shadow.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EMA state written by :meth:`state_dict`."""

        self.decay = float(state["decay"])
        self.enabled = self.decay > 0.0
        self.shadow = {key: value.clone() for key, value in state["shadow"].items()}
