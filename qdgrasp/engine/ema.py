"""Exponential moving average of model parameters."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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
        self.updates = 0
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
        self.updates += 1

    def to(self, device: torch.device | str) -> ModelEma:
        """Move the shadow copy onto ``device``.

        Needed after a resume: the shadow is restored from a CPU artifact while
        the model itself may live on an accelerator.
        """

        target = torch.device(device)
        self.shadow = {key: value.to(target) for key, value in self.shadow.items()}
        return self

    def state_dict(self) -> dict[str, Any]:
        """Serializable EMA state."""

        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {key: value.cpu() for key, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EMA state written by :meth:`state_dict`."""

        self.decay = float(state["decay"])
        self.enabled = self.decay > 0.0
        self.updates = int(state["updates"])
        self.shadow = {key: value.clone() for key, value in state["shadow"].items()}

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily expose the EMA shadow through ``model``.

        The training model remains the authoritative resume state.  Validation
        and publication borrow its storage only for the duration of this context
        and restore every live tensor even when the consumer raises.
        """

        if not self.enabled:
            yield
            return

        live = self._float_state(model)
        missing = sorted(set(live) - set(self.shadow))
        unexpected = sorted(set(self.shadow) - set(live))
        mismatched = sorted(key for key in set(live) & set(self.shadow) if live[key].shape != self.shadow[key].shape)
        if missing or unexpected or mismatched:
            raise ValueError(
                f"EMA/model state mismatch (missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched})"
            )

        backup = {key: value.detach().clone() for key, value in live.items()}
        try:
            for key, value in live.items():
                value.copy_(self.shadow[key].to(device=value.device, dtype=value.dtype))
            yield
        finally:
            for key, value in self._float_state(model).items():
                value.copy_(backup[key].to(device=value.device, dtype=value.dtype))
