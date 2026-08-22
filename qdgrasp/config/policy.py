"""Device policy: turn a requested :class:`RunConfig` into effective runtime values.

Every difference between what was requested and what will actually run is
recorded as an :class:`Adjustment` and logged explicitly, so a run bundle can be
audited without re-reading the code that produced it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from ..runtime import require_cuda
from .schema import ConfigError, RunConfig


LOGGER = logging.getLogger("qdgrasp.config")


@dataclass(frozen=True)
class Adjustment:
    """One requested value that the device policy had to change."""

    field: str
    requested: Any
    effective: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "requested": self.requested, "effective": self.effective, "reason": self.reason}


@dataclass(frozen=True)
class EffectiveRuntime:
    """Resolved runtime values plus the audit trail that produced them."""

    requested: dict[str, Any]
    device: torch.device
    accelerator: str
    device_index: int
    precision: str
    amp: bool
    workers: int
    seed: int
    deterministic: bool
    adjustments: tuple[Adjustment, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe requested/effective/adjustment record for the run bundle."""

        return {
            "requested": dict(self.requested),
            "effective": {
                "device": str(self.device),
                "accelerator": self.accelerator,
                "device_index": self.device_index,
                "precision": self.precision,
                "amp": self.amp,
                "workers": self.workers,
                "seed": self.seed,
                "deterministic": self.deterministic,
            },
            "adjustments": [item.to_dict() for item in self.adjustments],
        }


def _parse_device(device: str) -> tuple[str, int]:
    if device == "cpu":
        return "cpu", 0
    if device == "cuda":
        return "cuda", 0
    index = device.split(":", 1)[1]
    if not index.isdigit():
        raise ConfigError(f"invalid CUDA device index in '{device}'")
    return "cuda", int(index)


def resolve_runtime(config: RunConfig, *, expected_cuda_runtime: str = "12.8") -> EffectiveRuntime:
    """Apply the QDGrasp device policy without ever falling back to CPU.

    A requested CUDA device is validated against physical hardware through
    :func:`qdgrasp.runtime.require_cuda`; a missing GPU raises instead of
    silently downgrading the run.
    """

    accelerator, index = _parse_device(config.device)
    adjustments: list[Adjustment] = []

    if accelerator == "cuda":
        require_cuda(expected_runtime=expected_cuda_runtime)
        available = torch.cuda.device_count()
        if index >= available:
            raise ConfigError(f"requested {config.device} but only {available} CUDA device(s) are present")
        device = torch.device(f"cuda:{index}")
        amp = config.amp
    else:
        device = torch.device("cpu")
        amp = False
        if config.amp:
            adjustments.append(
                Adjustment("amp", True, False, "AMP is a CUDA-only path; CPU runs stay FP32 for parity reference")
            )

    precision = "16-mixed" if amp else "32-true"

    workers = config.workers
    max_workers = os.cpu_count() or 1
    if workers > max_workers:
        adjustments.append(Adjustment("workers", config.workers, max_workers, f"host exposes {max_workers} CPU(s)"))
        workers = max_workers

    for adjustment in adjustments:
        LOGGER.warning(
            "config adjusted: %s requested=%r effective=%r (%s)",
            adjustment.field,
            adjustment.requested,
            adjustment.effective,
            adjustment.reason,
        )

    return EffectiveRuntime(
        requested=config.to_document(),
        device=device,
        accelerator=accelerator,
        device_index=index,
        precision=precision,
        amp=amp,
        workers=workers,
        seed=config.seed,
        deterministic=config.deterministic,
        adjustments=tuple(adjustments),
    )
