"""TorchScript export and round-trip verification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def export_torchscript(model: nn.Module, path: str | Path, example: tuple[torch.Tensor, ...]) -> Path:
    """Trace ``model`` on ``example`` and save the TorchScript module."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)
    torch.jit.save(traced, str(target))
    return target


def verify_torchscript(
    model: nn.Module, path: str | Path, example: tuple[torch.Tensor, ...], *, atol: float = 1e-5
) -> dict[str, float]:
    """Compare eager and TorchScript outputs; raise when they diverge."""

    model.eval()
    loaded = torch.jit.load(str(path))
    with torch.no_grad():
        expected = model(*example)
        actual = loaded(*example)
    if len(expected) != len(actual):
        raise ValueError(f"TorchScript round-trip changed the output count: {len(expected)} vs {len(actual)}")
    deviations: dict[str, float] = {}
    names = ("translation", "rotation", "joints", "score")
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left.shape != right.shape:
            raise ValueError(f"TorchScript round-trip changed shape of output {index}: {left.shape} vs {right.shape}")
        deviation = float((left - right).abs().max())
        deviations[names[index] if index < len(names) else f"output_{index}"] = deviation
        if deviation > atol:
            raise ValueError(f"TorchScript round-trip deviation {deviation:.3e} exceeds atol {atol:.1e}")
    return deviations
