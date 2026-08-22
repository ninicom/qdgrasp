"""ONNX export; the ``onnx`` extra must be installed."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


OUTPUT_NAMES = ("translation", "rotation", "joints", "score")


def export_onnx(model: nn.Module, path: str | Path, example: tuple[torch.Tensor, ...], *, opset: int = 18) -> Path:
    """Export ``model`` to ONNX with a dynamic point-count axis."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        example,
        str(target),
        input_names=["points"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={"points": {0: "batch", 1: "points"}},
        opset_version=opset,
        dynamo=False,
    )
    return target


def verify_onnx(model: nn.Module, path: str | Path, example: tuple[torch.Tensor, ...], *, atol: float = 1e-4) -> dict[str, float]:
    """Compare eager and ONNX Runtime CPU outputs; raise when they diverge."""

    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("ONNX verification needs the 'export' extra: pip install qdgrasp[export]") from exc

    model.eval()
    with torch.no_grad():
        expected = model(*example)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"points": example[0].detach().cpu().numpy()})
    deviations: dict[str, float] = {}
    for name, left, right in zip(OUTPUT_NAMES, expected, outputs):
        deviation = float((left.detach().cpu() - torch.from_numpy(right)).abs().max())
        deviations[name] = deviation
        if deviation > atol:
            raise ValueError(f"ONNX round-trip deviation for {name} is {deviation:.3e} (atol {atol:.1e})")
    return deviations
