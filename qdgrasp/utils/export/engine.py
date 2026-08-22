# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import types
from pathlib import Path

import torch

from ultralytics.utils import TORCH_VERSION, ThreadingLocked
from ultralytics.utils.torch_utils import TORCH_2_4, TORCH_2_9


def best_onnx_opset(onnx: types.ModuleType, cuda: bool = False, quantize: int | str | None = None) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback."""
    if TORCH_2_4:  # _constants.ONNX_MAX_OPSET first defined in torch 1.13
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # use second-latest version for safety
        if TORCH_2_9:
            opset = min(opset, 20)  # legacy TorchScript exporter caps at opset 20 in torch 2.9+
        if cuda:
            opset -= 2  # fix CUDA ONNXRuntime NMS squeeze op errors
    else:
        version = ".".join(TORCH_VERSION.split(".")[:2])
        opset = {
            "1.8": 12,
            "1.9": 12,
            "1.10": 13,
            "1.11": 14,
            "1.12": 15,
            "1.13": 17,
            "2.0": 17,  # reduced from 18 to fix ONNX errors
            "2.1": 17,  # reduced from 19
            "2.2": 17,  # reduced from 19
            "2.3": 17,  # reduced from 19
            "2.4": 20,
            "2.5": 20,
            "2.6": 20,
            "2.7": 20,
            "2.8": 23,
        }.get(version, 12)
    if quantize == 8:
        opset = min(opset, 20)  # ONNX Runtime static INT8 quantization does not support opset>=21
    return min(opset, onnx.defs.onnx_opset_version())


@ThreadingLocked()
def torch2onnx(
    model: torch.nn.Module,
    im: torch.Tensor | tuple[torch.Tensor, ...],
    output_file: Path | str,
    opset: int = 14,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic: dict | None = None,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor | tuple[torch.Tensor, ...]): Example input tensor(s) for tracing.
        output_file (Path | str): Path to save the exported ONNX file.
        opset (int): ONNX opset version to use for export.
        input_names (list[str] | None): List of input tensor names. Defaults to ``["images"]``.
        output_names (list[str] | None): List of output tensor names. Defaults to ``["output0"]``.
        dynamic (dict | None): Dictionary specifying dynamic axes for inputs and outputs.

    Returns:
        (str): Path to the exported ONNX file.

    Notes:
        Setting `do_constant_folding=True` may cause issues with DNN inference for torch>=1.12.
    """
    if input_names is None:
        input_names = ["images"]
    if output_names is None:
        output_names = ["output0"]
    kwargs = {"dynamo": False} if TORCH_2_4 else {}
    torch.onnx.export(
        model,
        im,
        output_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic,
        **kwargs,
    )
    return str(output_file)
