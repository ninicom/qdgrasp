from __future__ import annotations

from unittest import mock

import pytest
import torch

from qdgrasp.runtime import environment_info, require_cuda


def test_environment_info_is_serializable() -> None:
    info = environment_info().to_dict()
    assert info["python"]
    assert info["torch"] == torch.__version__
    assert isinstance(info["cuda_available"], bool)


def test_require_cuda_fails_without_physical_device() -> None:
    with (
        mock.patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="CPU fallback is forbidden"),
    ):
        require_cuda()


def test_require_cuda_rejects_wrong_runtime() -> None:
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.device_count", return_value=1),
        mock.patch("torch.cuda.get_device_name", return_value="Mock GPU"),
        mock.patch.object(torch.version, "cuda", "12.7"),pytest.raises(RuntimeError, match="requires CUDA runtime 12.8")
    ):
        require_cuda()
