"""Runtime inspection and fail-closed device contracts."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class EnvironmentInfo:
    """Serializable runtime information used by install and CUDA gates."""

    python: str
    torch: str
    cuda_build: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None

    def to_dict(self) -> dict[str, str | bool | int | None]:
        """Return a JSON-safe representation."""

        return asdict(self)


def environment_info() -> EnvironmentInfo:
    """Inspect the active Python/PyTorch runtime without changing devices."""

    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    return EnvironmentInfo(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_build=torch.version.cuda,
        cuda_available=available,
        cuda_device_count=count,
        cuda_device_name=torch.cuda.get_device_name(0) if count else None,
    )


def require_cuda(*, expected_runtime: str = "12.8") -> torch.device:
    """Return CUDA device 0 or fail without falling back to CPU."""

    info = environment_info()
    if not info.cuda_available or info.cuda_device_count < 1:
        raise RuntimeError("QDGrasp CUDA execution requires a physical NVIDIA GPU; CPU fallback is forbidden.")
    if info.cuda_build != expected_runtime:
        raise RuntimeError(f"QDGrasp requires CUDA runtime {expected_runtime}, got {info.cuda_build!r}.")
    return torch.device("cuda:0")
