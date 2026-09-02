"""Deterministic seeding and RNG capture for reproducible runs and exact resume."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class RngSnapshot:
    """Serializable RNG state for Python, NumPy, CPU torch and CUDA torch."""

    python: str
    numpy: str
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Flatten to a ``torch.save``/``weights_only`` friendly mapping."""

        payload: dict[str, Any] = {
            "rng_python": self.python,
            "rng_numpy": self.numpy,
            "rng_torch_cpu": self.torch_cpu,
            "rng_torch_cuda_count": len(self.torch_cuda),
        }
        for index, state in enumerate(self.torch_cuda):
            payload[f"rng_torch_cuda_{index}"] = state
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RngSnapshot:
        """Rebuild a snapshot written by :meth:`to_payload`."""

        count = int(payload.get("rng_torch_cuda_count", 0))
        return cls(
            python=payload["rng_python"],
            numpy=payload["rng_numpy"],
            torch_cpu=payload["rng_torch_cpu"],
            torch_cuda=tuple(payload[f"rng_torch_cuda_{index}"] for index in range(count)),
        )


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and torch, and optionally enforce deterministic kernels."""

    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must fit in 32 bits, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return seed


@contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    """Run a block from a fixed RNG state without advancing its caller's streams.

    Validation code cannot always accept an explicit :class:`torch.Generator`.
    Temporarily seeding every process-local stream therefore gives it a stable
    draw while the snapshot/restore boundary keeps validation observational: a
    different validation cadence cannot change the following training step.

    Unlike :func:`seed_everything`, this helper deliberately leaves process-wide
    deterministic-kernel policy and ``PYTHONHASHSEED`` untouched.
    """

    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must fit in 32 bits, got {seed}")
    snapshot = capture_rng()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(snapshot)


def capture_rng() -> RngSnapshot:
    """Snapshot every RNG stream QDGrasp advances during a run."""

    python_state = random.getstate()
    numpy_state = cast(tuple[str, NDArray[np.uint32], int, int, float], np.random.get_state())
    return RngSnapshot(
        python=json.dumps([python_state[0], list(python_state[1]), python_state[2]]),
        numpy=json.dumps([numpy_state[0], [int(value) for value in numpy_state[1]], *numpy_state[2:]]),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=tuple(torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())),
    )


def restore_rng(snapshot: RngSnapshot) -> None:
    """Restore every RNG stream captured by :func:`capture_rng`."""

    python_state = json.loads(snapshot.python)
    random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
    numpy_state = json.loads(snapshot.numpy)
    np.random.set_state(
        (
            numpy_state[0],
            np.array(numpy_state[1], dtype=np.uint32),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        )
    )
    torch.set_rng_state(snapshot.torch_cpu.to(torch.uint8))
    for index, state in enumerate(snapshot.torch_cuda):
        if index < torch.cuda.device_count():
            torch.cuda.set_rng_state(state.to(torch.uint8), index)
