"""MJWarp CUDA backend tests (P3.4-05).

Everything here runs on a CPU host, so these test the contract and the
fail-closed behaviour, not the kernels. The kernels are exercised by the Kaggle
harness on a real device, and this file is explicit that it cannot substitute
for that.
"""

from __future__ import annotations

import inspect

import pytest

from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest
from qdgrasp.sim.batched.contracts import (
    BackendUnavailableError,
    BatchedContactBackend,
)
from qdgrasp.sim.batched.mjwarp_cuda import MjWarpCudaBackend, warp_is_available
from qdgrasp.sim.batched.mujoco_cpu import MuJoCoCpuBackend

pytestmark = pytest.mark.skipif(
    warp_is_available(),
    reason="these assert the CPU-host refusal path; run the Kaggle harness on a GPU",
)


def test_the_backend_refuses_to_exist_without_warp():
    # Not a warning, not a fallback: construction fails.
    with pytest.raises(BackendUnavailableError, match="mujoco_warp"):
        MjWarpCudaBackend("<mujoco/>")


def test_the_backend_refuses_a_non_cuda_device():
    with pytest.raises(BackendUnavailableError, match="CUDA-only"):
        MjWarpCudaBackend("<mujoco/>", device="cpu")


def test_there_is_no_cpu_code_path_through_this_class():
    # A CPU number reported under a CUDA schema is fabricated evidence, so the
    # class must have no branch that quietly degrades.
    source = inspect.getsource(MjWarpCudaBackend)
    assert "fallback" not in source.lower() or "not fall back" in source.lower()
    assert 'device="cpu"' not in source
    assert MjWarpCudaBackend.backend_id == "mjwarp_cuda"


def test_it_presents_the_same_surface_as_the_cpu_oracle():
    required = [
        name
        for name in dir(BatchedContactBackend)
        if not name.startswith("_")
    ]
    for name in required:
        assert hasattr(MjWarpCudaBackend, name), f"CUDA backend is missing {name}"
        assert hasattr(MuJoCoCpuBackend, name), f"CPU oracle is missing {name}"


def test_the_two_backends_agree_on_method_signatures():
    # A strategy must not be able to tell them apart, so the call shapes match.
    for name in ("compile", "reset", "step", "observe", "rollout", "export_finalists"):
        cpu = inspect.signature(getattr(MuJoCoCpuBackend, name))
        gpu = inspect.signature(getattr(MjWarpCudaBackend, name))
        assert list(cpu.parameters) == list(gpu.parameters), name


def test_exported_finalists_are_rewritten_as_cpu_requests():
    # The rewrite is what makes "GPU ranks, CPU admits" structural rather than a
    # convention someone has to remember.
    source = inspect.getsource(MjWarpCudaBackend.export_finalists)
    assert 'backend_request="cpu"' in source


def test_a_request_carries_its_backend_choice():
    request = DynamicGraspRequest(
        scene_state_ref="scene:micro#0",
        observation_ref="obs:micro/cam_top",
        target_object_id="target",
        robot_profile="leap_hand",
        strategy_id="batched_cem",
        safety_budget_id="micro-conservative-v1",
        horizon=10,
        control_dt=0.01,
        seed=0,
        backend_request="cuda",
    )
    assert request.backend_request == "cuda"
