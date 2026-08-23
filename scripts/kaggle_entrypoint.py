"""Kaggle GPU runner entrypoint for QDGrasp.

This script is designed to run in a Kaggle notebook environment with GPU acceleration (T4/P100).
It verifies GPU runtime, runs the full test suite, verifies Phase 3 gates, and executes GPU training.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
import torch

def setup_kaggle_env() -> dict[str, str | int | bool]:
    """Detect and configure GPU environment."""
    print("==================================================")
    print("      QDGrasp Kaggle GPU Environment Setup       ")
    print("==================================================")

    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_avail else 0
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"

    info = {
        "cuda_available": cuda_avail,
        "device_count": device_count,
        "device_name": device_name,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }

    print(f"CUDA Available : {cuda_avail}")
    print(f"Device Count   : {device_count}")
    print(f"Primary Device : {device_name}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"Python Version : {info['python_version']}")
    print("==================================================\n")
    return info

def run_tests() -> bool:
    """Run full test suite."""
    print("--> Running Full Pytest Test Suite...")
    res = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
    if res.returncode != 0:
        print("Test suite FAILED!")
        return False
    print("Test suite PASSED (100% tests passed)!\n")
    return True

def run_phase3_verification() -> bool:
    """Run Phase 3 data layer and pipeline verification."""
    print("--> Running Phase 3 Verification Suite (check_phase3.py)...")
    res = subprocess.run([sys.executable, "scripts/check_phase3.py"], check=False)
    if res.returncode != 0:
        print("Phase 3 verification FAILED!")
        return False
    print("Phase 3 verification PASSED!\n")
    return True

def run_gpu_training(device: str = "cuda") -> None:
    """Run GPU model training with QDGrasp API."""
    print(f"--> Running Model Training on {device}...")
    from qdgrasp.api import QDGrasp

    grasper = QDGrasp("qdgrasp-dummy-n.yaml", robot="leap_hand.yaml", seed=42)
    res = grasper.train(
        "configs/data/dgn_open_tiny.yaml",
        device=device,
        max_steps=20,
        batch_size=8,
        learning_rate=1e-3,
        run_name="kaggle_gpu_train",
        project_dir="runs/kaggle",
    )
    print(f"Training Complete! Final Metrics: {res.metrics}\n")

def main() -> None:
    env_info = setup_kaggle_env()
    device = "cuda" if env_info["cuda_available"] else "cpu"

    if not env_info["cuda_available"]:
        print("WARNING: CUDA is not available. Falling back to CPU mode.\n")

    tests_ok = run_tests()
    if not tests_ok:
        sys.exit(1)

    p3_ok = run_phase3_verification()
    if not p3_ok:
        sys.exit(1)

    run_gpu_training(device=device)
    print("All Kaggle benchmarks and verifications completed successfully!")

if __name__ == "__main__":
    main()
