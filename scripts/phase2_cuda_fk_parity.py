#!/usr/bin/env python3
"""Fail-closed CUDA FK parity verification for Phase 2.

This script must be executed on physical NVIDIA CUDA hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from qdgrasp.config.loader import load_robot_config
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.runtime import environment_info, require_cuda


def run_cuda_fk_parity(device_str: str = "cuda:0", out_path: Path | None = None) -> int:
    if not device_str.startswith("cuda"):
        raise SystemExit(f"--device must be a CUDA device, got {device_str!r}")
    require_cuda()
    device = torch.device(device_str)

    torch.manual_seed(42)
    B = 16

    results: dict[str, dict[str, float]] = {}
    profile_hashes: dict[str, str] = {}
    max_deviation_overall = 0.0

    hand_presets = ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml")

    for preset_name in hand_presets:
        cfg = load_robot_config(preset_name)
        profile_hashes[preset_name] = cfg.content_hash()
        spec = RobotSpec.from_config(cfg, sample_anchors=False)

        J = len(spec.actuated_joint_names)
        palm_pos_cpu = torch.randn(B, 3, dtype=torch.float32)
        palm_rot_cpu = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(B, 3, 3).clone()
        joints_cpu = torch.randn(B, J, dtype=torch.float32) * 0.2

        palm_pos_cuda = palm_pos_cpu.to(device)
        palm_rot_cuda = palm_rot_cpu.to(device)
        joints_cuda = joints_cpu.to(device)

        # CPU FP32 reference
        t_cpu = spec.forward_kinematics(palm_pos_cpu, palm_rot_cpu, joints_cpu)
        tips_cpu = spec.fingertip_positions(palm_pos_cpu, palm_rot_cpu, joints_cpu)

        # CUDA FP32 run
        t_cuda = spec.forward_kinematics(palm_pos_cuda, palm_rot_cuda, joints_cuda)
        tips_cuda = spec.fingertip_positions(palm_pos_cuda, palm_rot_cuda, joints_cuda)

        link_deviations: dict[str, float] = {}
        for link_name, mat_cpu in t_cpu.items():
            mat_cuda = t_cuda[link_name].cpu()
            dev = float((mat_cpu - mat_cuda).abs().max().item())
            link_deviations[link_name] = dev
            max_deviation_overall = max(max_deviation_overall, dev)

        tip_dev = float((tips_cpu - tips_cuda.cpu()).abs().max().item())
        max_deviation_overall = max(max_deviation_overall, tip_dev)

        results[preset_name] = {
            "max_link_deviation": max(link_deviations.values()) if link_deviations else 0.0,
            "fingertip_deviation": tip_dev,
        }

    evidence: dict[str, object] = {
        "schema": "qdgrasp/evidence/phase2-cuda-fk/v1",
        "environment": environment_info().to_dict(),
        "device": device_str,
        "profile_hashes": profile_hashes,
        "results": results,
        "max_deviation_overall": max_deviation_overall,
        "parity_threshold": 1e-4,
        "pass": max_deviation_overall <= 1e-4,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote CUDA FK parity evidence to {out_path}")

    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["pass"]:
        print(f"CUDA FK parity FAIL: max deviation {max_deviation_overall} > 1e-4", file=sys.stderr)
        return 1

    print("Phase 2 CUDA FK parity: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 CUDA FK parity verification.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device identifier.")
    parser.add_argument("--out", type=Path, default=None, help="Path to write evidence JSON.")
    args = parser.parse_args()

    return run_cuda_fk_parity(device_str=args.device, out_path=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
