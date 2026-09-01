#!/usr/bin/env python3
"""Overfit QDGrasp-Flow on a handful of samples (P4-10).

This measures one thing: whether the architecture can learn at all.  A model
that cannot drive the loss down on eight samples it sees repeatedly has a wiring
problem -- a dead gradient path, a detached tensor, a head that cannot reach its
target -- and no amount of data will fix it.

It is emphatically **not** a quality result.  Overfitting eight samples says
nothing about grasping, and ``ROADMAP-P4-001`` §7 forbids citing it as if it did.

The report includes palm and joint error alongside the loss, because a loss that
falls while the pose error does not is the signature of a model satisfying an
auxiliary term instead of the task.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdgrasp import __version__
from qdgrasp.models.flow import GraspFlowModel, model_semantics
from qdgrasp.models.losses import LossWeights, forward_and_loss, gradient_coverage
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.runtime import environment_info


def build_fixture(robot: RobotSpec, samples: int, points: int, seed: int, device: torch.device):
    """A small, fixed batch: random object clouds with reachable grasp labels.

    The labels are produced by running the profile's own forward kinematics on
    sampled palm poses and joint angles, so the target is guaranteed reachable.
    A target the hand cannot reach would make a failure to converge ambiguous
    between "the model cannot learn" and "there was nothing to learn".
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    joint_names = robot.actuated_joint_names
    limits = [robot.config.joint_limits[name] for name in joint_names]
    lower = torch.tensor([value[0] for value in limits], dtype=torch.float32)
    upper = torch.tensor([value[1] for value in limits], dtype=torch.float32)

    cloud = torch.randn(samples, points, 3, generator=generator) * 0.04
    palm_pos = torch.randn(samples, 3, generator=generator) * 0.05
    palm_rot = torch.tensor(Rotation.random(samples, random_state=seed).as_matrix(), dtype=torch.float32)
    fraction = torch.rand(samples, len(joint_names), generator=generator)
    joints = lower + fraction * (upper - lower)
    fingertips = robot.fingertip_positions(palm_pos, palm_rot, joints)
    success = (torch.rand(samples, generator=generator) > 0.5).float()

    return {
        "points": cloud.to(device),
        "palm_pos": palm_pos.to(device),
        "palm_rot": palm_rot.to(device),
        "joint_angles": joints.to(device),
        "fingertip_positions": fingertips.to(device),
        "success": success.to(device),
    }


def pose_errors(prediction, batch) -> dict[str, float]:
    from qdgrasp.models.losses import geodesic_rotation_error

    translation = torch.linalg.norm(prediction.palm_translation - batch["palm_pos"], dim=-1).mean()
    rotation = geodesic_rotation_error(prediction.palm_rotation, batch["palm_rot"]).mean()
    joint = (prediction.joint_angles - batch["joint_angles"]).abs().mean()
    fingertip = torch.linalg.norm(prediction.fingertips - batch["fingertip_positions"], dim=-1).mean()
    return {
        "palm_translation_m": float(translation.detach()),
        "palm_rotation_rad": float(rotation.detach()),
        "joint_abs_rad": float(joint.detach()),
        "fingertip_m": float(fingertip.detach()),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    ).stdout.strip()


def _git_commit() -> str:
    return _git("rev-parse", "HEAD") or "unknown"


def _worktree_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="leap_hand.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixed-noise",
        action="store_true",
        default=True,
        help="hold the flow's starting draw fixed, so the overfit asks a deterministic question",
    )
    parser.add_argument("--stochastic-noise", dest="fixed_noise", action="store_false")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--palm-threshold-m", type=float, default=0.06)
    parser.add_argument("--rotation-threshold-rad", type=float, default=0.05)
    parser.add_argument("--joint-threshold-rad", type=float, default=0.10)
    parser.add_argument("--fingertip-threshold-m", type=float, default=0.06)
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"{args.device} requested and CUDA is not available. A CPU run must not be reported as a "
            "CUDA one (ADR-0006)."
        )
        return 1

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    robot = RobotSpec.from_config(args.robot, sample_anchors=False)
    graph = robot.to_hand_graph(device=device)
    model = GraspFlowModel().to(device)
    batch = build_fixture(robot, args.samples, args.points, args.seed, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    sample_noise = None
    if args.fixed_noise:
        sample_noise = torch.randn(args.samples, model.flow_config.state_dimension, device=device, generator=generator)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(args.steps):
        optimizer.zero_grad()
        prediction, losses = forward_and_loss(
            model, robot, graph, weights=LossWeights(), generator=generator, sample_noise=sample_noise, **batch
        )
        total = losses.total
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(args.steps // 12, 1) == 0 or step == args.steps - 1:
            row = {"step": step, **losses.to_document(), **pose_errors(prediction, batch)}
            history.append(row)
            print(
                f"step {step:5d} loss={row['total']:8.4f} "
                f"palm={row['palm_translation_m']:.4f} m rot={row['palm_rotation_rad']:.4f} rad "
                f"joint={row['joint_abs_rad']:.4f} rad tip={row['fingertip_m']:.4f} m"
            )

    coverage = gradient_coverage(model)
    elapsed = time.perf_counter() - started
    first, last = history[0], history[-1]
    # The verdict is read off the *pose* errors, not the total.  The flow term
    # has an irreducible floor -- given an interpolated state and a time, the
    # velocity target `target - noise` is stochastic, so its conditional-mean
    # predictor keeps a non-zero MSE forever.  Measured here it settles near 1.0
    # while every pose term falls by one to two orders of magnitude, and a
    # criterion on the total would call that healthy run a wiring bug.
    thresholds = {
        "palm_translation_m": args.palm_threshold_m,
        "palm_rotation_rad": args.rotation_threshold_rad,
        "joint_abs_rad": args.joint_threshold_rad,
        "fingertip_m": args.fingertip_threshold_m,
    }
    met = {name: last[name] <= limit for name, limit in thresholds.items()}
    converged = all(met.values()) and last["total"] < first["total"]

    report = {
        # v2: the run records what produced it.  ``PLAN.md`` §9.1 superseded the
        # v1 evidence when the joint parameterization and the quality objective
        # changed, and evidence that cannot say which semantics it was measured
        # under is exactly how that happens quietly.
        "schema": "qdgrasp/phase4-overfit/v2",
        "robot": args.robot,
        "identity": {
            "qdgrasp_version": __version__,
            "git_commit": _git_commit(),
            "worktree_dirty": _worktree_dirty(),
            "robot_config_hash": robot.config.content_hash(),
            "model_semantics": model_semantics(),
            "environment": environment_info().to_dict(),
        },
        "device": str(device),
        "cuda": device.type == "cuda",
        "samples": args.samples,
        "points": args.points,
        "steps": args.steps,
        "seed": args.seed,
        "fixed_noise": bool(args.fixed_noise),
        "elapsed_s": elapsed,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "gradient_coverage": {"covered": sum(coverage.values()), "total": len(coverage)},
        "pose_thresholds": thresholds,
        "pose_thresholds_met": met,
        "first": first,
        "last": last,
        "history": history,
        "converged": bool(converged),
        "note": "Overfitting a fixed batch shows the architecture trains. It is not a grasping result.",
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")

    print(
        f"\nloss {first['total']:.4f} -> {last['total']:.4f} in {elapsed:.1f}s; "
        f"gradient coverage {report['gradient_coverage']['covered']}/{report['gradient_coverage']['total']}"
    )
    if sum(coverage.values()) != len(coverage):
        dead = sorted(name for name, ok in coverage.items() if not ok)
        print(f"parameters without a finite gradient: {dead[:10]}")
        return 1
    for name, limit in thresholds.items():
        print(f"  {name:22s} {last[name]:.4f} <= {limit:.4f}  {'ok' if met[name] else 'MISSED'}")
    if not converged:
        print("the architecture did not overfit the fixture; that is a wiring problem, not a data problem")
        return 1
    print("architecture trains: every pose error is under its pinned threshold, with full gradient coverage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
