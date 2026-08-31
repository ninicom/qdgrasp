#!/usr/bin/env python3
"""CPU gate for Phase 4 (``ROADMAP-P4-001`` §6).

The checker walks the work breakdown in §4 and reports each package as
``delivered``, ``open`` or ``blocked``.  Like the P3.5 gate it is written to
*fail* while packages remain open: P4 closes on a CUDA run and an independent
review, and a gate that returned zero without them would be describing a phase
that is not finished.

``--profile micro`` builds the model and runs a real forward/backward for both
active hands.  ``--profile contract`` checks only the static surface, for use
where torch cannot run.

Nothing here is a quality measurement.  §7 forbids citing any P4 number as a
grasping result, and this script deliberately reports structure -- does it
build, does it stay finite, does every parameter get a gradient -- rather than
anything that could be mistaken for one.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STATUS_DELIVERED = "delivered"
STATUS_OPEN = "open"
STATUS_BLOCKED = "blocked"

#: The active corpus, from the one list that governs it (ADR-0008).
ACTIVE_PROFILES = ("leap_hand.yaml", "wonik_allegro.yaml")


@dataclasses.dataclass
class PackageResult:
    package: str
    title: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == STATUS_DELIVERED


def _surface(package: str, title: str, module: str, names: tuple[str, ...]) -> PackageResult:
    try:
        imported = importlib.import_module(module)
    except Exception as error:  # noqa: BLE001 - an import failure is the finding, not a crash
        return PackageResult(package, title, STATUS_OPEN, f"{module} does not import: {error}")
    missing = [name for name in names if not hasattr(imported, name)]
    if missing:
        return PackageResult(package, title, STATUS_OPEN, f"{module} is missing {missing}")
    return PackageResult(package, title, STATUS_DELIVERED, f"{module} exposes {len(names)} declared names")


def _import_checks() -> list[PackageResult]:
    """Every P4 module must import and expose its public surface."""

    surfaces = {
        "P4-01": (
            "point tokenizer",
            "qdgrasp.models.tokenizer",
            ("TokenizerConfig", "tokenize_points", "pack_grid_coordinates", "scatter_tokens_to_points"),
        ),
        "P4-02": (
            "serialized point encoder",
            "qdgrasp.models.encoder",
            ("EncoderConfig", "PointEncoder", "WindowedBlock", "masked_mean"),
        ),
        "P4-03": (
            "HandGraph encoder",
            "qdgrasp.models.hand_graph",
            ("HandGraphEncoderConfig", "HandGraphEncoder", "MessagePassingLayer", "symmetrize"),
        ),
        "P4-04/05/07": (
            "conditioning, flow head and quality head",
            "qdgrasp.models.flow",
            ("FlowConfig", "GraspFlowModel", "CrossAttentionBlock", "rotation_from_9d", "clamp_to_limits"),
        ),
        "P4-06/09": (
            "FK consistency and loss assembly",
            "qdgrasp.models.losses",
            ("LOSS_TERMS", "LossWeights", "forward_and_loss", "geodesic_rotation_error", "gradient_coverage"),
        ),
        "P4-08": (
            "config schema and registry",
            "qdgrasp.models.config",
            ("FLOW_SCALES", "MODEL_TYPE", "FlowModelSettings", "QDGraspFlow", "build_qdgrasp_flow"),
        ),
    }
    return [_surface(package, title, module, names) for package, (title, module, names) in surfaces.items()]


def _preset_checks(root: Path) -> PackageResult:
    """Every declared scale must ship a preset that builds through the registry."""

    from qdgrasp.config.loader import load_model_config, load_robot_config
    from qdgrasp.config.registry import get_model_builder
    from qdgrasp.models.config import FLOW_SCALES, MODEL_TYPE

    missing = [scale for scale in FLOW_SCALES if not (root / f"qdgrasp/presets/qdgrasp-flow-{scale}.yaml").is_file()]
    if missing:
        return PackageResult("P4-08", "scale presets", STATUS_OPEN, f"no preset for scales {missing}")
    built = []
    robot_config = load_robot_config(ACTIVE_PROFILES[0])
    for scale in sorted(FLOW_SCALES):
        model_config = load_model_config(f"qdgrasp-flow-{scale}.yaml")
        if model_config.type != MODEL_TYPE:
            return PackageResult("P4-08", "scale presets", STATUS_OPEN, f"scale {scale} names type {model_config.type}")
        module = get_model_builder(model_config.type)(model_config, robot_config)
        built.append(f"{scale}={sum(p.numel() for p in module.parameters()) / 1e6:.1f}M")
    return PackageResult("P4-08", "scale presets", STATUS_DELIVERED, f"built {', '.join(built)} through the registry")


def _micro_checks() -> list[PackageResult]:
    """A real forward and backward for both active hands."""

    import torch

    from qdgrasp.models.config import FlowModelSettings, QDGraspFlow
    from qdgrasp.models.losses import gradient_coverage
    from qdgrasp.robot.spec import RobotSpec

    results: list[PackageResult] = []
    for profile in ACTIVE_PROFILES:
        robot = RobotSpec.from_config(profile, sample_anchors=False)
        module = QDGraspFlow(FlowModelSettings(), robot)
        joints = len(robot.actuated_joint_names)
        torch.manual_seed(0)
        batch = {
            "points": torch.randn(2, 256, 3) * 0.05,
            "palm_pos": torch.randn(2, 3) * 0.05,
            "palm_rot": torch.eye(3).expand(2, 3, 3).contiguous(),
            "joint_angles": torch.zeros(2, joints),
            "fingertip_positions": torch.zeros(2, len(robot.fingertip_links), 3),
            "success": torch.tensor([1.0, 0.0]),
        }
        loss = module.training_step(batch)
        loss.backward()
        coverage = gradient_coverage(module)
        covered = sum(coverage.values())
        prediction = module(batch["points"])
        lower = torch.tensor([robot.joint_limits[name][0] for name in robot.actuated_joint_names])
        upper = torch.tensor([robot.joint_limits[name][1] for name in robot.actuated_joint_names])
        rotation = prediction.palm_rotation
        orthonormal = torch.allclose(rotation.transpose(-1, -2) @ rotation, torch.eye(3).expand_as(rotation), atol=1e-4)
        determinant = torch.allclose(torch.linalg.det(rotation), torch.ones(rotation.shape[0]), atol=1e-4)
        in_limits = bool(
            (prediction.joint_angles >= lower - 1e-5).all() and (prediction.joint_angles <= upper + 1e-5).all()
        )
        ok = covered == len(coverage) and prediction.is_finite() and orthonormal and determinant and in_limits
        results.append(
            PackageResult(
                "P4-10",
                f"forward/backward on {profile}",
                STATUS_DELIVERED if ok else STATUS_OPEN,
                (
                    f"loss={float(loss.detach()):.4f}, gradient coverage {covered}/{len(coverage)}, "
                    f"finite={prediction.is_finite()}, SO(3)={orthonormal and determinant}, joints in limits={in_limits}"
                ),
            )
        )
    return results


def _evidence(root: Path) -> list[PackageResult]:
    """What has been measured, per hand, and what is only claimed."""

    directory = root / "evidence/phase4"
    reports = sorted(directory.glob("overfit-*-cpu.json")) if directory.is_dir() else []
    if not reports:
        return [PackageResult("P4-10", "CPU overfit evidence", STATUS_OPEN, f"no overfit record under {directory}")]

    results: list[PackageResult] = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        robot = report.get("robot", path.stem)
        if report.get("cuda"):
            # A CPU gate must never accept a file that claims to be CUDA
            # evidence; one here is either mislabelled or in the wrong place.
            results.append(
                PackageResult(
                    "P4-10",
                    f"CPU overfit evidence ({robot})",
                    STATUS_OPEN,
                    f"{path.name} claims cuda=true but sits in the CPU slot",
                )
            )
            continue
        last = report.get("last", {})
        detail = ", ".join(
            f"{name}={last[name]:.4f}"
            for name in ("palm_translation_m", "palm_rotation_rad", "joint_abs_rad", "fingertip_m")
            if name in last
        )
        results.append(
            PackageResult(
                "P4-10",
                f"CPU overfit evidence ({robot})",
                STATUS_DELIVERED if report.get("converged") else STATUS_OPEN,
                f"{detail} (architecture trains; not a grasping result)"
                if report.get("converged")
                else "recorded run did not converge",
            )
        )

    covered = {json.loads(path.read_text(encoding="utf-8")).get("robot") for path in reports}
    missing = [profile for profile in ACTIVE_PROFILES if profile not in covered]
    if missing:
        results.append(
            PackageResult(
                "P4-10",
                "CPU overfit coverage",
                STATUS_OPEN,
                f"no overfit record for {missing}; the active corpus is {list(ACTIVE_PROFILES)}",
            )
        )
    return results


def _outstanding(root: Path) -> list[PackageResult]:
    """Packages that cannot be closed from this machine, stated as such."""

    harness = root / "scripts/phase4_cuda_gate.py"
    notebook = root / "notebooks/phase4_cuda_gate.ipynb"
    cuda_evidence = (
        sorted((root / "evidence/phase4").glob("cuda-*.json")) if (root / "evidence/phase4").is_dir() else []
    )
    return [
        PackageResult(
            "P4-11a",
            "CUDA gate harness",
            STATUS_DELIVERED if harness.is_file() and notebook.is_file() else STATUS_OPEN,
            (
                "harness and notebook present; the harness refuses a CUDA label without CUDA"
                if harness.is_file() and notebook.is_file()
                else f"missing {'harness' if not harness.is_file() else 'notebook'}"
            ),
        ),
        PackageResult(
            "P4-11b",
            "CUDA gate evidence",
            STATUS_DELIVERED if cuda_evidence else STATUS_BLOCKED,
            (
                f"{len(cuda_evidence)} record(s) under evidence/phase4/"
                if cuda_evidence
                else "needs a real NVIDIA run; ADR-0006 forbids presenting a CPU run as CUDA evidence"
            ),
        ),
        PackageResult(
            "P4-12",
            "independent review",
            STATUS_BLOCKED,
            (
                "review packet and reviewer guide prepared"
                if (root / "scripts/phase4_review_packet.py").is_file()
                and (root / "docs/roadmap/PHASE4_REVIEWER_GUIDE.md").is_file()
                else "no review packet prepared"
            )
            + "; the verdict is still outstanding because the author of an artifact may not sign it",
        ),
    ]


def run_checks(profile: str, root: Path = Path(".")) -> list[PackageResult]:
    results = _import_checks()
    results.append(_preset_checks(root))
    if profile == "micro":
        results.extend(_micro_checks())
    results.extend(_evidence(root))
    results.extend(_outstanding(root))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="micro", choices=["micro", "contract"])
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    results = run_checks(args.profile, args.root)
    for result in results:
        marker = {STATUS_DELIVERED: "PASS", STATUS_OPEN: "OPEN", STATUS_BLOCKED: "BLOCKED"}[result.status]
        print(f"{marker:8s} [{result.package}] {result.title}: {result.detail}")

    delivered = sum(1 for item in results if item.passed)
    print(f"\n{delivered}/{len(results)} packages delivered (profile={args.profile})")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema": "qdgrasp/phase4-gate/v0",
            "profile": args.profile,
            "delivered": delivered,
            "total": len(results),
            "packages": [dataclasses.asdict(item) for item in results],
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outstanding = [item for item in results if not item.passed]
    if outstanding:
        print("Phase 4 status: in_progress -- " + ", ".join(sorted({item.package for item in outstanding})))
        return 1
    print("Phase 4 status: complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
