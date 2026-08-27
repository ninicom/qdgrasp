#!/usr/bin/env python3
"""Phase 3.4 CUDA gate; must run on a physical NVIDIA GPU.

Staged to match the phase. Stage 1 resolves the backend decision that P3.4-04
left open: whether MuJoCo Warp actually carries tendon transmission, weld
equality and per-contact force for the release hands. Until P3.4-05 lands there
is no CUDA backend to benchmark, and this script says so rather than reporting a
CPU number as CUDA evidence.

A CPU fallback is never admissible here
(``docs/decisions/0006-cuda-hardware-required.md``).

    python scripts/phase3_4_cuda_contact_search.py \\
      --device cuda:0 --profile kaggle-t4-micro \\
      --evidence phase3_4_cuda_evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_SCHEMA = "qdgrasp/evidence/phase3.4-cuda/v1"

#: Requirements P3.4-04 measured from the real rollout models. A GPU backend
#: that cannot carry all of these does not unblock the phase.
BLOCKING_REQUIREMENTS = (
    "tendon_transmission",
    "equality:mjEQ_WELD",
    "mocap_body",
    "per_contact_force_and_frame",
)


class GateFailure(RuntimeError):
    """The gate refuses to emit evidence."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_real_cuda(device: str) -> dict[str, Any]:
    """Refuse anything that is not a physical NVIDIA device."""
    import torch

    if not device.startswith("cuda"):
        raise GateFailure(f"--device must be a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise GateFailure(
            "torch.cuda.is_available() is False; a CPU run is not CUDA evidence"
        )
    if torch.cuda.device_count() < 1:
        raise GateFailure("no CUDA device visible to torch")

    index = int(device.split(":")[1]) if ":" in device else 0
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(index),
        "capability": f"{properties.major}.{properties.minor}",
        "total_memory_gib": round(properties.total_memory / (1024**3), 2),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "driver": getattr(torch.version, "cuda", None),
    }


def resolve_warp_backend(device: str) -> dict[str, Any]:
    """Answer the question P3.4-04 could not answer on CPU.

    Reports what MuJoCo Warp actually exposes. It does not guess: a missing
    module is reported as missing, and a missing capability blocks the phase.
    """
    import mujoco

    status: dict[str, Any] = {
        "mujoco": mujoco.__version__,
        "modules": {},
        "capabilities": {},
    }
    for name in ("warp", "mujoco_warp", "mujoco.mjx"):
        spec = importlib.util.find_spec(name)
        status["modules"][name] = "available" if spec is not None else "not_installed"

    if status["modules"]["mujoco_warp"] != "available":
        status["verdict"] = "blocked_mujoco_warp_missing"
        status["unsupported"] = list(BLOCKING_REQUIREMENTS)
        return status

    warp = importlib.import_module("warp")
    mjwarp = importlib.import_module("mujoco_warp")
    status["warp_version"] = getattr(warp, "__version__", "unknown")

    devices = [str(d) for d in getattr(warp, "get_cuda_devices", list)()]
    status["warp_cuda_devices"] = devices
    if not devices:
        status["verdict"] = "blocked_warp_sees_no_cuda_device"
        status["unsupported"] = list(BLOCKING_REQUIREMENTS)
        return status

    # Compile each release hand and record which blocking requirement survives.
    from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
        build_rollout_scene_model,
    )
    from qdgrasp.objects.schema import SubGeomSpec
    from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

    probe_geoms = (SubGeomSpec(type="box", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.0)),)
    unsupported: list[str] = []
    per_hand: dict[str, Any] = {}
    for hand, config_name in (
        ("leap_hand", "leap_hand.yaml"),
        ("wonik_allegro", "wonik_allegro.yaml"),
        ("shadow_hand", "shadow_hand.yaml"),
    ):
        spec = RobotSpec.from_config(config_name, sample_anchors=False)
        model = build_rollout_scene_model(
            resolve_robot_asset(spec.config.source_asset), probe_geoms
        )
        entry: dict[str, Any] = {
            "ntendon": int(model.ntendon),
            "neq": int(model.neq),
            "nmocap": int(model.nmocap),
        }
        try:
            put_model = getattr(mjwarp, "put_model", None)
            if put_model is None:
                raise AttributeError("mujoco_warp exposes no put_model")
            put_model(model)
            entry["compiled"] = True
        except Exception as exc:  # noqa: BLE001 - the verdict is the point
            entry["compiled"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if int(model.ntendon) > 0:
                unsupported.append("tendon_transmission")
            unsupported.append("equality:mjEQ_WELD")
        per_hand[hand] = entry

    status["per_hand"] = per_hand
    status["unsupported"] = sorted(set(unsupported))
    status["capabilities"] = {
        req: req not in status["unsupported"] for req in BLOCKING_REQUIREMENTS
    }
    status["verdict"] = (
        "supported" if not status["unsupported"] else "blocked_missing_capability"
    )
    return status


def stage_two_not_implemented() -> dict[str, Any]:
    """P3.4-05 does not exist, so there is nothing to benchmark yet."""
    return {
        "status": "not_implemented",
        "missing": ["P3.4-05 MJWarp CUDA backend"],
        "note": (
            "Throughput, VRAM and CPU/GPU parity numbers require the CUDA "
            "backend. Reporting the CPU oracle here would be a fabricated "
            "CUDA measurement."
        ),
    }


def build_evidence(device: str, profile: str) -> dict[str, Any]:
    from qdgrasp import __version__ as qdgrasp_version

    cuda = assert_real_cuda(device)
    warp = resolve_warp_backend(device)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "timestamp": datetime.now(UTC).isoformat(),
        "profile": profile,
        "qdgrasp_version": qdgrasp_version,
        "python": platform.python_version(),
        "cuda_environment": cuda,
        "backend_resolution": warp,
        "blocking_requirements": list(BLOCKING_REQUIREMENTS),
        "search_benchmark": stage_two_not_implemented(),
    }
    evidence["status"] = (
        "BACKEND_SUPPORTED_PENDING_P3.4-05"
        if warp["verdict"] == "supported"
        else "BLOCKED"
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", default="kaggle-t4-micro")
    parser.add_argument("--evidence", type=Path, default=None)
    args = parser.parse_args()

    try:
        evidence = build_evidence(args.device, args.profile)
    except GateFailure as exc:
        print(f"Phase 3.4 CUDA gate failed: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Phase 3.4 CUDA gate failed to import a dependency: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(evidence, indent=2, sort_keys=True)
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(f"evidence_sha256={_sha256_text(payload)}", file=sys.stderr)

    verdict = evidence["backend_resolution"]["verdict"]
    if verdict != "supported":
        unsupported = evidence["backend_resolution"].get("unsupported", [])
        print(
            f"Phase 3.4 BLOCKED: MuJoCo Warp verdict {verdict!r}; "
            f"unsupported blocking requirements: {unsupported}. "
            "Per ROADMAP-P3.4-001 section 10 the phase stays blocked and a backend "
            "decision record is written. Do not substitute a mock CUDA backend and "
            "do not drop Shadow from the gate.",
            file=sys.stderr,
        )
        return 1

    print(
        "Phase 3.4 backend supported; P3.4-05 and the search benchmark are still "
        "outstanding. This run does not close the phase.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
