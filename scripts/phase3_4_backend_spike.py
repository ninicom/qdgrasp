"""Phase 3.4 backend compatibility spike (P3.4-04).

Enumerates the MuJoCo features the release models actually use, so the
MJX/MuJoCo-Warp compatibility decision rests on a concrete requirement list
rather than on a general impression that "MJX supports most of MuJoCo".

This runs on CPU and needs no GPU. It reports what is *required*; whether a GPU
backend supports each item is filled in by running the same script in an
environment where `mujoco_warp` is importable, or recorded as `unknown`.

Plan section 6 makes the consequence explicit: if the GPU backend cannot carry
tendon transmission, weld equality or contact observation, Phase 3.4 stays
blocked and a backend decision record is written. It is not resolved by mocking
CUDA or by dropping Shadow from the gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from qdgrasp.dataset.pipeline.validators.mujoco_rollout import build_rollout_scene_model
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

REPO_ROOT = Path(__file__).resolve().parents[1]

HANDS = {
    "leap_hand": "leap_hand.yaml",
    "wonik_allegro": "wonik_allegro.yaml",
    "shadow_hand": "shadow_hand.yaml",
}

#: A single box stands in for the procedural target: the spike is about model
#: features, not about object geometry.
PROBE_GEOMS = (
    SubGeomSpec(type="box", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.0)),
)

def _enum_names(enum_type) -> dict[int, str]:
    """MuJoCo enums are pybind types, not iterable; read their member table."""
    return {int(value): name for name, value in enum_type.__members__.items()}


_TRN_NAMES = _enum_names(mujoco.mjtTrn)
_JNT_NAMES = _enum_names(mujoco.mjtJoint)
_EQ_NAMES = _enum_names(mujoco.mjtEq)
_GEOM_NAMES = _enum_names(mujoco.mjtGeom)
_INT_NAMES = _enum_names(mujoco.mjtIntegrator)
_SOLVER_NAMES = _enum_names(mujoco.mjtSolver)
_DYN_NAMES = _enum_names(mujoco.mjtDyn)


def _counts(values, names: dict[int, str]) -> dict[str, int]:
    counter = Counter(names.get(int(v), f"unknown({int(v)})") for v in values)
    return dict(sorted(counter.items()))


def enumerate_model_features(model: mujoco.MjModel) -> dict[str, Any]:
    """Everything a batched backend has to reproduce for this model."""
    return {
        "sizes": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "ngeom": int(model.ngeom),
            "ntendon": int(model.ntendon),
            "neq": int(model.neq),
            "nsensor": int(model.nsensor),
            "nmocap": int(model.nmocap),
            "nconmax": int(model.nconmax),
        },
        "actuator_transmission": _counts(model.actuator_trntype, _TRN_NAMES),
        "actuator_dynamics": _counts(model.actuator_dyntype, _DYN_NAMES),
        "joint_types": _counts(model.jnt_type, _JNT_NAMES),
        "equality_types": _counts(model.eq_type, _EQ_NAMES) if int(model.neq) else {},
        "geom_types": _counts(model.geom_type, _GEOM_NAMES),
        "options": {
            "integrator": _INT_NAMES.get(int(model.opt.integrator), "unknown"),
            "solver": _SOLVER_NAMES.get(int(model.opt.solver), "unknown"),
            "timestep": float(model.opt.timestep),
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "cone": int(model.opt.cone),
            "jacobian": int(model.opt.jacobian),
        },
        "uses_tendon": int(model.ntendon) > 0,
        "uses_equality_weld": any(
            int(t) == int(mujoco.mjtEq.mjEQ_WELD) for t in model.eq_type
        ),
        "uses_mocap": int(model.nmocap) > 0,
    }


def probe_contact_observation(model: mujoco.MjModel) -> dict[str, Any]:
    """Phase 3.4 reads per-contact forces, not just a contact count.

    A backend that reports contact existence but not resolved force cannot
    carry the safety budget, so record that this is a hard requirement.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)
    force = np.zeros(6)
    readable = False
    if int(data.ncon) > 0:
        mujoco.mj_contactForce(model, data, 0, force)
        readable = bool(np.all(np.isfinite(force)))
    return {
        "contacts_after_settle": int(data.ncon),
        "mj_contactForce_readable": readable,
        "requires_per_contact_force": True,
        "requires_per_contact_frame": True,
        "requires_contact_geom_pair_identity": True,
    }


def gpu_backend_status() -> dict[str, Any]:
    """What is importable here. Absence is a fact, not a failure of this script."""
    status: dict[str, Any] = {}
    for module in ("mujoco.mjx", "mujoco_warp", "warp"):
        available = importlib.util.find_spec(module) is not None
        status[module] = "available" if available else "not_installed"
    status["verdict"] = (
        "unknown_pending_gpu_environment"
        if status["mujoco_warp"] == "not_installed"
        else "resolvable_here"
    )
    return status


def run_spike() -> dict[str, Any]:
    per_hand: dict[str, Any] = {}
    for hand, config_name in HANDS.items():
        spec = RobotSpec.from_config(config_name, sample_anchors=False)
        xml_path = resolve_robot_asset(spec.config.source_asset)
        model = build_rollout_scene_model(xml_path, PROBE_GEOMS)
        per_hand[hand] = {
            "features": enumerate_model_features(model),
            "contact_observation": probe_contact_observation(model),
        }

    required: set[str] = set()
    for hand, payload in per_hand.items():
        features = payload["features"]
        for name in features["actuator_transmission"]:
            required.add(f"actuator_transmission:{name}")
        for name in features["equality_types"]:
            required.add(f"equality:{name}")
        for name in features["joint_types"]:
            required.add(f"joint:{name}")
        for name in features["geom_types"]:
            required.add(f"geom:{name}")
        required.add(f"integrator:{features['options']['integrator']}")
        required.add(f"solver:{features['options']['solver']}")
        if features["uses_tendon"]:
            required.add("tendon_transmission")
        if features["uses_mocap"]:
            required.add("mocap_body")

    return {
        "phase": "3.4",
        "work_package": "P3.4-04",
        "mujoco_version": mujoco.__version__,
        "per_hand": per_hand,
        "required_feature_set": sorted(required),
        "gpu_backend_status": gpu_backend_status(),
        "blocking_requirements": [
            "tendon_transmission",
            "equality:mjEQ_WELD",
            "mocap_body",
            "per_contact_force_and_frame",
        ],
        "decision_rule": (
            "If the GPU backend cannot carry every blocking requirement, Phase 3.4 "
            "stays blocked and a backend decision record is written. Mocking CUDA "
            "or dropping Shadow from the gate is not an accepted resolution."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write the report JSON here.")
    args = parser.parse_args()
    try:
        report = run_spike()
    except Exception as exc:  # noqa: BLE001 - the spike reports, it does not mask
        print(f"Phase 3.4 backend spike failed: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
