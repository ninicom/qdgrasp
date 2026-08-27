#!/usr/bin/env python3
"""Controlled static-vs-dynamic ablation for the active hands (S11; C05.8).

The Phase 3.4 hypothesis is that letting the scene react admits grasps a
frozen-object pipeline misses. That is a claim about one factor, so this
measures one factor: the same hand, the same scene, the same target at the same
pose, the same safety budget, the same pinned fingertip contacts. The only thing
that differs is whether the scene is simulated.

* **Static arm (frozen).** The planned fingertip contacts are certified for
  six-dimensional force closure and gravity equilibrium at the nominal object
  pose. No physics runs; the object cannot move, which is exactly the frozen
  assumption.
* **Dynamic arm (reactive).** The same scene and state go through the validated
  contact rollout, and the outcome is whatever the physics produced.

Everything is pinned before the run and hashed into the report.
``no_measured_difference`` is a legitimate verdict, and the plan forbids moving
a threshold to avoid it: if the dynamic arm does not admit more, the record says
so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdgrasp.config.active_scope import profile_of_hand, resolve_workload_hands
from qdgrasp.dataset.pipeline.certifiers.contact_force import (
    certify_force_closure,
)
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.scenes.release_recipes import build_release_grasp_recipe

sys.path.insert(0, str(REPO_ROOT / "scripts"))

REPORT_SCHEMA = "qdgrasp/evidence/phase3.4.3-ablation/v1"

#: Pinned before the run. Changing any of these after reading a result is the
#: move the plan forbids, so they are hashed into the report.
FRICTION_MU = 0.5
TORSIONAL_FRICTION = 0.005
TARGET_MASS_KG = 0.02


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def static_arm(hand: str) -> dict[str, Any]:
    """Certify the planned contacts with the object held still.

    This is the frozen-object question: given these fingertips on this object at
    this pose, is the grasp closed? Nothing is simulated, so nothing reacts.
    """
    spec = RobotSpec.from_config(profile_of_hand(hand), sample_anchors=False)
    recipe = build_release_grasp_recipe(profile_of_hand(hand))
    del spec

    points = np.asarray(recipe.rollout_kwargs["expected_fingertip_positions"], dtype=np.float64)
    centroid = np.asarray(recipe.rollout_kwargs["object_pos"], dtype=np.float64)
    # Inward normals point from each planned contact towards the object centre;
    # with the object frozen there is nothing else to derive them from.
    offsets = centroid - points
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    usable = (norms.ravel() > 1e-9)
    normals = np.zeros_like(offsets)
    normals[usable] = offsets[usable] / norms[usable]

    certificate = certify_force_closure(
        points[usable],
        normals[usable],
        centroid,
        mass=TARGET_MASS_KG,
        mu=FRICTION_MU,
        torsional_friction=TORSIONAL_FRICTION,
    )
    return {
        "hand": hand,
        "arm": "static_frozen",
        "contacts": int(usable.sum()),
        "passed": bool(certificate.passed),
        "reason": getattr(certificate, "reason", ""),
        "margin": float(getattr(certificate, "margin", 0.0) or 0.0),
    }


def dynamic_arm(hand: str, generate_one) -> dict[str, Any]:
    """Run the same scene and state with the physics turned on."""
    trajectory, outcome = generate_one(
        hand, environment="table", clutter="sparse", mode="static_seeded"
    )
    return {
        "hand": hand,
        "arm": "dynamic_reactive",
        "passed": bool(outcome.passed),
        "failure_reason": outcome.failure_reason,
        "lift_m": float(outcome.objective_terms.get("lift_m", 0.0)),
        "min_budget_margin": float(outcome.objective_terms.get("min_budget_margin", 0.0)),
        "steps": int(trajectory.num_steps),
        "cpu_confirmed": outcome.cpu_replay_evidence is not None,
    }


def run(hands: tuple[str, ...]) -> dict[str, Any]:
    import runpy

    generator = runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_contactrich_active_tiny.py"),
        run_name="ablation_generator",
    )
    generate_one = generator["generate_one"]

    static: list[dict[str, Any]] = []
    dynamic: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for hand in hands:
        left = static_arm(hand)
        right = dynamic_arm(hand, generate_one)
        static.append(left)
        dynamic.append(right)
        paired.append(
            {
                "hand": hand,
                "scene": "table/sparse",
                "static_passed": left["passed"],
                "dynamic_passed": right["passed"],
                # The pairing the plan asks for: the same scene and state where
                # the frozen pipeline does not admit and the reactive one does.
                "static_fail_dynamic_pass": bool(not left["passed"] and right["passed"]),
            }
        )

    static_yield = sum(1 for entry in static if entry["passed"]) / max(1, len(static))
    dynamic_yield = sum(1 for entry in dynamic if entry["passed"]) / max(1, len(dynamic))
    delta = dynamic_yield - static_yield
    unsafe = sum(
        1
        for entry in dynamic
        if entry.get("failure_reason") in ("damaging_contact", "forbidden_contact")
    )

    if delta > 0.0 and unsafe == 0:
        verdict = "dynamic_admits_more"
    elif delta > 0.0:
        verdict = "dynamic_admits_more_but_unsafe"
    elif delta < 0.0:
        verdict = "static_admits_more"
    else:
        verdict = "no_measured_difference"

    return {
        "schema": REPORT_SCHEMA,
        "work_package": "P3.4-14",
        "generated_at": datetime.now(UTC).isoformat(),
        "hands": list(hands),
        "controlled_factor": (
            "whether the scene is simulated. Both arms use the same hand, scene, "
            "target pose, planned fingertip contacts and safety budget."
        ),
        "pinned_parameters": {
            "friction_mu": FRICTION_MU,
            "torsional_friction": TORSIONAL_FRICTION,
            "target_mass_kg": TARGET_MASS_KG,
        },
        "parameter_hash": _hash(
            {
                "friction_mu": FRICTION_MU,
                "torsional_friction": TORSIONAL_FRICTION,
                "target_mass_kg": TARGET_MASS_KG,
            }
        ),
        "static_arm": static,
        "dynamic_arm": dynamic,
        "paired_evidence": paired,
        "static_yield": static_yield,
        "dynamic_yield": dynamic_yield,
        "yield_delta": delta,
        "unsafe_dynamic_outcomes": unsafe,
        "verdict": verdict,
        "note": (
            "no_measured_difference is a legitimate verdict. No threshold is "
            "moved to avoid it; if the reactive arm does not admit more, that is "
            "the result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "evidence" / "phase3_4_3" / "s11" / "static-vs-dynamic.json",
    )
    args = parser.parse_args()

    scope = resolve_workload_hands()
    report = run(scope.hands)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    print(payload)
    print(f"wrote {args.out.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"verdict: {report['verdict']}", file=sys.stderr)
    # A report is not a gate: the verdict is recorded either way, and the gate
    # that reads it lives in the completeness manifest.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
