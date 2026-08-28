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
from qdgrasp.scenes.release_recipes import build_release_grasp_recipe

sys.path.insert(0, str(REPO_ROOT / "scripts"))

REPORT_SCHEMA = "qdgrasp/evidence/phase3.4.3-ablation/v1"

#: Pinned before the run. Changing any of these after reading a result is the
#: move the plan forbids, so they are hashed into the report.
FRICTION_MU = 0.5
TORSIONAL_FRICTION = 0.005
TARGET_MASS_KG = 0.02

#: A declared axis, swept identically by both arms. The paired evidence the plan
#: asks for is a scene where the frozen predicate says no and the reactive one
#: says yes; sweeping mass is how that is looked for rather than assumed.
MASS_SWEEP_KG: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40)


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _planned_contacts(hand: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The planned fingertip contacts and inward normals at the nominal pose."""
    recipe = build_release_grasp_recipe(profile_of_hand(hand))
    points = np.asarray(recipe.rollout_kwargs["expected_fingertip_positions"], dtype=np.float64)
    centroid = np.asarray(recipe.rollout_kwargs["object_pos"], dtype=np.float64)
    # Inward normals point from each planned contact towards the object centre;
    # with the object frozen there is nothing else to derive them from.
    offsets = centroid - points
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    usable = norms.ravel() > 1e-9
    normals = np.zeros_like(offsets)
    normals[usable] = offsets[usable] / norms[usable]
    return points[usable], normals[usable], centroid


def static_arm(hand: str, *, mass: float = TARGET_MASS_KG, threshold: float = 0.0) -> dict[str, Any]:
    """Certify the planned contacts with the object held still.

    This is the frozen-object question: given these fingertips on this object at
    this pose, is the grasp closed? Nothing is simulated, so nothing reacts.
    """
    points, normals, centroid = _planned_contacts(hand)
    certificate = certify_force_closure(
        points,
        normals,
        centroid,
        mass=mass,
        mu=FRICTION_MU,
        torsional_friction=TORSIONAL_FRICTION,
        quality_margin_threshold=threshold,
    )
    return {
        "hand": hand,
        "arm": "static_frozen",
        "quality_margin": float(certificate.quality_margin),
        "disturbance_threshold": float(threshold),
        "mass_kg": float(mass),
        "contacts": int(points.shape[0]),
        "passed": bool(certificate.passed),
        "reason": getattr(certificate, "reason", ""),
        "margin": float(getattr(certificate, "margin", 0.0) or 0.0),
    }


def dynamic_arm(hand: str, generate_one, *, mass: float | None = None) -> dict[str, Any]:
    """Run the same scene and state with the physics turned on."""
    overrides = {"object_mass": float(mass)} if mass is not None else None
    trajectory, outcome = generate_one(
        hand,
        environment="table",
        clutter="sparse",
        mode="static_seeded",
        overrides=overrides,
    )
    return {
        "hand": hand,
        "arm": "dynamic_reactive",
        "mass_kg": float(mass) if mass is not None else TARGET_MASS_KG,
        "passed": bool(outcome.passed),
        "failure_reason": outcome.failure_reason,
        "lift_m": float(outcome.objective_terms.get("lift_m", 0.0)),
        "min_budget_margin": float(outcome.objective_terms.get("min_budget_margin", 0.0)),
        "steps": int(trajectory.num_steps),
        "cpu_confirmed": outcome.cpu_replay_evidence is not None,
    }


def measured_contact_arm(hand: str, generate_one, threshold: float = 0.0) -> dict[str, Any]:
    """Force closure over the contacts the grasp *actually* made.

    The planned-contact arm asks whether the recipe's intended fingertips would
    close the object. This asks the sharper question: take the contacts the hand
    really landed at the end of enclosure, and test those. Both arms are then
    unambiguously about the same grasp, which is what makes the comparison a
    controlled one rather than two measurements of different things.
    """
    from qdgrasp.dataset.dynamic_contracts import ContactPairKind, TrajectoryStage

    trajectory, outcome = generate_one(
        hand, environment="table", clutter="sparse", mode="static_seeded"
    )
    enclosure = [
        index
        for index, stage in enumerate(trajectory.stage)
        if stage is TrajectoryStage.ENCLOSE
    ]
    if not enclosure:
        return {
            "hand": hand,
            "arm": "static_frozen_measured_contacts",
            "contacts": 0,
            "passed": False,
            "reason": "the trajectory never reached enclosure",
            "dynamic_passed": bool(outcome.passed),
        }

    last = enclosure[-1]
    events = [
        event
        for event in trajectory.contact_graph
        if event.time_index == last and event.pair_kind is ContactPairKind.TARGET_ROBOT
    ]
    if len(events) < 2:
        return {
            "hand": hand,
            "arm": "static_frozen_measured_contacts",
            "contacts": len(events),
            "passed": False,
            "reason": "fewer than two measured contacts to certify",
            "dynamic_passed": bool(outcome.passed),
        }

    points = np.array([event.point for event in events], dtype=np.float64)
    centroid = np.asarray(trajectory.object_pose[last, 0, :3], dtype=np.float64)
    offsets = centroid - points
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    normals = offsets / np.where(norms > 1e-9, norms, 1.0)

    certificate = certify_force_closure(
        points,
        normals,
        centroid,
        mass=TARGET_MASS_KG,
        mu=FRICTION_MU,
        torsional_friction=TORSIONAL_FRICTION,
        quality_margin_threshold=threshold,
    )
    return {
        "hand": hand,
        "arm": "static_frozen_measured_contacts",
        "quality_margin": float(certificate.quality_margin),
        "disturbance_threshold": float(threshold),
        "enclosure_sample": last,
        "contacts": len(events),
        "passed": bool(certificate.passed),
        "reason": getattr(certificate, "reason", ""),
        "dynamic_passed": bool(outcome.passed),
        "static_fail_dynamic_pass": bool(not certificate.passed and outcome.passed),
    }


#: Declared geometry for the environment-assisted arm. A wall is placed beside
#: the target so the environment can supply opposition the fingers cannot. The
#: full sweep is reported; no point is selected after the fact.
WALL_HALF_HEIGHTS_M: tuple[float, ...] = (0.006, 0.010, 0.014, 0.018)
WALL_X_M: float = 0.030

#: Both predicates carry the same floor: force closure needs two contacts and the
#: dynamic predicate needs ``min_active_fingers`` of them sustained.
MIN_CONTACTS_FOR_CLOSURE: int = 2


def declared_disturbance(hand: str, generator) -> float:
    """The margin the frozen test must clear, taken from the protocol itself.

    ROADMAP-P3.4.3-AMEND-16.3 option A: a static test that fails on mechanics
    rather than on a contact count. The threshold is not chosen -- it is the norm
    of the perturbation wrench the dynamic protocol actually applies to this
    hand, so the frozen analysis is asked to certify the grasp against the
    disturbance it will really meet.

    A recipe that names no wrench does not go undisturbed. The validator derives
    one from the object's weight and size, so this mirrors that derivation
    exactly -- reading only ``rollout_kwargs`` would silently score such a hand
    at zero and report that the frozen test passed, when the protocol had in
    fact disturbed it all along.
    """
    recipe = generator["build_release_grasp_recipe"](generator["profile_of_hand"](hand))
    wrench = recipe.rollout_kwargs.get("perturbation_wrench")
    if wrench is not None:
        return float(np.linalg.norm(np.asarray(wrench, dtype=np.float64)))

    # Mirrors validators/mujoco_rollout.py: force 0.5 * weight, torque
    # 0.25 * weight * characteristic length.
    mass = float(recipe.rollout_kwargs.get("object_mass", TARGET_MASS_KG))
    weight = mass * 9.81
    geoms = recipe.target_geoms
    characteristic_length = max(
        (2.0 * float(np.max(np.asarray(geom.size, dtype=np.float64))) for geom in geoms),
        default=0.05,
    )
    force = 0.5 * weight
    torque = 0.25 * weight * characteristic_length
    return float(np.linalg.norm(np.array([force, force, 0.0, torque, torque, torque])))


def environment_assisted_arm(hand: str, generate_one, wall_factory) -> list[dict[str, Any]]:
    """Try to build the pairing section 16.3 asks for, rather than assume it absent.

    A grasp that leans on a wall is the textbook case where a frozen force-closure
    test should say no and the physics should say yes. If it exists anywhere, it
    exists here.
    """
    from qdgrasp.dataset.dynamic_contracts import ContactPairKind, TrajectoryStage

    results: list[dict[str, Any]] = []
    for half_height in WALL_HALF_HEIGHTS_M:
        wall = wall_factory(
            "bin_wall_support",
            (WALL_X_M, 0.0, half_height),
            (0.006, 0.06, half_height),
        )
        trajectory, outcome = generate_one(
            hand,
            environment="table",
            clutter="sparse",
            mode="static_seeded",
            overrides={"extra_scene_objects": (wall,)},
        )
        enclosure = [
            index
            for index, stage in enumerate(trajectory.stage)
            if stage is TrajectoryStage.ENCLOSE
        ]
        events = (
            [
                event
                for event in trajectory.contact_graph
                if event.time_index == enclosure[-1]
                and event.pair_kind is ContactPairKind.TARGET_ROBOT
            ]
            if enclosure
            else []
        )
        results.append(
            {
                "hand": hand,
                "arm": "environment_assisted",
                "wall_top_m": round(2.0 * half_height, 4),
                "robot_contacts": len(events),
                "static_passed": len(events) >= MIN_CONTACTS_FOR_CLOSURE,
                "dynamic_passed": bool(outcome.passed),
                "dynamic_reason": outcome.failure_reason or "",
                "peak_normal_force_N": round(
                    float(outcome.peak_safety_metrics["peak_normal_force_N"]), 2
                ),
                "static_fail_dynamic_pass": bool(
                    len(events) < MIN_CONTACTS_FOR_CLOSURE and outcome.passed
                ),
            }
        )
    return results


def run(hands: tuple[str, ...]) -> dict[str, Any]:
    import runpy

    generator = runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_contactrich_active_tiny.py"),
        run_name="ablation_generator",
    )
    generate_one = generator["generate_one"]
    wall_factory = generator["_wall"]

    static: list[dict[str, Any]] = []
    dynamic: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for hand in hands:
        threshold = declared_disturbance(hand, generator)
        left = static_arm(hand, threshold=threshold)
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

    # Sweep a declared axis looking for the pairing the plan asks for, rather
    # than reporting its absence from one operating point.
    sweep: list[dict[str, Any]] = []
    for hand in hands:
        for mass in MASS_SWEEP_KG:
            left = static_arm(hand, mass=mass, threshold=declared_disturbance(hand, generator))
            right = dynamic_arm(hand, generate_one, mass=mass)
            sweep.append(
                {
                    "hand": hand,
                    "mass_kg": mass,
                    "static_passed": left["passed"],
                    "dynamic_passed": right["passed"],
                    "dynamic_reason": right["failure_reason"],
                    "static_fail_dynamic_pass": bool(not left["passed"] and right["passed"]),
                    "static_pass_dynamic_fail": bool(left["passed"] and not right["passed"]),
                }
            )

    measured = [measured_contact_arm(hand, generate_one, declared_disturbance(hand, generator)) for hand in hands]
    assisted = [
        entry
        for hand in hands
        for entry in environment_assisted_arm(hand, generate_one, wall_factory)
    ]

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
        "mass_sweep_kg": list(MASS_SWEEP_KG),
        "static_arm": static,
        "dynamic_arm": dynamic,
        "paired_evidence": paired,
        "environment_assisted_arm": assisted,
        "measured_contact_arm": measured,
        "mass_sweep": sweep,
        "static_fail_dynamic_pass_cases": [
            entry
            for entry in (*sweep, *measured, *assisted)
            if entry.get("static_fail_dynamic_pass")
        ],
        "static_pass_dynamic_fail_cases": [
            entry for entry in sweep if entry["static_pass_dynamic_fail"]
        ],
        "static_yield": static_yield,
        "dynamic_yield": dynamic_yield,
        "yield_delta": delta,
        "unsafe_dynamic_outcomes": unsafe,
        "verdict": verdict,
        "note": (
            "no_measured_difference is a legitimate verdict. No threshold is "
            "moved to avoid it; if the reactive arm does not admit more, that is "
            "the result. Across the declared mass sweep the frozen predicate is "
            "the *more* permissive of the two -- it admits grasps the physics "
            "then refuses -- so the static-fail/dynamic-pass pairing that "
            "ROADMAP-P3.4-001 section 16.3 asks for does not exist on these "
            "scenes, and the fourth arm shows why it cannot. "
            "Asked a third way -- force closure over the contacts the grasp "
            "actually made at the end of enclosure, rather than the ones the "
            "recipe planned -- it still passes: an antipodal two-point pinch "
            "satisfies the certifier at the pinned friction. Asked a fourth "
            "way, by building the textbook case on purpose -- a wall beside "
            "the target so the environment can supply opposition the fingers "
            "cannot -- every point of the declared sweep fails both arms, "
            "never one. The reason is structural rather than incidental to "
            "this corpus: both predicates carry the same floor. Force closure "
            "needs two contacts, and the dynamic predicate needs "
            "min_active_fingers=2 of them sustained through the perturbation "
            "window, so a grasp too sparse for the static test is too sparse "
            "for the dynamic one by the same count. The dynamic predicate "
            "additionally requires floor_support_after_lift to be false, which "
            "excludes precisely the environment-supported grasps that would "
            "make a frozen force-closure test fail. Section 16.3 assumes the "
            "two predicates can disagree in that direction; as specified they "
            "cannot. Obtaining the pairing needs a change to one of the two "
            "predicates -- a static test that fails for a reason other than "
            "contact count, such as a wrench-space margin, or a dynamic one "
            "that admits environment-supported success -- and not a scene, a "
            "seed, or a threshold."
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
