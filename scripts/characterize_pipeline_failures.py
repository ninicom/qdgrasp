"""Characterization harness for Phase 3.2.1 (work package P3.2.1-01).

Runs the real `run_pipeline_chunk` over a pinned hand x recipe x object matrix
and freezes every candidate's stage accounting, failure reason and raw residual
telemetry into a JSON corpus.  The corpus is the control case that later
remediation work is measured against: a fix is only credited when it removes a
specific failure signature from this corpus while the unrelated signatures stay
put (see the plan's section 6, "Evidence schema và causal proof").

The harness never repairs, retries or tunes anything.  It records what the
pipeline does.

Baseline:

    timeout 3600 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
      PYTHONHASHSEED=0 .venv/bin/python scripts/characterize_pipeline_failures.py \\
      --label baseline

Comparison against a frozen corpus:

    .venv/bin/python scripts/characterize_pipeline_failures.py \\
      --label rc01-intervention --hypothesis RC-01 \\
      --intervention "contact_direction used for the autodiff Jacobian" \\
      --baseline evidence/phase3_2_1/baseline/corpus.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from qdgrasp.dataset.pipeline.contracts import ALLOWED_RECIPES, PipelineOutcome, get_recipe
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.runtime import environment_info

logger = logging.getLogger("characterize_pipeline_failures")

from qdgrasp.config import resolve_workload_hands

REPO_ROOT = Path(__file__).resolve().parents[1]

# This is a diagnostic corpus, not a release workload: it characterises where
# the pipeline fails, and a paused hand is part of what has to be characterised.
# Declaring that out loud is what ADR-0008 asks for; the result is non-release.
DEFAULT_SCOPE = resolve_workload_hands(
    ("leap_hand", "wonik_allegro", "shadow_hand"),
    experimental_shadow=True,
    purpose="failure characterisation across every known profile, including the paused one",
)
DEFAULT_HANDS = DEFAULT_SCOPE.hands
DEFAULT_RECIPES = ("surface_fixed_v1", "region_opposition_v1", "wrench_guided_v1")

ROBOT_CONFIGS = {
    "leap_hand": "leap_hand.yaml",
    "wonik_allegro": "wonik_allegro.yaml",
    "shadow_hand": "shadow_hand.yaml",
}

# Source files whose content defines the observed behaviour.  A corpus is only
# comparable to another corpus taken at the same hashes, or to one whose diff in
# these hashes is exactly the intervention under test.
PIPELINE_SOURCES = (
    "qdgrasp/dataset/pipeline/orchestrator.py",
    "qdgrasp/dataset/pipeline/contracts.py",
    "qdgrasp/dataset/pipeline/canonical_full_flow.py",
    "qdgrasp/dataset/pipeline/generated_reachable.py",
    "qdgrasp/dataset/pipeline/contact_state.py",
    "qdgrasp/dataset/pipeline/filter.py",
    "qdgrasp/dataset/pipeline/palm_hypotheses.py",
    "qdgrasp/dataset/pipeline/proposals/identity.py",
    "qdgrasp/dataset/pipeline/proposals/surface_fixed.py",
    "qdgrasp/dataset/pipeline/proposals/region_opposition.py",
    "qdgrasp/dataset/pipeline/proposals/wrench_guided.py",
    "qdgrasp/dataset/pipeline/proposals/width_mapper.py",
    "qdgrasp/dataset/pipeline/solvers/fixed_contact_dls.py",
    "qdgrasp/dataset/pipeline/solvers/joint_palm_dls.py",
    "qdgrasp/dataset/pipeline/solvers/normal_equations.py",
    "qdgrasp/dataset/pipeline/solvers/progress.py",
    "qdgrasp/dataset/pipeline/solvers/region_dls.py",
    "qdgrasp/dataset/pipeline/certifiers/contact_force.py",
    "qdgrasp/dataset/pipeline/certifiers/grasp_wrench.py",
    "qdgrasp/dataset/pipeline/observers/contact_load.py",
    "qdgrasp/dataset/pipeline/validators/mujoco_rollout.py",
    "qdgrasp/dataset/pipeline/validators/collision_admission.py",
    "qdgrasp/dataset/pipeline/validators/dynamic_predicate.py",
    "qdgrasp/dataset/pipeline/validators/scene_dynamic.py",
    "qdgrasp/robot/transmission/command.py",
    "qdgrasp/robot/transmission/contracts.py",
    "scripts/characterize_pipeline_failures.py",
)


# --------------------------------------------------------------------------
# Pinned diagnostic objects
# --------------------------------------------------------------------------


def _box_object(half_extent: float) -> dict[str, Any]:
    edge = 2.0 * half_extent
    mesh = trimesh.creation.box(extents=(edge, edge, edge))
    geoms = (
        SubGeomSpec(type="box", size=(half_extent, half_extent, half_extent)),
    )
    return {"mesh": mesh, "collision_geoms": geoms, "mass": 0.1}


def _cylinder_object(radius: float, half_height: float) -> dict[str, Any]:
    mesh = trimesh.creation.cylinder(radius=radius, height=2.0 * half_height, sections=64)
    geoms = (SubGeomSpec(type="cylinder", size=(radius, half_height)),)
    return {"mesh": mesh, "collision_geoms": geoms, "mass": 0.1}


# Diagnostic objects are analytic and hand-independent: the same geometry is
# presented to every hand so a failure signature cannot be blamed on object
# sampling noise.
OBJECT_BUILDERS = {
    # The 5 cm box that produced the 18/18 `IK: max_iter` finding.
    "box_50mm": lambda: _box_object(0.025),
    "box_70mm": lambda: _box_object(0.035),
    "cylinder_r25_h60": lambda: _cylinder_object(0.025, 0.030),
}
DEFAULT_OBJECTS = ("box_50mm",)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _mesh_hash(mesh: trimesh.Trimesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def collect_provenance(
    *, seed: int, candidates: int, run_dynamic: bool, objects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_dirty": dirty,
        "source_hashes": {
            name: _sha256_file(REPO_ROOT / name) for name in PIPELINE_SOURCES
        },
        "robot_profile_hashes": {
            hand: _sha256_file(REPO_ROOT / "qdgrasp" / "presets" / "robots" / cfg)
            for hand, cfg in ROBOT_CONFIGS.items()
        },
        "object_hashes": {
            name: _mesh_hash(spec["mesh"]) for name, spec in objects.items()
        },
        "environment": environment_info().to_dict(),
        "platform": platform.platform(),
        "seed": seed,
        "candidate_budget": candidates,
        "run_dynamic": run_dynamic,
    }


# --------------------------------------------------------------------------
# Telemetry extraction
# --------------------------------------------------------------------------


def _floats(value: Any) -> Any:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return [round(float(x), 9) for x in array.reshape(-1)]


def outcome_record(outcome: PipelineOutcome, index: int) -> dict[str, Any]:
    """Flatten one candidate's evidence — no derived verdicts, raw values only."""
    record: dict[str, Any] = {
        "candidate_index": index,
        "failure_stage": outcome.failure_stage,
        "failure_reason": outcome.failure_reason,
        "proposal_valid": bool(outcome.proposal_valid),
        "ik_valid": bool(outcome.ik_valid),
        "collision_valid": bool(outcome.collision_valid),
        "static_force_valid": bool(outcome.static_force_valid),
        "dynamic_valid": bool(outcome.dynamic_valid),
    }

    proposal = outcome.proposal
    if proposal is not None:
        record["proposal"] = {
            "candidate_id": proposal.candidate_id,
            "provenance": proposal.provenance,
            "target_points": _floats(proposal.target_points),
            "inward_normals": _floats(proposal.inward_normals),
            "face_ids": [int(f) for f in np.asarray(proposal.face_ids).reshape(-1)],
            "finger_ids": [int(f) for f in np.asarray(proposal.finger_ids).reshape(-1)],
            "active_fingers": (
                None
                if proposal.active_fingers is None
                else [bool(v) for v in np.asarray(proposal.active_fingers).reshape(-1)]
            ),
            "opposition_pairs": (
                None
                if proposal.opposition_pairs is None
                else [
                    [int(v) for v in pair]
                    for pair in np.asarray(proposal.opposition_pairs).reshape(-1, 2)
                ]
            ),
            "duplicate_face_ids": int(
                len(np.asarray(proposal.face_ids).reshape(-1))
                - len(np.unique(np.asarray(proposal.face_ids)))
            ),
        }

    kin = outcome.kinematics
    if kin is not None:
        active_mask = (
            np.ones_like(np.asarray(kin.position_residuals), dtype=bool)
            if proposal is None or proposal.active_fingers is None
            else np.asarray(proposal.active_fingers, dtype=bool)
        )
        active_position_residuals = np.asarray(kin.position_residuals)[active_mask]
        active_normal_residuals = np.asarray(kin.normal_residuals)[active_mask]
        record["kinematics"] = {
            "converged": bool(kin.converged),
            "reason": str(kin.reason),
            "iterations": None if kin.iterations is None else int(np.asarray(kin.iterations)),
            "q": _floats(kin.q),
            "palm_pos": _floats(kin.palm_pos),
            "palm_rot": _floats(kin.palm_rot),
            "position_residuals": _floats(kin.position_residuals),
            "normal_residuals": _floats(kin.normal_residuals),
            "max_position_residual": float(np.max(np.abs(kin.position_residuals))),
            "max_normal_residual": float(np.max(np.abs(kin.normal_residuals))),
            "active_max_position_residual": float(
                np.max(np.abs(active_position_residuals))
            ),
            "active_max_normal_residual": float(
                np.max(np.abs(active_normal_residuals))
            ),
            "surface_distances": _floats(kin.surface_distances),
            "solver_metrics": (
                None
                if kin.solver_metrics is None
                else {
                    name: _floats(value)
                    for name, value in sorted(kin.solver_metrics.items())
                }
            ),
            "palm_hypothesis_id": kin.palm_hypothesis_id,
            "palm_hypothesis_metrics": kin.palm_hypothesis_metrics,
        }

    cert = outcome.static_certificate
    if cert is not None:
        record["static_certificate"] = {
            "passed": bool(cert.passed),
            "cone_residual": float(cert.cone_residual),
            "quality_margin": float(cert.quality_margin),
            "object_wrench": _floats(cert.object_wrench),
            "force_solution": _floats(cert.force_solution),
        }

    collision = outcome.collision_admission
    if collision is not None:
        record["collision_admission"] = {
            "passed": bool(collision.passed),
            "reason": collision.reason,
            "contact_pairs": list(collision.contact_pairs),
            "max_penetration": float(collision.max_penetration),
            "min_hand_floor_clearance": float(
                collision.min_hand_floor_clearance
            ),
        }

    dyn = outcome.dynamic_validation
    if dyn is not None:
        record["dynamic_validation"] = {
            "passed": bool(dyn.passed),
            "failure_stage": dyn.failure_stage,
            "trajectory_metrics": {
                key: (float(val) if isinstance(val, (int, float, np.floating)) else val)
                for key, val in dyn.trajectory_metrics.items()
            },
            "per_finger_loads": _floats(dyn.per_finger_loads),
        }

    return record


# --------------------------------------------------------------------------
# Matrix execution
# --------------------------------------------------------------------------


def run_cell(
    *,
    hand: str,
    recipe_id: str,
    object_name: str,
    object_spec: dict[str, Any],
    seed: int,
    candidates: int,
    run_dynamic: bool,
) -> dict[str, Any]:
    spec = RobotSpec.from_config(ROBOT_CONFIGS[hand], sample_anchors=False)
    xml_path = resolve_robot_asset(spec.config.source_asset)
    if not xml_path.is_file():
        raise RuntimeError(f"robot asset unavailable: {xml_path}")

    rng = get_generator(seed, "characterize", hand, recipe_id, object_name)
    started = time.time()
    outcomes, reasons = run_pipeline_chunk(
        recipe_id=recipe_id,
        spec=spec,
        mesh=object_spec["mesh"],
        collision_geoms=object_spec["collision_geoms"],
        hand_xml_path=xml_path,
        rng=rng,
        num_candidates=candidates,
        object_mass=object_spec["mass"],
        run_dynamic=run_dynamic,
    )
    elapsed = time.time() - started

    records = [outcome_record(outcome, index) for index, outcome in enumerate(outcomes)]
    signature = Counter(
        f"{rec['failure_stage']}:{rec['failure_reason']}" for rec in records
    )
    return {
        "hand": hand,
        "recipe_id": recipe_id,
        "recipe": get_recipe(recipe_id),
        "object": object_name,
        "num_candidates": candidates,
        "num_outcomes": len(records),
        "elapsed_seconds": round(elapsed, 3),
        "stage_accounting": dict(reasons),
        "failure_signature": dict(sorted(signature.items())),
        "candidates": records,
    }


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    signature: Counter[str] = Counter()
    accounting: Counter[str] = Counter()
    total = 0
    for cell in cells:
        total += cell["num_outcomes"]
        signature.update(cell["failure_signature"])
        accounting.update(cell["stage_accounting"])
    reached_dynamic = sum(
        1
        for cell in cells
        for rec in cell["candidates"]
        if rec.get("dynamic_validation") is not None
    )
    positives = sum(
        1 for cell in cells for rec in cell["candidates"] if rec["dynamic_valid"]
    )
    return {
        "cells": len(cells),
        "total_candidates": total,
        "candidates_reaching_dynamic_rollout": reached_dynamic,
        "pipeline_generated_positives": positives,
        "failure_signature": dict(sorted(signature.items())),
        "stage_accounting": dict(sorted(accounting.items())),
    }


def compare_to_baseline(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Report what moved between two corpora, per cell and in aggregate.

    Only differences are reported; the caller decides whether a difference is
    the intended effect of the intervention or unrelated drift.
    """
    base_cells = {
        (c["hand"], c["recipe_id"], c["object"]): c for c in baseline["cells_detail"]
    }
    cur_cells = {
        (c["hand"], c["recipe_id"], c["object"]): c for c in current["cells_detail"]
    }

    cell_diffs = []
    for key in sorted(set(base_cells) | set(cur_cells)):
        base = base_cells.get(key)
        cur = cur_cells.get(key)
        if base is None or cur is None:
            cell_diffs.append(
                {
                    "cell": list(key),
                    "present_in_baseline": base is not None,
                    "present_in_current": cur is not None,
                }
            )
            continue
        if base["failure_signature"] != cur["failure_signature"]:
            cell_diffs.append(
                {
                    "cell": list(key),
                    "baseline_signature": base["failure_signature"],
                    "current_signature": cur["failure_signature"],
                }
            )

    changed_sources = sorted(
        name
        for name in set(baseline["provenance"]["source_hashes"])
        | set(current["provenance"]["source_hashes"])
        if baseline["provenance"]["source_hashes"].get(name)
        != current["provenance"]["source_hashes"].get(name)
    )

    return {
        "baseline_summary": baseline["summary"],
        "current_summary": current["summary"],
        "changed_cells": cell_diffs,
        "unchanged_cells": len(base_cells) - len(
            [d for d in cell_diffs if "baseline_signature" in d]
        ),
        "changed_source_files": changed_sources,
        "seed_matches": baseline["provenance"]["seed"] == current["provenance"]["seed"],
        "budget_matches": (
            baseline["provenance"]["candidate_budget"]
            == current["provenance"]["candidate_budget"]
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="baseline", help="Corpus label / output subdirectory.")
    parser.add_argument("--hands", nargs="+", default=list(DEFAULT_HANDS), choices=list(DEFAULT_HANDS))
    parser.add_argument(
        "--recipes", nargs="+", default=list(DEFAULT_RECIPES), choices=sorted(ALLOWED_RECIPES)
    )
    parser.add_argument(
        "--objects", nargs="+", default=list(DEFAULT_OBJECTS), choices=sorted(OBJECT_BUILDERS)
    )
    parser.add_argument("--candidates", type=int, default=2, help="Candidate budget per cell.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-dynamic",
        dest="run_dynamic",
        action="store_false",
        help="Skip MuJoCo rollouts (kinematic characterization only).",
    )
    parser.add_argument(
        "--hypothesis", default="", help="RC-* / H-* identifier this run is evidence for."
    )
    parser.add_argument(
        "--intervention",
        default="",
        help="The single variable changed relative to the baseline corpus.",
    )
    parser.add_argument(
        "--expected-signature",
        default="",
        help="Failure signature this run is expected to produce, recorded before the run.",
    )
    parser.add_argument("--baseline", default="", help="Frozen corpus JSON to compare against.")
    parser.add_argument("--output-root", default="evidence/phase3_2_1")
    args = parser.parse_args()

    objects = {name: OBJECT_BUILDERS[name]() for name in args.objects}
    provenance = collect_provenance(
        seed=args.seed,
        candidates=args.candidates,
        run_dynamic=args.run_dynamic,
        objects=objects,
    )

    cells: list[dict[str, Any]] = []
    for object_name in args.objects:
        for hand in args.hands:
            for recipe_id in args.recipes:
                logger.info("cell: %s x %s x %s", hand, recipe_id, object_name)
                cell = run_cell(
                    hand=hand,
                    recipe_id=recipe_id,
                    object_name=object_name,
                    object_spec=objects[object_name],
                    seed=args.seed,
                    candidates=args.candidates,
                    run_dynamic=args.run_dynamic,
                )
                logger.info("  signature: %s", cell["failure_signature"])
                cells.append(cell)

    corpus = {
        "schema": "qdgrasp/phase3-2-1-failure-corpus/v1",
        "label": args.label,
        "hypothesis": args.hypothesis,
        "intervention": args.intervention,
        "expected_signature": args.expected_signature,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance,
        "summary": summarize(cells),
        "cells_detail": cells,
    }

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        corpus["baseline_comparison"] = compare_to_baseline(baseline, corpus)
        corpus["baseline_path"] = args.baseline

    out_dir = REPO_ROOT / args.output_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corpus.json"
    out_path.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = corpus["summary"]
    logger.info("=== Characterization corpus '%s' ===", args.label)
    logger.info("  candidates:            %d", summary["total_candidates"])
    logger.info("  reached dynamic:       %d", summary["candidates_reaching_dynamic_rollout"])
    logger.info("  generated positives:   %d", summary["pipeline_generated_positives"])
    for sig, count in summary["failure_signature"].items():
        logger.info("  %-48s %d", sig, count)
    if "baseline_comparison" in corpus:
        changed = corpus["baseline_comparison"]["changed_cells"]
        logger.info("  cells changed vs baseline: %d", len(changed))
    logger.info("Corpus written to %s", out_path)


if __name__ == "__main__":
    main()
