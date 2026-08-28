#!/usr/bin/env python3
"""Release gate for the P3.2.1 full-pipeline correctness subphase."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.config import resolve_workload_hands
from qdgrasp.dataset.pipeline.canonical_full_flow import (
    build_canonical_full_flow_matrix,
)
from qdgrasp.dataset.pipeline.generated_reachable import (
    build_generated_reachable_object,
    generated_reachable_rng,
)
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.dataset.pipeline.validators.dynamic_predicate import RolloutProtocol
from qdgrasp.dataset.rng import get_generator
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

REPO_ROOT = Path(__file__).resolve().parents[1]
# Phase 3.2.1 predates ADR-0008 and its checker verifies what that phase
# published, which covered three hands. Reproducing that check is not new
# three-hand coverage, and it declares itself as such.
_SCOPE = resolve_workload_hands(
    ("leap_hand", "wonik_allegro", "shadow_hand"),
    experimental_shadow=True,
    purpose="re-checking the pre-ADR-0008 Phase 3.2.1 artifact as published",
)
HANDS = _SCOPE.hands
CONTRACT_TESTS = (
    "tests/test_no_positive_substitution.py",
    "tests/test_contact_state.py",
    "tests/test_active_mask_invariance.py",
    "tests/test_fixed_contact_dls.py",
    "tests/test_joint_palm_dls.py",
    "tests/test_solver_progress.py",
    "tests/test_proposal_identity.py",
    "tests/test_region_opposition.py",
    "tests/test_surface_fixed.py",
    "tests/test_palm_hypotheses.py",
    "tests/test_collision_admission.py",
    "tests/test_task_space_command.py",
    "tests/test_dynamic_predicate.py",
    "tests/test_physics_rollout.py::test_zero_actuated_damping_fails_before_mujoco_step",
    "tests/test_physics_rollout.py::test_task_command_with_too_few_active_fingers_fails_before_step",
    "tests/test_generated_reachable.py::test_generated_reachable_contains_geometry_but_no_grasp_oracle",
    "tests/test_canonical_full_flow.py",
)
PIPELINE_SOURCES = (
    "qdgrasp/dataset/pipeline/orchestrator.py",
    "qdgrasp/dataset/pipeline/contracts.py",
    "qdgrasp/dataset/pipeline/contact_state.py",
    "qdgrasp/dataset/pipeline/palm_hypotheses.py",
    "qdgrasp/dataset/pipeline/generated_reachable.py",
    "qdgrasp/dataset/pipeline/canonical_full_flow.py",
    "qdgrasp/dataset/pipeline/proposals/region_opposition.py",
    "qdgrasp/dataset/pipeline/proposals/width_mapper.py",
    "qdgrasp/dataset/pipeline/solvers/fixed_contact_dls.py",
    "qdgrasp/dataset/pipeline/solvers/region_dls.py",
    "qdgrasp/dataset/pipeline/solvers/joint_palm_dls.py",
    "qdgrasp/dataset/pipeline/certifiers/contact_force.py",
    "qdgrasp/dataset/pipeline/certifiers/grasp_wrench.py",
    "qdgrasp/dataset/pipeline/validators/collision_admission.py",
    "qdgrasp/dataset/pipeline/validators/dynamic_predicate.py",
    "qdgrasp/dataset/pipeline/validators/mujoco_rollout.py",
    "qdgrasp/robot/transmission/command.py",
    "scripts/check_phase3_2_1.py",
)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_scalar(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 12)
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh_hash(mesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def collect_provenance(canonical_candidates: int) -> dict[str, Any]:
    protocol = _json_scalar(dataclasses.asdict(RolloutProtocol()))
    canonical = build_canonical_full_flow_matrix()
    return {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_hashes": {
            name: _sha256_file(REPO_ROOT / name) for name in PIPELINE_SOURCES
        },
        "robot_profile_hashes": {
            hand: _sha256_file(
                REPO_ROOT / "qdgrasp" / "presets" / "robots" / f"{hand}.yaml"
            )
            for hand in HANDS
        },
        "generated_object_hashes": {
            hand: _mesh_hash(build_generated_reachable_object(hand).mesh)
            for hand in HANDS
        },
        "canonical_object_hashes": {
            item.name: _mesh_hash(item.mesh) for item in canonical
        },
        "rollout_protocol": protocol,
        "rollout_protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "seed": 42,
        "generated_candidate_budget": 16,
        "canonical_candidate_budget": canonical_candidates,
    }


def _outcome_payload(outcome) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recipe_id": outcome.recipe_id,
        "failure_stage": outcome.failure_stage,
        "failure_reason": outcome.failure_reason,
        "proposal_valid": outcome.proposal_valid,
        "ik_valid": outcome.ik_valid,
        "collision_valid": outcome.collision_valid,
        "static_force_valid": outcome.static_force_valid,
        "dynamic_valid": outcome.dynamic_valid,
    }
    if outcome.proposal is not None:
        payload["proposal_id"] = outcome.proposal.candidate_id
        payload["active_fingers"] = _json_scalar(outcome.proposal.active_fingers)
    if outcome.kinematics is not None:
        payload["palm_hypothesis_id"] = outcome.kinematics.palm_hypothesis_id
        payload["q"] = _json_scalar(outcome.kinematics.q)
        payload["palm_pos"] = _json_scalar(outcome.kinematics.palm_pos)
        payload["palm_rot"] = _json_scalar(outcome.kinematics.palm_rot)
        payload["position_residuals"] = _json_scalar(
            outcome.kinematics.position_residuals
        )
        payload["normal_residuals"] = _json_scalar(
            outcome.kinematics.normal_residuals
        )
    if outcome.collision_admission is not None:
        payload["collision"] = {
            "reason": outcome.collision_admission.reason,
            "max_penetration": outcome.collision_admission.max_penetration,
            "min_hand_floor_clearance": (
                outcome.collision_admission.min_hand_floor_clearance
            ),
        }
    if outcome.static_certificate is not None:
        payload["static"] = {
            "passed": outcome.static_certificate.passed,
            "cone_residual": outcome.static_certificate.cone_residual,
            "quality_margin": outcome.static_certificate.quality_margin,
            "object_wrench": outcome.static_certificate.object_wrench,
        }
    if outcome.dynamic_validation is not None:
        payload["dynamic"] = {
            "passed": outcome.dynamic_validation.passed,
            "failure_stage": outcome.dynamic_validation.failure_stage,
            "trajectory_metrics": outcome.dynamic_validation.trajectory_metrics,
            "per_finger_loads": outcome.dynamic_validation.per_finger_loads,
        }
    return _json_scalar(payload)


def _cell_payload(hand: str, object_name: str, outcomes, accounting) -> dict[str, Any]:
    signature = Counter(
        f"{outcome.failure_stage}:{outcome.failure_reason}" for outcome in outcomes
    )
    return {
        "hand": hand,
        "object": object_name,
        "stage_accounting": dict(sorted(accounting.items())),
        "failure_signature": dict(sorted(signature.items())),
        "outcomes": [_outcome_payload(outcome) for outcome in outcomes],
    }


def run_generated_matrix() -> list[dict[str, Any]]:
    cells = []
    for hand in HANDS:
        obj = build_generated_reachable_object(hand)
        spec = RobotSpec.from_config(f"{hand}.yaml", sample_anchors=False)
        outcomes, accounting = run_pipeline_chunk(
            recipe_id="region_opposition_v1",
            spec=spec,
            mesh=obj.mesh,
            collision_geoms=obj.collision_geoms,
            hand_xml_path=resolve_robot_asset(spec.config.source_asset),
            rng=generated_reachable_rng(hand),
            num_candidates=obj.candidate_budget,
            object_mass=obj.mass,
            object_pos=obj.object_pos,
            run_dynamic=True,
        )
        if accounting["accepted"] < 1:
            raise AssertionError(f"generated-reachable has no full positive for {hand}")
        cells.append(_cell_payload(hand, "generated_reachable", outcomes, accounting))
        print(f"generated {hand}: accepted={accounting['accepted']}", flush=True)
    return cells


def run_canonical_matrix(candidate_budget: int = 8) -> list[dict[str, Any]]:
    cells = []
    for hand in HANDS:
        spec = RobotSpec.from_config(f"{hand}.yaml", sample_anchors=False)
        xml_path = resolve_robot_asset(spec.config.source_asset)
        for obj in build_canonical_full_flow_matrix():
            outcomes, accounting = run_pipeline_chunk(
                recipe_id="region_opposition_v1",
                spec=spec,
                mesh=obj.mesh,
                collision_geoms=obj.collision_geoms,
                hand_xml_path=xml_path,
                rng=get_generator(42, "p3.2.1", "canonical", hand, obj.name),
                num_candidates=candidate_budget,
                object_mass=obj.mass,
                object_pos=obj.object_pos,
                run_dynamic=True,
            )
            cells.append(_cell_payload(hand, obj.name, outcomes, accounting))
            print(
                f"canonical {hand}/{obj.name}: accepted={accounting['accepted']}",
                flush=True,
            )
    if len(cells) != len(HANDS) * 4:
        raise AssertionError("canonical full-flow matrix is incomplete")
    return cells


def _write_manifest(path: Path, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded + b"\n")
    return hashlib.sha256(encoded).hexdigest()


def _run_contract_tests() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *CONTRACT_TESTS],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--skip-contract-tests", action="store_true")
    parser.add_argument("--canonical-candidates", type=int, default=8)
    args = parser.parse_args()
    if args.canonical_candidates < 1:
        parser.error("--canonical-candidates must be positive")

    if not args.skip_contract_tests:
        _run_contract_tests()

    temporary = None
    if args.staging_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="qdgrasp-p3-2-1-")
        staging = Path(temporary.name)
    else:
        staging = args.staging_dir.resolve()

    provenance = collect_provenance(args.canonical_candidates)
    run_a = {
        "schema": "qdgrasp/p3.2.1-staging/v1",
        "provenance": provenance,
        "generated": run_generated_matrix(),
    }
    run_b = {
        "schema": "qdgrasp/p3.2.1-staging/v1",
        "provenance": provenance,
        "generated": run_generated_matrix(),
    }
    hash_a = _write_manifest(staging / "run-a" / "manifest.json", run_a)
    hash_b = _write_manifest(staging / "run-b" / "manifest.json", run_b)
    if hash_a != hash_b or run_a != run_b:
        raise AssertionError("two-run generated regeneration is not deterministic")

    canonical = run_canonical_matrix(args.canonical_candidates)
    canonical_payload = {
        "schema": "qdgrasp/p3.2.1-canonical-evidence/v1",
        "provenance": provenance,
        "candidate_budget": args.canonical_candidates,
        "cells": canonical,
    }
    canonical_hash = _write_manifest(
        staging / "canonical" / "manifest.json", canonical_payload
    )
    summary = {
        "generated_positive_hands": [cell["hand"] for cell in run_a["generated"]],
        "generated_two_run_sha256": hash_a,
        "canonical_cells": len(canonical),
        "canonical_sha256": canonical_hash,
        "staging_dir": str(staging),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
