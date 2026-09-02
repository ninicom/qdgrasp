from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from qdgrasp.dataset.pipeline.contracts import KinematicSolution, PipelineOutcome

_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_dgn_open_tiny",
    Path(__file__).resolve().parents[1] / "scripts" / "generate_dgn_open_tiny.py",
)
assert _GENERATOR_SPEC is not None and _GENERATOR_SPEC.loader is not None
_GENERATOR = importlib.util.module_from_spec(_GENERATOR_SPEC)
_GENERATOR_SPEC.loader.exec_module(_GENERATOR)
outcome_to_sample = _GENERATOR.outcome_to_sample
loaded_qdgrasp_source_hashes = _GENERATOR.loaded_qdgrasp_source_hashes


def _spec():
    return SimpleNamespace(
        actuated_joint_names=("j0", "j1"),
        fingertip_links=("tip0", "tip1"),
    )


def test_generator_provenance_covers_effective_split_object_and_robot_sources() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hashes = loaded_qdgrasp_source_hashes(repo_root)

    assert {
        "scripts/generate_dgn_open_tiny.py",
        "qdgrasp/dataset/split.py",
        "qdgrasp/dataset/render.py",
        "qdgrasp/objects/generate.py",
        "qdgrasp/robot/spec.py",
        "qdgrasp/dataset/pipeline/orchestrator.py",
    } <= set(hashes)


def test_static_pass_dynamic_fail_remains_negative_and_points_use_object_frame():
    outcome = PipelineOutcome(
        proposal_valid=True,
        ik_valid=True,
        collision_valid=True,
        static_force_valid=True,
        dynamic_valid=False,
        failure_stage="dynamic_squeeze",
        failure_reason="rollout_failed_squeeze",
        recipe_id="surface_fixed_v1",
    )
    mesh = trimesh.creation.box(extents=(0.05, 0.05, 0.05))

    sample = outcome_to_sample(
        outcome,
        spec=_spec(),
        mesh=mesh,
        rng=np.random.default_rng(42),
        object_id="box",
        robot_name="mock",
        recipe_id="surface_fixed_v1",
    )

    assert float(sample["success"]) == 0.0
    assert float(sample["quality"]) == 0.0
    assert not sample["dynamic_valid"]
    assert not sample["kinematics_valid"]
    assert not sample["pose_target_valid"]
    assert not sample["joint_target_valid"]
    assert not sample["fk_target_valid"]
    assert sample["frame"] == "object"
    points = sample["points"].numpy()
    assert np.all(np.abs(points) <= 0.025001)
    assert np.all(np.min(np.abs(np.abs(points) - 0.025), axis=1) < 1e-5)


def test_generator_rejects_dynamic_positive_without_rollout_evidence():
    malformed = PipelineOutcome(
        proposal_valid=True,
        ik_valid=True,
        collision_valid=True,
        static_force_valid=True,
        dynamic_valid=True,
        failure_stage="none",
        failure_reason="passed",
        recipe_id="surface_fixed_v1",
    )

    with pytest.raises(RuntimeError, match="lacks passing rollout evidence"):
        outcome_to_sample(
            malformed,
            spec=_spec(),
            mesh=trimesh.creation.box(extents=(0.05, 0.05, 0.05)),
            rng=np.random.default_rng(42),
            object_id="box",
            robot_name="mock",
            recipe_id="surface_fixed_v1",
        )


def test_nonconverged_solver_output_remains_an_explicit_measured_target():
    kinematics = KinematicSolution(
        q=np.zeros(2),
        palm_pos=np.array([0.01, 0.02, 0.03]),
        palm_rot=np.eye(3),
        achieved_contacts=np.zeros((2, 3)),
        achieved_normals=np.zeros((2, 3)),
        position_residuals=np.ones(2),
        normal_residuals=np.ones(2),
        converged=np.array(False),
        reason=np.array("max_iter"),
    )
    outcome = PipelineOutcome(
        proposal_valid=True,
        ik_valid=False,
        collision_valid=False,
        static_force_valid=False,
        dynamic_valid=False,
        failure_stage="ik",
        failure_reason="max_iter",
        recipe_id="surface_fixed_v1",
        kinematics=kinematics,
    )

    sample = outcome_to_sample(
        outcome,
        spec=_spec(),
        mesh=trimesh.creation.box(extents=(0.05, 0.05, 0.05)),
        rng=np.random.default_rng(42),
        object_id="box",
        robot_name="mock",
        recipe_id="surface_fixed_v1",
    )

    assert not sample["ik_valid"]
    assert all(
        sample[name] for name in ("kinematics_valid", "pose_target_valid", "joint_target_valid", "fk_target_valid")
    )
