from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from qdgrasp.config import ConfigError
from qdgrasp.dataset.pipeline.contracts import PipelineOutcome


_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_dgn_open_tiny",
    Path(__file__).resolve().parents[1] / "scripts" / "generate_dgn_open_tiny.py",
)
assert _GENERATOR_SPEC is not None and _GENERATOR_SPEC.loader is not None
_GENERATOR = importlib.util.module_from_spec(_GENERATOR_SPEC)
_GENERATOR_SPEC.loader.exec_module(_GENERATOR)
outcome_to_sample = _GENERATOR.outcome_to_sample


def _spec():
    return SimpleNamespace(
        actuated_joint_names=("j0", "j1"),
        fingertip_links=("tip0", "tip1"),
    )


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


def test_generator_rejects_blocked_robot_before_creating_output(tmp_path: Path):
    output = tmp_path / "release"

    with pytest.raises(ConfigError, match="fixed-tendon underactuation"):
        _GENERATOR.generate_tiny_dataset(
            output_dir=output,
            samples_per_pair=1,
            recipe_id="surface_fixed_v1",
        )

    assert not output.exists()
