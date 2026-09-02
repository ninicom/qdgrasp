"""Regression guard for P3.2.1-00: the generator may not fabricate positives.

REV-20260823-009 found `scripts/generate_dgn_open_tiny.py` replacing a real
pipeline outcome with a hand-built positive (hardcoded joint states, a
hand-assembled `KinematicSolution`/`StaticCertificate`, and a rollout run
outside `run_pipeline_chunk`).  Every sample written to a shard must instead
come from the outcome list the orchestrator returned, unmodified.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from qdgrasp.dataset.pipeline.contracts import PipelineOutcome

GENERATOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_dgn_open_tiny.py"
GENERATOR_SOURCE = GENERATOR_PATH.read_text(encoding="utf-8")
GENERATOR_TREE = ast.parse(GENERATOR_SOURCE)


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_dgn_open_tiny", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in generator")


def test_generator_does_not_import_evidence_fabrication_machinery() -> None:
    """The generator must not build kinematic/static/dynamic evidence itself."""
    forbidden = {
        "validate_grasp_rollout",
        "solve_dls_ik_batch",
        "solve_region_dls_ik_batch",
        "KinematicSolution",
        "StaticCertificate",
        "DynamicValidation",
        "ContactProposal",
    }
    leaked = forbidden & _imported_names(GENERATOR_TREE)
    assert not leaked, (
        f"generator imports evidence-construction symbols {sorted(leaked)}; "
        "all stage evidence must originate inside run_pipeline_chunk"
    )


def test_generator_never_mutates_the_outcome_list() -> None:
    """No element of the orchestrator's outcome list may be replaced or appended."""
    for node in ast.walk(GENERATOR_TREE):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "outcomes"
                ):
                    raise AssertionError(
                        f"generator assigns into outcomes[...] at line {node.lineno}"
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "outcomes"
            and node.func.attr in {"append", "insert", "extend", "pop", "remove", "clear"}
        ):
            raise AssertionError(
                f"generator calls outcomes.{node.func.attr}() at line {node.lineno}"
            )


def test_every_shard_sample_comes_from_outcome_to_sample() -> None:
    """`samples` may only be appended from an `outcome_to_sample(...)` result."""
    generate = _function(GENERATOR_TREE, "generate_tiny_dataset")
    appends = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "samples"
    ]
    assert appends, "generator no longer appends to samples; test needs updating"
    assigned_from_conversion: set[str] = set()
    for node in ast.walk(generate):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "outcome_to_sample"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_from_conversion.add(target.id)
    for call in appends:
        (arg,) = call.args
        origin_ok = (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Name)
            and arg.func.id == "outcome_to_sample"
        ) or (isinstance(arg, ast.Name) and arg.id in assigned_from_conversion)
        assert origin_ok, (
            f"samples.append at line {call.lineno} does not carry an "
            "outcome_to_sample() result"
        )


def test_generator_defines_no_hardcoded_joint_state_vectors() -> None:
    """Hand-built joint vectors were how the fabricated positive was seeded."""
    def _is_number(node: ast.expr) -> bool:
        # Negative literals parse as UnaryOp(USub, Constant), not Constant.
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            node = node.operand
        return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))

    for node in ast.walk(GENERATOR_TREE):
        if isinstance(node, ast.List) and len(node.elts) >= 8 and all(_is_number(elt) for elt in node.elts):
            raise AssertionError(
                f"generator contains a hardcoded numeric vector of length "
                f"{len(node.elts)} at line {node.lineno}"
            )


def test_positive_count_tracks_measured_dynamic_validity() -> None:
    """Shard positive counts must be read off the serialized dynamic verdict."""
    generate = _function(GENERATOR_TREE, "generate_tiny_dataset")
    increments = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "positives"
    ]
    assert increments, "positive counting removed from generator"
    for node in increments:
        source = ast.get_source_segment(GENERATOR_SOURCE, node.value) or ""
        assert "dynamic_valid" in source, (
            f"positives incremented from {source!r}, not from the measured "
            "dynamic verdict"
        )


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        actuated_joint_names=("j0", "j1"),
        fingertip_links=("tip0", "tip1"),
    )


def test_outcome_to_sample_rejects_a_positive_without_rollout_evidence() -> None:
    """A dynamic-valid outcome with no passing rollout must not serialize."""
    module = _load_generator()
    outcome = PipelineOutcome(
        proposal_valid=True,
        ik_valid=True,
        collision_valid=True,
        static_force_valid=True,
        dynamic_valid=True,
        failure_stage="none",
        failure_reason="",
        recipe_id="wrench_guided_v1",
    )
    with pytest.raises(RuntimeError, match="lacks passing rollout evidence"):
        module.outcome_to_sample(
            outcome,
            spec=_spec(),
            mesh=trimesh.creation.box(extents=(0.05, 0.05, 0.05)),
            rng=np.random.default_rng(0),
            object_id="prim_box_01",
            robot_name="leap_hand",
            recipe_id="wrench_guided_v1",
        )
