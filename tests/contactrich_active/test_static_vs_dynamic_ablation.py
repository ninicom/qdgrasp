"""The amended section 16.3 criterion: a frozen test that fails on mechanics.

These are invariants, not a snapshot of today's numbers. A test that asserted
"leap_hand yields a pair" would go red the moment a recipe changed, and would be
asserting a project state rather than a property.
"""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import numpy as np
import pytest

from qdgrasp.dataset.pipeline.certifiers.contact_force import certify_force_closure

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ablation():
    spec = importlib.util.spec_from_file_location(
        "phase3_4_3_ablation", REPO_ROOT / "scripts" / "phase3_4_3_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _antipodal_pinch():
    """Two opposed contacts on a small object: closed, but barely."""
    points = np.array([[0.014, 0.0, 0.02], [-0.014, 0.0, 0.02]])
    centroid = np.array([0.0, 0.0, 0.02])
    offsets = centroid - points
    normals = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    return points, normals, centroid


def test_zero_threshold_is_the_historical_behaviour():
    """65 requirements passed under the old certifier; the default must not move.

    If this fails, every piece of evidence gathered before the amendment is
    invalidated, which is precisely what the default of 0.0 exists to prevent.
    """
    points, normals, centroid = _antipodal_pinch()
    without = certify_force_closure(points, normals, centroid, mass=0.02)
    with_zero = certify_force_closure(
        points, normals, centroid, mass=0.02, quality_margin_threshold=0.0
    )
    assert without.passed == with_zero.passed
    assert without.quality_margin == pytest.approx(with_zero.quality_margin)


def test_a_threshold_above_the_margin_refuses_the_grasp():
    points, normals, centroid = _antipodal_pinch()
    baseline = certify_force_closure(points, normals, centroid, mass=0.02)
    assert baseline.passed, "the pinch should clear the positive-margin test"

    strict = certify_force_closure(
        points,
        normals,
        centroid,
        mass=0.02,
        quality_margin_threshold=baseline.quality_margin * 2.0,
    )
    assert not strict.passed
    # The margin still travels, so a caller can see how far short it fell.
    assert strict.quality_margin == pytest.approx(baseline.quality_margin)


def test_a_threshold_below_the_margin_leaves_the_grasp_alone():
    points, normals, centroid = _antipodal_pinch()
    baseline = certify_force_closure(points, normals, centroid, mass=0.02)
    lenient = certify_force_closure(
        points,
        normals,
        centroid,
        mass=0.02,
        quality_margin_threshold=baseline.quality_margin * 0.5,
    )
    assert lenient.passed == baseline.passed


@pytest.mark.parametrize("hand", ["leap_hand", "wonik_allegro"])
def test_every_active_hand_faces_a_real_disturbance(hand: str):
    """A recipe that names no wrench is still disturbed, and must score as such.

    Reading only ``rollout_kwargs`` scores such a hand at zero and reports that
    the frozen test passed, when the protocol had been disturbing it all along.
    That mistake was made once; this keeps it made only once.
    """
    ablation = _load_ablation()
    generator = runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_contactrich_active_tiny.py"),
        run_name="ablation_generator",
    )
    threshold = ablation.declared_disturbance(hand, generator)
    assert threshold > 0.0, f"{hand} would be certified against no disturbance at all"


def test_the_derived_wrench_matches_the_validator_formula():
    """The threshold is derived from the protocol, never chosen by hand."""
    ablation = _load_ablation()
    generator = runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_contactrich_active_tiny.py"),
        run_name="ablation_generator",
    )
    recipe = generator["build_release_grasp_recipe"](
        generator["profile_of_hand"]("leap_hand")
    )
    assert "perturbation_wrench" not in recipe.rollout_kwargs

    mass = float(recipe.rollout_kwargs.get("object_mass", ablation.TARGET_MASS_KG))
    weight = mass * 9.81
    length = max(
        2.0 * float(np.max(np.asarray(geom.size, dtype=np.float64)))
        for geom in recipe.target_geoms
    )
    expected = float(
        np.linalg.norm(
            np.array(
                [
                    0.5 * weight,
                    0.5 * weight,
                    0.0,
                    0.25 * weight * length,
                    0.25 * weight * length,
                    0.25 * weight * length,
                ]
            )
        )
    )
    assert ablation.declared_disturbance("leap_hand", generator) == pytest.approx(expected)
