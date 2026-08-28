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

    Superseded WRK-R1: the threshold this used to check was dimensionally
    incommensurable with the margin it was compared against (RRV-03), so the
    check now runs against the snapshot the two arms actually share.
    """
    ablation = _load_ablation()
    generator = runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_contactrich_active_tiny.py"),
        run_name="ablation_generator",
    )
    snapshot = ablation.build_snapshot(hand, generator, mass=0.02)
    assert np.linalg.norm(np.array(snapshot.applied_wrench)) > 0.0, (
        f"{hand} would be certified against no disturbance at all"
    )
    assert snapshot.applied_wrench_hash


def test_the_two_arms_share_one_snapshot_at_every_sweep_point():
    """RRV-04: moving the sweep axis must move both sides of the comparison.

    The defect was a sweep that varied the dynamic mass while holding the static
    threshold at the original one, so the arms described different experiments
    at every point but the first.
    """
    ablation = _load_ablation()
    generator = runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_contactrich_active_tiny.py"),
        run_name="ablation_generator",
    )
    first = ablation.build_snapshot("leap_hand", generator, mass=0.02)
    heavier = ablation.build_snapshot("leap_hand", generator, mass=0.40)

    assert first.object_mass_kg != heavier.object_mass_kg
    assert first.digest() != heavier.digest(), "the snapshot must track the mass"
    assert first.applied_wrench_hash != heavier.applied_wrench_hash, (
        "a derived disturbance must follow the mass it is derived from"
    )


def test_the_resistance_arm_reads_its_disturbance_from_the_snapshot():
    ablation = _load_ablation()
    generator = runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_contactrich_active_tiny.py"),
        run_name="ablation_generator",
    )
    snapshot = ablation.build_snapshot("leap_hand", generator, mass=0.02)
    arm = ablation.resistance_arm(snapshot)
    assert arm["physics_mode"] == "frozen"
    assert arm["snapshot_hash"] == snapshot.digest()
    assert arm["force_limit_N"] > 0.0
    assert arm["status"] == "solved"
