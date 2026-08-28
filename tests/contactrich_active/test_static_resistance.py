"""WRK-R1/WRK-R2: a frozen criterion with units, and a controlled comparison.

RRV-03 found the old section 16.3 metric comparing a normalized unit-primitive
GWS margin against a raw wrench norm mixing N and Nm. These tests pin the
properties that make the replacement mean something: alpha is dimensionless, so
it survives a change of length unit; it moves the right way with the force
budget and the disturbance; and it is refused outright when there is no force cap
to make it physical.
"""

from __future__ import annotations

import numpy as np
import pytest

from qdgrasp.dataset.pipeline.certifiers.static_resistance import (
    EQUILIBRIUM_TOLERANCE,
    certify_static_resistance,
)


def _four_contact_grasp(scale: float = 1.0):
    points = np.array(
        [[0.02, 0, 0.02], [-0.02, 0, 0.02], [0, 0.02, 0.02], [0, -0.02, 0.02]]
    ) * scale
    centroid = np.array([0.0, 0.0, 0.02]) * scale
    offsets = centroid - points
    normals = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    return points, normals, centroid


def _certify(scale=1.0, *, mu=0.5, limit=25.0, wrench=None, mass=0.02):
    points, normals, centroid = _four_contact_grasp(scale)
    if wrench is None:
        wrench = np.array([0.1, 0.1, 0.0, 0.001, 0.001, 0.001])
    return certify_static_resistance(
        points,
        normals,
        centroid,
        mass=mass,
        mu=mu,
        disturbance_wrench=wrench,
        force_limits=limit,
        characteristic_length=0.04 * scale,
    )


def test_a_solvable_grasp_reports_an_equilibrium():
    certificate = _certify()
    assert certificate.status == "solved"
    assert certificate.equilibrium_residual < EQUILIBRIUM_TOLERANCE
    assert certificate.alpha > 0.0


def test_alpha_survives_a_change_of_length_unit():
    """Dimensionless means dimensionless: rescale the scene, keep the answer."""
    base = _certify(scale=1.0)
    wrench = np.array([0.1, 0.1, 0.0, 0.001, 0.001, 0.001])
    scaled_wrench = np.concatenate([wrench[:3], wrench[3:] * 10.0])
    scaled = _certify(scale=10.0, wrench=scaled_wrench)
    assert scaled.alpha == pytest.approx(base.alpha, rel=1e-9)


def test_alpha_rises_with_the_force_budget():
    alphas = [_certify(limit=limit).alpha for limit in (1.0, 5.0, 25.0, 100.0)]
    assert alphas == sorted(alphas)
    assert alphas[0] < alphas[-1]


def test_alpha_falls_as_the_disturbance_grows():
    wrench = np.array([0.1, 0.1, 0.0, 0.001, 0.001, 0.001])
    alphas = [_certify(wrench=wrench * k).alpha for k in (0.5, 1.0, 2.0, 4.0)]
    assert alphas == sorted(alphas, reverse=True)
    # alpha is a multiple of the disturbance, so doubling it halves the multiple.
    assert alphas[2] == pytest.approx(alphas[1] / 2.0, rel=1e-6)


def test_without_a_force_cap_there_is_no_certificate():
    """An unbounded LP squeezes as hard as it likes and certifies anything."""
    certificate = _certify(limit=0.0)
    assert certificate.status == "no_force_bound"
    assert not certificate.passed


def test_a_zero_disturbance_has_no_multiple():
    certificate = _certify(wrench=np.zeros(6))
    assert certificate.status == "no_disturbance"
    assert not certificate.passed


def test_no_contacts_resist_nothing():
    certificate = certify_static_resistance(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros(3),
        mass=0.02,
        mu=0.5,
        disturbance_wrench=np.array([0.1, 0.1, 0.0, 0.001, 0.001, 0.001]),
        force_limits=25.0,
        characteristic_length=0.04,
    )
    assert certificate.status == "no_contacts"
    assert not certificate.passed


def test_the_pass_line_is_one_times_the_declared_disturbance():
    """alpha >= 1 means it holds what it will actually meet, and nothing weaker."""
    strong = _certify(limit=100.0)
    assert strong.passed
    huge = np.array([1e4, 1e4, 0.0, 1e2, 1e2, 1e2])
    weak = _certify(limit=1.0, wrench=huge)
    assert weak.alpha < 1.0
    assert not weak.passed


def test_both_arms_fork_from_one_snapshot_and_differ_in_one_factor():
    """WRK-R2: a controlled comparison differs in physics_mode and nothing else."""
    from qdgrasp.dataset.pipeline.candidate_snapshot import (
        CandidateSnapshot,
        one_factor_diff,
    )

    snapshot = CandidateSnapshot(
        hand="leap_hand",
        scene="table/sparse",
        seed=0,
        object_mass_kg=0.02,
        friction_mu=0.5,
        torsional_friction=0.005,
        characteristic_length_m=0.04,
        horizon_steps=300,
        applied_wrench=(0.1, 0.1, 0.0, 0.001, 0.001, 0.001),
        applied_wrench_hash="deadbeef",
        contact_points=((0.02, 0.0, 0.02),),
        contact_normals=((-1.0, 0.0, 0.0),),
        centroid=(0.0, 0.0, 0.02),
        force_limit_N=25.0,
        safety_budget_id="leap_hand/table",
        recipe_id="release_grasp",
    )
    frozen = snapshot.fork("frozen")
    reactive = snapshot.fork("reactive")
    assert frozen["snapshot_hash"] == reactive["snapshot_hash"]
    assert one_factor_diff(frozen, reactive) == ("physics_mode",)

    with pytest.raises(ValueError):
        snapshot.fork("something_else")


def test_the_disturbance_resolver_is_shared_not_copied():
    """WRK-R2: one policy, one implementation.

    The ablation used to re-derive this beside the validator's copy, and the
    copies drifted silently: one of them scored a hand at zero disturbance while
    the other was disturbing it.
    """
    import inspect

    from qdgrasp.dataset.pipeline.validators import mujoco_rollout
    from qdgrasp.dataset.pipeline.validators.disturbance import (
        resolve_perturbation_wrench,
    )

    source = inspect.getsource(mujoco_rollout)
    assert "resolve_perturbation_wrench(" in source
    assert "0.25 * object_weight" not in source, "the validator kept its own copy"

    class _Geom:
        size = np.array([0.02, 0.02, 0.02])

    derived = resolve_perturbation_wrench(None, object_mass=0.02, collision_geoms=[_Geom()])
    assert np.linalg.norm(derived) > 0.0, "a recipe naming no wrench is still disturbed"
    declared = resolve_perturbation_wrench(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), object_mass=0.02, collision_geoms=[_Geom()]
    )
    assert declared[0] == pytest.approx(1.0)
