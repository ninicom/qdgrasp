"""MVP-01: the pinch prior is reproducible and fitted only on train sizes."""

from __future__ import annotations

import numpy as np
import pytest

from qdgrasp.mvp.config import MvpScopeConfig, load_mvp_scope
from qdgrasp.mvp.prior import (
    DEFAULT_PRIOR_PATH,
    PinchPriorTable,
    build_pinch_prior_table,
)
from qdgrasp.robot.spec import RobotSpec


@pytest.fixture(scope="module")
def scope() -> MvpScopeConfig:
    return load_mvp_scope()


@pytest.fixture(scope="module")
def prior() -> PinchPriorTable:
    return PinchPriorTable.load(DEFAULT_PRIOR_PATH)


def test_prior_knots_cover_exactly_the_train_widths(scope: MvpScopeConfig, prior: PinchPriorTable) -> None:
    fitted = sorted(knot.half_width for knot in prior.knots)
    expected = sorted(variant.half_width for variant in scope.train_variants)
    assert fitted == pytest.approx(expected)
    held_out = {variant.half_width for variant in scope.heldout_variants}
    assert not held_out & set(fitted), "a held-out width was fitted into the prior"


def test_prior_document_round_trips(prior: PinchPriorTable) -> None:
    rebuilt = PinchPriorTable.from_document(prior.to_document())
    assert rebuilt.content_hash() == prior.content_hash()


def test_prior_refits_to_the_committed_table(scope: MvpScopeConfig, prior: PinchPriorTable) -> None:
    """Re-running the fit reproduces the shipped artifact to solver tolerance."""

    spec = RobotSpec.from_config(scope.robot_profile, sample_anchors=False)
    refitted = build_pinch_prior_table(spec, [variant.half_width for variant in scope.train_variants])
    for fresh, stored in zip(refitted.knots, prior.knots):
        assert fresh.half_width == pytest.approx(stored.half_width)
        np.testing.assert_allclose(fresh.open_q, stored.open_q, atol=1e-4)
        np.testing.assert_allclose(fresh.squeeze_q, stored.squeeze_q, atol=1e-4)


def test_interpolation_is_exact_at_knots_and_clamped_outside(prior: PinchPriorTable) -> None:
    for knot in prior.knots:
        command = prior.command(knot.half_width)
        np.testing.assert_allclose(command.open_q, knot.open_q, atol=1e-9)
        np.testing.assert_allclose(command.squeeze_q, knot.squeeze_q, atol=1e-9)
    widest = prior.knots[-1]
    np.testing.assert_allclose(prior.command(widest.half_width + 0.01).squeeze_q, widest.squeeze_q, atol=1e-9)
    narrowest = prior.knots[0]
    np.testing.assert_allclose(
        prior.command(max(narrowest.half_width - 0.01, 1e-4)).open_q, narrowest.open_q, atol=1e-9
    )


def test_interpolated_command_lies_between_its_neighbours(prior: PinchPriorTable) -> None:
    left, right = prior.knots[1], prior.knots[2]
    middle = prior.command(0.5 * (left.half_width + right.half_width))
    expected = 0.5 * (left.squeeze_q + right.squeeze_q)
    np.testing.assert_allclose(middle.squeeze_q, expected, atol=1e-9)


def test_synergy_directions_are_unit_and_disjoint(prior: PinchPriorTable) -> None:
    directions = prior.synergy_directions()
    assert directions.shape == (2, len(prior.joint_names))
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-9)
    assert float(np.abs(directions[0] @ directions[1])) < 1e-9
