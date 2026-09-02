"""Negative mutations for the P3.2.1-09 measured dynamic predicate."""

from __future__ import annotations

import dataclasses

import pytest

from qdgrasp.dataset.pipeline.validators.dynamic_predicate import (
    DynamicPredicateEvidence,
    RolloutProtocol,
    evaluate_dynamic_success,
)

BASELINE = DynamicPredicateEvidence(
    stable=True,
    actuator_tracking_pass=True,
    palm_tracking_pass=True,
    active_contact_sustained=True,
    palm_support=False,
    floor_support_after_lift=False,
    penetration_pass=True,
    lift_pass=True,
    disturbance_survival_pass=True,
    friction_cone_pass=True,
)


def test_measured_baseline_passes_independently_of_scenario_name() -> None:
    assert evaluate_dynamic_success(BASELINE) == (True, "none")


@pytest.mark.parametrize(
    ("mutation", "value", "stage"),
    [
        ("stable", False, "simulation_instability"),
        ("actuator_tracking_pass", False, "actuator_tracking"),
        ("palm_tracking_pass", False, "palm_tracking"),
        ("active_contact_sustained", False, "active_contact"),
        ("palm_support", True, "palm_support"),
        ("floor_support_after_lift", True, "floor_support"),
        ("penetration_pass", False, "penetration"),
        ("lift_pass", False, "lift"),
        ("disturbance_survival_pass", False, "perturbation"),
        ("friction_cone_pass", False, "friction_cone"),
    ],
)
def test_each_negative_mutation_fails_at_its_own_stage(
    mutation: str, value: bool, stage: str
) -> None:
    verdict = evaluate_dynamic_success(dataclasses.replace(BASELINE, **{mutation: value}))
    assert verdict == (False, stage)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contact_window_fraction", 0.0),
        ("minimum_contact_duty_cycle", float("nan")),
        ("palm_position_tolerance", 0.0),
        ("lift_success_fraction", 1.1),
        ("cone_tolerance", float("inf")),
        ("gains_source", ""),
    ],
)
def test_invalid_rollout_protocol_mutations_fail_admission(field, value):
    protocol = dataclasses.replace(RolloutProtocol(), **{field: value})
    assert protocol.validation_error() is not None
