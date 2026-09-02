"""Pinned rollout protocol and measured dynamic success predicate (P3.2.1-09)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutProtocol:
    """Thresholds and sources that define one reproducible rollout verdict."""

    gains_source: str = "compiled_mjcf"
    timestep_source: str = "compiled_model"
    contact_window_fraction: float = 0.25
    minimum_contact_duty_cycle: float = 0.8
    minimum_contact_impulse_ratio: float = 0.5
    actuator_tracking_range_fraction: float = 0.10
    joint_tracking_range_fraction: float = 0.15
    palm_position_tolerance: float = 0.005
    palm_rotation_tolerance: float = 0.08726646259971647  # 5 degrees
    root_mocap_position_tolerance: float = 0.005
    lift_success_fraction: float = 0.5
    cone_tolerance: float = 1e-6

    def validation_error(self) -> str | None:
        """Return the first invalid controller/predicate threshold, if any."""
        finite_positive = {
            "contact_window_fraction": self.contact_window_fraction,
            "minimum_contact_duty_cycle": self.minimum_contact_duty_cycle,
            "minimum_contact_impulse_ratio": self.minimum_contact_impulse_ratio,
            "actuator_tracking_range_fraction": self.actuator_tracking_range_fraction,
            "joint_tracking_range_fraction": self.joint_tracking_range_fraction,
            "palm_position_tolerance": self.palm_position_tolerance,
            "palm_rotation_tolerance": self.palm_rotation_tolerance,
            "root_mocap_position_tolerance": self.root_mocap_position_tolerance,
            "lift_success_fraction": self.lift_success_fraction,
            "cone_tolerance": self.cone_tolerance,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                return name
        bounded_unit = {
            "contact_window_fraction": self.contact_window_fraction,
            "minimum_contact_duty_cycle": self.minimum_contact_duty_cycle,
            "minimum_contact_impulse_ratio": self.minimum_contact_impulse_ratio,
            "lift_success_fraction": self.lift_success_fraction,
        }
        for name, value in bounded_unit.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                return name
        if not self.gains_source or not self.timestep_source:
            return "protocol_source"
        return None


@dataclass(frozen=True)
class DynamicPredicateEvidence:
    stable: bool
    actuator_tracking_pass: bool
    palm_tracking_pass: bool
    active_contact_sustained: bool
    palm_support: bool
    floor_support_after_lift: bool
    penetration_pass: bool
    lift_pass: bool
    disturbance_survival_pass: bool
    friction_cone_pass: bool


def evaluate_dynamic_success(
    evidence: DynamicPredicateEvidence,
) -> tuple[bool, str]:
    """Return a fail-closed verdict with stable failure-stage precedence."""
    checks = (
        (evidence.stable, "simulation_instability"),
        (evidence.actuator_tracking_pass, "actuator_tracking"),
        (evidence.palm_tracking_pass, "palm_tracking"),
        (evidence.active_contact_sustained, "active_contact"),
        (not evidence.palm_support, "palm_support"),
        (not evidence.floor_support_after_lift, "floor_support"),
        (evidence.penetration_pass, "penetration"),
        (evidence.lift_pass, "lift"),
        (evidence.disturbance_survival_pass, "perturbation"),
        (evidence.friction_cone_pass, "friction_cone"),
    )
    for passed, failure_stage in checks:
        if not passed:
            return False, failure_stage
    return True, "none"
