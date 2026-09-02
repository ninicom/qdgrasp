"""Transmission layer for direct-drive and underactuated robotic hands."""

from .command import project_joint_delta_to_actuator_command
from .contracts import ActuatorCommand, TransmissionModel, TransmissionState
from .direct import DirectJointTransmission
from .fixed_tendon import FixedTendonTransmission
from .model import (
    compute_finite_difference_moment_matrix,
    create_transmission_model_from_spec_and_mjcf,
    extract_moment_matrix,
)

__all__ = [
    "ActuatorCommand",
    "DirectJointTransmission",
    "FixedTendonTransmission",
    "TransmissionModel",
    "TransmissionState",
    "compute_finite_difference_moment_matrix",
    "create_transmission_model_from_spec_and_mjcf",
    "extract_moment_matrix",
    "project_joint_delta_to_actuator_command",
]
