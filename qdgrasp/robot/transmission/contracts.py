"""Typed contracts for robot transmission states, commands, and models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import mujoco


@dataclass(frozen=True)
class TransmissionState:
    """Snapshot of kinematic joint states and corresponding actuator coordinates."""

    joint_names: Tuple[str, ...]
    actuator_names: Tuple[str, ...]
    joint_position: np.ndarray  # [B, J] or [J]
    actuator_coordinate: np.ndarray  # [B, U] or [U] (joint angle or tendon length)
    moment_matrix: np.ndarray  # [B, U, J] or [U, J] (dl/dq in spec named joint order)
    rank: np.ndarray  # [B] or scalar int

    def __post_init__(self) -> None:
        j_pos = np.asarray(self.joint_position)
        a_coord = np.asarray(self.actuator_coordinate)
        m_mat = np.asarray(self.moment_matrix)
        J = len(self.joint_names)
        U = len(self.actuator_names)

        if j_pos.ndim == 1:
            if j_pos.shape[0] != J:
                raise ValueError(f"joint_position length {j_pos.shape[0]} != joint_names {J}")
        elif j_pos.ndim == 2:
            if j_pos.shape[1] != J:
                raise ValueError(f"joint_position shape {j_pos.shape} incompatible with joint_names {J}")
        else:
            raise ValueError(f"joint_position must be 1D or 2D, got shape {j_pos.shape}")

        if a_coord.ndim == 1:
            if a_coord.shape[0] != U:
                raise ValueError(f"actuator_coordinate length {a_coord.shape[0]} != actuator_names {U}")
        elif a_coord.ndim == 2:
            if a_coord.shape[1] != U:
                raise ValueError(f"actuator_coordinate shape {a_coord.shape} incompatible with actuator_names {U}")
        else:
            raise ValueError(f"actuator_coordinate must be 1D or 2D, got shape {a_coord.shape}")

        if m_mat.ndim == 2:
            if m_mat.shape != (U, J):
                raise ValueError(f"moment_matrix shape {m_mat.shape} != ({U}, {J})")
        elif m_mat.ndim == 3:
            if m_mat.shape[1:] != (U, J):
                raise ValueError(f"moment_matrix shape {m_mat.shape} != (B, {U}, {J})")
        else:
            raise ValueError(f"moment_matrix must be 2D or 3D, got shape {m_mat.shape}")


@dataclass(frozen=True)
class ActuatorCommand:
    """Actuator-space control command with projection residuals and saturation diagnostics."""

    control_target: np.ndarray  # [B, U] or [U]
    projected_joint_delta: np.ndarray  # [B, J] or [J]
    controllable_residual: np.ndarray  # [B] or scalar float
    nullspace_residual: np.ndarray  # [B] or scalar float
    saturated: np.ndarray  # [B, U] or [U] (bool)
    reason: np.ndarray  # [B] or str ("converged", "nullspace_rejection", "actuator_saturation")


class TransmissionModel(ABC):
    """Abstract interface for robot transmission mapping between joint states and actuators."""

    @property
    @abstractmethod
    def joint_names(self) -> Tuple[str, ...]:
        """Names of actuated joints in canonical RobotSpec order."""
        ...

    @property
    @abstractmethod
    def actuator_names(self) -> Tuple[str, ...]:
        """Names of actuators in canonical control order."""
        ...

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_actuators(self) -> int:
        return len(self.actuator_names)

    @property
    @abstractmethod
    def actuator_ctrlrange(self) -> np.ndarray:
        """Actuator control range [U, 2]."""
        ...

    @abstractmethod
    def extract_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> TransmissionState:
        """Extract current joint positions, actuator coordinates, and moment matrix."""
        ...

    @abstractmethod
    def project_joint_delta(
        self,
        joint_delta: np.ndarray,
        current_state: TransmissionState | None = None,
        *,
        max_nullspace_residual: float = 0.05,
    ) -> ActuatorCommand:
        """Project desired kinematic joint delta into actuator-space command."""
        ...
