"""Fixed-tendon transmission strategy for underactuated hands (Shadow Hand)."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from .command import project_joint_delta_to_actuator_command
from .contracts import ActuatorCommand, TransmissionModel, TransmissionState
from .model import compute_finite_difference_moment_matrix, extract_moment_matrix


class FixedTendonTransmission(TransmissionModel):
    """Underactuated transmission combining direct joint and coupled fixed-tendon actuators."""

    def __init__(
        self,
        joint_names: Sequence[str],
        actuator_names: Sequence[str],
        model: mujoco.MjModel,
    ) -> None:
        self._joint_names = tuple(joint_names)
        self._actuator_names = tuple(actuator_names)
        self._model = model

        # Cache control ranges
        ctrl_ranges = []
        for a_name in self._actuator_names:
            a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name)
            if a_id < 0:
                raise ValueError(f"Actuator '{a_name}' not found in compiled model")
            if bool(model.actuator_ctrllimited[a_id]):
                lo, hi = float(model.actuator_ctrlrange[a_id][0]), float(model.actuator_ctrlrange[a_id][1])
            else:
                lo, hi = -float("inf"), float("inf")
            ctrl_ranges.append([lo, hi])
        self._actuator_ctrlrange = np.array(ctrl_ranges, dtype=np.float64)

        # Precompute moment matrix
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self._M = extract_moment_matrix(model, data, self._joint_names, self._actuator_names)
        self._rank = int(np.linalg.matrix_rank(self._M))

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return self._actuator_names

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        return self._actuator_ctrlrange

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def moment_matrix(self) -> np.ndarray:
        return self._M.copy()

    def verify_finite_difference(
        self,
        eps: float = 1e-6,
        atol: float = 1e-5,
    ) -> bool:
        """Verify that analytic moment matrix matches numerical central finite differences."""
        data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, data)
        M_fd = compute_finite_difference_moment_matrix(
            self._model, data, self._joint_names, self._actuator_names, eps=eps
        )
        max_diff = float(np.max(np.abs(self._M - M_fd)))
        return bool(max_diff <= atol)

    def extract_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> TransmissionState:
        """Extract current joint positions, actuator coordinates, and moment matrix."""
        j_pos = np.zeros(len(self._joint_names), dtype=np.float64)
        for j_idx, j_name in enumerate(self._joint_names):
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            qpos_adr = int(model.jnt_qposadr[j_id])
            j_pos[j_idx] = float(data.qpos[qpos_adr])

        act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name)
            for a_name in self._actuator_names
        ]
        a_coords = np.array([float(data.actuator_length[a_id]) for a_id in act_ids], dtype=np.float64)

        return TransmissionState(
            joint_names=self._joint_names,
            actuator_names=self._actuator_names,
            joint_position=j_pos,
            actuator_coordinate=a_coords,
            moment_matrix=self._M.copy(),
            rank=self._rank,
        )

    def project_joint_delta(
        self,
        joint_delta: np.ndarray,
        current_state: TransmissionState | None = None,
        *,
        max_nullspace_residual: float = 0.05,
    ) -> ActuatorCommand:
        """Project desired kinematic joint delta into actuator-space command."""
        curr_coords = current_state.actuator_coordinate if current_state is not None else None
        return project_joint_delta_to_actuator_command(
            joint_delta=joint_delta,
            moment_matrix=self._M,
            actuator_ctrlrange=self._actuator_ctrlrange,
            current_actuator_coordinates=curr_coords,
            max_nullspace_residual=max_nullspace_residual,
        )
