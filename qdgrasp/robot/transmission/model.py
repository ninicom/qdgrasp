"""Transmission models and extraction from compiled MuJoCo models."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple
import numpy as np
import mujoco

from qdgrasp.config.schema import ConfigError
from qdgrasp.robot.spec import RobotSpec
from .contracts import ActuatorCommand, TransmissionModel, TransmissionState


def extract_moment_matrix(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Sequence[str],
    actuator_names: Sequence[str],
) -> np.ndarray:
    """
    Extract the dense moment matrix M = dl/dq [U, J] from compiled MuJoCo CSR data.
    Columns correspond strictly to joint_names in given order.
    """
    U = len(actuator_names)
    J = len(joint_names)
    M = np.zeros((U, J), dtype=np.float64)

    # Map DOF address to joint index
    dof_to_jidx: dict[int, int] = {}
    for j_idx, j_name in enumerate(joint_names):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id < 0:
            raise ConfigError(f"Joint '{j_name}' not found in compiled MuJoCo model")
        dof_adr = int(model.jnt_dofadr[j_id])
        dof_to_jidx[dof_adr] = j_idx

    # Map actuator name to actuator ID
    for a_idx, a_name in enumerate(actuator_names):
        a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name)
        if a_id < 0:
            raise ConfigError(f"Actuator '{a_name}' not found in compiled MuJoCo model")
        adr = int(data.moment_rowadr[a_id])
        nnz = int(data.moment_rownnz[a_id])
        for k in range(nnz):
            col_dof = int(data.moment_colind[adr + k])
            val = float(data.actuator_moment[adr + k])
            if col_dof in dof_to_jidx:
                j_idx = dof_to_jidx[col_dof]
                M[a_idx, j_idx] = val

    return M


def compute_finite_difference_moment_matrix(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Sequence[str],
    actuator_names: Sequence[str],
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute numerical central finite-difference moment matrix dl/dq [U, J]."""
    U = len(actuator_names)
    J = len(joint_names)
    M_fd = np.zeros((U, J), dtype=np.float64)

    actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name)
        for a_name in actuator_names
    ]

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        for j_name in joint_names
    ]

    for j_idx, j_id in enumerate(joint_ids):
        qpos_adr = int(model.jnt_qposadr[j_id])
        orig_val = float(data.qpos[qpos_adr])

        # Positive perturbation
        data.qpos[qpos_adr] = orig_val + eps
        mujoco.mj_forward(model, data)
        l_plus = np.array([data.actuator_length[a_id] for a_id in actuator_ids], dtype=np.float64)

        # Negative perturbation
        data.qpos[qpos_adr] = orig_val - eps
        mujoco.mj_forward(model, data)
        l_minus = np.array([data.actuator_length[a_id] for a_id in actuator_ids], dtype=np.float64)

        # Restore
        data.qpos[qpos_adr] = orig_val
        mujoco.mj_forward(model, data)

        M_fd[:, j_idx] = (l_plus - l_minus) / (2.0 * eps)

    return M_fd


def create_transmission_model_from_spec_and_mjcf(
    spec: RobotSpec,
    mjcf_model: mujoco.MjModel,
) -> TransmissionModel:
    """Factory creating appropriate TransmissionModel (DirectJoint or FixedTendon) from spec and model."""
    from .direct import DirectJointTransmission
    from .fixed_tendon import FixedTendonTransmission

    joint_names = tuple(spec.actuated_joint_names)

    # Actuator names from spec or model
    if hasattr(spec, "actuators") and spec.actuators:
        # Match actuator names in compiled model
        # Check if spec actuator names exist in model or need prefix
        actuator_names_list: list[str] = []
        for name in spec.actuators.keys():
            if mujoco.mj_name2id(mjcf_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0:
                actuator_names_list.append(name)
            else:
                    # Try common prefix like rh_A_
                    prefixed = f"rh_A_{name.replace('rh_', '')}" if name.startswith("rh_") else f"A_{name}"
                    if mujoco.mj_name2id(mjcf_model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefixed) >= 0:
                        actuator_names_list.append(prefixed)
                    else:
                        raise ConfigError(
                            f"Cannot resolve spec actuator '{name}' in compiled MuJoCo model. "
                            f"Tried exact match '{name}' and prefixed '{prefixed}'."
                        )
            actuator_names = tuple(actuator_names_list)
    else:
        actuator_names = tuple(
            mujoco.mj_id2name(mjcf_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator_{a}"
            for a in range(mjcf_model.nu)
        )

    # Inspect transmission types
    has_tendon = False
    for a_name in actuator_names:
        a_id = mujoco.mj_name2id(mjcf_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name)
        trntype = int(mjcf_model.actuator_trntype[a_id])
        if trntype == int(mujoco.mjtTrn.mjTRN_TENDON):
            has_tendon = True
        elif trntype != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ConfigError(
                f"Unsupported transmission type {trntype} for actuator '{a_name}'. "
                "Only mjTRN_JOINT (0) and mjTRN_TENDON (3) are supported."
            )

    if has_tendon:
        return FixedTendonTransmission(
            joint_names=joint_names,
            actuator_names=actuator_names,
            model=mjcf_model,
        )
    return DirectJointTransmission(
        joint_names=joint_names,
        actuator_names=actuator_names,
        model=mjcf_model,
    )
