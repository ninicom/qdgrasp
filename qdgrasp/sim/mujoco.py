"""MuJoCo simulation wrapper for kinematics, grasping and dynamic evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

from ..config.schema import ConfigError


class MujocoSim:
    """Wrapper around MuJoCo MjModel and MjData for deterministic simulation."""

    def __init__(self, xml_source: str | Path | mujoco.MjModel) -> None:
        if isinstance(xml_source, mujoco.MjModel):
            self.model = xml_source
        else:
            p = Path(xml_source)
            if not p.is_file():
                raise ConfigError(f"MuJoCo XML file not found: {p}")
            self.model = mujoco.MjModel.from_xml_path(str(p))

        self.data = mujoco.MjData(self.model)
        self.forward()

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nu(self) -> int:
        return int(self.model.nu)

    @property
    def nbody(self) -> int:
        return int(self.model.nbody)

    def forward(self) -> None:
        """Run kinematics forward pass."""
        mujoco.mj_forward(self.model, self.data)

    def step(self, steps: int = 1) -> None:
        """Advance physics simulation by ``steps`` time steps."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        """Reset simulation state to initial configuration."""
        mujoco.mj_resetData(self.model, self.data)
        self.forward()

    def set_joint_qpos(self, joint_name: str, value: float) -> None:
        """Set position of named joint."""
        j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id < 0:
            raise ConfigError(f"joint '{joint_name}' not found in MuJoCo model")
        qpos_adr = self.model.jnt_qposadr[j_id]
        self.data.qpos[qpos_adr] = value

    def get_body_pos(self, body_name: str) -> np.ndarray:
        """Get 3D world position of named body."""
        b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if b_id < 0:
            raise ConfigError(f"body '{body_name}' not found in MuJoCo model")
        return np.array(self.data.xpos[b_id], dtype=np.float32)

    def get_body_mat(self, body_name: str) -> np.ndarray:
        """Get 3x3 world rotation matrix of named body."""
        b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if b_id < 0:
            raise ConfigError(f"body '{body_name}' not found in MuJoCo model")
        return np.array(self.data.xmat[b_id], dtype=np.float32).reshape(3, 3)

    def get_contact_count(self) -> int:
        """Return number of active contact pairs."""
        return int(self.data.ncon)
