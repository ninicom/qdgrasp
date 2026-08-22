"""MJCF importer and introspection using MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mujoco

from ..config.schema import ConfigError


@dataclass
class MJCFBody:
    name: str
    id: int
    parent_id: int
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    geom_names: list[str] = field(default_factory=list)


@dataclass
class MJCFJoint:
    name: str
    id: int
    body_id: int
    body_name: str
    type: int  # mujoco.mjtJoint
    limits: tuple[float, float] | None
    axis: tuple[float, float, float]
    pos: tuple[float, float, float]


@dataclass
class MJCFActuator:
    name: str
    id: int
    trntype: int
    trnid: tuple[int, int]
    ctrl_range: tuple[float, float] | None
    force_range: tuple[float, float] | None


class MJCFModel:
    """Introspected MuJoCo model wrapping an MjModel."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ConfigError(f"MJCF file not found: {self.path}")
        try:
            self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(self.path))
        except Exception as exc:
            raise ConfigError(f"failed to load MJCF {self.path}: {exc}") from exc

        self.name = self.path.stem
        self.nq = int(self.model.nq)
        self.nu = int(self.model.nu)
        self.nbody = int(self.model.nbody)
        self.njnt = int(self.model.njnt)

        self.bodies: dict[str, MJCFBody] = {}
        for b_id in range(self.nbody):
            b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b_id) or f"body_{b_id}"
            pos = tuple(float(x) for x in self.model.body_pos[b_id])
            quat = tuple(float(x) for x in self.model.body_quat[b_id])
            parent_id = int(self.model.body_parentid[b_id])
            self.bodies[b_name] = MJCFBody(
                name=b_name,
                id=b_id,
                parent_id=parent_id,
                pos=(pos[0], pos[1], pos[2]),
                quat=(quat[0], quat[1], quat[2], quat[3]),
            )

        self.joints: dict[str, MJCFJoint] = {}
        for j_id in range(self.njnt):
            j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or f"joint_{j_id}"
            b_id = int(self.model.jnt_bodyid[j_id])
            b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b_id) or f"body_{b_id}"
            j_type = int(self.model.jnt_type[j_id])
            axis = tuple(float(x) for x in self.model.jnt_axis[j_id])
            pos = tuple(float(x) for x in self.model.jnt_pos[j_id])
            j_limited = bool(self.model.jnt_limited[j_id]) if hasattr(self.model, "jnt_limited") else True
            if j_limited:
                rng = self.model.jnt_range[j_id]
                limits = (float(rng[0]), float(rng[1]))
            else:
                limits = None
            self.joints[j_name] = MJCFJoint(
                name=j_name,
                id=j_id,
                body_id=b_id,
                body_name=b_name,
                type=j_type,
                limits=limits,
                axis=(axis[0], axis[1], axis[2]),
                pos=(pos[0], pos[1], pos[2]),
            )

        self.actuators: dict[str, MJCFActuator] = {}
        for a_id in range(self.nu):
            a_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_id) or f"actuator_{a_id}"
            trntype = int(self.model.actuator_trntype[a_id])
            trnid = (int(self.model.actuator_trnid[a_id][0]), int(self.model.actuator_trnid[a_id][1]))
            ctrl_limited = bool(self.model.actuator_ctrllimited[a_id]) if hasattr(self.model, "actuator_ctrllimited") else False
            ctrl_range = None
            if ctrl_limited or hasattr(self.model, "actuator_ctrlrange"):
                cr = self.model.actuator_ctrlrange[a_id]
                ctrl_range = (float(cr[0]), float(cr[1]))
            force_range = None
            if hasattr(self.model, "actuator_forcerange"):
                fr = self.model.actuator_forcerange[a_id]
                force_range = (float(fr[0]), float(fr[1]))
            self.actuators[a_name] = MJCFActuator(
                name=a_name,
                id=a_id,
                trntype=trntype,
                trnid=trnid,
                ctrl_range=ctrl_range,
                force_range=force_range,
            )

        self.mesh_names: list[str] = []
        for m_id in range(self.model.nmesh):
            m_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MESH, m_id) or f"mesh_{m_id}"
            self.mesh_names.append(m_name)

    def validate_semantic_bodies(
        self,
        *,
        palm_body: str,
        base_body: str | None = None,
        fingertip_bodies: Sequence[str] = (),
    ) -> None:
        """Ensure declared palm/fingertip body names exist in the model without guessing."""
        if palm_body not in self.bodies:
            raise ConfigError(
                f"declared palm_body '{palm_body}' does not exist in MJCF bodies {list(self.bodies.keys())}"
            )
        if base_body is not None and base_body not in self.bodies:
            raise ConfigError(f"declared base_body '{base_body}' not in MJCF bodies")
        for tip in fingertip_bodies:
            if tip not in self.bodies:
                raise ConfigError(f"declared fingertip_body '{tip}' not in MJCF bodies")


def parse_mjcf(path: str | Path) -> MJCFModel:
    """Parse and introspect an MJCF XML file via MuJoCo."""
    return MJCFModel(path)
