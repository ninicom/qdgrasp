"""MJCF importer and introspection using MuJoCo."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mujoco

from ..config.schema import ConfigError


@dataclass
class MJCFBody:
    name: str
    id: int
    parent_id: int
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    mass: float = 0.0
    inertia: tuple[float, float, float] = (0.0, 0.0, 0.0)
    geom_names: list[str] = field(default_factory=list)
    mesh_files: list[Path] = field(default_factory=list)


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


@dataclass
class MJCFMimicTendon:
    """A MuJoCo fixed tendon expressed as named joint coefficients."""

    name: str
    kind: str
    joint_coefficients: tuple[tuple[str, float], ...]


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
            inertia = tuple(float(x) for x in self.model.body_inertia[b_id])
            self.bodies[b_name] = MJCFBody(
                name=b_name,
                id=b_id,
                parent_id=parent_id,
                pos=(pos[0], pos[1], pos[2]),
                quat=(quat[0], quat[1], quat[2], quat[3]),
                mass=float(self.model.body_mass[b_id]),
                inertia=(inertia[0], inertia[1], inertia[2]),
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

        self.tendons = self._extract_tendons()

        self.mesh_names: list[str] = []
        for m_id in range(self.model.nmesh):
            m_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MESH, m_id) or f"mesh_{m_id}"
            self.mesh_names.append(m_name)

        self.mesh_files: dict[str, Path] = self._declared_mesh_files()
        self._attach_geometry()

    def _extract_tendons(self) -> dict[str, MJCFMimicTendon]:
        """Extract declared tendon coupling without inferring a joint relation."""

        root = ET.parse(self.path).getroot()
        tendon_root = root.find("tendon")
        tendons: dict[str, MJCFMimicTendon] = {}
        if tendon_root is None:
            return tendons
        for element in tendon_root:
            name = element.get("name")
            if not name:
                raise ConfigError(f"{self.path}: tendon is missing a name")
            terms: list[tuple[str, float]] = []
            for joint in element.findall("joint"):
                joint_name = joint.get("joint")
                if not joint_name or joint_name not in self.joints:
                    raise ConfigError(
                        f"{self.path}: tendon '{name}' references unknown joint '{joint_name}'"
                    )
                try:
                    coefficient = float(joint.get("coef", "1"))
                except ValueError as exc:
                    raise ConfigError(f"{self.path}: tendon '{name}' has an invalid coefficient") from exc
                terms.append((joint_name, coefficient))
            tendons[name] = MJCFMimicTendon(
                name=name,
                kind=element.tag,
                joint_coefficients=tuple(terms),
            )
        return tendons

    def _declared_mesh_files(self) -> dict[str, Path]:
        """Map every declared mesh asset name to its file on disk.

        ``MjModel`` keeps the compiled vertices but not the source file name, so
        the paths come from the XML: ``<compiler meshdir=...>`` plus each
        ``<mesh file=...>``.  A mesh without an explicit ``name`` takes the file
        stem, which is what MuJoCo itself does.
        """

        root = ET.parse(self.path).getroot()
        compiler = root.find("compiler")
        mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
        base_dir = self.path.parent / mesh_dir
        declared: dict[str, Path] = {}
        for element in root.iter("mesh"):
            file_name = element.get("file")
            if not file_name:
                continue
            declared[element.get("name") or Path(file_name).stem] = base_dir / file_name
        return declared

    def _attach_geometry(self) -> None:
        """Record each body's geoms and the mesh files they reference."""

        for g_id in range(int(self.model.ngeom)):
            b_id = int(self.model.geom_bodyid[g_id])
            b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b_id) or f"body_{b_id}"
            body = self.bodies.get(b_name)
            if body is None:
                continue
            g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g_id) or f"geom_{g_id}"
            body.geom_names.append(g_name)
            if int(self.model.geom_type[g_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_id = int(self.model.geom_dataid[g_id])
            if mesh_id < 0:
                continue
            mesh_name = self.mesh_names[mesh_id]
            mesh_path = self.mesh_files.get(mesh_name)
            if mesh_path is None:
                raise ConfigError(
                    f"{self.path}: geom '{g_name}' references mesh '{mesh_name}' "
                    "that is not declared with a file in the MJCF assets"
                )
            if mesh_path not in body.mesh_files:
                body.mesh_files.append(mesh_path)

    def validate_semantic_bodies(
        self,
        *,
        palm_body: str,
        base_body: str | None = None,
        wrist_body: str | None = None,
        fingertip_bodies: Sequence[str] = (),
        contact_bodies: Sequence[str] = (),
    ) -> None:
        """Ensure every declared semantic body exists, without guessing any of them.

        Covers the same five roles as the URDF path.  All released profiles are
        MJCF, so a role validated only on the URDF side would be unchecked in
        practice.
        """

        if palm_body not in self.bodies:
            raise ConfigError(
                f"declared palm_body '{palm_body}' does not exist in MJCF bodies {list(self.bodies.keys())}"
            )
        for role, name in (("base_body", base_body), ("wrist_body", wrist_body)):
            if name is not None and name not in self.bodies:
                raise ConfigError(f"declared {role} '{name}' not in MJCF bodies")
        for role, names in (("fingertip_body", fingertip_bodies), ("contact_body", contact_bodies)):
            for name in names:
                if name not in self.bodies:
                    raise ConfigError(f"declared {role} '{name}' not in MJCF bodies")


def parse_mjcf(path: str | Path) -> MJCFModel:
    """Parse and introspect an MJCF XML file via MuJoCo."""
    return MJCFModel(path)
