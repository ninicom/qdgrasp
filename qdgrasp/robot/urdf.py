"""URDF parser using standard library xml.etree."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..config.schema import ConfigError


@dataclass
class URDFMesh:
    filename: str
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class URDFLink:
    name: str
    visual_meshes: list[URDFMesh] = field(default_factory=list)
    collision_meshes: list[URDFMesh] = field(default_factory=list)
    mass: float = 0.0
    inertia: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    inertial_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertial_origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class URDFJoint:
    name: str
    type: str  # "revolute", "continuous", "prismatic", "fixed"
    parent: str
    child: str
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    limit_lower: float | None = None
    limit_upper: float | None = None
    limit_effort: float | None = None
    limit_velocity: float | None = None
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0

    @property
    def is_movable(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")


def _parse_vec3(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if not text:
        return default
    parts = [float(x) for x in text.strip().split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 floats, got {text}")
    return (parts[0], parts[1], parts[2])


class URDFModel:
    """Parsed URDF representation and kinematic tree."""

    def __init__(
        self,
        name: str,
        links: dict[str, URDFLink],
        joints: dict[str, URDFJoint],
        root_link: str,
    ) -> None:
        self.name = name
        self.links = links
        self.joints = joints
        self.root_link = root_link

        # Validate tree structure
        self.parent_map: dict[str, tuple[str, str]] = {}  # child_link -> (parent_link, joint_name)
        self.children_map: dict[str, list[tuple[str, str]]] = {name: [] for name in links}

        for j_name, joint in joints.items():
            if joint.child not in links:
                raise ConfigError(f"joint '{j_name}' references unknown child link '{joint.child}'")
            if joint.parent not in links:
                raise ConfigError(f"joint '{j_name}' references unknown parent link '{joint.parent}'")
            if joint.child in self.parent_map:
                raise ConfigError(f"link '{joint.child}' has multiple parent joints in URDF")
            self.parent_map[joint.child] = (joint.parent, j_name)
            self.children_map[joint.parent].append((joint.child, j_name))

    @property
    def movable_joints(self) -> list[str]:
        return [name for name, j in self.joints.items() if j.is_movable]

    @property
    def fixed_joints(self) -> list[str]:
        return [name for name, j in self.joints.items() if j.type == "fixed"]

    def topological_links(self) -> list[str]:
        """Return link names sorted in topological order starting from root_link."""
        visited: list[str] = []
        queue = [self.root_link]
        while queue:
            current = queue.pop(0)
            visited.append(current)
            for child, _ in self.children_map[current]:
                queue.append(child)
        # In case of disconnected links, add remaining
        for name in self.links:
            if name not in visited:
                visited.append(name)
        return visited

    def validate_semantic_links(
        self,
        *,
        palm_link: str,
        base_link: str | None = None,
        wrist_link: str | None = None,
        fingertip_links: Sequence[str] = (),
        contact_links: Sequence[str] = (),
    ) -> None:
        """Strictly validate that declared semantic links exist.

        Refuses to silently guess any missing semantic links.
        """
        if not palm_link or palm_link not in self.links:
            raise ConfigError(
                f"declared palm_link '{palm_link}' does not exist in URDF links {list(self.links.keys())}; "
                "semantic links must not be guessed"
            )
        if base_link is not None and base_link not in self.links:
            raise ConfigError(f"declared base_link '{base_link}' not in URDF links")
        if wrist_link is not None and wrist_link not in self.links:
            raise ConfigError(f"declared wrist_link '{wrist_link}' not in URDF links")
        for tip in fingertip_links:
            if tip not in self.links:
                raise ConfigError(f"declared fingertip_link '{tip}' not in URDF links")
        for contact in contact_links:
            if contact not in self.links:
                raise ConfigError(f"declared contact_link '{contact}' not in URDF links")


def parse_urdf(source: str | Path | ET.Element) -> URDFModel:
    """Parse a URDF XML string, file path, or ElementTree element."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file():
            tree = ET.parse(str(p))
            root = tree.getroot()
        else:
            root = ET.fromstring(str(source))
    else:
        root = source

    if root.tag != "robot":
        raise ConfigError(f"expected root tag <robot>, got <{root.tag}>")

    robot_name = root.get("name", "unnamed_robot")
    links: dict[str, URDFLink] = {}
    joints: dict[str, URDFJoint] = {}

    for elem in root:
        if elem.tag == "link":
            link_name = elem.get("name")
            if not link_name:
                raise ConfigError("link element missing name attribute")
            visual_meshes: list[URDFMesh] = []
            collision_meshes: list[URDFMesh] = []

            for vis in elem.findall("visual"):
                geom = vis.find("geometry")
                if geom is not None:
                    mesh_elem = geom.find("mesh")
                    if mesh_elem is not None:
                        fn = mesh_elem.get("filename", "")
                        scale_str = mesh_elem.get("scale")
                        scale = _parse_vec3(scale_str, (1.0, 1.0, 1.0))
                        origin_elem = vis.find("origin")
                        xyz = _parse_vec3(origin_elem.get("xyz") if origin_elem is not None else None)
                        rpy = _parse_vec3(origin_elem.get("rpy") if origin_elem is not None else None)
                        visual_meshes.append(URDFMesh(filename=fn, origin_xyz=xyz, origin_rpy=rpy, scale=scale))

            for col in elem.findall("collision"):
                geom = col.find("geometry")
                if geom is not None:
                    mesh_elem = geom.find("mesh")
                    if mesh_elem is not None:
                        fn = mesh_elem.get("filename", "")
                        scale_str = mesh_elem.get("scale")
                        scale = _parse_vec3(scale_str, (1.0, 1.0, 1.0))
                        origin_elem = col.find("origin")
                        xyz = _parse_vec3(origin_elem.get("xyz") if origin_elem is not None else None)
                        rpy = _parse_vec3(origin_elem.get("rpy") if origin_elem is not None else None)
                        collision_meshes.append(URDFMesh(filename=fn, origin_xyz=xyz, origin_rpy=rpy, scale=scale))

            mass = 0.0
            inertia = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            iner_xyz = (0.0, 0.0, 0.0)
            iner_rpy = (0.0, 0.0, 0.0)

            inertial = elem.find("inertial")
            if inertial is not None:
                mass_elem = inertial.find("mass")
                if mass_elem is not None and mass_elem.get("value"):
                    mass = float(mass_elem.get("value"))
                origin_elem = inertial.find("origin")
                if origin_elem is not None:
                    iner_xyz = _parse_vec3(origin_elem.get("xyz"))
                    iner_rpy = _parse_vec3(origin_elem.get("rpy"))
                iner_elem = inertial.find("inertia")
                if iner_elem is not None:
                    ixx = float(iner_elem.get("ixx", 0.0))
                    ixy = float(iner_elem.get("ixy", 0.0))
                    ixz = float(iner_elem.get("ixz", 0.0))
                    iyy = float(iner_elem.get("iyy", 0.0))
                    iyz = float(iner_elem.get("iyz", 0.0))
                    izz = float(iner_elem.get("izz", 0.0))
                    inertia = (ixx, ixy, ixz, iyy, iyz, izz)

            links[link_name] = URDFLink(
                name=link_name,
                visual_meshes=visual_meshes,
                collision_meshes=collision_meshes,
                mass=mass,
                inertia=inertia,
                inertial_origin_xyz=iner_xyz,
                inertial_origin_rpy=iner_rpy,
            )

        elif elem.tag == "joint":
            joint_name = elem.get("name")
            if not joint_name:
                raise ConfigError("joint element missing name attribute")
            joint_type = elem.get("type", "revolute")

            parent_elem = elem.find("parent")
            child_elem = elem.find("child")
            if parent_elem is None or not parent_elem.get("link"):
                raise ConfigError(f"joint '{joint_name}' missing parent link")
            if child_elem is None or not child_elem.get("link"):
                raise ConfigError(f"joint '{joint_name}' missing child link")

            parent_link = parent_elem.get("link")
            child_link = child_elem.get("link")

            origin_elem = elem.find("origin")
            orig_xyz = _parse_vec3(origin_elem.get("xyz") if origin_elem is not None else None)
            orig_rpy = _parse_vec3(origin_elem.get("rpy") if origin_elem is not None else None)

            axis_elem = elem.find("axis")
            axis_xyz = _parse_vec3(axis_elem.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0))

            limit_lower = None
            limit_upper = None
            limit_effort = None
            limit_velocity = None

            limit_elem = elem.find("limit")
            if limit_elem is not None:
                if limit_elem.get("lower") is not None:
                    limit_lower = float(limit_elem.get("lower"))
                if limit_elem.get("upper") is not None:
                    limit_upper = float(limit_elem.get("upper"))
                if limit_elem.get("effort") is not None:
                    limit_effort = float(limit_elem.get("effort"))
                if limit_elem.get("velocity") is not None:
                    limit_velocity = float(limit_elem.get("velocity"))

            mimic_joint = None
            mimic_mult = 1.0
            mimic_offset = 0.0
            mimic_elem = elem.find("mimic")
            if mimic_elem is not None:
                mimic_joint = mimic_elem.get("joint")
                if mimic_elem.get("multiplier") is not None:
                    mimic_mult = float(mimic_elem.get("multiplier"))
                if mimic_elem.get("offset") is not None:
                    mimic_offset = float(mimic_elem.get("offset"))

            joints[joint_name] = URDFJoint(
                name=joint_name,
                type=joint_type,
                parent=parent_link,
                child=child_link,
                origin_xyz=orig_xyz,
                origin_rpy=orig_rpy,
                axis=axis_xyz,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                limit_effort=limit_effort,
                limit_velocity=limit_velocity,
                mimic_joint=mimic_joint,
                mimic_multiplier=mimic_mult,
                mimic_offset=mimic_offset,
            )

    if not links:
        raise ConfigError(f"URDF '{robot_name}' contains no links")

    # Find root link (link that is never a child of any joint)
    child_links = {j.child for j in joints.values()}
    root_candidates = [l for l in links if l not in child_links]
    if not root_candidates:
        # Pick first link as fallback if cyclic
        root_link = next(iter(links.keys()))
    else:
        root_link = root_candidates[0]

    return URDFModel(name=robot_name, links=links, joints=joints, root_link=root_link)
