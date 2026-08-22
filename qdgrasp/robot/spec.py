"""RobotSpec: unified representation of hand kinematics, meshes and semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..config.loader import load_robot_config, resolve_document_path
from ..config.schema import ConfigError
from .graph import HandGraph
from .kinematics import (
    compute_joint_transform,
    invert_rigid_transform,
    quaternion_to_rotation_matrix,
    rpy_to_rotation_matrix,
    transform_points,
)
from .meshes import load_mesh, resolve_mesh_path, sample_mesh_surface
from .mjcf import parse_mjcf
from .normalize import normalize_urdf
from .schema import ActuatorSpec, MimicSpec, RobotConfigV2
from .urdf import URDFJoint, URDFLink, URDFModel, parse_urdf


def _matrix_to_tuple(matrix: torch.Tensor) -> tuple[tuple[float, float, float], ...]:
    """Freeze a 3x3 rotation tensor into a hashable nested tuple."""

    values = matrix.reshape(3, 3).tolist()
    return tuple(tuple(float(value) for value in row) for row in values)


def _topological_order(links: dict[str, "LinkSpec"], preferred: Sequence[str]) -> list[str]:
    """Order links so that every parent precedes its children.

    The order is derived from ``parent_link`` rather than inherited from the
    parser, so forward kinematics never depends on a parser happening to emit
    bodies in tree order.  ``preferred`` only breaks ties, keeping the source
    file's ordering among siblings.
    """

    remaining = [name for name in preferred if name in links]
    remaining += [name for name in links if name not in set(remaining)]

    ordered: list[str] = []
    placed: set[str] = set()
    while remaining:
        progressed = False
        deferred: list[str] = []
        for name in remaining:
            parent = links[name].parent_link
            if parent is None or parent not in links or parent in placed:
                ordered.append(name)
                placed.add(name)
                progressed = True
            else:
                deferred.append(name)
        if not progressed:
            raise ConfigError(f"kinematic tree contains a cycle among links {sorted(deferred)}")
        remaining = deferred
    return ordered


@dataclass
class LinkSpec:
    name: str
    parent_link: str | None
    parent_joint: str | None
    joint_type: str  # "revolute", "prismatic", "fixed"
    origin_xyz: tuple[float, float, float]
    origin_rotation: tuple[tuple[float, float, float], ...]
    axis: tuple[float, float, float]
    mass: float
    inertia: tuple[float, float, float, float, float, float]
    mesh_paths: list[Path] = field(default_factory=list)
    surface_anchors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    surface_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    semantic_tag: int = 0  # 0: other/intermediate, 1: palm, 2: base, 3: wrist, 4: fingertip, 5: contact


class RobotSpec:
    """Complete specification of a dexterous hand for kinematics, learning and simulation."""

    def __init__(
        self,
        config: RobotConfigV2,
        links: dict[str, LinkSpec],
        topological_links: list[str],
        actuated_joint_names: tuple[str, ...],
        joint_limits: dict[str, tuple[float, float]],
        mimic_joints: dict[str, MimicSpec],
    ) -> None:
        self.config = config
        self.links = links
        self.topological_links = _topological_order(links, topological_links)
        self.actuated_joint_names = actuated_joint_names
        self.joint_limits = joint_limits
        self.mimic_joints = mimic_joints

        self.palm_link = config.palm_link
        self.base_link = config.base_link or config.palm_link
        self.wrist_link = config.wrist_link
        self.fingertip_links = config.fingertip_links
        self.contact_links = config.contact_links

        # Precompute fingertip anchor indices
        self.fingertip_indices = tuple(
            self.topological_links.index(tip) for tip in self.fingertip_links if tip in self.topological_links
        )

    @classmethod
    def from_config(cls, reference: str | Path | RobotConfigV2, *, sample_anchors: bool = True, anchor_count_per_link: int = 16) -> "RobotSpec":
        """Build a RobotSpec from a RobotConfigV2 or YAML preset/file."""
        if isinstance(reference, RobotConfigV2):
            config = reference
        else:
            config = load_robot_config(reference)
            if not isinstance(config, RobotConfigV2):
                raise ConfigError(f"RobotSpec requires RobotConfigV2, got {type(config).__name__}")

        asset_path = Path(config.source_asset)
        if not asset_path.is_file():
            # Try resolving relative to workspace
            cand = Path(resolve_document_path(config.source_asset)) if not asset_path.is_file() else asset_path
            if cand.is_file():
                asset_path = cand
            else:
                raise ConfigError(f"source asset '{config.source_asset}' does not exist")

        base_dir = asset_path.parent

        if config.format == "urdf" or asset_path.suffix == ".urdf":
            urdf_model = parse_urdf(asset_path)
            urdf_model.validate_semantic_links(
                palm_link=config.palm_link,
                base_link=config.base_link,
                wrist_link=config.wrist_link,
                fingertip_links=config.fingertip_links,
                contact_links=config.contact_links,
            )

            links_dict: dict[str, LinkSpec] = {}
            for link_name, u_link in urdf_model.links.items():
                parent_info = urdf_model.parent_map.get(link_name)
                parent_link = parent_info[0] if parent_info else None
                joint_name = parent_info[1] if parent_info else None
                u_joint = urdf_model.joints.get(joint_name) if joint_name else None

                joint_type = u_joint.type if u_joint else "fixed"
                origin_xyz = u_joint.origin_xyz if u_joint else (0.0, 0.0, 0.0)
                origin_rpy = u_joint.origin_rpy if u_joint else (0.0, 0.0, 0.0)
                origin_rotation = _matrix_to_tuple(rpy_to_rotation_matrix(origin_rpy))
                axis = u_joint.axis if u_joint else (1.0, 0.0, 0.0)

                # Resolve visual meshes and sample anchors
                mesh_paths: list[Path] = []
                anchors_list: list[np.ndarray] = []
                normals_list: list[np.ndarray] = []

                for m in u_link.visual_meshes:
                    # No blanket except here: a mesh that cannot be resolved or
                    # loaded must fail the profile, otherwise the zero-missing-mesh
                    # guarantee is satisfied by silently dropping the mesh.
                    try:
                        resolved = resolve_mesh_path(
                            m.filename,
                            base_dir=base_dir,
                            mesh_root=config.mesh_root,
                            package_roots=config.package_roots,
                        )
                    except ConfigError as exc:
                        raise ConfigError(f"link '{link_name}': {exc}") from exc
                    mesh_paths.append(resolved)
                    if sample_anchors:
                        mesh_obj = load_mesh(resolved)
                        if m.scale != (1.0, 1.0, 1.0):
                            mesh_obj.apply_scale(m.scale)
                        pts, nrms = sample_mesh_surface(mesh_obj, count=anchor_count_per_link, seed=0)
                        anchors_list.append(pts)
                        normals_list.append(nrms)

                if anchors_list:
                    surf_anchors = np.concatenate(anchors_list, axis=0)
                    surf_normals = np.concatenate(normals_list, axis=0)
                else:
                    surf_anchors = np.zeros((0, 3), dtype=np.float32)
                    surf_normals = np.zeros((0, 3), dtype=np.float32)

                # Determine semantic tag
                if link_name == config.palm_link:
                    tag = 1
                elif link_name == config.base_link:
                    tag = 2
                elif link_name == config.wrist_link:
                    tag = 3
                elif link_name in config.fingertip_links:
                    tag = 4
                elif link_name in config.contact_links:
                    tag = 5
                else:
                    tag = 0

                links_dict[link_name] = LinkSpec(
                    name=link_name,
                    parent_link=parent_link,
                    parent_joint=joint_name,
                    joint_type=joint_type,
                    origin_xyz=origin_xyz,
                    origin_rotation=origin_rotation,
                    axis=axis,
                    mass=u_link.mass,
                    inertia=u_link.inertia,
                    mesh_paths=mesh_paths,
                    surface_anchors=surf_anchors,
                    surface_normals=surf_normals,
                    semantic_tag=tag,
                )

            topological_links = urdf_model.topological_links()

        elif config.format == "mjcf" or asset_path.suffix == ".xml":
            mjcf_model = parse_mjcf(asset_path)
            mjcf_model.validate_semantic_bodies(
                palm_body=config.palm_link,
                base_body=config.base_link,
                wrist_body=config.wrist_link,
                fingertip_bodies=config.fingertip_links,
                contact_bodies=config.contact_links,
            )

            links_dict = {}
            id_to_name = {b.id: name for name, b in mjcf_model.bodies.items()}

            for b_name, b in mjcf_model.bodies.items():
                parent_name = id_to_name.get(b.parent_id) if b.parent_id != b.id else None
                # Check joints belonging to this body
                b_joints = [j for j in mjcf_model.joints.values() if j.body_name == b_name]
                if len(b_joints) > 1:
                    # A body carrying several joints needs their transforms composed
                    # in order.  No hand in the pinned corpus does this, so rather
                    # than compose it untested, refuse loudly.
                    raise ConfigError(
                        f"body '{b_name}' carries {len(b_joints)} joints "
                        f"({[j.name for j in b_joints]}); multi-joint bodies are not supported yet"
                    )
                if b_joints:
                    j = b_joints[0]
                    if any(abs(value) > 1e-12 for value in j.pos):
                        # A joint anchor offset composes as T_origin . T_anchor .
                        # T_motion . T_anchor^-1; every pinned hand has a zero
                        # anchor, so refuse instead of silently ignoring it.
                        raise ConfigError(
                            f"joint '{j.name}' declares a non-zero anchor offset {j.pos}; "
                            "joint anchor offsets are not supported yet"
                        )
                    joint_name = j.name
                    joint_type = "revolute" if j.type == 3 else "prismatic" if j.type == 2 else "fixed"
                    axis = j.axis
                else:
                    joint_name = None
                    joint_type = "fixed"
                    axis = (1.0, 0.0, 0.0)

                origin_xyz = b.pos
                # Use the quaternion directly: the Euler detour is degenerate at
                # |pitch| == 90 deg, which is exactly where several Menagerie
                # bodies sit (Allegro 'palm', Shadow 'rh_forearm').
                origin_rotation = _matrix_to_tuple(quaternion_to_rotation_matrix(b.quat))

                # Determine semantic tag
                if b_name == config.palm_link:
                    tag = 1
                elif b_name == config.base_link:
                    tag = 2
                elif b_name == config.wrist_link:
                    tag = 3
                elif b_name in config.fingertip_links:
                    tag = 4
                elif b_name in config.contact_links:
                    tag = 5
                else:
                    tag = 0

                links_dict[b_name] = LinkSpec(
                    name=b_name,
                    parent_link=parent_name,
                    parent_joint=joint_name,
                    joint_type=joint_type,
                    origin_xyz=origin_xyz,
                    origin_rotation=origin_rotation,
                    axis=axis,
                    mass=b.mass,
                    inertia=(b.inertia[0], 0.0, 0.0, b.inertia[1], 0.0, b.inertia[2]),
                    mesh_paths=list(b.mesh_files),
                    # Empty rather than a block of zeros: claiming
                    # ``anchor_count_per_link`` anchors that are all at the origin
                    # feeds a false anchor count into the graph features.  Real
                    # anchors need the geom-level pose offsets, and surface
                    # sampling is not reproducible across trimesh versions (dev
                    # runs 5.0.0, the cu128 lock pins 4.12.2), so they are left to
                    # a later phase rather than baked in environment-dependent.
                    surface_anchors=np.zeros((0, 3), dtype=np.float32),
                    surface_normals=np.zeros((0, 3), dtype=np.float32),
                    semantic_tag=tag,
                )

            topological_links = list(mjcf_model.bodies.keys())
        else:
            raise ConfigError(f"unsupported format '{config.format}'")

        return cls(
            config=config,
            links=links_dict,
            topological_links=topological_links,
            actuated_joint_names=config.joints,
            joint_limits=config.joint_limits,
            mimic_joints=config.mimic_joints,
        )

    def to_hand_graph(self, device: torch.device | str = "cpu") -> HandGraph:
        """Construct the variable-length HandGraph without NxN expansion."""
        L = len(self.topological_links)
        node_features_list: list[list[float]] = []

        link_to_idx = {name: i for i, name in enumerate(self.topological_links)}

        edges_src: list[int] = []
        edges_dst: list[int] = []
        edge_features_list: list[list[float]] = []

        for idx, link_name in enumerate(self.topological_links):
            link = self.links[link_name]
            ox, oy, oz = link.origin_xyz
            rotation = link.origin_rotation
            # First two columns of the rotation: a continuous 6D encoding, unlike
            # RPY which is discontinuous and degenerate at the Euler singularity.
            r6 = [
                rotation[0][0], rotation[1][0], rotation[2][0],
                rotation[0][1], rotation[1][1], rotation[2][1],
            ]
            ixx, _, _, iyy, _, izz = link.inertia
            num_anchors = float(len(link.surface_anchors))

            # 17-dim node feature: [pos(3), rot6d(6), mass(1), inertia_diag(3),
            # semantic_tag(1), anchor_count(1), num_joints(1), index_norm(1)]
            feat = [
                ox, oy, oz,
                *r6,
                link.mass,
                ixx, iyy, izz,
                float(link.semantic_tag),
                num_anchors,
                float(len(self.actuated_joint_names)),
                float(idx) / max(L - 1, 1),
            ]
            node_features_list.append(feat)

            if link.parent_link is not None and link.parent_link in link_to_idx:
                p_idx = link_to_idx[link.parent_link]
                edges_src.append(p_idx)
                edges_dst.append(idx)

                # Edge features
                j_type_val = 0.0 if link.joint_type == "revolute" else 1.0 if link.joint_type == "prismatic" else 2.0
                ax, ay, az = link.axis

                j_name = link.parent_joint
                if j_name and j_name in self.joint_limits:
                    l_lim, u_lim = self.joint_limits[j_name]
                else:
                    l_lim, u_lim = -np.pi, np.pi

                if j_name and j_name in self.mimic_joints:
                    m_spec = self.mimic_joints[j_name]
                    m_mult, m_off = m_spec.multiplier, m_spec.offset
                    is_mimic = 1.0
                else:
                    m_mult, m_off = 1.0, 0.0
                    is_mimic = 0.0

                # 9-dim edge feature: [j_type(1), axis(3), lower_lim(1), upper_lim(1), is_mimic(1), mimic_mult(1), mimic_off(1)]
                edge_feat = [j_type_val, ax, ay, az, l_lim, u_lim, is_mimic, m_mult, m_off]
                edge_features_list.append(edge_feat)

        node_features = torch.tensor(node_features_list, dtype=torch.float32, device=device)
        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
            edge_features = torch.tensor(edge_features_list, dtype=torch.float32, device=device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_features = torch.zeros((0, 9), dtype=torch.float32, device=device)

        palm_idx = link_to_idx.get(self.palm_link, 0)
        fingertip_indices = tuple(link_to_idx[tip] for tip in self.fingertip_links if tip in link_to_idx)

        return HandGraph(
            node_names=tuple(self.topological_links),
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            palm_index=palm_idx,
            fingertip_indices=fingertip_indices,
            actuated_joint_names=self.actuated_joint_names,
        )

    def forward_kinematics(
        self,
        palm_pos: torch.Tensor,  # [B, 3]
        palm_rot: torch.Tensor,  # [B, 3, 3] or [B, 9]
        joint_angles: torch.Tensor | Mapping[str, torch.Tensor],  # [B, J] or dict
    ) -> dict[str, torch.Tensor]:
        """Compute differentiable forward kinematics for all links.

        Returns:
            Dictionary mapping link_name -> global transform Tensor [B, 4, 4].
        """
        if palm_pos.ndim == 1:
            palm_pos = palm_pos.unsqueeze(0)
        B = palm_pos.shape[0]
        device = palm_pos.device
        dtype = palm_pos.dtype

        if palm_rot.ndim == 1:
            palm_rot = palm_rot.unsqueeze(0)
        if palm_rot.shape[-1] == 9:
            # Reshape 9D to [B, 3, 3] and orthonormalize via SVD/Gram-Schmidt
            R_raw = palm_rot.view(B, 3, 3)
            # Gram-Schmidt on column vectors
            c0 = F.normalize(R_raw[:, :, 0], dim=-1)
            c1 = R_raw[:, :, 1] - (c0 * R_raw[:, :, 1]).sum(dim=-1, keepdim=True) * c0
            c1 = F.normalize(c1, dim=-1)
            c2 = torch.cross(c0, c1, dim=-1)
            R_palm = torch.stack([c0, c1, c2], dim=-1)
        else:
            R_palm = palm_rot.to(device=device, dtype=dtype)

        # Map input joint angles to full joint mapping
        full_q: dict[str, torch.Tensor] = {}
        if isinstance(joint_angles, torch.Tensor):
            if joint_angles.ndim == 1:
                joint_angles = joint_angles.unsqueeze(0)
            for j_idx, j_name in enumerate(self.actuated_joint_names):
                if j_idx < joint_angles.shape[1]:
                    full_q[j_name] = joint_angles[:, j_idx]
                else:
                    full_q[j_name] = torch.zeros(B, dtype=dtype, device=device)
        else:
            for j_name in self.actuated_joint_names:
                if j_name in joint_angles:
                    val = joint_angles[j_name]
                    if val.ndim == 0:
                        val = val.unsqueeze(0).expand(B)
                    elif val.ndim == 1 and val.shape[0] != B:
                        val = val.expand(B)
                    full_q[j_name] = val.to(device=device, dtype=dtype)
                else:
                    full_q[j_name] = torch.zeros(B, dtype=dtype, device=device)

        # Apply mimic joint equations
        for m_name, m_spec in self.mimic_joints.items():
            if m_spec.target_joint in full_q:
                full_q[m_name] = full_q[m_spec.target_joint] * m_spec.multiplier + m_spec.offset
            else:
                full_q[m_name] = torch.zeros(B, dtype=dtype, device=device)

        T_palm = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(B, 4, 4).clone()
        T_palm[:, :3, :3] = R_palm
        T_palm[:, :3, 3] = palm_pos

        # Pass 1: every link in the frame of its own kinematic root.  Links above
        # the palm in the tree are ordinary members of this chain, so they are no
        # longer silently re-parented onto the palm.
        root_transforms: dict[str, torch.Tensor] = {}
        for link_name in self.topological_links:
            link = self.links[link_name]
            joint_name = link.parent_joint
            q_val = (
                full_q.get(joint_name, torch.zeros(B, dtype=dtype, device=device))
                if joint_name
                else torch.zeros(B, dtype=dtype, device=device)
            )
            T_local = compute_joint_transform(
                joint_type=link.joint_type,
                axis=link.axis,
                origin_xyz=link.origin_xyz,
                origin_rotation=link.origin_rotation,
                q=q_val.to(device=device, dtype=dtype),
            )
            parent = link.parent_link
            if parent is None or parent not in self.links:
                root_transforms[link_name] = T_local
            elif parent in root_transforms:
                root_transforms[link_name] = torch.bmm(root_transforms[parent], T_local)
            else:
                raise ConfigError(
                    f"link '{link_name}' is visited before its parent '{parent}'; "
                    "the kinematic order is not topological"
                )

        if self.palm_link not in root_transforms:
            raise ConfigError(f"palm link '{self.palm_link}' is not part of the kinematic tree")

        # Pass 2: pin the palm at the requested pose and carry the whole tree with
        # it, so the caller's palm frame is honoured exactly.
        T_palm_in_root_inverse = invert_rigid_transform(root_transforms[self.palm_link])
        rebase = torch.bmm(T_palm, T_palm_in_root_inverse)
        return {name: torch.bmm(rebase, T) for name, T in root_transforms.items()}

    def fingertip_positions(
        self,
        palm_pos: torch.Tensor,
        palm_rot: torch.Tensor,
        joint_angles: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute [B, num_fingertips, 3] world positions of the declared fingertips."""
        transforms = self.forward_kinematics(palm_pos, palm_rot, joint_angles)
        missing = [tip for tip in self.fingertip_links if tip not in transforms]
        if missing:
            raise ConfigError(
                f"profile '{self.config.name}' declares fingertip links {missing} "
                "that are absent from the kinematic tree"
            )
        tip_positions = [transforms[tip][:, :3, 3] for tip in self.fingertip_links]
        if not tip_positions:
            return torch.zeros((palm_pos.shape[0], 0, 3), dtype=palm_pos.dtype, device=palm_pos.device)
        return torch.stack(tip_positions, dim=1)
