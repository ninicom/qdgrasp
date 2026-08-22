"""Variable-length HandGraph representation for cross-embodiment grasping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class HandGraph:
    """Graph representation of hand morphology without NxN dense expansion.

    Attributes:
        node_names: List of link/node names of length L.
        node_features: FloatTensor of shape [L, node_dim] encoding link kinematics and semantics.
        edge_index: LongTensor of shape [2, E] encoding parent-child kinematic joints.
        edge_features: FloatTensor of shape [E, edge_dim] encoding joint types, axes and limits.
        palm_index: Index of the palm/root node in node_names.
        fingertip_indices: Tuple of indices of the fingertip nodes.
        actuated_joint_names: Ordered tuple of actuated named joints.
    """

    node_names: tuple[str, ...]
    node_features: torch.Tensor  # [L, D_node]
    edge_index: torch.Tensor  # [2, E]
    edge_features: torch.Tensor  # [E, D_edge]
    palm_index: int
    fingertip_indices: tuple[int, ...]
    actuated_joint_names: tuple[str, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1]) if self.edge_index.numel() > 0 else 0

    def to(self, device: torch.device | str) -> "HandGraph":
        return HandGraph(
            node_names=self.node_names,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            palm_index=self.palm_index,
            fingertip_indices=self.fingertip_indices,
            actuated_joint_names=self.actuated_joint_names,
        )

    def memory_bytes(self) -> int:
        """Total memory consumed by tensors in this graph."""
        return int(
            self.node_features.element_size() * self.node_features.nelement()
            + self.edge_index.element_size() * self.edge_index.nelement()
            + self.edge_features.element_size() * self.edge_features.nelement()
        )
