"""Variable-length HandGraph encoder (P4-03).

The point of cross-embodiment is that the model reads the hand it is given
rather than one it was compiled for.  LEAP arrives as 18 nodes and 17 edges,
Allegro as 22 and 21; both carry ``node_dim=17``, ``edge_dim=9`` and 16 actuated
joints, and neither count may be baked in.

So this is message passing on the edge list, not a fixed-size MLP over a padded
adjacency.  Nothing here allocates ``L x L``: messages are gathered along
``edge_index`` and scattered back, which costs ``O(E)`` for a tree-shaped hand
where ``E = L - 1``.

Edges are made symmetric before propagation.  A kinematic chain is directed
parent-to-child, but information has to travel both ways -- a fingertip needs to
know what the palm is doing as much as the reverse -- and one-directional
message passing on a tree only ever moves information toward the leaves.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from qdgrasp.robot.graph import HandGraph


@dataclasses.dataclass(frozen=True)
class HandGraphEncoderConfig:
    """Width and depth of the graph encoder."""

    channels: int = 128
    layers: int = 3
    node_dim: int = 17
    edge_dim: int = 9

    def validate(self) -> None:
        if self.channels <= 0 or self.layers <= 0:
            raise ValueError("channels and layers must be positive")
        if self.node_dim <= 0 or self.edge_dim <= 0:
            raise ValueError("node_dim and edge_dim must be positive")


def symmetrize(edge_index: torch.Tensor, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the reverse of every edge, tagging direction on the features.

    The direction flag matters: a joint seen from the child is not the same
    relation as the same joint seen from the parent, and collapsing the two
    would make the encoder unable to tell a hand from its mirror.
    """

    if edge_index.numel() == 0:
        return edge_index, edge_features
    reversed_index = edge_index.flip(0)
    forward_flag = edge_features.new_ones((edge_features.shape[0], 1))
    backward_flag = -forward_flag
    return (
        torch.cat([edge_index, reversed_index], dim=1),
        torch.cat(
            [
                torch.cat([edge_features, forward_flag], dim=-1),
                torch.cat([edge_features, backward_flag], dim=-1),
            ],
            dim=0,
        ),
    )


class MessagePassingLayer(nn.Module):
    """One round of gather-along-edges, scatter-to-nodes, update."""

    def __init__(self, channels: int, edge_channels: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(channels * 2 + edge_channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.update = nn.Sequential(nn.Linear(channels * 2, channels), nn.GELU(), nn.Linear(channels, channels))
        self.norm = nn.LayerNorm(channels)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(nodes + self.update(torch.cat([nodes, torch.zeros_like(nodes)], dim=-1)))
        source, target = edge_index[0], edge_index[1]
        messages = self.message(torch.cat([nodes[source], nodes[target], edge_features], dim=-1))
        aggregated = torch.zeros_like(nodes)
        aggregated.index_add_(0, target, messages)
        counts = torch.zeros(nodes.shape[0], 1, device=nodes.device, dtype=nodes.dtype)
        counts.index_add_(0, target, torch.ones_like(messages[:, :1]))
        aggregated = aggregated / counts.clamp(min=1.0)
        return self.norm(nodes + self.update(torch.cat([nodes, aggregated], dim=-1)))


@dataclasses.dataclass
class HandGraphEmbedding:
    """What the graph encoder hands to the rest of the model."""

    #: Per-node features, ``[L, C]``.
    nodes: torch.Tensor
    #: Pooled hand descriptor, ``[C]``.
    summary: torch.Tensor
    #: Palm node feature, ``[C]``.
    palm: torch.Tensor
    #: Fingertip node features, ``[K, C]``.
    fingertips: torch.Tensor
    #: Number of actuated joints this hand exposes.
    joint_count: int


class HandGraphEncoder(nn.Module):
    """Encode a hand's morphology without assuming its size."""

    def __init__(self, config: HandGraphEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or HandGraphEncoderConfig()
        self.config.validate()
        channels = self.config.channels
        self.node_input = nn.Linear(self.config.node_dim, channels)
        # +1 for the direction flag added by symmetrize.
        self.edge_input = nn.Linear(self.config.edge_dim + 1, channels)
        self.layers = nn.ModuleList(MessagePassingLayer(channels, channels) for _ in range(self.config.layers))
        self.output_norm = nn.LayerNorm(channels)

    @property
    def output_channels(self) -> int:
        return self.config.channels

    def forward(self, graph: HandGraph) -> HandGraphEmbedding:
        node_features = graph.node_features
        if node_features.shape[-1] != self.config.node_dim:
            raise ValueError(
                f"graph node_dim is {node_features.shape[-1]}, encoder was built for {self.config.node_dim}"
            )
        edge_index, edge_features = symmetrize(graph.edge_index, graph.edge_features)
        nodes = self.node_input(node_features)
        edges = (
            self.edge_input(edge_features) if edge_index.numel() else edge_features.new_zeros((0, self.config.channels))
        )
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edges)
        nodes = self.output_norm(nodes)
        return HandGraphEmbedding(
            nodes=nodes,
            summary=nodes.mean(dim=0),
            palm=nodes[graph.palm_index],
            fingertips=nodes[list(graph.fingertip_indices)],
            joint_count=len(graph.actuated_joint_names),
        )
