"""A traceable façade over QDGrasp-Flow (``PLAN.md`` §9.8).

``GraspFlowModel.forward`` cannot be exported and should not be: it takes a
``HandGraph`` and a ``RobotSpec``, draws its own noise, and returns a dataclass.
A tracer records the draw as a constant and the graph objects as nothing at all,
so what comes out looks like a model and is a photograph of one sample.

This adapter fixes each of those in the only place where it is safe to: at the
export boundary.

*The hand is bound, not passed.*  A published artifact serves one hand, so the
graph tensors become buffers of the adapter and the joint order becomes part of
its recorded metadata.

*The noise is an input.*  Generation stays stochastic; where the sample comes
from becomes the caller's decision, which is also what makes the export
reproducible enough to compare against eager.

*The topology is a declared constant.*  Tokenisation uses the dense form, and
its token axis is then padded to one capacity fixed at construction.  A trace
whose token axis follows the input is only valid at point counts congruent to
the traced one -- the windowed encoder pads its tail by a Python integer -- and
that is the kind of constraint nobody discovers until a runtime returns the
wrong answer.  With a declared capacity the graph is the same graph for every
cloud it accepts, and a cloud it does not accept is refused here rather than in
the tracer's arithmetic.
"""

from __future__ import annotations

import torch
from torch import nn

from ..models.flow import GraspFlowModel
from ..models.tokenizer import tokenize_points_dense
from ..robot.graph import HandGraph
from ..robot.spec import RobotSpec


class FlowExportAdapter(nn.Module):
    """``(points, noise) -> (translation, rotation, joints, score)``, tensors only."""

    def __init__(self, model: GraspFlowModel, robot: RobotSpec, *, max_points: int = 2048) -> None:
        super().__init__()
        if max_points < 1:
            raise ValueError(f"max_points must be positive, got {max_points}")
        self.model = model
        self.max_points = int(max_points)
        window = model.point_encoder.config.window
        #: Token slots the exported graph always carries.  A cloud yields at
        #: most one token per point, so this bounds every input it accepts.
        self.token_capacity = -(-self.max_points // window) * window
        self.joint_names: tuple[str, ...] = tuple(robot.actuated_joint_names)
        self.robot_name = str(robot.config.name)
        self.frame = str(robot.config.frame)

        graph = robot.to_hand_graph()
        self._graph_fields = tuple(
            name for name, value in vars(graph).items() if isinstance(value, torch.Tensor)
        )
        for name in self._graph_fields:
            self.register_buffer(f"graph_{name}", getattr(graph, name).clone(), persistent=False)
        self._graph_extras = {
            name: value for name, value in vars(graph).items() if not isinstance(value, torch.Tensor)
        }

        limits = [robot.config.joint_limits[name] for name in self.joint_names]
        self.register_buffer("joint_lower", torch.tensor([value[0] for value in limits]), persistent=False)
        self.register_buffer("joint_upper", torch.tensor([value[1] for value in limits]), persistent=False)

    @property
    def state_dimension(self) -> int:
        return self.model.flow_config.state_dimension

    def _hand_graph(self) -> HandGraph:
        fields = {name: getattr(self, f"graph_{name}") for name in self._graph_fields}
        return HandGraph(**{**self._graph_extras, **fields})

    def forward(self, points: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Generate one grasp per row from an explicit starting point.

        Args:
            points: ``[B, N, 3]`` object points, in the frame the bundle declares.
            noise: ``[B, state_dimension]`` starting state of the flow.

        Returns:
            ``(translation [B, 3], rotation [B, 3, 3], joints [B, J], score [B])``.
        """

        if not torch.jit.is_tracing() and points.shape[1] > self.max_points:
            raise ValueError(
                f"this adapter was built for at most {self.max_points} points and was given "
                f"{points.shape[1]}; rebuild it with a larger max_points rather than letting the export "
                "silently drop geometry"
            )
        token_positions, token_mask = tokenize_points_dense(points, self.model.tokenizer_config)
        token_positions, token_mask = self._pad_to_capacity(token_positions, token_mask)
        token_features = self.model.point_encoder(token_positions, token_mask)
        keys = self.model.point_projection(token_features)

        hand = self.model.hand_encoder(self._hand_graph())
        queries = self.model.hand_projection(hand.nodes).unsqueeze(0).expand(points.shape[0], -1, -1)
        for block in self.model.conditioning:
            queries = block(queries, keys, token_mask)

        from ..models.encoder import masked_mean

        pooled = self.model.conditioning_pool(queries.mean(dim=1) + masked_mean(keys, token_mask))
        state = self.model.sample_state(pooled, noise=noise)

        translation = state[:, :3]
        from ..models.flow import clamp_to_limits, rotation_from_9d

        rotation = rotation_from_9d(state[:, 3:12])
        joints = clamp_to_limits(state[:, 12 : 12 + len(self.joint_names)], self.joint_lower, self.joint_upper)
        score = torch.sigmoid(self.model.quality(pooled, state))
        return translation, rotation, joints, score

    def _pad_to_capacity(
        self, token_positions: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Give the token axis the fixed length the exported graph declares."""

        tokens = token_positions.shape[1]
        missing = self.token_capacity - tokens
        if missing == 0:
            return token_positions, token_mask
        if missing < 0:
            return token_positions[:, : self.token_capacity], token_mask[:, : self.token_capacity]
        pad_positions = token_positions.new_zeros((token_positions.shape[0], missing, 3))
        pad_mask = token_mask.new_zeros((token_mask.shape[0], missing))
        return (
            torch.cat([token_positions, pad_positions], dim=1),
            torch.cat([token_mask, pad_mask], dim=1),
        )

    def example_inputs(self, *, points: int = 512, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """A deterministic pair a tracer can be handed."""

        generator = torch.Generator().manual_seed(0)
        return (
            torch.randn(batch, points, 3, generator=generator) * 0.05,
            torch.randn(batch, self.state_dimension, generator=generator),
        )

    def output_schema(self) -> dict[str, object]:
        """What each returned tensor is, recorded beside the artifact."""

        return {
            "outputs": ["palm_translation", "palm_rotation", "joint_angles", "quality_score"],
            "layouts": ["[B, 3]", "[B, 3, 3]", f"[B, {len(self.joint_names)}]", "[B]"],
            "joint_names": list(self.joint_names),
            "robot_name": self.robot_name,
            "frame": self.frame,
            "inputs": ["points", "noise"],
            "input_layouts": ["[B, N, 3]", f"[B, {self.state_dimension}]"],
            "max_points": self.max_points,
            "token_capacity": self.token_capacity,
        }


__all__ = ("FlowExportAdapter",)
