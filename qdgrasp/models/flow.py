"""World-edge conditioning and the rectified-flow grasp head (P4-04..07).

The flow generates an *executable* grasp directly: palm translation, palm
rotation as 9D, and named joint angles.  Nothing downstream has to run IK to
turn the output into something a hand can be commanded with, which is the point
-- an IK solver on the default inference path is a second system whose failures
get attributed to the model.

Four decisions are load-bearing.

*Conditioning is hand-queries-over-point-keys.*  Cross-attention runs with the
hand's ``L <= 24`` nodes as queries and the object's ``T`` tokens as keys, so the
attention matrix is ``[L, T]``.  The other direction would be ``[T, L]`` which is
the same size, but querying *from* the hand is what makes the conditioning
variable-length in the hand rather than in the object.

*Rotation is projected, not normalised.*  A 9D output is turned into a rotation
by Gram-Schmidt on its first two columns, which is differentiable everywhere and
needs no SVD.  Normalising a quaternion instead would reintroduce the double
cover the 9D representation exists to avoid.

*Joints are masked, then clamped to the profile's limits.*  The head emits a
fixed ``max_joints`` channels and the hand's own joint count selects from them,
so one checkpoint serves hands with different joint counts.  The clamp is a
``tanh`` onto the named limits rather than a hard cut, because a hard cut has
zero gradient exactly where the model most needs to be pushed back in range.

*Fingertips come from FK, not from a head.*  Predicting them separately would
let the auxiliary loss disagree with the pose that was actually generated.
Running the profile's differentiable FK on the generated pose means the keypoint
loss can only be satisfied by fixing the pose.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from qdgrasp.models.encoder import EncoderConfig, PointEncoder, masked_mean
from qdgrasp.models.hand_graph import HandGraphEmbedding, HandGraphEncoder, HandGraphEncoderConfig
from qdgrasp.models.tokenizer import TokenizerConfig, tokenize_points
from qdgrasp.robot.graph import HandGraph
from qdgrasp.robot.spec import RobotSpec


def rotation_from_9d(raw: torch.Tensor) -> torch.Tensor:
    """Project a raw 9-vector onto SO(3) by Gram-Schmidt.

    Differentiable everywhere the first two columns are independent, which a
    random initialisation satisfies with probability one.  The third column is
    the cross product, so ``det = +1`` by construction rather than by luck.
    """

    if raw.shape[-1] != 9:
        raise ValueError(f"expected a 9-vector, got {raw.shape[-1]}")
    columns = raw.reshape(*raw.shape[:-1], 3, 3)
    first = torch.nn.functional.normalize(columns[..., 0], dim=-1, eps=1e-8)
    second = columns[..., 1] - (first * columns[..., 1]).sum(-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def clamp_to_limits(raw: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Squash raw joint outputs into ``[lower, upper]`` with a live gradient."""

    centre = (upper + lower) / 2.0
    half = (upper - lower) / 2.0
    return centre + half * torch.tanh(raw)


@dataclasses.dataclass(frozen=True)
class FlowConfig:
    """Shape of the conditioning and the flow head."""

    channels: int = 192
    heads: int = 4
    conditioning_layers: int = 2
    flow_layers: int = 3
    #: Channels the flow head reserves for joints.  A hand with fewer actuated
    #: joints uses a prefix of them; one with more is refused rather than
    #: silently truncated.
    max_joints: int = 32
    #: Euler steps used to integrate the straight-path velocity field.
    flow_steps: int = 5
    time_bands: int = 6

    def validate(self) -> None:
        if self.channels % self.heads:
            raise ValueError(f"channels={self.channels} must be divisible by heads={self.heads}")
        for name in ("conditioning_layers", "flow_layers", "max_joints", "flow_steps", "time_bands"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def state_dimension(self) -> int:
        """Translation (3) + rotation 9D (9) + joint channels."""

        return 3 + 9 + self.max_joints


class CrossAttentionBlock(nn.Module):
    """Hand nodes attend to object tokens; ``[L, T]``, never ``[T, T]``."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(channels)
        self.norm_key = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.norm_query(queries),
            self.norm_key(keys),
            self.norm_key(keys),
            key_padding_mask=key_mask <= 0.0,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.mlp(self.norm_mlp(queries))


class TimeEmbedding(nn.Module):
    """Fourier embedding of the flow time, at fixed frequencies."""

    def __init__(self, bands: int, channels: int) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(bands, dtype=torch.float32) * torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.project = nn.Sequential(nn.Linear(1 + 2 * bands, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        scaled = time.unsqueeze(-1) * self.frequencies
        return self.project(torch.cat([time.unsqueeze(-1), scaled.sin(), scaled.cos()], dim=-1))


class VelocityField(nn.Module):
    """Predicts ``dx/dt`` for the straight path between noise and the grasp."""

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        channels = config.channels
        self.state_input = nn.Linear(config.state_dimension, channels)
        self.time = TimeEmbedding(config.time_bands, channels)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(channels), nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels)
            )
            for _ in range(config.flow_layers)
        )
        self.output = nn.Linear(channels, config.state_dimension)
        # Start near zero so an untrained field is a small perturbation rather
        # than a shove into a region the clamps then have to rescue.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, time: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        hidden = self.state_input(state) + self.time(time) + conditioning
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class CandidateQualityHead(nn.Module):
    """Score one grasp candidate in the context that produced it.

    A grasp score is only useful for ranking when it depends on the candidate.
    Keeping the observation and latent state as separate arguments makes that
    contract explicit at every call site and lets a batch contain positive and
    negative grasps that share exactly the same observation.
    """

    def __init__(self, conditioning_channels: int, state_dimension: int) -> None:
        super().__init__()
        self.conditioning_channels = conditioning_channels
        self.state_dimension = state_dimension
        inputs = conditioning_channels + state_dimension
        hidden = max(conditioning_channels // 2, 1)
        self.network = nn.Sequential(
            nn.LayerNorm(inputs),
            nn.Linear(inputs, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, conditioning: torch.Tensor, candidate_state: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim != 2 or candidate_state.ndim != 2:
            raise ValueError(
                "quality expects conditioning [B, C] and candidate_state [B, D], got "
                f"{tuple(conditioning.shape)} and {tuple(candidate_state.shape)}"
            )
        if conditioning.shape[0] != candidate_state.shape[0]:
            raise ValueError(
                "quality conditioning and candidate_state must have the same batch size, got "
                f"{conditioning.shape[0]} and {candidate_state.shape[0]}"
            )
        if conditioning.shape[1] != self.conditioning_channels or candidate_state.shape[1] != self.state_dimension:
            raise ValueError(
                "quality feature dimensions must be "
                f"C={self.conditioning_channels}, D={self.state_dimension}; got "
                f"C={conditioning.shape[1]}, D={candidate_state.shape[1]}"
            )
        return self.network(torch.cat([conditioning, candidate_state], dim=-1)).squeeze(-1)


@dataclasses.dataclass
class GraspPrediction:
    """One generated, executable grasp plus what it implies."""

    palm_translation: torch.Tensor  # [B, 3]
    palm_rotation: torch.Tensor  # [B, 3, 3]
    joint_angles: torch.Tensor  # [B, J]
    fingertips: torch.Tensor  # [B, K, 3]
    quality_logit: torch.Tensor  # [B]
    raw_state: torch.Tensor  # [B, state_dim]

    def is_finite(self) -> bool:
        return all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                self.palm_translation,
                self.palm_rotation,
                self.joint_angles,
                self.fingertips,
                self.quality_logit,
            )
        )


class GraspFlowModel(nn.Module):
    """Object points plus a hand graph in; an executable grasp out."""

    def __init__(
        self,
        encoder: EncoderConfig | None = None,
        hand: HandGraphEncoderConfig | None = None,
        flow: FlowConfig | None = None,
        tokenizer: TokenizerConfig | None = None,
    ) -> None:
        super().__init__()
        self.flow_config = flow or FlowConfig()
        self.flow_config.validate()
        self.tokenizer_config = tokenizer or TokenizerConfig()
        self.tokenizer_config.validate()

        channels = self.flow_config.channels
        self.point_encoder = PointEncoder(encoder or EncoderConfig())
        self.hand_encoder = HandGraphEncoder(hand or HandGraphEncoderConfig())
        self.point_projection = nn.Linear(self.point_encoder.config.output_channels, channels)
        self.hand_projection = nn.Linear(self.hand_encoder.output_channels, channels)
        self.conditioning = nn.ModuleList(
            CrossAttentionBlock(channels, self.flow_config.heads) for _ in range(self.flow_config.conditioning_layers)
        )
        self.conditioning_pool = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels))
        self.velocity = VelocityField(self.flow_config)
        self.quality = CandidateQualityHead(channels, self.flow_config.state_dimension)

    # -- conditioning ------------------------------------------------------

    def encode(
        self,
        points: torch.Tensor,
        graph: HandGraph,
        point_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HandGraphEmbedding]:
        """Fuse the object and the hand into one conditioning vector per sample."""

        token_positions, token_mask = self._tokenize_observation(points, point_mask)
        token_features = self.point_encoder(token_positions, token_mask)
        keys = self.point_projection(token_features)

        hand = self.hand_encoder(graph)
        queries = self.hand_projection(hand.nodes).unsqueeze(0).expand(points.shape[0], -1, -1)
        for block in self.conditioning:
            queries = block(queries, keys, token_mask)
        pooled_hand = queries.mean(dim=1)
        pooled_object = masked_mean(keys, token_mask)
        return self.conditioning_pool(pooled_hand + pooled_object), hand

    def _tokenize_observation(
        self, points: torch.Tensor, point_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize real points only, excluding padded coordinates entirely.

        The regular tokenizer has no point-level mask because its public contract
        receives a dense cloud.  Dataset batches are padded, though, and a zero
        placeholder must not become a real voxel at the origin.  Ragged rows are
        therefore tokenized independently and repadded at the *token* level,
        where the encoder already has an explicit mask.
        """

        if point_mask is None:
            tokenized = tokenize_points(points, self.tokenizer_config)
            return tokenized.token_positions, tokenized.token_mask

        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape [B, N, 3], got {tuple(points.shape)}")
        mask = torch.as_tensor(point_mask, device=points.device)
        if mask.ndim == 1 and points.shape[0] == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != points.shape[:2]:
            raise ValueError(
                f"point_mask must have shape {tuple(points.shape[:2])}, got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            if not bool(torch.all((mask == 0) | (mask == 1))):
                raise ValueError("point_mask values must be boolean or 0/1")
            mask = mask.to(torch.bool)
        valid_counts = mask.sum(dim=1)
        if bool(torch.any(valid_counts == 0)):
            raise ValueError("each observation needs at least one valid point")

        rows: list[torch.Tensor] = []
        for index in range(points.shape[0]):
            tokenized = tokenize_points(points[index, mask[index]].unsqueeze(0), self.tokenizer_config)
            rows.append(tokenized.token_positions[0])

        max_tokens = max(row.shape[0] for row in rows)
        positions = points.new_zeros((points.shape[0], max_tokens, 3))
        token_mask = points.new_zeros((points.shape[0], max_tokens))
        for index, row in enumerate(rows):
            count = row.shape[0]
            positions[index, :count] = row
            token_mask[index, :count] = 1.0
        return positions, token_mask

    # -- generation --------------------------------------------------------

    def sample_state(
        self,
        conditioning: torch.Tensor,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the velocity field with a fixed-step Euler solver.

        ``noise`` pins the starting point.  Sampling is otherwise stochastic by
        design -- that is what a generative head is for -- but a diagnostic that
        asks "can this model reproduce one specific grasp" has to hold the draw
        fixed, or it is asking the model to map every noise vector to the same
        answer and calling the failure a wiring bug.
        """

        batch = conditioning.shape[0]
        if noise is not None:
            state = noise.to(device=conditioning.device, dtype=conditioning.dtype)
        else:
            state = torch.randn(
                batch,
                self.flow_config.state_dimension,
                device=conditioning.device,
                dtype=conditioning.dtype,
                generator=generator,
            )
        steps = self.flow_config.flow_steps
        step_size = 1.0 / steps
        for index in range(steps):
            time = torch.full((batch,), index * step_size, device=conditioning.device, dtype=conditioning.dtype)
            state = state + step_size * self.velocity(state, time, conditioning)
        return state

    def decode(self, state: torch.Tensor, robot: RobotSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Turn a raw flow state into a translation, a rotation and joints."""

        joint_count = len(robot.actuated_joint_names)
        if joint_count > self.flow_config.max_joints:
            raise ValueError(
                f"{robot.config.name} has {joint_count} actuated joints, beyond the head's "
                f"max_joints={self.flow_config.max_joints}; widen the head rather than truncating the hand"
            )
        translation = state[:, :3]
        rotation = rotation_from_9d(state[:, 3:12])
        raw_joints = state[:, 12 : 12 + joint_count]
        lower, upper = self._joint_limits(robot, state.device, state.dtype)
        return translation, rotation, clamp_to_limits(raw_joints, lower, upper)

    @staticmethod
    def _joint_limits(robot: RobotSpec, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        limits = [robot.config.joint_limits[name] for name in robot.actuated_joint_names]
        lower = torch.tensor([value[0] for value in limits], device=device, dtype=dtype)
        upper = torch.tensor([value[1] for value in limits], device=device, dtype=dtype)
        return lower, upper

    def forward(
        self,
        points: torch.Tensor,
        graph: HandGraph,
        robot: RobotSpec,
        generator: torch.Generator | None = None,
        *,
        point_mask: torch.Tensor | None = None,
    ) -> GraspPrediction:
        conditioning, _hand = self.encode(points, graph, point_mask=point_mask)
        state = self.sample_state(conditioning, generator=generator)
        translation, rotation, joints = self.decode(state, robot)
        fingertips = robot.fingertip_positions(translation, rotation, joints)
        return GraspPrediction(
            palm_translation=translation,
            palm_rotation=rotation,
            joint_angles=joints,
            fingertips=fingertips,
            quality_logit=self.quality(conditioning, state),
            raw_state=state,
        )

    # -- training ----------------------------------------------------------

    def velocity_target(
        self, target_state: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The straight path from noise to target, and its constant velocity.

        Rectified flow's whole simplification is that the target velocity does
        not depend on time: it is ``target - noise`` everywhere on the segment.
        """

        interpolated = noise + time.unsqueeze(-1) * (target_state - noise)
        return interpolated, target_state - noise

    def encode_target(
        self,
        batch_palm_pos: torch.Tensor,
        batch_palm_rot: torch.Tensor,
        batch_joints: torch.Tensor,
        robot: RobotSpec,
        *,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """Pack a physical grasp into the exact latent coordinates ``decode`` reads.

        Joint channels use the inverse of :func:`clamp_to_limits`.  Clamping the
        normalized coordinate just inside ``(-1, 1)`` keeps targets at a declared
        joint limit finite while retaining sub-microradian round-trip accuracy for
        the shipped profiles.
        """

        batch = batch_palm_pos.shape[0]
        joint_count = len(robot.actuated_joint_names)
        if batch_joints.ndim != 2 or batch_joints.shape != (batch, joint_count):
            raise ValueError(
                f"joint target for {robot.config.name} must have shape {(batch, joint_count)}, "
                f"got {tuple(batch_joints.shape)}"
            )
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must lie in (0, 1), got {epsilon}")

        lower, upper = self._joint_limits(robot, batch_joints.device, batch_joints.dtype)
        tolerance = 1e-6
        if bool(torch.any(batch_joints < lower - tolerance)) or bool(torch.any(batch_joints > upper + tolerance)):
            raise ValueError(f"joint target for {robot.config.name} lies outside its declared limits")
        centre = (upper + lower) / 2.0
        half = (upper - lower) / 2.0
        normalized = ((batch_joints - centre) / half).clamp(-1.0 + epsilon, 1.0 - epsilon)
        latent_joints = torch.atanh(normalized)

        joints = torch.zeros(
            batch, self.flow_config.max_joints, device=batch_palm_pos.device, dtype=batch_palm_pos.dtype
        )
        joints[:, :joint_count] = latent_joints.to(device=batch_palm_pos.device, dtype=batch_palm_pos.dtype)
        return torch.cat([batch_palm_pos, batch_palm_rot.reshape(batch, 9), joints], dim=-1)
