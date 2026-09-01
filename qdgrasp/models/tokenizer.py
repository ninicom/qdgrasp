"""Point tokenizer: packed integer keys, pure Torch, no dense N x N (P4-01).

The tokenizer turns a raw point cloud into an ordered set of voxel tokens plus
the map back to the points that made them.  Three constraints from ``PLAN.md``
§4 shape it, and each one is enforced rather than documented.

*Packed keys are checked for overflow.*  Three grid coordinates are packed into
one integer so that "same cell" is an integer equality instead of a distance
query.  A grid fine enough to overflow the packing silently aliases distant
points onto one token, so the packing refuses instead.

*No hash collisions.*  The key is a positional encoding of the grid coordinate,
not a hash of it.  Two points in different cells cannot collide, because the
mapping is injective by construction over the declared grid.

*No custom C++/CUDA, and no ``[N, N]``.*  Everything is sort, unique and scatter,
which Torch provides for every backend the project ships to.  Nothing here
allocates a tensor whose size is quadratic in the point count.
"""

from __future__ import annotations

import dataclasses

import torch

#: int64 has 63 usable bits; three coordinates therefore get 21 bits each.  A
#: grid dimension beyond 2**21 cells cannot be packed and is refused.
_BITS_PER_AXIS = 21
_MAX_GRID = 1 << _BITS_PER_AXIS


class TokenizerError(ValueError):
    """The point cloud or the grid cannot be tokenised as asked."""


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    """Voxel grid the points are quantised onto."""

    #: Edge length of one voxel, in metres.
    voxel_size: float = 0.005
    #: Half-extent of the region the grid covers, in metres.  Points outside are
    #: clamped onto the boundary cell rather than dropped, so the token count
    #: never depends silently on how far an outlier flew.
    extent: float = 0.5

    def validate(self) -> None:
        if not self.voxel_size > 0.0:
            raise TokenizerError(f"voxel_size must be positive, got {self.voxel_size}")
        if not self.extent > 0.0:
            raise TokenizerError(f"extent must be positive, got {self.extent}")
        if self.grid_size > _MAX_GRID:
            raise TokenizerError(
                f"grid of {self.grid_size} cells per axis exceeds the {_MAX_GRID} that fit in "
                f"{_BITS_PER_AXIS} bits; packing it would alias distant points onto one token"
            )

    @property
    def grid_size(self) -> int:
        return int(2.0 * self.extent / self.voxel_size) + 1


@dataclasses.dataclass
class TokenizedPoints:
    """Voxel tokens and the exact map back to the points that formed them."""

    #: Mean position of the points in each token, [B, T, 3].
    token_positions: torch.Tensor
    #: 1 where the token is real, 0 where it is padding, [B, T].
    token_mask: torch.Tensor
    #: Token index of every input point, [B, N].  Padding points get ``-1``.
    point_to_token: torch.Tensor
    #: Number of points in each token, [B, T].
    token_counts: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.token_positions.shape[0])

    @property
    def max_tokens(self) -> int:
        return int(self.token_positions.shape[1])


def pack_grid_coordinates(coordinates: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Pack non-negative integer grid coordinates into one int64 key.

    Injective over ``[0, grid_size)^3``: the key is a base-``grid_size``
    positional encoding, so distinct cells cannot share a key.  That is the
    difference between this and a spatial hash, and it is the property the
    "no hash collision" requirement asks for.
    """

    if grid_size > _MAX_GRID:
        raise TokenizerError(f"grid_size {grid_size} exceeds the packable maximum {_MAX_GRID}")
    coordinates = coordinates.to(torch.int64)
    if not torch.jit.is_tracing() and (
        bool(torch.any(coordinates < 0)) or bool(torch.any(coordinates >= grid_size))
    ):
        raise TokenizerError("grid coordinates must lie within [0, grid_size)")
    return (coordinates[..., 0] * grid_size + coordinates[..., 1]) * grid_size + coordinates[..., 2]


def unpack_grid_key(keys: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Inverse of :func:`pack_grid_coordinates`, used to prove injectivity."""

    keys = keys.to(torch.int64)
    z = keys % grid_size
    y = (keys // grid_size) % grid_size
    x = keys // (grid_size * grid_size)
    return torch.stack([x, y, z], dim=-1)


def quantize(points: torch.Tensor, config: TokenizerConfig) -> torch.Tensor:
    """Map metric points onto integer grid coordinates, clamping to the extent."""

    config.validate()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise TokenizerError(f"points must have shape [B, N, 3], got {tuple(points.shape)}")
    if not torch.jit.is_tracing() and not torch.all(torch.isfinite(points)):
        raise TokenizerError("points contain NaN or Inf")
    shifted = (points + config.extent) / config.voxel_size
    return torch.clamp(torch.floor(shifted), 0, config.grid_size - 1).to(torch.int64)


def tokenize_points_dense(points: torch.Tensor, config: TokenizerConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise without a Python loop or a data-dependent shape.

    Same tokens as :func:`tokenize_points`, in the same order, but the token
    axis is padded to the point count instead of to the number of occupied
    voxels.  That upper bound is what makes the routine traceable: the number of
    tokens a cloud produces is a property of the data, and a tracer that reads
    it turns one observation's voxel layout into the exported model's topology.

    Padding slots carry a zero mask, exactly as the sparse form's do, so the
    encoder sees the same keys and the same masked mean -- only the amount of
    padding differs.

    Returns:
        ``(token_positions [B, N, 3], token_mask [B, N])``.
    """

    config.validate()
    coordinates = quantize(points, config)
    keys = pack_grid_coordinates(coordinates, config.grid_size)  # [B, N]

    sorted_keys, order = torch.sort(keys, dim=1)
    starts = torch.ones_like(sorted_keys, dtype=torch.bool)
    starts[:, 1:] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
    token_index = torch.cumsum(starts.to(torch.int64), dim=1) - 1  # [B, N]

    gathered = torch.gather(points, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    sums = torch.zeros_like(points)
    sums.scatter_add_(1, token_index.unsqueeze(-1).expand(-1, -1, 3), gathered)
    counts = torch.zeros(keys.shape, dtype=points.dtype, device=points.device)
    counts.scatter_add_(1, token_index, torch.ones_like(counts))

    occupied = counts > 0
    means = sums / counts.clamp(min=1.0).unsqueeze(-1)
    return means, occupied.to(points.dtype)


def tokenize_points(points: torch.Tensor, config: TokenizerConfig) -> TokenizedPoints:
    """Group points into voxel tokens.

    The whole routine is sort/unique/scatter over a ``[B, N]`` key tensor.  Peak
    memory is linear in the point count, which is what makes doubling the token
    budget cost about twice the memory rather than four times it.
    """

    config.validate()
    coordinates = quantize(points, config)
    keys = pack_grid_coordinates(coordinates, config.grid_size)  # [B, N]
    batch_size, point_count = keys.shape
    device = points.device

    token_positions: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    assignments = torch.full((batch_size, point_count), -1, dtype=torch.int64, device=device)

    per_batch_tokens = []
    for index in range(batch_size):
        unique_keys, inverse, counts = torch.unique(keys[index], sorted=True, return_inverse=True, return_counts=True)
        assignments[index] = inverse
        token_count = int(unique_keys.shape[0])
        per_batch_tokens.append(token_count)

        summed = torch.zeros((token_count, 3), dtype=points.dtype, device=device)
        summed.index_add_(0, inverse, points[index])
        means = summed / counts.unsqueeze(-1).to(points.dtype)
        token_positions.append(means)
        token_counts.append(counts)

    max_tokens = max(per_batch_tokens)
    padded_positions = torch.zeros((batch_size, max_tokens, 3), dtype=points.dtype, device=device)
    padded_counts = torch.zeros((batch_size, max_tokens), dtype=torch.int64, device=device)
    mask = torch.zeros((batch_size, max_tokens), dtype=points.dtype, device=device)
    for index, count in enumerate(per_batch_tokens):
        padded_positions[index, :count] = token_positions[index]
        padded_counts[index, :count] = token_counts[index]
        mask[index, :count] = 1.0

    return TokenizedPoints(
        token_positions=padded_positions,
        token_mask=mask,
        point_to_token=assignments,
        token_counts=padded_counts,
    )


def scatter_tokens_to_points(token_features: torch.Tensor, tokenized: TokenizedPoints) -> torch.Tensor:
    """Broadcast per-token features back onto the points that formed them.

    This is what keeps raw-point resolution available after pooling: the encoder
    works on tokens, and every point can still read the feature of its own cell.
    """

    if token_features.ndim != 3:
        raise TokenizerError(f"token_features must have shape [B, T, C], got {tuple(token_features.shape)}")
    batch_size, _, channels = token_features.shape
    assignments = tokenized.point_to_token
    if assignments.shape[0] != batch_size:
        raise TokenizerError("token features and point assignments disagree about the batch size")
    index = assignments.clamp(min=0).unsqueeze(-1).expand(-1, -1, channels)
    gathered = torch.gather(token_features, 1, index)
    valid = (assignments >= 0).unsqueeze(-1).to(gathered.dtype)
    return gathered * valid
