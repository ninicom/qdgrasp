"""Serialized point encoder with shifted local windows (P4-02).

``PLAN.md`` §6 requires a profile showing the model never builds an ``N x N``
tensor, and that doubling the token count does not roughly quadruple memory.
Global attention over raw points fails both, so the encoder does not use it.

Tokens arrive already serialized: :func:`~qdgrasp.models.tokenizer.tokenize_points`
sorts by packed grid key, which is an x-major ordering of the voxel grid, so
neighbouring indices are usually neighbouring cells.  Attention then runs inside
fixed-size windows of that order, and alternate blocks shift the window by half
its width so information crosses the boundaries a fixed partition would create.

The cost is ``B * T * W`` rather than ``B * T^2``.  With ``W`` fixed, doubling
``T`` doubles the work -- which is the property the gate measures, not a claim
the docstring makes.

A single pooled token carries global context at ``O(T)``, so "the encoder cannot
see the whole object" is not the price paid for dropping global attention.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class EncoderConfig:
    """Shape of the point encoder.  ``channels`` is per-stage width."""

    channels: tuple[int, ...] = (32, 64, 128, 192)
    depths: tuple[int, ...] = (1, 1, 2, 2)
    window: int = 32
    heads: int = 4
    #: Number of Fourier bands used to embed a token's position.
    position_bands: int = 8

    def validate(self) -> None:
        if len(self.channels) != len(self.depths):
            raise ValueError("channels and depths must describe the same number of stages")
        if not self.channels or any(width <= 0 for width in self.channels):
            raise ValueError("every stage needs a positive width")
        if any(depth <= 0 for depth in self.depths):
            raise ValueError("every stage needs at least one block")
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if any(width % self.heads for width in self.channels):
            raise ValueError(f"every stage width must be divisible by heads={self.heads}")
        if self.position_bands <= 0:
            raise ValueError("position_bands must be positive")

    @property
    def output_channels(self) -> int:
        return self.channels[-1]


class FourierPositionEmbedding(nn.Module):
    """Sine/cosine embedding of a metric position, at fixed frequencies.

    The bands are geometric and *not* learned, so a position means the same
    thing to a checkpoint trained yesterday and one trained today.  Learned
    frequencies would make the embedding a moving target across runs.
    """

    def __init__(self, bands: int, channels: int) -> None:
        super().__init__()
        self.bands = bands
        frequencies = 2.0 ** torch.arange(bands, dtype=torch.float32) * torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.project = nn.Linear(3 + 3 * 2 * bands, channels)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        scaled = positions.unsqueeze(-1) * self.frequencies  # [B, T, 3, bands]
        features = torch.cat([positions, scaled.sin().flatten(-2), scaled.cos().flatten(-2)], dim=-1)
        return self.project(features)


def _window_partition(x: torch.Tensor, window: int, shift: int) -> tuple[torch.Tensor, int, int]:
    """Split ``[B, T, C]`` into ``[B * W_count, window, C]``, padding the tail."""

    batch, tokens, channels = x.shape
    if shift:
        x = torch.roll(x, shifts=-shift, dims=1)
    pad = (window - tokens % window) % window
    if pad:
        x = torch.cat([x, x.new_zeros(batch, pad, channels)], dim=1)
    total = x.shape[1]
    return x.reshape(batch * (total // window), window, channels), pad, total


def _window_merge(windows: torch.Tensor, batch: int, tokens: int, pad: int, total: int, shift: int) -> torch.Tensor:
    """Inverse of :func:`_window_partition`."""

    channels = windows.shape[-1]
    x = windows.reshape(batch, total, channels)
    if pad:
        x = x[:, : total - pad]
    if shift:
        x = torch.roll(x, shifts=shift, dims=1)
    return x[:, :tokens]


class WindowedBlock(nn.Module):
    """Pre-norm attention inside one window, then an MLP.

    Masked tokens are excluded from attention rather than merely zeroed, because
    a padded token that participates in the softmax still moves every real token
    it shares a window with.
    """

    def __init__(self, channels: int, heads: int, window: int, shift: int) -> None:
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm_attention = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        normed = self.norm_attention(x)
        windows, pad, total = _window_partition(normed, self.window, self.shift)

        mask_channels = mask.unsqueeze(-1)
        mask_windows, _, _ = _window_partition(mask_channels, self.window, self.shift)
        key_padding = mask_windows.squeeze(-1) <= 0.0  # [B*W_count, window]
        # A window of nothing but padding has no valid key; attending in it
        # produces NaN, so it is allowed to attend to itself and masked out after.
        empty = key_padding.all(dim=-1)
        key_padding = torch.where(empty.unsqueeze(-1), torch.zeros_like(key_padding), key_padding)

        attended, _ = self.attention(windows, windows, windows, key_padding_mask=key_padding, need_weights=False)
        attended = torch.where(empty.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(attended), attended)
        merged = _window_merge(attended, batch, tokens, pad, total, self.shift)
        x = x + merged * mask.unsqueeze(-1)
        return x + self.mlp(self.norm_mlp(x)) * mask.unsqueeze(-1)


class PointEncoder(nn.Module):
    """Stages of windowed blocks over serialized voxel tokens."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        self.config.validate()
        self.embedding = FourierPositionEmbedding(self.config.position_bands, self.config.channels[0])

        stages: list[nn.Module] = []
        projections: list[nn.Module] = []
        for stage, (width, depth) in enumerate(zip(self.config.channels, self.config.depths, strict=True)):
            previous = self.config.channels[stage - 1] if stage else width
            projections.append(nn.Linear(previous, width) if previous != width else nn.Identity())
            blocks = nn.ModuleList(
                WindowedBlock(
                    width,
                    self.config.heads,
                    self.config.window,
                    shift=(self.config.window // 2) if index % 2 else 0,
                )
                for index in range(depth)
            )
            stages.append(blocks)
        self.stages = nn.ModuleList(stages)
        self.projections = nn.ModuleList(projections)
        self.final_norm = nn.LayerNorm(self.config.output_channels)

    def forward(self, token_positions: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, T, 3]`` positions into ``[B, T, C]`` token features."""

        x = self.embedding(token_positions) * token_mask.unsqueeze(-1)
        for projection, blocks in zip(self.projections, self.stages, strict=True):
            x = projection(x) * token_mask.unsqueeze(-1)
            for block in blocks:
                x = block(x, token_mask)
        return self.final_norm(x) * token_mask.unsqueeze(-1)


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid tokens, ``[B, T, C] -> [B, C]``, safe for an empty mask."""

    weights = mask.unsqueeze(-1)
    total = (features * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return total / count
