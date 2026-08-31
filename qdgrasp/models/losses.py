"""Loss assembly for QDGrasp-Flow (P4-09).

The total is the sum of its logged terms, checked on construction, for the same
reason the RL contract enforces it in P3.5: a loss with an unlogged component
cannot be attributed when it moves.

The flow term is the one that actually trains the generator.  Rectified flow's
simplification is that the target velocity along the straight segment from noise
to data is constant -- ``target - noise`` -- so the objective is a plain
regression onto that, with the time sampled uniformly.  The pose terms are
supervision on the *decoded* sample, and the FK term ties the two together: it
can only be reduced by moving the palm and the joints, because the fingertips
are computed from them rather than predicted alongside them.

Rotation error is geodesic, not elementwise.  Two rotation matrices can differ
in every entry and describe nearly the same orientation; an elementwise loss
reports that as large, and the gradient it produces points somewhere unhelpful.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from qdgrasp.models.flow import GraspFlowModel, GraspPrediction
from qdgrasp.robot.graph import HandGraph
from qdgrasp.robot.spec import RobotSpec

#: Every term the total may contain.  An unknown key is a mistake, not a feature.
LOSS_TERMS: tuple[str, ...] = (
    "flow_velocity",
    "palm_translation",
    "palm_rotation",
    "joint",
    "fk_consistency",
    "quality",
)


@dataclasses.dataclass(frozen=True)
class LossWeights:
    """Relative weight of each term."""

    flow_velocity: float = 1.0
    palm_translation: float = 1.0
    palm_rotation: float = 1.0
    joint: float = 1.0
    fk_consistency: float = 1.0
    quality: float = 0.1

    def validate(self) -> None:
        for name in LOSS_TERMS:
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"loss weight {name} must be non-negative, got {value}")


@dataclasses.dataclass
class LossBreakdown:
    """Per-term losses whose sum is the total, by construction."""

    terms: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        unknown = sorted(set(self.terms) - set(LOSS_TERMS))
        if unknown:
            raise ValueError(f"unknown loss terms: {unknown}")

    @property
    def total(self) -> torch.Tensor:
        return sum(self.terms.values())  # type: ignore[return-value]

    def to_document(self) -> dict[str, float]:
        document = {name: float(self.terms.get(name, torch.zeros(())).detach()) for name in LOSS_TERMS}
        document["total"] = float(self.total.detach())
        return document


def geodesic_rotation_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Angle between two rotations, in radians, safe at the branch points.

    ``arccos`` has infinite derivative at ``+-1``, which is exactly where a
    well-fit model lives, so the argument is clamped just inside the domain.
    """

    relative = predicted.transpose(-1, -2) @ target
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(cosine)


def compute_losses(
    model: GraspFlowModel,
    prediction: GraspPrediction,
    conditioning: torch.Tensor,
    *,
    palm_pos: torch.Tensor,
    palm_rot: torch.Tensor,
    joint_angles: torch.Tensor,
    fingertip_positions: torch.Tensor,
    success: torch.Tensor,
    weights: LossWeights | None = None,
    generator: torch.Generator | None = None,
) -> LossBreakdown:
    """Assemble every term for one batch."""

    settings = weights or LossWeights()
    settings.validate()

    target_state = model.encode_target(palm_pos, palm_rot, joint_angles)
    noise = torch.randn(target_state.shape, device=target_state.device, dtype=target_state.dtype, generator=generator)
    time = torch.rand(target_state.shape[0], device=target_state.device, dtype=target_state.dtype, generator=generator)
    interpolated, velocity_target = model.velocity_target(target_state, noise, time)
    velocity_prediction = model.velocity(interpolated, time, conditioning)

    terms = {
        "flow_velocity": settings.flow_velocity * nn.functional.mse_loss(velocity_prediction, velocity_target),
        "palm_translation": settings.palm_translation * nn.functional.mse_loss(prediction.palm_translation, palm_pos),
        "palm_rotation": settings.palm_rotation * geodesic_rotation_error(prediction.palm_rotation, palm_rot).mean(),
        "joint": settings.joint * nn.functional.mse_loss(prediction.joint_angles, joint_angles),
        "fk_consistency": settings.fk_consistency * nn.functional.mse_loss(prediction.fingertips, fingertip_positions),
        "quality": settings.quality
        * nn.functional.binary_cross_entropy_with_logits(
            prediction.quality_logit, success.to(prediction.quality_logit.dtype)
        ),
    }
    return LossBreakdown(terms=terms)


def forward_and_loss(
    model: GraspFlowModel,
    robot: RobotSpec,
    graph: HandGraph,
    *,
    points: torch.Tensor,
    palm_pos: torch.Tensor,
    palm_rot: torch.Tensor,
    joint_angles: torch.Tensor,
    fingertip_positions: torch.Tensor,
    success: torch.Tensor,
    weights: LossWeights | None = None,
    generator: torch.Generator | None = None,
    sample_noise: torch.Tensor | None = None,
) -> tuple[GraspPrediction, LossBreakdown]:
    """One training step's forward pass, sharing conditioning between both uses.

    The conditioning is computed once and used both to sample and to evaluate
    the velocity field.  Recomputing it would double the encoder cost and, more
    importantly, let the two paths drift apart under dropout.
    """

    conditioning, _hand = model.encode(points, graph)
    state = model.sample_state(conditioning, generator=generator, noise=sample_noise)
    translation, rotation, joints = model.decode(state, robot)
    prediction = GraspPrediction(
        palm_translation=translation,
        palm_rotation=rotation,
        joint_angles=joints,
        fingertips=robot.fingertip_positions(translation, rotation, joints),
        quality_logit=model.quality(conditioning).squeeze(-1),
        raw_state=state,
    )
    losses = compute_losses(
        model,
        prediction,
        conditioning,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        joint_angles=joint_angles,
        fingertip_positions=fingertip_positions,
        success=success,
        weights=weights,
        generator=generator,
    )
    return prediction, losses


def gradient_coverage(model: nn.Module) -> dict[str, bool]:
    """Which trainable parameters received a finite gradient.

    ``PLAN.md`` §6 requires that *every* trainable parameter gets one.  A
    parameter that never does is dead weight in a checkpoint and a silent hole
    in an ablation.
    """

    return {
        name: parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
