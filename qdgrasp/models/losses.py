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


def _validity_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Normalize one explicit sample-validity field to a strict boolean vector."""

    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    value = torch.as_tensor(mask, device=device)
    if value.ndim == 0 and batch_size == 1:
        value = value.reshape(1)
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must have shape {(batch_size,)}, got {tuple(value.shape)}")
    if value.dtype != torch.bool:
        if not bool(torch.all((value == 0) | (value == 1))):
            raise ValueError(f"{name} values must be boolean or 0/1")
        value = value.to(torch.bool)
    return value


def masked_sample_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    name: str = "target_valid",
) -> torch.Tensor:
    """Average per-sample values over valid samples only.

    All trailing dimensions are first averaged within a sample.  An empty mask
    returns a differentiable zero, so a placeholder-only batch is finite and can
    participate in a normal backward pass without manufacturing a gradient.
    """

    if values.ndim < 1:
        raise ValueError("masked_sample_mean expects a batch dimension")
    valid = _validity_mask(mask, batch_size=values.shape[0], device=values.device, name=name)
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1) if values.ndim > 1 else values
    selected = torch.where(valid, per_sample, torch.zeros_like(per_sample))
    return selected.sum() / valid.sum().clamp(min=1).to(selected.dtype)


def _replace_invalid(values: torch.Tensor, valid: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    """Replace invalid rows before arithmetic so NaN placeholders cannot leak through ``0 * NaN``."""

    shape = (valid.shape[0],) + (1,) * (values.ndim - 1)
    return torch.where(valid.reshape(shape), values, replacement)


def compute_losses(
    model: GraspFlowModel,
    prediction: GraspPrediction,
    conditioning: torch.Tensor,
    robot: RobotSpec,
    *,
    palm_pos: torch.Tensor,
    palm_rot: torch.Tensor,
    joint_angles: torch.Tensor,
    fingertip_positions: torch.Tensor,
    success: torch.Tensor,
    weights: LossWeights | None = None,
    generator: torch.Generator | None = None,
    kinematics_valid: torch.Tensor | None = None,
    pose_target_valid: torch.Tensor | None = None,
    joint_target_valid: torch.Tensor | None = None,
    fk_target_valid: torch.Tensor | None = None,
) -> LossBreakdown:
    """Assemble every term, excluding samples without the corresponding target."""

    settings = weights or LossWeights()
    settings.validate()

    batch_size = palm_pos.shape[0]
    device = palm_pos.device
    pose_valid = _validity_mask(
        pose_target_valid, batch_size=batch_size, device=device, name="pose_target_valid"
    )
    joint_valid = _validity_mask(
        joint_target_valid, batch_size=batch_size, device=device, name="joint_target_valid"
    )
    fk_valid = _validity_mask(fk_target_valid, batch_size=batch_size, device=device, name="fk_target_valid")
    # Old callers carry no validity fields and remain all-valid.  Once individual
    # fields are supplied, the natural fallback for the full flow target is the
    # intersection of pose and joint validity.
    if kinematics_valid is None:
        kinematics_mask = pose_valid & joint_valid
    else:
        kinematics_mask = _validity_mask(
            kinematics_valid, batch_size=batch_size, device=device, name="kinematics_valid"
        )
    flow_valid = kinematics_mask & pose_valid & joint_valid

    safe_palm_pos = _replace_invalid(palm_pos, pose_valid, torch.zeros_like(palm_pos))
    identity = torch.eye(3, device=palm_rot.device, dtype=palm_rot.dtype).expand_as(palm_rot)
    safe_palm_rot = _replace_invalid(palm_rot, pose_valid, identity)
    lower, upper = model._joint_limits(robot, joint_angles.device, joint_angles.dtype)
    centre = ((lower + upper) / 2.0).unsqueeze(0).expand_as(joint_angles)
    safe_joints = _replace_invalid(joint_angles, joint_valid, centre)
    safe_fingertips = _replace_invalid(
        fingertip_positions, fk_valid, torch.zeros_like(fingertip_positions)
    )

    target_state = model.encode_target(safe_palm_pos, safe_palm_rot, safe_joints, robot)
    noise = torch.randn(target_state.shape, device=target_state.device, dtype=target_state.dtype, generator=generator)
    time = torch.rand(target_state.shape[0], device=target_state.device, dtype=target_state.dtype, generator=generator)
    interpolated, velocity_target = model.velocity_target(target_state, noise, time)
    velocity_prediction = model.velocity(interpolated, time, conditioning)
    target_quality_logit = model.quality(conditioning, target_state)
    quality_target = torch.as_tensor(success, device=target_quality_logit.device, dtype=target_quality_logit.dtype)
    if quality_target.ndim == 2 and quality_target.shape[1] == 1:
        quality_target = quality_target[:, 0]
    if quality_target.shape != (batch_size,):
        raise ValueError(f"success must have shape {(batch_size,)}, got {tuple(quality_target.shape)}")
    valid_quality_targets = quality_target[flow_valid]
    if valid_quality_targets.numel() and (
        not bool(torch.isfinite(valid_quality_targets).all())
        or bool(torch.any((valid_quality_targets < 0.0) | (valid_quality_targets > 1.0)))
    ):
        raise ValueError("success targets with valid kinematics must be finite values in [0, 1]")
    safe_success = torch.where(
        flow_valid,
        quality_target,
        torch.zeros_like(target_quality_logit),
    )

    terms = {
        "flow_velocity": settings.flow_velocity
        * masked_sample_mean(
            (velocity_prediction - velocity_target).square(), flow_valid, name="kinematics_valid"
        ),
        "palm_translation": settings.palm_translation
        * masked_sample_mean(
            (prediction.palm_translation - safe_palm_pos).square(), pose_valid, name="pose_target_valid"
        ),
        "palm_rotation": settings.palm_rotation
        * masked_sample_mean(
            geodesic_rotation_error(prediction.palm_rotation, safe_palm_rot),
            pose_valid,
            name="pose_target_valid",
        ),
        "joint": settings.joint
        * masked_sample_mean(
            (prediction.joint_angles - safe_joints).square(), joint_valid, name="joint_target_valid"
        ),
        "fk_consistency": settings.fk_consistency
        * masked_sample_mean(
            (prediction.fingertips - safe_fingertips).square(), fk_valid, name="fk_target_valid"
        ),
        "quality": settings.quality
        * masked_sample_mean(
            nn.functional.binary_cross_entropy_with_logits(
                target_quality_logit, safe_success, reduction="none"
            ),
            flow_valid,
            name="kinematics_valid",
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
    point_mask: torch.Tensor | None = None,
    kinematics_valid: torch.Tensor | None = None,
    pose_target_valid: torch.Tensor | None = None,
    joint_target_valid: torch.Tensor | None = None,
    fk_target_valid: torch.Tensor | None = None,
) -> tuple[GraspPrediction, LossBreakdown]:
    """One training step's forward pass, sharing conditioning between both uses.

    The conditioning is computed once and used both to sample and to evaluate
    the velocity field.  Recomputing it would double the encoder cost and, more
    importantly, let the two paths drift apart under dropout.
    """

    conditioning, _hand = model.encode(points, graph, point_mask=point_mask)
    state = model.sample_state(conditioning, generator=generator, noise=sample_noise)
    translation, rotation, joints = model.decode(state, robot)
    prediction = GraspPrediction(
        palm_translation=translation,
        palm_rotation=rotation,
        joint_angles=joints,
        fingertips=robot.fingertip_positions(translation, rotation, joints),
        quality_logit=model.quality(conditioning, state),
        raw_state=state,
    )
    losses = compute_losses(
        model,
        prediction,
        conditioning,
        robot,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        joint_angles=joint_angles,
        fingertip_positions=fingertip_positions,
        success=success,
        weights=weights,
        generator=generator,
        kinematics_valid=kinematics_valid,
        pose_target_valid=pose_target_valid,
        joint_target_valid=joint_target_valid,
        fk_target_valid=fk_target_valid,
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
