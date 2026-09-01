"""Regression tests for target validity, joint coordinates and grasp quality."""

from __future__ import annotations

import pytest
import torch

from qdgrasp.models.flow import GraspFlowModel
from qdgrasp.models.losses import LossWeights, forward_and_loss
from qdgrasp.robot.spec import RobotSpec

ACTIVE_PROFILES = ("leap_hand.yaml", "wonik_allegro.yaml")


@pytest.mark.parametrize("profile", ACTIVE_PROFILES)
def test_joint_latent_round_trips_interior_and_limit_values(profile: str) -> None:
    model = GraspFlowModel()
    robot = RobotSpec.from_config(profile, sample_anchors=False)
    lower, upper = model._joint_limits(robot, torch.device("cpu"), torch.float32)
    fractions = torch.tensor([[0.0], [0.02], [0.5], [0.98], [1.0]])
    joints = lower + fractions * (upper - lower)
    palm_pos = torch.zeros(len(fractions), 3)
    palm_rot = torch.eye(3).expand(len(fractions), 3, 3).contiguous()

    state = model.encode_target(palm_pos, palm_rot, joints, robot)
    _translation, _rotation, decoded = model.decode(state, robot)

    torch.testing.assert_close(decoded, joints, atol=1e-5, rtol=0.0)
    assert torch.isfinite(state).all()


def test_quality_scores_the_candidate_and_supports_opposite_labels_for_one_observation() -> None:
    torch.manual_seed(0)
    model = GraspFlowModel()
    channels = model.flow_config.channels
    dimension = model.flow_config.state_dimension
    conditioning = torch.randn(1, channels).expand(2, -1).contiguous()
    candidates = torch.stack([torch.zeros(dimension), torch.ones(dimension)])

    logits = model.quality(conditioning, candidates)
    labels = torch.tensor([1.0, 0.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    assert logits.shape == (2,)
    assert not torch.allclose(logits[0], logits[1])
    assert all(parameter.grad is not None for parameter in model.quality.parameters())


def test_quality_supervision_is_attached_to_the_encoded_target_not_a_random_draw() -> None:
    torch.manual_seed(4)
    model = GraspFlowModel().eval()
    robot = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    lower, upper = model._joint_limits(robot, torch.device("cpu"), torch.float32)
    joints = torch.stack([lower + 0.3 * (upper - lower), lower + 0.7 * (upper - lower)])
    palm_pos = torch.zeros(2, 3)
    palm_rot = torch.eye(3).expand(2, 3, 3).contiguous()
    points = torch.randn(1, 64, 3).expand(2, -1, -1).contiguous()
    success = torch.tensor([1.0, 0.0])

    prediction, losses = forward_and_loss(
        model,
        robot,
        robot.to_hand_graph(),
        points=points,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        joint_angles=joints,
        fingertip_positions=robot.fingertip_positions(palm_pos, palm_rot, joints),
        success=success,
        sample_noise=torch.randn(2, model.flow_config.state_dimension),
        weights=LossWeights(
            flow_velocity=0.0,
            palm_translation=0.0,
            palm_rotation=0.0,
            joint=0.0,
            fk_consistency=0.0,
            quality=1.0,
        ),
    )
    conditioning, _ = model.encode(points, robot.to_hand_graph())
    target_state = model.encode_target(palm_pos, palm_rot, joints, robot)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        model.quality(conditioning, target_state), success
    )
    random_draw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model.quality(conditioning, prediction.raw_state), success
    )

    torch.testing.assert_close(losses.terms["quality"], expected)
    assert not torch.allclose(losses.terms["quality"], random_draw_loss)


def test_padded_points_do_not_create_an_origin_token() -> None:
    torch.manual_seed(0)
    model = GraspFlowModel().eval()
    robot = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    real = torch.randn(1, 64, 3) * 0.05
    padded = torch.cat([real, torch.zeros(1, 32, 3)], dim=1)
    mask = torch.cat([torch.ones(1, 64, dtype=torch.bool), torch.zeros(1, 32, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        expected, _ = model.encode(real, robot.to_hand_graph())
        actual, _ = model.encode(padded, robot.to_hand_graph(), point_mask=mask)

    torch.testing.assert_close(actual, expected)


def _targets(robot: RobotSpec, count: int) -> dict[str, torch.Tensor]:
    joint_count = len(robot.actuated_joint_names)
    lower = torch.tensor([robot.config.joint_limits[name][0] for name in robot.actuated_joint_names])
    upper = torch.tensor([robot.config.joint_limits[name][1] for name in robot.actuated_joint_names])
    joints = ((lower + upper) / 2.0).expand(count, joint_count).clone()
    palm_pos = torch.linspace(0.01, 0.03, count).unsqueeze(-1).expand(count, 3).clone()
    palm_rot = torch.eye(3).expand(count, 3, 3).contiguous()
    return {
        "palm_pos": palm_pos,
        "palm_rot": palm_rot,
        "joint_angles": joints,
        "fingertip_positions": robot.fingertip_positions(palm_pos, palm_rot, joints),
        "success": torch.arange(count, dtype=torch.float32).remainder(2),
    }


def test_an_invalid_placeholder_does_not_change_geometric_or_quality_losses() -> None:
    torch.manual_seed(0)
    model = GraspFlowModel().eval()
    robot = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    graph = robot.to_hand_graph()
    targets = _targets(robot, 2)
    points = torch.randn(2, 96, 3, generator=torch.Generator().manual_seed(1)) * 0.05
    sample_noise = torch.randn(
        2, model.flow_config.state_dimension, generator=torch.Generator().manual_seed(2)
    )

    def losses_for(count: int, valid: torch.Tensor):
        _prediction, losses = forward_and_loss(
            model,
            robot,
            graph,
            points=points[:count],
            palm_pos=targets["palm_pos"][:count],
            palm_rot=targets["palm_rot"][:count],
            joint_angles=targets["joint_angles"][:count],
            fingertip_positions=targets["fingertip_positions"][:count],
            success=targets["success"][:count],
            sample_noise=sample_noise[:count],
            generator=torch.Generator().manual_seed(3),
            kinematics_valid=valid,
            pose_target_valid=valid,
            joint_target_valid=valid,
            fk_target_valid=valid,
        )
        return losses.terms

    alone = losses_for(1, torch.tensor([True]))
    beside_placeholder = losses_for(2, torch.tensor([True, False]))
    for name in ("palm_translation", "palm_rotation", "joint", "fk_consistency", "quality"):
        torch.testing.assert_close(beside_placeholder[name], alone[name], atol=1e-6, rtol=1e-6)


def test_placeholder_only_batch_has_finite_zero_losses_and_gradients() -> None:
    torch.manual_seed(0)
    model = GraspFlowModel()
    robot = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    targets = _targets(robot, 2)
    for name in ("palm_pos", "palm_rot", "joint_angles", "fingertip_positions", "success"):
        targets[name].fill_(float("nan"))
    invalid = torch.zeros(2, dtype=torch.bool)

    _prediction, losses = forward_and_loss(
        model,
        robot,
        robot.to_hand_graph(),
        points=torch.randn(2, 64, 3) * 0.05,
        **targets,
        weights=LossWeights(),
        kinematics_valid=invalid,
        pose_target_valid=invalid,
        joint_target_valid=invalid,
        fk_target_valid=invalid,
    )
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert float(losses.total.detach()) == 0.0
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() and not bool(torch.any(gradient)) for gradient in gradients)
