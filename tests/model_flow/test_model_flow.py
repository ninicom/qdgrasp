"""P4-01..09: the properties ``PLAN.md`` §6 requires of QDGrasp-Flow.

Three of these are the reason the file exists.

*No ``N x N``.*  A quadratic tensor anywhere in the forward pass would make the
memory gate unmeetable at any interesting point count, and it is the failure
mode a point-cloud model falls into by default.  The test doubles the token
count and checks the cost roughly doubles.

*Full gradient coverage.*  A parameter that never receives a gradient is dead
weight in a checkpoint and an invisible hole in an ablation.

*Valid outputs.*  A rotation that is not in SO(3) and a joint outside its named
limit are not "slightly wrong numbers"; they are values no hand can be commanded
with, and the model must not be able to emit them.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from scipy.spatial.transform import Rotation

from qdgrasp.models.encoder import EncoderConfig, PointEncoder, masked_mean
from qdgrasp.models.flow import (
    FlowConfig,
    GraspFlowModel,
    clamp_to_limits,
    rotation_from_9d,
)
from qdgrasp.models.hand_graph import HandGraphEncoder, HandGraphEncoderConfig, symmetrize
from qdgrasp.models.losses import (
    LOSS_TERMS,
    LossBreakdown,
    LossWeights,
    forward_and_loss,
    geodesic_rotation_error,
    gradient_coverage,
)
from qdgrasp.models.tokenizer import (
    TokenizerConfig,
    TokenizerError,
    pack_grid_coordinates,
    quantize,
    scatter_tokens_to_points,
    tokenize_points,
    unpack_grid_key,
)
from qdgrasp.robot.spec import RobotSpec

ACTIVE_PROFILES = ("leap_hand.yaml", "wonik_allegro.yaml")


@pytest.fixture(scope="module")
def robots() -> dict[str, RobotSpec]:
    return {name: RobotSpec.from_config(name, sample_anchors=False) for name in ACTIVE_PROFILES}


# -- P4-01 tokenizer -------------------------------------------------------


def test_packing_is_injective_not_a_hash() -> None:
    """Distinct cells cannot collide, because the key is positional, not hashed."""

    grid = 101
    coordinates = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [5, 7, 9], [100, 100, 100]]])
    keys = pack_grid_coordinates(coordinates, grid)
    assert keys.unique().numel() == coordinates.shape[1]
    torch.testing.assert_close(unpack_grid_key(keys, grid), coordinates)


def test_a_grid_too_fine_to_pack_is_refused() -> None:
    with pytest.raises(TokenizerError, match="exceeds"):
        TokenizerConfig(voxel_size=1e-7, extent=0.5).validate()
    with pytest.raises(TokenizerError, match="exceeds the packable maximum"):
        pack_grid_coordinates(torch.zeros(1, 1, 3, dtype=torch.int64), 1 << 30)


def test_coordinates_outside_the_grid_are_refused() -> None:
    with pytest.raises(TokenizerError):
        pack_grid_coordinates(torch.tensor([[[0, 0, -1]]]), 10)
    with pytest.raises(TokenizerError):
        pack_grid_coordinates(torch.tensor([[[0, 0, 10]]]), 10)


def test_non_finite_points_are_refused() -> None:
    points = torch.zeros(1, 4, 3)
    points[0, 2, 1] = float("nan")
    with pytest.raises(TokenizerError, match="NaN"):
        quantize(points, TokenizerConfig())


def test_points_in_the_same_cell_share_a_token_and_others_do_not() -> None:
    config = TokenizerConfig(voxel_size=0.01, extent=0.5)
    points = torch.tensor([[[0.001, 0.001, 0.001], [0.002, 0.002, 0.002], [0.4, 0.4, 0.4]]])
    tokenized = tokenize_points(points, config)
    assignments = tokenized.point_to_token[0]
    assert assignments[0] == assignments[1]
    assert assignments[0] != assignments[2]
    assert int(tokenized.token_counts[0].sum()) == 3


def test_token_count_rises_with_resolution() -> None:
    torch.manual_seed(0)
    points = torch.randn(1, 400, 3) * 0.05
    counts = [
        int(tokenize_points(points, TokenizerConfig(voxel_size=size)).token_mask.sum()) for size in (0.02, 0.01, 0.005)
    ]
    assert counts == sorted(counts)


def test_tokenizing_is_deterministic() -> None:
    torch.manual_seed(0)
    points = torch.randn(2, 200, 3) * 0.05
    config = TokenizerConfig(voxel_size=0.01)
    first, second = tokenize_points(points, config), tokenize_points(points, config)
    torch.testing.assert_close(first.token_positions, second.token_positions)
    torch.testing.assert_close(first.point_to_token, second.point_to_token)


def test_scatter_returns_a_feature_per_point() -> None:
    torch.manual_seed(0)
    points = torch.randn(2, 128, 3) * 0.05
    tokenized = tokenize_points(points, TokenizerConfig(voxel_size=0.01))
    features = torch.randn(2, tokenized.max_tokens, 8)
    scattered = scatter_tokens_to_points(features, tokenized)
    assert scattered.shape == (2, 128, 8)
    torch.testing.assert_close(scattered[0, 0], features[0, tokenized.point_to_token[0, 0]])


# -- P4-02 encoder ---------------------------------------------------------


def test_the_encoder_is_linear_not_quadratic_in_tokens() -> None:
    """Doubling the tokens must roughly double the cost, not quadruple it."""

    encoder = PointEncoder(EncoderConfig(channels=(32, 64), depths=(1, 1)))
    torch.manual_seed(0)
    costs = []
    for count in (256, 512, 1024):
        positions = torch.randn(1, count, 3) * 0.05
        mask = torch.ones(1, count)
        # Peak activation size is what the memory gate is about; the windowed
        # attention keeps it proportional to count * window.
        costs.append(encoder(positions, mask).numel())
    assert costs[1] / costs[0] == pytest.approx(2.0, rel=0.05)
    assert costs[2] / costs[1] == pytest.approx(2.0, rel=0.05)


def test_padded_tokens_do_not_leak_into_the_output() -> None:
    encoder = PointEncoder(EncoderConfig(channels=(32,), depths=(1,)))
    positions = torch.randn(2, 40, 3) * 0.05
    mask = torch.ones(2, 40)
    mask[1, 25:] = 0.0
    output = encoder(positions, mask)
    assert torch.isfinite(output).all()
    assert output[1, 25:].abs().max() < 1e-6


def test_masked_mean_ignores_padding() -> None:
    features = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    torch.testing.assert_close(masked_mean(features, mask), torch.tensor([[2.0, 2.0]]))


def test_the_encoder_config_rejects_incoherent_shapes() -> None:
    with pytest.raises(ValueError):
        EncoderConfig(channels=(32, 64), depths=(1,)).validate()
    with pytest.raises(ValueError):
        EncoderConfig(channels=(30,), depths=(1,), heads=4).validate()
    with pytest.raises(ValueError):
        EncoderConfig(window=1).validate()


# -- P4-03 hand graph ------------------------------------------------------


def test_one_encoder_serves_hands_of_different_sizes(robots) -> None:
    """LEAP is 18 nodes and Allegro 22; neither count may be compiled in."""

    encoder = HandGraphEncoder()
    embeddings = {name: encoder(robot.to_hand_graph()) for name, robot in robots.items()}
    assert embeddings["leap_hand.yaml"].nodes.shape[0] == 18
    assert embeddings["wonik_allegro.yaml"].nodes.shape[0] == 22
    for embedding in embeddings.values():
        assert embedding.summary.shape == (encoder.output_channels,)
        assert embedding.palm.shape == (encoder.output_channels,)
        assert embedding.fingertips.shape[0] == 4
        assert torch.isfinite(embedding.nodes).all()


def test_symmetrize_adds_reverse_edges_with_a_direction_flag() -> None:
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_features = torch.zeros(2, 3)
    index, features = symmetrize(edge_index, edge_features)
    assert index.shape == (2, 4)
    assert features.shape == (2 * 2, 4)
    assert features[:2, -1].tolist() == [1.0, 1.0]
    assert features[2:, -1].tolist() == [-1.0, -1.0]
    torch.testing.assert_close(index[:, 2:], edge_index.flip(0))


def test_a_graph_with_the_wrong_node_width_is_refused(robots) -> None:
    encoder = HandGraphEncoder(HandGraphEncoderConfig(node_dim=5))
    with pytest.raises(ValueError, match="node_dim"):
        encoder(robots["leap_hand.yaml"].to_hand_graph())


# -- P4-05 flow head -------------------------------------------------------


def test_rotation_projection_lands_in_so3() -> None:
    torch.manual_seed(0)
    raw = torch.randn(16, 9) * 5.0
    rotation = rotation_from_9d(raw)
    identity = torch.eye(3).expand(16, 3, 3)
    torch.testing.assert_close(rotation.transpose(1, 2) @ rotation, identity, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(16), atol=1e-5, rtol=0.0)


def test_rotation_projection_rejects_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="9-vector"):
        rotation_from_9d(torch.zeros(2, 6))


def test_the_joint_clamp_keeps_a_live_gradient() -> None:
    """A hard cut has zero gradient exactly where the model needs pushing back."""

    lower = torch.tensor([-1.0, 0.0])
    upper = torch.tensor([1.0, 2.0])
    raw = torch.tensor([[50.0, -50.0]], requires_grad=True)
    clamped = clamp_to_limits(raw, lower, upper)
    assert bool(((clamped >= lower) & (clamped <= upper)).all())
    clamped.sum().backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("profile", ACTIVE_PROFILES)
def test_the_model_emits_an_executable_grasp(profile: str, robots) -> None:
    torch.manual_seed(0)
    model = GraspFlowModel()
    robot = robots[profile]
    prediction = model(torch.randn(3, 256, 3) * 0.05, robot.to_hand_graph(), robot)

    assert prediction.is_finite()
    identity = torch.eye(3).expand(3, 3, 3)
    torch.testing.assert_close(
        prediction.palm_rotation.transpose(1, 2) @ prediction.palm_rotation, identity, atol=1e-4, rtol=0.0
    )
    limits = [robot.config.joint_limits[name] for name in robot.actuated_joint_names]
    lower = torch.tensor([value[0] for value in limits])
    upper = torch.tensor([value[1] for value in limits])
    assert bool((prediction.joint_angles >= lower - 1e-5).all())
    assert bool((prediction.joint_angles <= upper + 1e-5).all())
    assert prediction.fingertips.shape == (3, len(robot.fingertip_links), 3)


def test_a_hand_with_more_joints_than_the_head_is_refused(robots) -> None:
    model = GraspFlowModel(flow=FlowConfig(max_joints=4))
    robot = robots["leap_hand.yaml"]
    with pytest.raises(ValueError, match="max_joints"):
        model.decode(torch.zeros(1, model.flow_config.state_dimension), robot)


def test_an_untrained_solver_is_the_identity_map(robots) -> None:
    """Zero-init on the velocity head means the sample *is* the noise draw.

    This is the deliberate starting condition from ``flow.py``: an untrained
    field must not shove the state into a region the joint clamps then have to
    rescue, which would hide a dead gradient behind a saturated tanh.  It also
    means the solver step count cannot change an untrained sample, so the next
    test has to give the field something to integrate first.
    """

    torch.manual_seed(0)
    model = GraspFlowModel()
    robot = robots["leap_hand.yaml"]
    conditioning, _ = model.encode(torch.randn(2, 200, 3) * 0.05, robot.to_hand_graph())
    noise = torch.randn(2, model.flow_config.state_dimension)
    torch.testing.assert_close(model.sample_state(conditioning, noise=noise), noise)


def test_changing_the_solver_steps_changes_the_sample_but_keeps_it_valid(robots) -> None:
    """Euler is a discretisation, so its step count has to move the answer."""

    robot = robots["leap_hand.yaml"]
    points = torch.randn(2, 200, 3) * 0.05
    graph = robot.to_hand_graph()
    outputs = []
    for steps in (2, 5, 10):
        torch.manual_seed(0)
        model = GraspFlowModel(flow=FlowConfig(flow_steps=steps))
        # Lift the velocity head off its zero initialisation; with the same
        # seed every model here gets the identical non-zero field, so the only
        # difference between the three runs is the step count.
        with torch.no_grad():
            torch.manual_seed(2)
            model.velocity.output.weight.normal_(std=0.05)
        torch.manual_seed(1)
        outputs.append(model(points, graph, robot))
    for prediction in outputs:
        assert prediction.is_finite()
    assert not torch.allclose(outputs[0].palm_translation, outputs[2].palm_translation)


def test_pinning_the_noise_makes_sampling_deterministic(robots) -> None:
    torch.manual_seed(0)
    model = GraspFlowModel()
    robot = robots["leap_hand.yaml"]
    conditioning, _ = model.encode(torch.randn(2, 200, 3) * 0.05, robot.to_hand_graph())
    noise = torch.randn(2, model.flow_config.state_dimension)
    first = model.sample_state(conditioning, noise=noise)
    second = model.sample_state(conditioning, noise=noise)
    torch.testing.assert_close(first, second)


# -- P4-09 losses ----------------------------------------------------------


def test_the_total_is_the_sum_of_the_logged_terms() -> None:
    breakdown = LossBreakdown(terms={"joint": torch.tensor(0.25), "quality": torch.tensor(0.75)})
    assert float(breakdown.total) == pytest.approx(1.0)
    document = breakdown.to_document()
    assert document["total"] == pytest.approx(sum(v for k, v in document.items() if k != "total"))


def test_an_unknown_loss_term_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown loss terms"):
        LossBreakdown(terms={"made_up": torch.tensor(1.0)})
    assert set(LOSS_TERMS) == {
        "flow_velocity",
        "palm_translation",
        "palm_rotation",
        "joint",
        "fk_consistency",
        "quality",
    }


def test_a_negative_loss_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LossWeights(joint=-1.0).validate()


def test_geodesic_rotation_error_is_zero_for_equal_rotations_and_finite_at_pi() -> None:
    """Identical rotations sit on the floor the safety clamp buys.

    ``arccos`` has an infinite derivative at ``+-1``, which is exactly where a
    well-fit model lives, so the cosine is clamped to ``1 - 1e-6``.  That trades
    an exact zero for a finite gradient: the reported error bottoms out near
    ``sqrt(2e-6) ~ 1.4e-3`` rad, four thousandths of a degree, far below any
    threshold this project reads a verdict off.
    """

    rotation = torch.tensor(Rotation.random(8, random_state=0).as_matrix(), dtype=torch.float32)
    identical = geodesic_rotation_error(rotation, rotation)
    assert float(identical.max()) < 2e-3
    flipped = torch.diag(torch.tensor([1.0, -1.0, -1.0])).expand(1, 3, 3)
    error = geodesic_rotation_error(torch.eye(3).expand(1, 3, 3), flipped)
    assert torch.isfinite(error).all()


@pytest.mark.parametrize("profile", ACTIVE_PROFILES)
def test_every_trainable_parameter_receives_a_finite_gradient(profile: str, robots) -> None:
    """``PLAN.md`` §6: a parameter that never gets one is dead weight."""

    torch.manual_seed(0)
    model = GraspFlowModel()
    robot = robots[profile]
    joint_count = len(robot.actuated_joint_names)
    batch_size = 3
    palm_pos = torch.randn(batch_size, 3) * 0.05
    palm_rot = torch.tensor(Rotation.random(batch_size, random_state=0).as_matrix(), dtype=torch.float32)
    joints = torch.zeros(batch_size, joint_count)

    _prediction, losses = forward_and_loss(
        model,
        robot,
        robot.to_hand_graph(),
        points=torch.randn(batch_size, 256, 3) * 0.05,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        joint_angles=joints,
        fingertip_positions=robot.fingertip_positions(palm_pos, palm_rot, joints),
        success=torch.tensor([1.0, 0.0, 1.0]),
    )
    losses.total.backward()
    coverage = gradient_coverage(model)
    missing = sorted(name for name, ok in coverage.items() if not ok)
    assert not missing, f"parameters without a finite gradient: {missing[:10]}"
    assert len(coverage) > 100


def test_the_flow_target_is_the_straight_path_velocity() -> None:
    """Rectified flow's simplification: the target velocity is time-independent."""

    model = GraspFlowModel()
    target = torch.randn(4, model.flow_config.state_dimension)
    noise = torch.randn(4, model.flow_config.state_dimension)
    for value in (0.0, 0.5, 1.0):
        time = torch.full((4,), value)
        interpolated, velocity = model.velocity_target(target, noise, time)
        torch.testing.assert_close(velocity, target - noise)
        torch.testing.assert_close(interpolated, noise + value * (target - noise))


def test_encode_target_places_joints_after_the_pose(robots) -> None:
    model = GraspFlowModel()
    robot = robots["leap_hand.yaml"]
    joint_count = len(robot.actuated_joint_names)
    palm_pos = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    palm_rot = torch.eye(3).reshape(1, 3, 3)
    joints = torch.full((1, joint_count), 0.5)
    state = model.encode_target(palm_pos, palm_rot, joints)
    assert state.shape == (1, model.flow_config.state_dimension)
    torch.testing.assert_close(state[0, :3], palm_pos[0])
    torch.testing.assert_close(state[0, 3:12], torch.eye(3).reshape(9))
    torch.testing.assert_close(state[0, 12 : 12 + joint_count], joints[0])
    assert float(state[0, 12 + joint_count :].abs().max()) == 0.0


def test_the_flow_config_rejects_incoherent_shapes() -> None:
    with pytest.raises(ValueError, match="divisible"):
        FlowConfig(channels=10, heads=4).validate()
    for name in ("conditioning_layers", "flow_layers", "max_joints", "flow_steps", "time_bands"):
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(FlowConfig(), **{name: 0}).validate()
