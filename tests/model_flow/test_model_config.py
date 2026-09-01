"""P4-08: the preset is the model, and a typo in it is an error.

``ModelConfig.params`` is an open mapping.  Without a strict reader, a preset
that says ``flow_step: 5`` instead of ``flow_steps: 5`` would build a model with
the default step count and no one would know the run did not mean what the file
said.  So every test here is about a way that could happen.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from qdgrasp.config.loader import load_model_config, load_robot_config
from qdgrasp.config.registry import get_model_builder, registered_models
from qdgrasp.config.schema import ConfigError
from qdgrasp.models.config import (
    FLOW_SCALES,
    MODEL_TYPE,
    OVERRIDABLE_PARAMS,
    FlowModelSettings,
    QDGraspFlow,
)
from qdgrasp.robot.spec import RobotSpec

PRESETS = tuple(f"qdgrasp-flow-{scale}.yaml" for scale in ("n", "s", "m"))


@pytest.fixture(scope="module")
def leap() -> RobotSpec:
    return RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)


def test_the_builder_is_registered_under_its_declared_type() -> None:
    assert MODEL_TYPE in registered_models()
    assert get_model_builder(MODEL_TYPE) is not None


@pytest.mark.parametrize("preset", PRESETS)
def test_every_shipped_preset_builds_and_runs(preset: str) -> None:
    model_config = load_model_config(preset)
    robot_config = load_robot_config("leap_hand.yaml")
    assert model_config.type == MODEL_TYPE
    module = get_model_builder(model_config.type)(model_config, robot_config)
    with torch.no_grad():
        prediction = module(torch.randn(2, 128, 3) * 0.05)
    assert prediction.is_finite()
    assert prediction.joint_angles.shape == (2, 16)


def test_scales_are_ordered_by_capacity(leap: RobotSpec) -> None:
    """``s`` and `m` are config-only in P4, so their only claim is being bigger."""

    counts = []
    for scale in ("n", "s", "m"):
        module = QDGraspFlow(FlowModelSettings(scale=scale), leap)
        counts.append(sum(p.numel() for p in module.parameters()))
    assert counts == sorted(counts)
    assert len(set(counts)) == 3


def test_an_unknown_parameter_is_refused_not_ignored() -> None:
    with pytest.raises(ConfigError, match="unknown QDGrasp-Flow parameters"):
        FlowModelSettings.from_params({"scale": "n", "flow_step": 5})
    with pytest.raises(ConfigError, match="unknown QDGrasp-Flow scale"):
        FlowModelSettings.from_params({"scale": "xl"})


def test_widths_cannot_be_set_from_a_preset() -> None:
    """Shape belongs to the scale table, named once, not to a preset."""

    for name in ("channels", "flow_channels", "encoder_channels", "graph_layers", "heads"):
        assert name not in OVERRIDABLE_PARAMS
        with pytest.raises(ConfigError):
            FlowModelSettings.from_params({"scale": "n", name: 64})


def test_an_impossible_override_is_refused_by_the_config_it_feeds() -> None:
    with pytest.raises(Exception, match="exceeds"):
        FlowModelSettings.from_params({"scale": "n", "voxel_size": 1e-7})
    with pytest.raises(Exception):
        FlowModelSettings.from_params({"scale": "n", "flow_steps": 0})


def test_a_hand_too_wide_for_the_head_is_refused_at_build(leap: RobotSpec) -> None:
    with pytest.raises(ConfigError, match="max_joints"):
        QDGraspFlow(FlowModelSettings(max_joints=8), leap)


def test_a_v1_robot_profile_is_refused() -> None:
    """FK consistency needs kinematics, which robot/v1 does not carry."""

    model_config = load_model_config("qdgrasp-flow-n.yaml")
    v1 = load_robot_config("dummy-hand.yaml")
    with pytest.raises(ConfigError, match="robot/v2"):
        get_model_builder(MODEL_TYPE)(model_config, v1)


def test_results_name_what_produced_them(leap: RobotSpec) -> None:
    module = QDGraspFlow(FlowModelSettings(grasps=5), leap)
    results = module.predict_results(torch.randn(200, 3) * 0.05)
    assert len(results) == 5
    assert bool((results.score[:-1] >= results.score[1:]).all())
    # Built without a document, so provenance falls back to the settings, and
    # says so rather than emitting an empty string.
    assert results.model_hash.startswith("settings:")
    # Nothing was transferred, so both roles name the same profile.
    assert results.training_robot_hash == "robot-profile:leap_hand"
    assert results.runtime_robot_hash == results.training_robot_hash
    assert results.joint_names == tuple(leap.actuated_joint_names)


def test_a_document_built_model_reports_the_document_hash() -> None:
    model_config = load_model_config("qdgrasp-flow-n.yaml")
    robot_config = load_robot_config("leap_hand.yaml")
    module = get_model_builder(MODEL_TYPE)(model_config, robot_config)
    results = module.predict_results(torch.randn(200, 3) * 0.05)
    assert results.model_hash == model_config.content_hash()
    assert results.training_robot_hash == robot_config.content_hash()
    assert results.runtime_robot_hash == robot_config.content_hash()


def test_predict_refuses_a_batched_cloud(leap: RobotSpec) -> None:
    module = QDGraspFlow(FlowModelSettings(), leap)
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        module.predict_results(torch.randn(2, 200, 3))


def test_training_and_validation_steps_run(leap: RobotSpec) -> None:
    module = QDGraspFlow(FlowModelSettings(), leap)
    joints = len(leap.actuated_joint_names)
    batch = {
        "points": torch.randn(3, 128, 3) * 0.05,
        "palm_pos": torch.randn(3, 3) * 0.05,
        "palm_rot": torch.eye(3).expand(3, 3, 3).contiguous(),
        "joint_angles": torch.zeros(3, joints),
        "fingertip_positions": torch.zeros(3, len(leap.fingertip_links), 3),
        "success": torch.tensor([1.0, 0.0, 1.0]),
    }
    loss = module.training_step(batch)
    assert loss.requires_grad and torch.isfinite(loss)
    metrics = module.validation_step(batch)
    assert {"total", "palm_translation_m", "palm_rotation_rad", "joint_abs_rad"} <= set(metrics)
    assert all(torch.isfinite(value) for value in metrics.values())


def test_an_incomplete_batch_is_named_not_guessed(leap: RobotSpec) -> None:
    module = QDGraspFlow(FlowModelSettings(), leap)
    with pytest.raises(KeyError, match="fingertip_positions"):
        module.training_step({"points": torch.randn(1, 32, 3)})


def test_the_graph_travels_with_the_weights(leap: RobotSpec) -> None:
    """A device move must carry the hand, or the first forward on GPU dies."""

    module = QDGraspFlow(FlowModelSettings(), leap)
    graph = module.graph
    assert graph.node_features.device == next(module.parameters()).device
    assert graph.num_nodes == 18
    assert graph.actuated_joint_names == tuple(leap.actuated_joint_names)
    # Derived from the profile, so it must not be frozen into a checkpoint.
    assert not any(key.startswith("graph_") for key in module.state_dict())


def test_the_scale_table_is_the_single_place_shapes_are_written() -> None:
    for name, scale in FLOW_SCALES.items():
        assert len(scale.encoder_channels) == len(scale.encoder_depths), name
        assert scale.flow_channels % scale.heads == 0, name
        assert all(width % scale.heads == 0 for width in scale.encoder_channels), name
    settings = FlowModelSettings(scale="n")
    assert dataclasses.asdict(settings)["scale"] == "n"
    assert settings.encoder().channels == FLOW_SCALES["n"].encoder_channels
