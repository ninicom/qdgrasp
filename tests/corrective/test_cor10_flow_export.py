"""COR-10: the export path had only ever been exercised on the dummy model.

``GraspFlowModel.forward`` takes a ``HandGraph`` and a ``RobotSpec``, draws its
own noise and returns a dataclass.  None of those survive a tracer: the objects
are not tensors, the draw makes the trace a recording of one sample, and the
dataclass is a Python container the runtime cannot receive.  The export tests
passed because the model they exported was the dummy one, which has none of
these properties.

An export that cannot be produced is a missing feature.  An export produced by
tracing a stochastic function is worse, because it looks like a model and is a
constant.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from _corrective_support import characterization

ACTIVE_PROFILES = ("leap_hand.yaml", "wonik_allegro.yaml")

#: Deliberately not all multiples of the encoder's window: a graph whose token
#: axis follows the input is only correct at counts congruent to the traced one,
#: and that is exactly the constraint an export must not carry silently.
POINT_COUNTS = (1, 129, 200, 512, 1024)


def _adapter(profile: str, max_points: int = 1024):
    from qdgrasp.export import FlowExportAdapter
    from qdgrasp.models.flow import GraspFlowModel
    from qdgrasp.robot.spec import RobotSpec

    torch.manual_seed(0)
    model = GraspFlowModel().eval()
    robot = RobotSpec.from_config(profile, sample_anchors=False)
    return FlowExportAdapter(model, robot, max_points=max_points).eval(), model, robot


@pytest.mark.parametrize("profile", ACTIVE_PROFILES)
@characterization("COR-10", note="no tensor-only export adapter exists for the flow model")
def test_the_flow_model_traces_to_a_tensor_only_module(profile: str) -> None:
    adapter, model, robot = _adapter(profile)
    points, noise = adapter.example_inputs(points=256)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            eager = adapter(points, noise)
            traced = torch.jit.trace(adapter, (points, noise))
            replayed = traced(points, noise)

    assert isinstance(eager, tuple) and all(isinstance(item, torch.Tensor) for item in eager)
    translation, rotation, joints, score = eager
    assert translation.shape == (1, 3)
    assert rotation.shape == (1, 3, 3)
    assert joints.shape == (1, len(robot.actuated_joint_names))
    assert score.shape == (1,)
    for expected, actual in zip(eager, replayed, strict=True):
        assert torch.allclose(expected, actual, atol=1e-5)

    # The adapter is the model, not a recording of it: same conditioning, same
    # explicit noise, same answer as the eager path it wraps.
    with torch.no_grad():
        conditioning, _hand = model.encode(points, robot.to_hand_graph())
        state = model.sample_state(conditioning, noise=noise)
        reference = model.decode(state, robot)
    for expected, actual in zip(reference, eager[:3], strict=True):
        assert torch.allclose(expected, actual, atol=1e-5)


@pytest.mark.parametrize("profile", ACTIVE_PROFILES)
@characterization("COR-10", note="a traced token topology is only valid at the traced cloud size")
def test_one_trace_answers_for_every_cloud_size_it_declares(profile: str) -> None:
    adapter, _model, _robot = _adapter(profile)
    points, noise = adapter.example_inputs(points=256)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            traced = torch.jit.trace(adapter, (points, noise))
            for count in POINT_COUNTS:
                cloud, _ = adapter.example_inputs(points=count)
                for expected, actual in zip(adapter(cloud, noise), traced(cloud, noise), strict=True):
                    assert torch.allclose(expected, actual, atol=1e-5), f"{profile} disagrees at {count} points"


@characterization("COR-10", note="an unsupported cloud size fell into the tracer's arithmetic")
def test_a_cloud_beyond_the_declared_capacity_is_refused() -> None:
    adapter, _model, _robot = _adapter("leap_hand.yaml", max_points=256)
    too_many, noise = adapter.example_inputs(points=512)

    with pytest.raises(ValueError, match="max_points"):
        adapter(too_many, noise)

    schema = adapter.output_schema()
    assert schema["max_points"] == 256
    assert schema["token_capacity"] % 32 == 0
    assert schema["outputs"] == ["palm_translation", "palm_rotation", "joint_angles", "quality_score"]
    assert schema["joint_names"] == list(_adapter("leap_hand.yaml")[2].actuated_joint_names)
