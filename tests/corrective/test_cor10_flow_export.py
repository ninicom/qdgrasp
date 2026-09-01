"""COR-10: the export path has only ever been exercised on the dummy model.

``GraspFlowModel.forward`` takes a ``HandGraph`` and a ``RobotSpec``, draws its
own noise and returns a dataclass.  None of those survive a tracer: the objects
are not tensors, the draw makes the trace a recording of one sample, and the
dataclass is a Python container the runtime cannot receive.  The export tests
pass because the model they export is the dummy one, which has none of these
properties.

An export that cannot be produced is a missing feature.  An export produced by
tracing a stochastic function is worse, because it looks like a model and is a
constant.
"""

from __future__ import annotations

import torch
from _corrective_support import characterization


@characterization("COR-10", note="no tensor-only export adapter exists for the flow model")
def test_the_flow_model_exports_through_a_tensor_only_adapter() -> None:
    import qdgrasp.export as export_package
    from qdgrasp.models.flow import GraspFlowModel
    from qdgrasp.robot.spec import RobotSpec

    assert hasattr(export_package, "FlowExportAdapter"), (
        "PLAN.md §9.8 asks for an export adapter that takes points and explicit noise and returns a stable "
        "tuple of tensors; tracing GraspFlowModel.forward records one random draw as if it were the model"
    )

    torch.manual_seed(0)
    model = GraspFlowModel().eval()
    robot = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    adapter = export_package.FlowExportAdapter(model, robot)

    points = torch.randn(1, 128, 3, generator=torch.Generator().manual_seed(1)) * 0.05
    noise = torch.randn(1, model.flow_config.state_dimension, generator=torch.Generator().manual_seed(2))

    with torch.no_grad():
        eager = adapter(points, noise)
        traced = torch.jit.trace(adapter, (points, noise))
        replayed = traced(points, noise)

    assert isinstance(eager, tuple) and all(isinstance(item, torch.Tensor) for item in eager)
    for expected, actual in zip(eager, replayed):
        assert torch.allclose(expected, actual, atol=1e-5)

    # Dynamic point count: a traced token topology would silently keep the first.
    wider = torch.randn(1, 512, 3, generator=torch.Generator().manual_seed(3)) * 0.05
    with torch.no_grad():
        assert all(
            torch.allclose(expected, actual, atol=1e-5)
            for expected, actual in zip(adapter(wider, noise), traced(wider, noise))
        )
