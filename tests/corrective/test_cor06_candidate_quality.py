"""COR-06: the quality head cannot see the grasp it is scoring.

``quality`` reads the pooled conditioning, which is a function of the object and
the hand alone.  Every candidate generated for one object therefore receives the
same score, and ranking K grasps is ranking K copies of one number.  A ranking
test written against that passes on ties, which is why the head survived.
"""

from __future__ import annotations

import torch
from _corrective_support import characterization


@characterization("COR-06", note="quality is a function of the observation only")
def test_two_candidates_for_one_object_can_score_differently() -> None:
    """Same object, same hand, two draws: two grasps, and two scores."""

    from qdgrasp.models.flow import GraspFlowModel
    from qdgrasp.robot.spec import RobotSpec

    torch.manual_seed(0)
    model = GraspFlowModel()
    robot = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    graph = robot.to_hand_graph()
    points = torch.randn(1, 128, 3, generator=torch.Generator().manual_seed(1)) * 0.05

    first = model(points, graph, robot, generator=torch.Generator().manual_seed(2))
    second = model(points, graph, robot, generator=torch.Generator().manual_seed(3))

    assert not torch.allclose(first.raw_state, second.raw_state), "the two candidates must actually differ"
    assert not torch.allclose(first.quality_logit, second.quality_logit), (
        "two different grasps on the same object received the same quality score; the head is conditioned on "
        "the observation only, so ranking candidates ranks copies of one number"
    )
