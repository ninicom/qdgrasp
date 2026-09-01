"""COR-05: the flow's target and the flow's output are different variables.

``encode_target`` packs the physical joint angles straight into the state
vector.  ``decode`` reads that channel through ``centre + half * tanh(raw)``.
So the state the flow is trained to reach decodes to a *different* pose than the
one it was built from, and the flow term and the joint/FK terms are pulling
towards two solutions at once.

The round-trip is the cheapest possible statement of the invariant: encode a
pose, decode it, get the pose back.
"""

from __future__ import annotations

import pytest
import torch
from _corrective_support import characterization

ACTIVE_PROFILES = ("leap_hand.yaml", "wonik_allegro.yaml")


@pytest.mark.parametrize("profile", ACTIVE_PROFILES)
@characterization("COR-05", note="encode writes physical joints, decode applies a tanh squash")
def test_encode_then_decode_returns_the_joints_it_was_given(profile: str) -> None:
    from qdgrasp.models.flow import GraspFlowModel
    from qdgrasp.robot.spec import RobotSpec

    model = GraspFlowModel()
    robot = RobotSpec.from_config(profile, sample_anchors=False)
    limits = [robot.config.joint_limits[name] for name in robot.actuated_joint_names]
    lower = torch.tensor([value[0] for value in limits])
    upper = torch.tensor([value[1] for value in limits])

    # Interior, near-lower and near-upper poses: a parameterization that is only
    # correct in the middle of the range is not correct.
    fractions = torch.tensor([[0.5], [0.02], [0.98]])
    joints = lower + (upper - lower) * fractions
    palm_pos = torch.zeros(3, 3)
    palm_rot = torch.eye(3).expand(3, 3, 3).contiguous()

    state = model.encode_target(palm_pos, palm_rot, joints, robot)
    _translation, _rotation, decoded = model.decode(state, robot)
    error = (decoded - joints).abs().max()

    assert float(error) < 1e-5, (
        f"{profile}: decode(encode(q)) is off by {float(error):.6f} rad, so the flow's regression target and "
        "the pose the model emits are two different parameterizations of the hand"
    )
