"""COR-04: a placeholder is stored as a number and then regressed onto.

When the generator has no kinematics for a proposal it writes zeros and an
identity rotation, and nothing downstream can tell that apart from a measured
grasp at the origin with no rotation.  The loss regresses every sample, so a
failed proposal pulls the palm towards the origin with the same authority as a
successful one.

The fix is not a filter with a threshold.  It is a validity flag per target, so
that "we do not know where the palm was" stops being spelled ``0, 0, 0``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from _corrective_support import characterization

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "datasets/dgn-open-tiny"

VALIDITY_FIELDS = ("kinematics_valid", "pose_target_valid", "joint_target_valid", "fk_target_valid")


@characterization("COR-04", note="samples carry no validity flags")
def test_a_sample_says_which_of_its_targets_are_measurements() -> None:
    from qdgrasp.models.data import FlowDataset

    dataset = FlowDataset(DATASET, split="train", robot="leap_hand")
    assert len(dataset) > 0
    first = dataset[0]

    missing = [field for field in VALIDITY_FIELDS if field not in first]
    assert not missing, (
        f"samples carry no {missing}; validity is currently inferred from the value itself, so a placeholder "
        "at the origin is indistinguishable from a grasp measured at the origin"
    )


@characterization("COR-04", note="a placeholder contributes to the pose loss")
def test_adding_a_placeholder_does_not_change_the_pose_loss() -> None:
    """The same valid sample must produce the same palm loss, alone or beside a placeholder."""

    from qdgrasp.models.flow import GraspFlowModel
    from qdgrasp.models.losses import forward_and_loss
    from qdgrasp.robot.spec import RobotSpec

    torch.manual_seed(0)
    model = GraspFlowModel()
    robot = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    joint_count = len(robot.actuated_joint_names)

    points = torch.randn(2, 128, 3, generator=torch.Generator().manual_seed(1)) * 0.05
    palm_pos = torch.tensor([[0.10, 0.02, 0.05], [0.0, 0.0, 0.0]])
    palm_rot = torch.eye(3).expand(2, 3, 3).contiguous()
    joints = torch.zeros(2, joint_count)
    fingertips = robot.fingertip_positions(palm_pos, palm_rot, joints)
    success = torch.tensor([1.0, 0.0])
    noise = torch.randn(2, model.flow_config.state_dimension, generator=torch.Generator().manual_seed(2))

    def palm_term(count: int) -> torch.Tensor:
        _prediction, losses = forward_and_loss(
            model,
            robot,
            robot.to_hand_graph(),
            points=points[:count],
            palm_pos=palm_pos[:count],
            palm_rot=palm_rot[:count],
            joint_angles=joints[:count],
            fingertip_positions=fingertips[:count],
            success=success[:count],
            generator=torch.Generator().manual_seed(3),
            sample_noise=noise[:count],
        )
        return losses.terms["palm_translation"].detach()

    # Sample 1 is a placeholder: no palm, no rotation, no joints, not a success.
    assert torch.allclose(palm_term(1), palm_term(2), atol=1e-6), (
        "adding a proposal with no measured kinematics changed the palm-translation loss, so the placeholder "
        "is producing geometric gradient as if it were a measurement"
    )
