import pytest
import numpy as np
import torch
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline.solvers.region_dls import solve_region_dls_ik_batch

@pytest.fixture
def mock_spec():
    class MockConfig:
        name = "mock_hand"
        palm_link = "palm"
        base_link = "palm"
        wrist_link = "wrist"
        fingertip_links = ["tip_0"]
        contact_links = []
        joints = ("j_0",)
        joint_limits = {"j_0": (-2.0, 2.0)}
        mimic_joints = {}

    class MockSpec(RobotSpec):
        def __init__(self):
            self.config = MockConfig()
            self.actuated_joint_names = self.config.joints
            self.joint_limits = self.config.joint_limits
            self.fingertip_links = self.config.fingertip_links
            self.palm_link = self.config.palm_link

        def forward_kinematics(self, palm_pos, palm_rot, joint_angles):
            B = palm_pos.shape[0]
            device = palm_pos.device

            # Simple mock FK:
            # tip_0 pos = palm_pos + (q0, 0, 0)
            # Normal is always (0, 0, -1)

            T0 = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
            T0[:, 0, 3] = joint_angles["j_0"] if isinstance(joint_angles, dict) else joint_angles[:, 0]
            T0[:, :3, 2] = torch.tensor([0.0, 0.0, -1.0], device=device)
            T0[:, :3, 3] += palm_pos

            return {"tip_0": T0}

    return MockSpec()

def test_region_dls_converges_in_region(mock_spec):
    """
    Test that the solver stops searching when the finger is inside the region,
    even if it's not exactly on the anchor point.
    """
    B = 1
    K = 1

    palm_pos = np.zeros((B, 3))
    palm_rot = np.stack([np.eye(3)] * B)

    # Target anchor is at (0.5, 0.0, 0.0)
    target_contacts = np.zeros((B, K, 3))
    target_contacts[0, 0] = [0.5, 0.0, 0.0]

    target_normals = np.zeros((B, K, 3))
    target_normals[:] = [0.0, 0.0, -1.0]

    # Start the joint slightly off the anchor, but within region_radius
    init_q = np.array([[0.51]])

    sol = solve_region_dls_ik_batch(
        spec=mock_spec,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        target_contacts=target_contacts,
        target_normals=target_normals,
        init_q=init_q,
        max_iter=10,
        region_radius=0.02
    )

    # Since 0.51 is within 0.02 of 0.5, the dynamic target should be 0.51 itself (distance 0 to current)
    # The solver should converge immediately (0 error) or at least not move to 0.5.

    assert np.all(sol.converged)
    # The joint shouldn't have moved much from 0.51
    np.testing.assert_allclose(sol.q[0, 0], 0.51, atol=1e-3)

    # Now start far away (e.g., 0.8)
    init_q = np.array([[0.8]])
    sol = solve_region_dls_ik_batch(
        spec=mock_spec,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        target_contacts=target_contacts,
        target_normals=target_normals,
        init_q=init_q,
        max_iter=50,
        region_radius=0.02
    )

    assert np.all(sol.converged)
    # Should move towards the region and stop at the boundary (0.5 + 0.02 = 0.52)
    # Since the gradient pushes it towards the region boundary.
    np.testing.assert_allclose(sol.q[0, 0], 0.52, atol=5e-2)
