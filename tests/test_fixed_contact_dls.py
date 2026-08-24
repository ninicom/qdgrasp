import pytest
import numpy as np
import torch
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch

@pytest.fixture
def mock_spec():
    class MockConfig:
        name = "mock_hand"
        palm_link = "palm"
        base_link = "palm"
        wrist_link = "wrist"
        fingertip_links = ["tip_0", "tip_1"]
        contact_links = []
        joints = ("j_0", "j_1")
        joint_limits = {"j_0": (-1.0, 1.0), "j_1": (-1.0, 1.0)}
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
            # tip_1 pos = palm_pos + (0, q1, 0)
            # Normal is always (0, 0, -1)

            T0 = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
            T0[:, 0, 3] = joint_angles["j_0"] if isinstance(joint_angles, dict) else joint_angles[:, 0]
            T0[:, :3, 2] = torch.tensor([0.0, 0.0, -1.0], device=device)
            T0[:, :3, 3] += palm_pos

            T1 = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
            T1[:, 1, 3] = joint_angles["j_1"] if isinstance(joint_angles, dict) else joint_angles[:, 1]
            T1[:, :3, 2] = torch.tensor([0.0, 0.0, -1.0], device=device)
            T1[:, :3, 3] += palm_pos

            return {"tip_0": T0, "tip_1": T1}

    return MockSpec()

def test_fixed_contact_dls_batching(mock_spec):
    B = 3
    K = 2

    palm_pos = np.zeros((B, 3))
    palm_rot = np.stack([np.eye(3)] * B)

    target_contacts = np.zeros((B, K, 3))
    # We set target to something achievable
    target_contacts[0, 0] = [0.5, 0.0, 0.0]
    target_contacts[0, 1] = [0.0, -0.3, 0.0]
    target_contacts[1, 0] = [0.1, 0.0, 0.0]
    target_contacts[1, 1] = [0.0, 0.8, 0.0]
    target_contacts[2, 0] = [-0.5, 0.0, 0.0]
    target_contacts[2, 1] = [0.0, 0.9, 0.0]

    target_normals = target_contacts / np.linalg.norm(target_contacts, axis=-1, keepdims=True)

    sol = solve_dls_ik_batch(
        spec=mock_spec,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        target_contacts=target_contacts,
        target_normals=target_normals,
        # q=0 places both mock tips exactly at their fallback parent origin,
        # where normalize(parent->tip) has an undefined direction derivative.
        # Production profiles use configured contact axes; seed this fallback
        # mock inside the target ray so the test exercises batching, not that
        # artificial normalization singularity.
        init_q=np.stack(
            [target_contacts[:, 0, 0], target_contacts[:, 1, 1]], axis=1
        ) * 0.1,
        max_iter=30
    )

    assert sol.q.shape == (B, 2)
    assert sol.converged.shape == (B,)
    assert sol.achieved_contacts.shape == (B, K, 3)

    # Check that it converged
    assert np.all(sol.converged)

    # Check values roughly match targets
    np.testing.assert_allclose(sol.achieved_contacts, target_contacts, atol=1e-2)


def test_convergence_is_per_candidate_and_requires_every_tip(mock_spec):
    palm_pos = np.zeros((2, 3))
    palm_rot = np.stack([np.eye(3)] * 2)
    targets = np.array(
        [
            [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0]],
            [[0.4, 0.0, 0.0], [0.0, 3.0, 0.0]],
        ]
    )
    normals = targets / np.linalg.norm(targets, axis=-1, keepdims=True)

    sol = solve_dls_ik_batch(
        mock_spec,
        palm_pos,
        palm_rot,
        targets,
        normals,
        init_q=np.array([[0.04, 0.04], [0.04, 0.3]]),
        max_iter=30,
    )

    assert sol.converged.tolist() == [True, False]
    assert np.all(sol.position_residuals[0] < 0.005)
    assert np.any(sol.position_residuals[1] >= 0.005)
    assert sol.iterations is not None


def test_position_match_does_not_bypass_normal_alignment(mock_spec):
    targets = np.array([[[0.4, 0.0, 0.0], [0.0, 0.4, 0.0]]])
    wrong_normals = -targets / np.linalg.norm(targets, axis=-1, keepdims=True)

    sol = solve_dls_ik_batch(
        mock_spec,
        np.zeros((1, 3)),
        np.eye(3)[None],
        targets,
        wrong_normals,
        init_q=np.array([[0.4, 0.4]]),
        max_iter=3,
    )

    assert not sol.converged[0]
    assert np.all(sol.position_residuals[0] < 0.005)
    assert np.any(sol.normal_residuals[0] > np.deg2rad(30.0))
