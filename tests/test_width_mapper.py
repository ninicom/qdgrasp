import numpy as np

from qdgrasp.dataset.pipeline.proposals.width_mapper import WidthMapper
from qdgrasp.robot.spec import RobotSpec


def test_opposition_seed_keeps_shadow_inactive_fingers_canonical_open():
    spec = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    mapper = WidthMapper(spec)
    active = np.array([True, False, False, False, True], dtype=bool)

    seed = mapper.map_width_to_opposition_qpos(0.043, active_fingers=active)
    canonical = mapper.get_canonical_open_qpos()

    inactive_prefixes = ("rh_MF", "rh_RF", "rh_LF")
    inactive_indices = [
        index
        for index, name in enumerate(spec.actuated_joint_names)
        if name.startswith(inactive_prefixes)
    ]
    assert inactive_indices
    np.testing.assert_allclose(
        seed[inactive_indices], canonical[inactive_indices], atol=1e-6, rtol=0.0
    )


def test_opposition_seed_rejects_mask_without_non_thumb_finger():
    spec = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    mapper = WidthMapper(spec)
    active = np.array([False, False, False, True], dtype=bool)

    with np.testing.assert_raises_regex(ValueError, "active non-thumb"):
        mapper.map_width_to_opposition_qpos(0.04, active_fingers=active)
