from __future__ import annotations

import numpy as np
import pytest
import trimesh

from qdgrasp.dataset.pipeline.filter import filter_grasp_candidate
from qdgrasp.dataset.pipeline.ik import solve_dls_ik
from qdgrasp.dataset.pipeline.sample import sample_grasp_candidates
from qdgrasp.dataset.rng import get_generator
from qdgrasp.robot.spec import RobotSpec


@pytest.fixture
def test_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.04, 0.04, 0.04))


@pytest.mark.parametrize("preset", ["leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"])
def test_sample_grasp_candidates_all_hands(preset: str, test_mesh: trimesh.Trimesh) -> None:
    spec = RobotSpec.from_config(preset, sample_anchors=False)
    rng = get_generator(42, preset)

    candidates = sample_grasp_candidates(spec, test_mesh, rng, num_candidates=5)
    assert len(candidates) == 5

    for cand in candidates:
        assert cand.palm_pos.shape == (3,)
        assert cand.palm_rot.shape == (3, 3)
        assert cand.target_contacts.shape == (len(spec.fingertip_links), 3)
        assert cand.standoff >= 0.04


@pytest.mark.parametrize("preset", ["leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"])
def test_solve_dls_ik_and_joint_limits(preset: str, test_mesh: trimesh.Trimesh) -> None:
    spec = RobotSpec.from_config(preset, sample_anchors=False)
    rng = get_generator(123, preset)

    cand = sample_grasp_candidates(spec, test_mesh, rng, num_candidates=1)[0]
    res = solve_dls_ik(spec, cand.palm_pos, cand.palm_rot, cand.target_contacts, max_iter=20)

    assert len(res.q) == len(spec.actuated_joint_names)
    assert res.fingertip_positions.shape == (len(spec.fingertip_links), 3)

    # Check projection strictly respects joint limits
    for j_idx, j_name in enumerate(spec.actuated_joint_names):
        lo, hi = spec.joint_limits[j_name]
        assert lo - 1e-4 <= res.q[j_idx] <= hi + 1e-4


def test_filter_grasp_candidate_rejections(test_mesh: trimesh.Trimesh) -> None:
    spec = RobotSpec.from_config("wonik_allegro.yaml", sample_anchors=False)
    rng = get_generator(777)
    cand = sample_grasp_candidates(spec, test_mesh, rng, num_candidates=1)[0]
    res_ik = solve_dls_ik(spec, cand.palm_pos, cand.palm_rot, cand.target_contacts, max_iter=10)

    # 1. Normal candidate should pass
    filter_res = filter_grasp_candidate(spec, cand.palm_pos, cand.palm_rot, res_ik.q, test_mesh)
    assert filter_res.valid is True
    assert filter_res.reason == "passed"

    # 2. Joint limit violation
    bad_q = res_ik.q.copy()
    bad_q[0] = 99.0
    res_lim = filter_grasp_candidate(spec, cand.palm_pos, cand.palm_rot, bad_q, test_mesh)
    assert res_lim.valid is False
    assert "joint_limit_violation" in res_lim.reason

    # 3. Excessive penetration
    deep_palm_pos = test_mesh.centroid  # Palm right in center of object
    res_pen = filter_grasp_candidate(spec, deep_palm_pos, cand.palm_rot, res_ik.q, test_mesh, max_penetration=0.001)
    assert res_pen.valid is False
    assert "excessive_penetration" in res_pen.reason
