from __future__ import annotations

import pytest
import torch

from qdgrasp.api import GraspResults
from qdgrasp.geometry import is_rotation_matrix, rot6d_to_matrix


def make_results(count: int = 3, joints: int = 2) -> GraspResults:
    generator = torch.Generator().manual_seed(11)
    return GraspResults(
        translation=torch.randn(count, 3, generator=generator),
        rotation=rot6d_to_matrix(torch.randn(count, 6, generator=generator)),
        joint_names=tuple(f"joint_{index}" for index in range(joints)),
        joint_values=torch.zeros(count, joints),
        score=torch.linspace(1.0, 0.1, count),
        seed_points=torch.randn(count, 3, generator=generator),
        frame="palm",
        model_hash="a" * 64,
        training_robot_hash="b" * 64,
        runtime_robot_hash="b" * 64,
    )


def test_rot6d_output_is_in_so3() -> None:
    matrices = rot6d_to_matrix(torch.randn(7, 6))
    assert is_rotation_matrix(matrices)


def test_shape_contract_is_enforced() -> None:
    with pytest.raises(ValueError, match="joint_values"):
        GraspResults(
            translation=torch.zeros(2, 3),
            rotation=rot6d_to_matrix(torch.randn(2, 6)),
            joint_names=("a", "b"),
            joint_values=torch.zeros(2, 3),
            score=torch.zeros(2),
            seed_points=torch.zeros(2, 3),
            frame="palm",
            model_hash="a",
            training_robot_hash="b",
            runtime_robot_hash="b",
        )


def test_invalid_rotation_is_rejected() -> None:
    with pytest.raises(ValueError, match="SO\\(3\\)"):
        GraspResults(
            translation=torch.zeros(1, 3),
            rotation=torch.zeros(1, 3, 3),
            joint_names=("a",),
            joint_values=torch.zeros(1, 1),
            score=torch.zeros(1),
            seed_points=torch.zeros(1, 3),
            frame="palm",
            model_hash="a",
            training_robot_hash="b",
            runtime_robot_hash="b",
        )


def test_conversions_preserve_values() -> None:
    results = make_results()
    assert len(results) == 3
    assert results.cpu().device.type == "cpu"
    assert results.to("cpu").summary() == results.summary()
    arrays = results.numpy()
    assert arrays["translation"].shape == (3, 3)
    assert list(arrays["joint_names"]) == list(results.joint_names)


def test_save_and_load_round_trip(tmp_path) -> None:
    results = make_results()
    path = results.save(tmp_path / "grasps")
    assert path.suffix == ".npz"
    restored = GraspResults.load(path)
    assert torch.equal(restored.translation, results.translation)
    assert torch.equal(restored.rotation, results.rotation)
    assert restored.joint_names == results.joint_names
    assert restored.metadata() == results.metadata()


def test_plot_is_deferred_to_the_robot_layer() -> None:
    with pytest.raises(NotImplementedError, match="Phase 2"):
        make_results().plot()
