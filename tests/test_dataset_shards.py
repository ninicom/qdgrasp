from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.batch import GraspBatch
from qdgrasp.dataset.loader import DgnOpenDataset
from qdgrasp.dataset.shards import read_shard_file, write_shard_file
from qdgrasp.engine.sampling import collate_samples


def _make_dummy_samples(n: int = 4) -> list[dict[str, object]]:
    samples = []
    for i in range(n):
        samples.append(
            {
                "points": torch.randn(128, 3),
                "palm_pos": torch.randn(3),
                "palm_rot": torch.eye(3),
                "joint_angles": torch.randn(16),
                "fingertip_positions": torch.randn(4, 3),
                "success": torch.tensor(1.0 if i % 2 == 0 else 0.0),
                "quality": torch.tensor(0.05),
                "object_id": f"obj_{i}",
                "robot_name": "leap_hand",
                "kinematics_valid": True,
                "pose_target_valid": True,
                "joint_target_valid": True,
                "fk_target_valid": True,
            }
        )
    return samples


def test_write_and_read_shard_round_trip() -> None:
    samples = _make_dummy_samples(6)
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "shard_001.pt"
        sha = write_shard_file(samples, shard_path)

        loaded = read_shard_file(shard_path, expected_sha256=sha)
        assert len(loaded) == 6
        assert loaded[0]["object_id"] == "obj_0"

        with pytest.raises(ConfigError, match="integrity mismatch"):
            read_shard_file(shard_path, expected_sha256="wrong_hash")


def test_grasp_batch_collate_and_to() -> None:
    samples = _make_dummy_samples(4)
    batch = GraspBatch.collate(samples)

    assert batch.batch_size == 4
    assert batch.points.shape == (4, 128, 3)
    assert batch.palm_pos.shape == (4, 3)
    assert batch.palm_rot.shape == (4, 3, 3)
    assert len(batch.object_ids) == 4
    assert bool(batch.kinematics_valid.all())
    assert batch.pose_target_valid.dtype == torch.bool

    batch_pinned = batch.pin_memory()
    if torch.cuda.is_available():
        assert batch_pinned.points.is_pinned()
    else:
        assert batch_pinned.points is batch.points


def test_dgn_open_dataset_load_shards(verified_corpus) -> None:
    """The public loader opens a verified corpus and pads with an honest mask.

    Built on the real shards rather than a hand-written manifest: the loader now
    goes through ``DatasetArtifact.open_verified``, so a fixture that satisfies
    only the manifest schema is no longer a corpus it would accept -- which is
    the point of having one entry point.
    """

    subsampled = DgnOpenDataset(
        dataset_root=verified_corpus,
        split="train",
        robot_name="leap_hand",
        point_count=256,
    )
    expected = next(
        shard.num_samples
        for shard in subsampled.manifest_spec.shards
        if shard.split == "train" and shard.robot_name == "leap_hand"
    )
    assert len(subsampled) == expected
    item = subsampled[0]
    assert item["points"].shape == (256, 3)
    assert bool(item["point_mask"].all()), "a subsampled cloud has no padding to mask"
    assert item["robot_name"] == "leap_hand"
    assert item["robot_profile_hash"] == subsampled.manifest_spec.robot_profile_hashes["leap_hand"]
    assert {"kinematics_valid", "pose_target_valid", "joint_target_valid", "fk_target_valid"} <= set(item)
    batch = collate_samples([subsampled[0], subsampled[1]])
    for field in ("kinematics_valid", "pose_target_valid", "joint_target_valid", "fk_target_valid"):
        assert batch[field].shape == (2,)
        assert batch[field].dtype == torch.bool

    padded = DgnOpenDataset(
        dataset_root=verified_corpus,
        split="val",
        robot_name="leap_hand",
        point_count=2048,
    )
    assert len(padded) > 0
    padded_item = padded[0]
    assert padded_item["points"].shape == (2048, 3)
    assert int(padded_item["point_mask"].sum()) == 1024
    assert not bool(padded_item["point_mask"][1024:].any())


def test_the_loader_refuses_a_robot_outside_the_config_allowlist(verified_corpus) -> None:
    with pytest.raises(ConfigError, match="allowlist"):
        DgnOpenDataset(
            dataset_root=verified_corpus,
            split="train",
            robot_name="wonik_allegro",
            allowed_robot_names=["leap_hand.yaml"],
        )
