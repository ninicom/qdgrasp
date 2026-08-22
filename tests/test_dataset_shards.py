from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.batch import GraspBatch
from qdgrasp.dataset.loader import DgnOpenDataset
from qdgrasp.dataset.manifest import DatasetManifestSpec, ShardMetadata, save_dataset_manifest
from qdgrasp.dataset.shards import read_shard_file, write_shard_file


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

    batch_pinned = batch.pin_memory()
    if torch.cuda.is_available():
        assert batch_pinned.points.is_pinned()
    else:
        assert batch_pinned.points is batch.points


def test_dgn_open_dataset_load_shards() -> None:
    samples_train = _make_dummy_samples(4)
    samples_val = _make_dummy_samples(2)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        p_train = root / "train_shard_0.pt"
        p_val = root / "val_shard_0.pt"

        sha_train = write_shard_file(samples_train, p_train)
        sha_val = write_shard_file(samples_val, p_val)

        manifest = DatasetManifestSpec(
            dataset_id="test_tiny",
            generator_version="0.1.0a1",
            seed=42,
            environment_fingerprint={"torch": "2.11"},
            robot_profile_hashes={"leap": "abc"},
            splits={"train": ["obj_0", "obj_1", "obj_2", "obj_3"], "val": ["obj_4", "obj_5"]},
            shards=[
                ShardMetadata(
                    filename="train_shard_0.pt",
                    sha256=sha_train,
                    num_samples=4,
                    positive_samples=2,
                    robot_name="leap_hand",
                    split="train",
                ),
                ShardMetadata(
                    filename="val_shard_0.pt",
                    sha256=sha_val,
                    num_samples=2,
                    positive_samples=1,
                    robot_name="leap_hand",
                    split="val",
                ),
            ],
            success_criteria={"min_contacts": 1.0},
        )
        save_dataset_manifest(manifest, root / "dataset_manifest.json")

        ds_train = DgnOpenDataset(dataset_root=root, split="train", point_count=256)
        assert len(ds_train) == 4
        item = ds_train[0]
        assert item["points"].shape == (256, 3)  # Padded to point_count

        ds_val = DgnOpenDataset(dataset_root=root, split="val", point_count=128)
        assert len(ds_val) == 2
