import hashlib

import numpy as np
import pytest

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.scene_loader import SceneDataset, audit_scene_dataset
from qdgrasp.dataset.scene_manifest import (
    SceneDatasetManifest,
    SceneShardMetadata,
    save_scene_manifest,
)
from qdgrasp.dataset.scene_shards import read_scene_shard, write_scene_shard


def _positive_grasp():
    return {
        "record_type": "grasp",
        "scene_id": "scene-1",
        "target_object_id": "object-1",
        "robot_profile": "leap",
        "candidate_id": "candidate-1",
        "dynamic_valid": True,
        "label_stage": "dynamic_valid",
        "failure_reason": "none",
        "static_certificate": {"passed": True},
        "swept_clearance_metrics": {"passed": True, "minimum_clearance": 0.01},
        "dynamic_trajectory_evidence": {
            "validated_stages": ["initial", "squeeze", "lift", "perturbation"],
            "per_finger_loads": np.ones((2, 6)),
            "measured_target_lift": 0.05,
        },
        "scene_state_hashes": {
            "initial": "1" * 64,
            "squeeze": "2" * 64,
            "lift": "3" * 64,
            "perturbation": "4" * 64,
        },
        "protocol_hash": "a" * 64,
        "recipe_hash": "b" * 64,
        "source_hash": "c" * 64,
    }


def _manifest(shard_hash, *, release_blocked=False):
    return SceneDatasetManifest(
        dataset_id="scene-tiny-test",
        generator_version="test",
        generator_commit="deadbeef",
        generator_worktree_dirty=False,
        seed=42,
        splits={"train": ["scene-1"]},
        scene_spec_hashes={"scene-1": "d" * 64},
        camera_calibration_hashes={"cam-1": "e" * 64},
        environment_hashes={"table": "f" * 64},
        source_licenses={"native": "CC0-1.0"},
        shards=(
            SceneShardMetadata(
                filename="shards/train-grasp.jsonl",
                sha256=shard_hash,
                num_records=1,
                record_type="grasp",
                split="train",
            ),
        ),
        success_criteria={"minimum_target_lift": 0.025},
        release_blocked=release_blocked,
    )


def test_scene_shard_is_deterministic_and_integrity_checked(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_hash = write_scene_shard([_positive_grasp()], first, record_type="grasp")
    second_hash = write_scene_shard([_positive_grasp()], second, record_type="grasp")
    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    assert read_scene_shard(
        first, record_type="grasp", expected_sha256=first_hash, expected_records=1
    )[0]["dynamic_valid"]

    first.write_bytes(first.read_bytes() + b" ")
    with pytest.raises(ConfigError, match="integrity mismatch"):
        read_scene_shard(first, record_type="grasp", expected_sha256=first_hash)


@pytest.mark.parametrize(
    "mutation",
    ["fixture", "stages", "loads", "state_hash", "static", "clearance", "identity_hash"],
)
def test_scene_positive_admission_rejects_incomplete_or_mock_evidence(tmp_path, mutation):
    record = _positive_grasp()
    if mutation == "fixture":
        record["source_class"] = "test_fixture_only"
    elif mutation == "stages":
        record["dynamic_trajectory_evidence"]["validated_stages"] = ["lift"]
    elif mutation == "loads":
        record["dynamic_trajectory_evidence"]["per_finger_loads"] = []
    elif mutation == "state_hash":
        record["scene_state_hashes"]["lift"] = "fake"
    elif mutation == "static":
        record["static_certificate"]["passed"] = False
    elif mutation == "clearance":
        record["swept_clearance_metrics"]["passed"] = False
    else:
        record["protocol_hash"] = "fake"
    with pytest.raises(ConfigError):
        write_scene_shard([record], tmp_path / "bad.jsonl", record_type="grasp")


def test_scene_loader_verifies_manifest_split_count_and_release_state(tmp_path):
    shard_path = tmp_path / "shards" / "train-grasp.jsonl"
    shard_hash = write_scene_shard([_positive_grasp()], shard_path, record_type="grasp")
    save_scene_manifest(_manifest(shard_hash), tmp_path / "scene_manifest.json")
    dataset = SceneDataset(tmp_path, split="train", record_type="grasp")
    assert len(dataset) == 1
    assert dataset[0]["candidate_id"] == "candidate-1"

    save_scene_manifest(
        _manifest(shard_hash, release_blocked=True), tmp_path / "scene_manifest.json"
    )
    with pytest.raises(ConfigError, match="release is blocked"):
        SceneDataset(tmp_path, split="train")
    assert len(SceneDataset(tmp_path, split="train", allow_incomplete=True)) == 1


def test_manifest_rejects_split_leakage_and_shard_path_traversal():
    payload = _manifest("0" * 64).model_dump()
    payload["splits"] = {"train": ["scene-1"], "val": ["scene-1"]}
    with pytest.raises(ValueError, match="leak"):
        SceneDatasetManifest.model_validate(payload)

    with pytest.raises(ValueError, match="relative POSIX"):
        SceneShardMetadata(
            filename="../escape.jsonl",
            sha256="0" * 64,
            num_records=1,
            record_type="grasp",
            split="train",
        )


def test_manifest_bytes_are_stable(tmp_path):
    manifest = _manifest("0" * 64)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    save_scene_manifest(manifest, first)
    save_scene_manifest(manifest, second)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()


def test_release_audit_resolves_positive_stage_hashes_across_shards(tmp_path):
    grasp = _positive_grasp()
    state_records = [
        {
            "record_type": "scene_state",
            "scene_id": "scene-1",
            "stage": stage,
            "state_hash": state_hash,
            "lineage_hash": (str(index + 5) * 64)[:64],
        }
        for index, (stage, state_hash) in enumerate(grasp["scene_state_hashes"].items())
    ]
    observation = {
        "record_type": "observation",
        "scene_id": "scene-1",
        "camera_id": "cam-1",
        "calibration_hash": "e" * 64,
    }
    shard_specs = []
    for record_type, records in (
        ("scene_state", state_records),
        ("observation", [observation]),
        ("grasp", [grasp]),
    ):
        filename = f"shards/train-{record_type}.jsonl"
        digest = write_scene_shard(records, tmp_path / filename, record_type=record_type)
        shard_specs.append(
            SceneShardMetadata(
                filename=filename,
                sha256=digest,
                num_records=len(records),
                record_type=record_type,
                split="train",
            )
        )
    manifest = _manifest(shard_specs[-1].sha256).model_copy(update={"shards": tuple(shard_specs)})
    save_scene_manifest(manifest, tmp_path / "scene_manifest.json")
    assert audit_scene_dataset(tmp_path) == {
        "scene_state": 4,
        "observation": 1,
        "grasp": 1,
        "positive_grasp": 1,
    }

    state_records.pop()
    state_meta = shard_specs[0]
    digest = write_scene_shard(
        state_records, tmp_path / state_meta.filename, record_type="scene_state"
    )
    shard_specs[0] = state_meta.model_copy(
        update={"sha256": digest, "num_records": len(state_records)}
    )
    save_scene_manifest(
        manifest.model_copy(update={"shards": tuple(shard_specs)}),
        tmp_path / "scene_manifest.json",
    )
    with pytest.raises(ConfigError, match="absent scene-state hashes"):
        audit_scene_dataset(tmp_path)
