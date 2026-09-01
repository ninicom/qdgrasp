from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from qdgrasp.config.loader import load_robot_config
from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.manifest import (
    DatasetManifestSpec,
    ShardMetadata,
    save_dataset_manifest,
)
from qdgrasp.dataset.shards import write_shard_file
from qdgrasp.objects.generate import generate_box
from qdgrasp.objects.manifest import create_object_asset, save_object_asset

_CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_dataset_manifest",
    Path(__file__).resolve().parents[1] / "scripts" / "check_dataset_manifest.py",
)
assert _CHECKER_SPEC is not None and _CHECKER_SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_CHECKER_SPEC)
_CHECKER_SPEC.loader.exec_module(_CHECKER)
audit_dataset_manifest = _CHECKER.audit_dataset_manifest


def _sample(object_id: str, success: bool) -> dict[str, object]:
    """A complete training sample.

    The audit reads the tensors now, not only the labels: a fixture that carries
    a verdict but no grasp cannot demonstrate that the audit accepts a corpus,
    because there is nothing in it a trainer could consume.
    """

    profile = load_robot_config("robots/leap_hand.yaml")
    joints = len(profile.joints)
    fingertips = len(getattr(profile, "fingertip_links", ()))
    return {
        "points": torch.zeros((8, 3), dtype=torch.float32),
        "palm_pos": torch.zeros(3, dtype=torch.float32),
        "palm_rot": torch.eye(3, dtype=torch.float32),
        "joint_angles": torch.zeros(joints, dtype=torch.float32),
        "fingertip_positions": torch.zeros((fingertips, 3), dtype=torch.float32),
        "quality": torch.tensor(float(success), dtype=torch.float32),
        "success": torch.tensor(float(success), dtype=torch.float32),
        "dynamic_valid": success,
        "static_force_valid": True,
        "collision_valid": True,
        "ik_valid": True,
        "proposal_valid": True,
        "recipe_id": "surface_fixed_v1",
        "frame": "object",
        "proposal_module": "surface_fixed",
        "solver_module": "fixed_contact_dls",
        "certifier_version": "gws-gravity-v1",
        "dynamic_protocol_version": "mocap-weld-v3",
        "success_schema_version": "dynamic-only-v1",
        "object_id": object_id,
        "robot_name": "leap_hand",
        "failure_stage": "none" if success else "dynamic_squeeze",
        "failure_reason": "passed" if success else "rollout_failed_squeeze",
    }


def _make_release(tmp_path: Path) -> tuple[Path, DatasetManifestSpec]:
    repo_root = tmp_path
    root = repo_root / "datasets" / "fixture"
    object_dir = root / "objects"
    source_path = repo_root / "scripts" / "fixture_generator.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("VERSION = 1\n", encoding="utf-8")

    object_hashes = {}
    for index, object_id in enumerate(("train_box", "val_box")):
        mesh, geoms, params, mass, inertia = generate_box(
            np.random.default_rng(index), size_range=(0.04, 0.04)
        )
        mesh_bytes, object_manifest = create_object_asset(
            object_id,
            "primitive",
            "box",
            mesh,
            geoms,
            params,
            mass,
            inertia,
        )
        manifest_path = save_object_asset(mesh_bytes, object_manifest, object_dir)
        object_hashes[object_id] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    shard_metadata = []
    for split, object_id in (("train", "train_box"), ("val", "val_box")):
        filename = f"shards/{split}_leap_hand.pt"
        shard_hash = write_shard_file(
            [_sample(object_id, True), _sample(object_id, False)], root / filename
        )
        shard_metadata.append(
            ShardMetadata(
                filename=filename,
                sha256=shard_hash,
                num_samples=2,
                positive_samples=1,
                robot_name="leap_hand",
                split=split,
                recipe_id="surface_fixed_v1",
            )
        )

    profile = load_robot_config("robots/leap_hand.yaml")
    manifest = DatasetManifestSpec(
        dataset_id="audit-fixture",
        generator_version="test",
        generator_commit="fixture-commit",
        generator_worktree_dirty=False,
        seed=42,
        environment_fingerprint={"python": "test"},
        robot_profile_hashes={"leap_hand": profile.content_hash()},
        object_manifest_hashes=object_hashes,
        generator_source_hashes={
            "scripts/fixture_generator.py": hashlib.sha256(source_path.read_bytes()).hexdigest()
        },
        recipe_id="surface_fixed_v1",
        proposal_module="surface_fixed",
        solver_module="fixed_contact_dls",
        certifier_version="gws-gravity-v1",
        dynamic_protocol_version="mocap-weld-v3",
        splits={"train": ["train_box"], "val": ["val_box"]},
        shards=shard_metadata,
        success_criteria={
            "min_contacts": 2.0,
            "max_penetration": 0.002,
            "min_lift_ratio": 0.5,
        },
    )
    save_dataset_manifest(manifest, root / "dataset_manifest.json")
    return root, manifest


def test_manifest_audit_recounts_labels_and_accepts_complete_fixture(tmp_path):
    root, _ = _make_release(tmp_path)

    summary = audit_dataset_manifest(root)

    assert summary["total_samples"] == 4
    assert summary["total_positives"] == 2


def test_manifest_audit_rejects_declared_positive_count_drift(tmp_path):
    root, manifest = _make_release(tmp_path)
    bad_shard = manifest.shards[0].model_copy(update={"positive_samples": 0})
    bad_manifest = manifest.model_copy(update={"shards": [bad_shard, manifest.shards[1]]})
    save_dataset_manifest(bad_manifest, root / "dataset_manifest.json")

    with pytest.raises(ConfigError, match="positive count mismatch"):
        audit_dataset_manifest(root)


def test_manifest_audit_rejects_stale_release_file(tmp_path):
    root, _ = _make_release(tmp_path)
    (root / "shards" / "stale.pt").write_bytes(b"stale")

    with pytest.raises(ConfigError, match="release file set mismatch"):
        audit_dataset_manifest(root)
