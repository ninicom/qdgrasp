import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.scene_manifest import (
    SceneDatasetManifest,
    SceneShardMetadata,
    save_scene_manifest,
)
from qdgrasp.dataset.scene_shards import write_scene_shard
from qdgrasp.scenes.adapters import get_adapter
from qdgrasp.scenes.adapters._common import calibration_hash


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _source_manifest(root: Path, dataset_id: str, **extra):
    _write_json(
        root / "source_manifest.json",
        {
            "dataset_id": dataset_id,
            "version": "fixture-v1",
            "license": "fixture-license",
            "source_url": "https://example.invalid/fixture",
            **extra,
        },
    )


def _png(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _graspnet_layout(root: Path, dataset_id: str, *, dgn2=False):
    _source_manifest(
        root,
        dataset_id,
        **({"splits": {"train": ["scene_0000"]}} if dgn2 else {}),
    )
    scene = root / "scenes" / "scene_0000"
    camera = scene / "kinect"
    (camera / "annotations").mkdir(parents=True)
    (scene / "object_id_list.txt").write_text("0\n", encoding="utf-8")
    np.save(camera / "camera_poses.npy", np.eye(4)[None])
    alignment = np.eye(4)
    alignment[0, 3] = 1.0
    np.save(camera / "cam0_wrt_table.npy", alignment)
    intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    np.save(camera / "camK.npy", intrinsics)
    (camera / "annotations" / "0000.xml").write_text(
        "<root><obj><obj_id>0</obj_id><pos_in_world>0 0 0.1</pos_in_world>"
        "<ori_in_world>1 0 0 0</ori_in_world></obj></root>",
        encoding="utf-8",
    )
    _png(camera / "rgb" / "0000.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _png(camera / "depth" / "0000.png", np.full((2, 2), 100, dtype=np.uint16))
    _png(camera / "label" / "0000.png", np.array([[1, 1], [0, 0]], dtype=np.uint8))
    if dgn2:
        asset = root / "meshdata" / "000" / "simplified.obj"
        asset.parent.mkdir(parents=True)
        asset.write_text("v 0 0 0\n", encoding="utf-8")
        grasp_dir = root / "dex_grasps_new" / "scene_0000" / "leap_hand"
        grasp_dir.mkdir(parents=True)
        np.savez(
            grasp_dir / "000.npz",
            point=np.zeros((2, 3)),
            translation=np.zeros((2, 3)),
            rotation=np.repeat(np.eye(3)[None], 2, axis=0),
            joint_0=np.zeros(2),
        )
    else:
        asset = root / "models" / "000" / "nontextured.ply"
        asset.parent.mkdir(parents=True)
        asset.write_text("ply\n", encoding="utf-8")
        labels = root / "grasp_label"
        labels.mkdir()
        np.savez(labels / "000_labels.npz", points=np.zeros((1, 3)))
    return alignment, intrinsics


def _graspclutter_layout(root: Path):
    _source_manifest(root, "graspclutter6d", environment="bin")
    for filename, values in {
        "grasp_train_scene_ids.json": [0],
        "grasp_test_scene_ids.json": [],
        "ycbv_train_scene_ids.json": [],
        "ycbv_test_scene_ids.json": [],
    }.items():
        _write_json(root / "split_info" / filename, values)
    scene = root / "scenes" / "000000"
    camera = {
        "cam_K": [100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0],
        "cam_R_w2c": np.eye(3).reshape(-1).tolist(),
        "cam_t_w2c": [0.0, 0.0, 0.0],
        "depth_scale": 1.0,
    }
    gt = {
        "obj_id": 1,
        "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
        "cam_t_m2c": [0.0, 0.0, 100.0],
    }
    _write_json(scene / "scene_camera.json", {"1": camera})
    _write_json(scene / "scene_gt.json", {"1": [gt]})
    _write_json(scene / "scene_gt_info.json", {"1": [{}]})
    _png(scene / "rgb" / "000001.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _png(scene / "depth" / "000001.png", np.full((2, 2), 100, dtype=np.uint16))
    _png(scene / "label" / "000001.png", np.ones((2, 2), dtype=np.uint8))
    _png(
        scene / "visible_mask" / "000001_000000.png",
        np.array([[255, 255], [0, 0]], dtype=np.uint8),
    )
    asset = root / "models_obj_m" / "obj_000001.obj"
    asset.parent.mkdir(parents=True)
    asset.write_text("v 0 0 0\n", encoding="utf-8")
    labels = root / "grasp_label"
    labels.mkdir()
    np.savez(labels / "obj_000001_labels.npz", points=np.zeros((1, 3)))


def _native_layout(root: Path):
    asset = root / "assets" / "object.manifest.json"
    asset.parent.mkdir(parents=True)
    asset.write_text("{}", encoding="utf-8")
    rgb = root / "observations" / "rgb.png"
    _png(rgb, np.zeros((2, 2, 3), dtype=np.uint8))
    transform = np.eye(4)
    intrinsics = np.eye(3)
    spec = {
        "scene_id": "native-1",
        "source_dataset": "native",
        "source_version": "1",
        "source_split": "train",
        "environment": "table",
        "objects": [
            {
                "object_id": "object-1",
                "asset_ref": "assets/object.manifest.json",
                "T_world_object": transform.tolist(),
            }
        ],
        "supports": [],
        "cameras": [
            {
                "camera_id": "cam-1",
                "intrinsics": intrinsics.tolist(),
                "T_world_camera": transform.tolist(),
            }
        ],
        "redistributable": True,
    }
    spec_hash = (
        __import__("hashlib").sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    state = {
        "record_type": "scene_state",
        "scene_id": "native-1",
        "stage": "initial",
        "state_hash": "1" * 64,
        "lineage_hash": "2" * 64,
        "scene_spec": spec,
    }
    observation = {
        "record_type": "observation",
        "scene_id": "native-1",
        "camera_id": "cam-1",
        "frame_id": "0",
        "timestamp": 0.0,
        "T_world_camera": transform.tolist(),
        "calibration_hash": calibration_hash(intrinsics, transform),
        "rgb_ref": "observations/rgb.png",
        "visibility_by_object": {"object-1": 1.0},
    }
    grasp = {
        "record_type": "grasp",
        "scene_id": "native-1",
        "dynamic_valid": False,
        "label_stage": "rejected",
    }
    shards = []
    for record_type, records in (
        ("scene_state", [state]),
        ("observation", [observation]),
        ("grasp", [grasp]),
    ):
        filename = f"shards/train-{record_type}.jsonl"
        digest = write_scene_shard(records, root / filename, record_type=record_type)
        shards.append(
            SceneShardMetadata(
                filename=filename,
                sha256=digest,
                num_records=1,
                record_type=record_type,
                split="train",
            )
        )
    save_scene_manifest(
        SceneDatasetManifest(
            dataset_id="native-fixture",
            generator_version="fixture-v1",
            generator_commit="fixture",
            generator_worktree_dirty=False,
            seed=1,
            splits={"train": ["native-1"]},
            scene_spec_hashes={"native-1": spec_hash},
            camera_calibration_hashes={"cam-1": calibration_hash(intrinsics, transform)},
            environment_hashes={"table": "3" * 64},
            source_licenses={"native": "CC0-1.0"},
            shards=tuple(shards),
            success_criteria={"minimum_target_lift": 0.025},
            release_blocked=True,
        ),
        root / "scene_manifest.json",
    )


def test_registry_is_an_allowlist():
    for name in ("native", "graspnet1b", "dexgraspnet2", "graspclutter6d"):
        assert get_adapter(name) is not None
    with pytest.raises(ValueError, match="Unknown scene adapter"):
        get_adapter("invalid_adapter_name")


def test_empty_or_unproven_root_never_fabricates_index(tmp_path):
    adapter = get_adapter("graspnet1b")
    assert not adapter.probe(str(tmp_path)).is_valid
    assert adapter.index(str(tmp_path), "train").scene_keys == []
    (tmp_path / "scenes").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "grasp_label").mkdir()
    assert not adapter.probe(str(tmp_path)).is_valid
    with pytest.raises(ConfigError, match="source manifest"):
        adapter.load_scene(str(tmp_path), "scene_0000")


def test_graspnet_scene_observation_external_label_and_audit(tmp_path):
    alignment, _ = _graspnet_layout(tmp_path, "graspnet-1billion")
    adapter = get_adapter("graspnet1b")
    assert adapter.probe(str(tmp_path)).num_scenes == 1
    assert adapter.index(str(tmp_path), "train").scene_keys == ["scene_0000"]
    scene = adapter.load_scene(str(tmp_path), "scene_0000")
    np.testing.assert_allclose(scene.objects[0].T_world_object[:3, 3], [1.0, 0.0, 0.1])
    np.testing.assert_allclose(scene.cameras[0].T_world_camera, alignment)
    observation = adapter.load_observation(str(tmp_path), "scene_0000", "kinect", "0")
    assert observation.visibility_by_object == {"000": 0.5}
    grasps = adapter.load_external_grasps(str(tmp_path), "scene_0000")
    assert grasps.grasps[0]["label_stage"] == "external_label"
    assert "dynamic_valid" not in grasps.grasps[0]
    audit = adapter.audit(str(tmp_path), "scene_0000")
    assert audit.is_complete and len(audit.record_hash) == 64


def test_dexgraspnet2_preserves_hand_and_upstream_unreplayed_boundary(tmp_path):
    _graspnet_layout(tmp_path, "dexgraspnet2", dgn2=True)
    adapter = get_adapter("dexgraspnet2")
    assert adapter.index(str(tmp_path), "train").scene_keys == ["scene_0000"]
    scene = adapter.load_scene(str(tmp_path), "scene_0000")
    assert scene.objects[0].asset_ref.endswith("meshdata/000/simplified.obj")
    labels = adapter.load_external_grasps(str(tmp_path), "scene_0000")
    assert labels.gripper_type == "leap_hand"
    assert labels.grasps[0]["source_validation"] == "upstream_simulation_unreplayed"
    assert labels.grasps[0]["num_grasps"] == 2
    assert adapter.audit(str(tmp_path), "scene_0000").is_complete


def test_graspclutter_bop_pose_visibility_split_and_license(tmp_path):
    _graspclutter_layout(tmp_path)
    adapter = get_adapter("graspclutter6d")
    info = adapter.probe(str(tmp_path))
    assert info.is_valid and info.license_type == "fixture-license"
    assert adapter.index(str(tmp_path), "grasp_train").scene_keys == ["000000"]
    scene = adapter.load_scene(str(tmp_path), "000000")
    assert scene.environment == "bin"
    np.testing.assert_allclose(scene.objects[0].T_world_object[:3, 3], [0.0, 0.0, 0.1])
    observation = adapter.load_observation(str(tmp_path), "000000", "frame_000001", "1")
    assert observation.visibility_by_object == {"obj_000001": 0.5}
    assert adapter.load_external_grasps(str(tmp_path), "000000").grasps[0]["label_stage"] == "external_label"
    assert adapter.audit(str(tmp_path), "000000").is_complete


def test_native_adapter_reads_verified_shards_and_rejects_missing_reference(tmp_path):
    _native_layout(tmp_path)
    adapter = get_adapter("native")
    assert adapter.probe(str(tmp_path)).num_scenes == 1
    assert adapter.index(str(tmp_path), "train").scene_keys == ["native-1"]
    scene = adapter.load_scene(str(tmp_path), "native-1")
    assert scene.objects[0].asset_ref.startswith(str(tmp_path.resolve()))
    observation = adapter.load_observation(str(tmp_path), "native-1", "cam-1", "0")
    assert observation.visibility_by_object == {"object-1": 1.0}
    assert adapter.audit(str(tmp_path), "native-1").is_complete
    Path(observation.rgb_ref).unlink()
    with pytest.raises(ConfigError, match="missing"):
        adapter.load_observation(str(tmp_path), "native-1", "cam-1", "0")
    assert not adapter.audit(str(tmp_path), "native-1").is_complete


def test_graspclutter_rejects_camera_frame_mismatch(tmp_path):
    _graspclutter_layout(tmp_path)
    adapter = get_adapter("graspclutter6d")
    with pytest.raises(ConfigError, match="camera/frame mismatch"):
        adapter.load_observation(str(tmp_path), "000000", "frame_000002", "1")


def test_graspnet_rejects_zero_norm_object_quaternion(tmp_path):
    _graspnet_layout(tmp_path, "graspnet-1billion")
    annotation = tmp_path / "scenes/scene_0000/kinect/annotations/0000.xml"
    annotation.write_text(
        "<root><obj><obj_id>0</obj_id><pos_in_world>0 0 0.1</pos_in_world>"
        "<ori_in_world>0 0 0 0</ori_in_world></obj></root>",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="zero-norm quaternion"):
        get_adapter("graspnet1b").load_scene(str(tmp_path), "scene_0000")


def test_external_audit_fails_when_observation_evidence_is_removed(tmp_path):
    _graspnet_layout(tmp_path, "graspnet-1billion")
    adapter = get_adapter("graspnet1b")
    (tmp_path / "scenes/scene_0000/kinect/label/0000.png").unlink()
    audit = adapter.audit(str(tmp_path), "scene_0000")
    assert not audit.is_complete
    assert audit.record_hash == ""
