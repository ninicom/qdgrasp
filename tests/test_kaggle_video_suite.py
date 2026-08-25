from __future__ import annotations

import copy
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.scene_manifest import load_scene_manifest
from qdgrasp.scenes.release_recipes import build_release_grasp_recipe
from scripts import render_4view_rollout as video_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "qdgrasp-scene-tiny"


class _RecorderStub:
    instances: ClassVar[list[_RecorderStub]] = []

    def __init__(self, candidate_id, *, frame_stride, width, height):
        self.candidate_id = candidate_id
        self.frame_stride = frame_stride
        self.width = width
        self.height = height
        self.stages: list[str] = []
        self.steps: list[str] = []
        self.instances.append(self)

    def observe_stage(self, stage, model, data):
        del model, data
        self.stages.append(stage)

    def observe_step(self, stage, model, data):
        del model, data
        self.steps.append(stage)

    def finish(self, verdict, metrics):
        assert verdict == "PASS"
        assert metrics["has_palm_contact"] == 0.0
        assert metrics["floor_support"] == 0.0
        return [np.zeros((8, 8, 3), dtype=np.uint8)]


def _write_stub(path: Path, frames, *, fps: int) -> None:
    assert frames and fps > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"measured-video")


def test_video_suite_loads_only_the_three_admitted_release_controls():
    records = video_suite._load_positive_records(DATASET_ROOT)
    assert len(records) == 3
    assert {record["robot_profile"] for record in records} == video_suite.EXPECTED_ROBOTS
    assert all(record["source_class"] == "native_measured_release" for record in records)
    assert all(record["dynamic_valid"] for record in records)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("source_class", "test_fixture_only", "not an admitted native measured release"),
        ("recipe_hash", "0" * 64, "recipe hash"),
        ("protocol_hash", "0" * 64, "protocol hash"),
        ("source_hash", "0" * 64, "source hash"),
    ],
)
def test_video_suite_rejects_identity_mutations(field, replacement, message):
    record = copy.deepcopy(video_suite._load_positive_records(DATASET_ROOT)[0])
    recipe = build_release_grasp_recipe(record["robot_profile"])
    record[field] = replacement
    with pytest.raises(ConfigError, match=message):
        video_suite._verify_record_identity(
            record,
            recipe,
            source_hash=load_scene_manifest(DATASET_ROOT / "scene_manifest.json").scene_spec_hashes[
                record["scene_id"]
            ],
        )


@pytest.mark.parametrize("robot_profile", sorted(video_suite.EXPECTED_ROBOTS))
def test_video_replays_exact_validator_path_and_derives_pass_manifest(
    monkeypatch, tmp_path, robot_profile
):
    _RecorderStub.instances.clear()
    monkeypatch.setattr(video_suite, "_RolloutRecorder", _RecorderStub)
    monkeypatch.setattr(video_suite, "_write_video", _write_stub)
    record = next(
        item
        for item in video_suite._load_positive_records(DATASET_ROOT)
        if item["robot_profile"] == robot_profile
    )
    result = video_suite.render_release_record(
        record,
        dataset_root=DATASET_ROOT,
        output_dir=tmp_path,
        frame_stride=25,
        width=32,
        height=24,
    )
    assert result["actual_outcome"] == "PASS"
    assert result["category"] == "pass"
    assert result["failure_stage"] == "none"
    assert result["has_palm_contact"] is False
    assert result["floor_support"] is False
    assert result["final_active_fingers"] >= 2
    assert result["measured_target_lift"] > 0.0
    assert result["scene_state_hashes"] == record["scene_state_hashes"]
    assert Path(result["video_path"]).read_bytes() == b"measured-video"
    assert _RecorderStub.instances[0].stages == ["initial", "squeeze", "lift", "perturbation"]
    assert {"approach", "squeeze", "lift", "perturbation"}.issubset(_RecorderStub.instances[0].steps)


@pytest.mark.parametrize("mutation", ("state_hash", "trajectory_evidence"))
def test_video_replay_rejects_mutated_admission_evidence(monkeypatch, tmp_path, mutation):
    monkeypatch.setattr(video_suite, "_RolloutRecorder", _RecorderStub)
    monkeypatch.setattr(video_suite, "_write_video", _write_stub)
    record = copy.deepcopy(video_suite._load_positive_records(DATASET_ROOT)[0])
    if mutation == "state_hash":
        record["scene_state_hashes"]["lift"] = "0" * 64
        message = "state hashes drifted"
    else:
        record["dynamic_trajectory_evidence"]["max_penetration"] += 1.0
        message = "dynamic_trajectory_evidence.max_penetration"
    with pytest.raises(ConfigError, match=message):
        video_suite.render_release_record(
            record,
            dataset_root=DATASET_ROOT,
            output_dir=tmp_path,
            frame_stride=25,
            width=32,
            height=24,
        )


def test_video_source_has_no_legacy_fake_pass_path():
    source = (REPO_ROOT / "scripts" / "render_4view_rollout.py").read_text(encoding="utf-8")
    assert "scenario_cfg" not in source
    assert "actual_success = bool(lift_achieved" not in source
    assert "model.geom_pos[floor_geom_id][2] = -10.0" not in source
    assert '"q_close"' not in source
    assert "EXPECTED_ROBOTS" in source
    assert "run_scene_grasp_rollout" in source
