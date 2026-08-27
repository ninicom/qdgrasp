"""S11 — the dataset contract and the loader that enforces it (G09, C05, C06).

**B-07**: the manifest mixed pause metadata into the sample list, so counts,
hashes and the split contract disagreed with what was on disk. Coverage status
and samples are separate blocks now, and every count is recomputed from the
shard records rather than trusted.

**B-17**: coverage was asserted. The grid is declared before the run and checked
cell by cell after it, so an artifact with the right hashes and the wrong physics
fails.

**B-20**: nothing stood between the artifact and a consumer. The public loader
refuses a v1 schema, a dirty worktree, a hash mismatch, a shard path that
escapes the root, a split with a group on both sides, and a blocked artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qdgrasp.config.active_scope import ACTIVE_HANDS, PAUSED_HANDS
from qdgrasp.dataset.contactrich_active import (
    REQUIRED_SPLITS,
    DatasetRejected,
    dataset_card,
    load,
    verify,
)
from qdgrasp.dataset.dynamic_contracts import (
    CONTACTRICH_MANIFEST_SCHEMA_V2,
    DYNAMIC_TRAJECTORY_SCHEMA_V2,
    DynamicSearchOutcome,
    TrajectoryStage,
    TrajectoryTimebase,
)
from qdgrasp.dataset.dynamic_shards import write_trajectory_shard

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PERIOD_S = 0.01


def _trajectory(steps: int = 4):
    from qdgrasp.dataset.dynamic_contracts import DynamicGraspTrajectory

    palm = np.zeros((steps, 7))
    palm[:, 3] = 1.0
    pose = np.zeros((steps, 1, 7))
    pose[:, :, 3] = 1.0
    return DynamicGraspTrajectory(
        time=np.arange(steps, dtype=float) * SAMPLE_PERIOD_S,
        palm_pose=palm,
        joint_state=np.zeros((steps, 16)),
        actuator_command=np.zeros((steps, 16)),
        object_pose=pose,
        object_velocity=np.zeros((steps, 1, 6)),
        stage=tuple([TrajectoryStage.APPROACH] * steps),
        timebase=TrajectoryTimebase(simulator_dt=SAMPLE_PERIOD_S, sample_every=1),
    )


def _outcome():
    return DynamicSearchOutcome(
        trajectory_ref="t:0",
        passed=False,
        failure_stage="lift",
        failure_reason="insufficient_lift",
    )


def build_artifact(root: Path, **manifest_overrides) -> Path:
    """A minimal well-formed artifact, so each test can break exactly one thing."""
    shards = []
    for split, count in (("train", 2), ("val", 1)):
        path = root / "shards" / f"{split}.json"
        digest = write_trajectory_shard(
            path, [(_trajectory(), _outcome()) for _ in range(count)]
        )
        shards.append(
            {"path": f"shards/{split}.json", "split": split, "count": count, "sha256": digest}
        )

    manifest = {
        "schema": CONTACTRICH_MANIFEST_SCHEMA_V2,
        "dataset_id": "QDGrasp-ContactRich-Active-Tiny",
        "version": "2.0.0",
        "worktree_dirty": False,
        "scope": {
            "active_hands": list(ACTIVE_HANDS),
            "paused_hands": list(PAUSED_HANDS),
            "selected_hands": list(ACTIVE_HANDS),
            "three_hand_coverage": False,
            "historical_p3_4_state": "paused_by_ADR-0008",
            "governing_decision": "ADR-0008",
        },
        "coverage_status": {hand: "paused_by_ADR-0008" for hand in PAUSED_HANDS},
        "coverage": {
            "declared_cells": 4,
            "observed_cells": 4,
            "positives_by_hand": dict.fromkeys(ACTIVE_HANDS, 1),
        },
        "counts": {
            "samples": 3,
            "train": 2,
            "val": 1,
            "dispositions": {"positive": 2, "negative": 1},
        },
        "shards": shards,
        "splits": {
            "train": {"count": 2, "groups": ["g1", "g2"]},
            "val": {"count": 1, "groups": ["g3"]},
        },
        "release_blocked": False,
        "blocked_reasons": [],
        "license": "AGPL-3.0-or-later",
        "limitations": ["Simulation-only contact."],
    }
    manifest.update(manifest_overrides)
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


# -- the loader refuses what it should ------------------------------------


def test_a_well_formed_artifact_loads(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    assert verify(root) == []
    dataset = load(root)
    assert dataset.dataset_id == "QDGrasp-ContactRich-Active-Tiny"
    assert set(dataset.active_hands) == set(ACTIVE_HANDS)
    assert dataset.three_hand_coverage is False
    assert len(dataset.split("train")) == 2
    assert len(dataset.split("val")) == 1


def test_a_v1_schema_is_never_release_ready(tmp_path: Path) -> None:
    root = build_artifact(tmp_path, schema=DYNAMIC_TRAJECTORY_SCHEMA_V2)
    problems = verify(root)
    assert any("not the release schema" in p for p in problems)
    with pytest.raises(DatasetRejected):
        load(root)


def test_a_dirty_worktree_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path, worktree_dirty=True)
    assert any("dirty worktree" in p for p in verify(root))


def test_a_hash_mismatch_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["shards"][0]["sha256"] = "0" * 64
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("hash" in p for p in verify(root))


def test_a_count_that_disagrees_with_the_shards_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["samples"] = 99
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("but the shards hold" in p for p in verify(root))


def test_a_shard_path_that_escapes_the_root_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["shards"][0]["path"] = "../../etc/passwd"
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("escapes the dataset root" in p for p in verify(root))


def test_split_leakage_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["splits"]["val"]["groups"] = ["g1"]
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("split leakage" in p for p in verify(root))


def test_an_empty_split_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["val"] = 0
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("is empty" in p for p in verify(root))


def test_a_passing_negative_control_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["dispositions"]["unexpected_control_outcome"] = 2
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("does not control anything" in p for p in verify(root))


def test_an_incomplete_coverage_grid_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["coverage"]["observed_cells"] = 3
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("coverage grid incomplete" in p for p in verify(root))


def test_a_hand_without_a_positive_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["coverage"]["positives_by_hand"][ACTIVE_HANDS[0]] = 0
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("no CPU-confirmed positive" in p for p in verify(root))


def test_a_three_hand_claim_is_refused(tmp_path: Path) -> None:
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["scope"]["three_hand_coverage"] = True
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("three_hand_coverage" in p for p in verify(root))


def test_a_blocked_artifact_is_inspectable_but_never_promoted(tmp_path: Path) -> None:
    root = build_artifact(
        tmp_path, release_blocked=True, blocked_reasons=["the CUDA gate has not run"]
    )
    assert any("release_blocked=true" in p for p in verify(root))
    # Inspection is allowed; it does not make the artifact release-ready.
    assert verify(root, allow_blocked=True) == []
    dataset = load(root, allow_blocked=True)
    assert dataset.release_blocked is True


# -- coverage status is not sample data -----------------------------------


def test_pause_metadata_is_not_counted_as_a_sample(tmp_path: Path) -> None:
    # v1 filed the paused hand's status alongside real trajectories, so the
    # counts described something that was not on disk (blocker B-07).
    root = build_artifact(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["coverage_status"]) == set(PAUSED_HANDS)
    assert manifest["counts"]["samples"] == sum(
        shard["count"] for shard in manifest["shards"]
    )
    assert not any(
        hand in json.dumps(manifest["counts"]) for hand in PAUSED_HANDS
    )


def test_required_splits_are_declared_once() -> None:
    assert REQUIRED_SPLITS == ("train", "val")


# -- the dataset card discloses the scope ---------------------------------


def test_the_card_discloses_the_pause_and_the_limitations(tmp_path: Path) -> None:
    root = build_artifact(
        tmp_path, release_blocked=True, blocked_reasons=["the CUDA gate has not run"]
    )
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    card = dataset_card(manifest)
    assert "Three-hand coverage: **no**" in card
    assert "paused_by_ADR-0008" in card
    assert "not a claim about what a" in card or "Nothing here is a claim" in card
    assert "release_blocked = true" in card
    assert "AGPL-3.0-or-later" in card


# -- the shipped artifact, if it has been generated -----------------------

ARTIFACT_ROOT = REPO_ROOT / "datasets" / "contactrich-active-tiny"


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "dataset_manifest.json").is_file(),
    reason="the artifact has not been generated in this checkout",
)
def test_the_generated_artifact_is_well_formed() -> None:
    problems = [p for p in verify(ARTIFACT_ROOT, allow_blocked=True) if "dirty worktree" not in p]
    assert problems == [], problems


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "dataset_manifest.json").is_file(),
    reason="the artifact has not been generated in this checkout",
)
def test_the_generated_artifact_covers_the_declared_grid() -> None:
    manifest = json.loads(
        (ARTIFACT_ROOT / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    coverage = manifest["coverage"]
    assert coverage["observed_cells"] == coverage["declared_cells"]
    assert set(coverage["environment_classes"]) == {"table", "tray", "bin"}
    assert set(coverage["clutter_tiers"]) == {"sparse", "denser"}
    assert len(coverage["generation_modes"]) == 3
    for hand in ACTIVE_HANDS:
        assert coverage["positives_by_hand"].get(hand, 0) >= 1


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "dataset_manifest.json").is_file(),
    reason="the artifact has not been generated in this checkout",
)
def test_the_generated_artifact_has_working_negative_controls() -> None:
    manifest = json.loads(
        (ARTIFACT_ROOT / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    dispositions = manifest["counts"]["dispositions"]
    assert dispositions.get("unexpected_control_outcome", 0) == 0
    assert dispositions.get("unexpected_control_reason", 0) == 0
    assert dispositions.get("negative", 0) >= len(ACTIVE_HANDS)


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "dataset_manifest.json").is_file(),
    reason="the artifact has not been generated in this checkout",
)
def test_the_generated_artifact_declares_itself_blocked() -> None:
    # It is blocked, and it says why: the CUDA gate has not run and no
    # independent review has been issued.
    manifest = json.loads(
        (ARTIFACT_ROOT / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["release_blocked"] is True
    assert manifest["blocked_reasons"]


# -- storage does not grow with the integrator timestep --------------------


def test_the_stored_contact_stream_is_one_reading_per_contact_per_sample() -> None:
    """C06.1: a finer timestep is a simulation choice, not more data.

    The observer reads contacts every integrator step, because impulse, work and
    duration are only honest if it does. What gets *stored* is one reading per
    contact pair per episode per recorded sample -- so the payload is bounded by
    the sample count, not by how finely the simulator was stepped.
    """
    from qdgrasp.dataset.dynamic_contracts import ContactClass, ContactPairKind
    from qdgrasp.dynamic.wrapped_rollout import _sparsify_contacts

    from .test_taxonomy_and_terminal import _event

    # Forty readings of the same contact inside one sample, as a fine timestep
    # would produce, plus one in the next sample.
    dense = [
        _event(0, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL,
               budget_margin=0.9 - step * 0.01, simulator_step=step)
        for step in range(40)
    ]
    dense.append(
        _event(1, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL,
               budget_margin=0.5, simulator_step=40)
    )
    sparse = _sparsify_contacts(dense, steps=2)
    assert len(sparse) == 2

    # The reading that survives is the worst one, because that is the one the
    # safety verdict was made on.
    assert sparse[0].budget_margin == pytest.approx(0.9 - 39 * 0.01)
    assert sparse[0].time_index == 0
    assert sparse[1].time_index == 1


def test_halving_the_timestep_does_not_double_the_stored_stream() -> None:
    from qdgrasp.dataset.dynamic_contracts import ContactClass, ContactPairKind
    from qdgrasp.dynamic.wrapped_rollout import _sparsify_contacts

    from .test_taxonomy_and_terminal import _event

    def stream(readings_per_sample: int):
        return [
            _event(sample, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL,
                   budget_margin=0.5, simulator_step=sample * readings_per_sample + step)
            for sample in range(10)
            for step in range(readings_per_sample)
        ]

    coarse = _sparsify_contacts(stream(5), steps=10)
    fine = _sparsify_contacts(stream(10), steps=10)
    assert len(coarse) == len(fine) == 10


def test_separate_contact_pairs_are_both_kept() -> None:
    from qdgrasp.dataset.dynamic_contracts import ContactClass, ContactPairKind
    from qdgrasp.dynamic.wrapped_rollout import _sparsify_contacts

    from .test_taxonomy_and_terminal import _event

    events = [
        _event(0, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL,
               geom_a="tip_0", body_a="distal_0"),
        _event(0, ContactPairKind.TARGET_SUPPORT, ContactClass.SUPPORT_ASSISTED,
               geom_a="target_geom", geom_b="table", body_a="target", body_b="table"),
    ]
    assert len(_sparsify_contacts(events, steps=1)) == 2


def test_a_recontact_is_kept_apart_from_the_episode_before_it() -> None:
    from qdgrasp.dataset.dynamic_contracts import ContactClass, ContactPairKind
    from qdgrasp.dynamic.wrapped_rollout import _sparsify_contacts

    from .test_taxonomy_and_terminal import _event

    events = [
        _event(0, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL,
               episode_index=0),
        _event(0, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL,
               episode_index=1),
    ]
    assert len(_sparsify_contacts(events, steps=1)) == 2
