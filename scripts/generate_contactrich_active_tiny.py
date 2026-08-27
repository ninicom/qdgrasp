#!/usr/bin/env python3
"""Generate QDGrasp-ContactRich-Active-Tiny (S11; G09, C05, C06).

A successor artifact, not an overwrite. ContactRich v1 stays where it is with
its own verdict; this one has a new id, a new schema and a scope of two hands.

Three things the first dataset got wrong are fixed here.

**The manifest counted metadata as data** (blocker B-07). A paused hand's
``paused_by_ADR-0008`` entry sat in the same list as real trajectories, so the
sample count, the shard hashes and the split contract disagreed with what was on
disk. Coverage status and samples are now separate blocks, and every count is
computed from the shard records and cross-checked against the shard header.

**Coverage was asserted rather than enumerated** (blocker B-17). The declared
grid -- two active hands, three environment classes, two clutter tiers, three
generation modes -- is written down before the run and checked cell by cell
after it, so a dataset with the right hashes but the wrong physics fails.

**Negative controls had no predicate** (part of C05.12). Each control now
declares the failure it is supposed to produce. A control that passes is not
quietly counted as a negative: it is recorded as ``unexpected_control_outcome``,
which is a gate failure, because a control that does not control anything is
evidence about the harness rather than about the hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdgrasp.config.active_scope import (
    ACTIVE_HANDS,
    GOVERNING_DECISION,
    PAUSED_HANDS,
    profile_of_hand,
    require_release_scope,
    resolve_workload_hands,
)
from qdgrasp.dataset.dynamic_contracts import (
    CONTACTRICH_MANIFEST_SCHEMA_V2,
    ContactSafetyBudget,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
)
from qdgrasp.dataset.dynamic_shards import write_trajectory_shard
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
    RolloutSceneObject,
)
from qdgrasp.dynamic.safety import SceneRoles
from qdgrasp.dynamic.self_contact import (
    SelfContactPolicy,
    build_self_contact_policy,
    resolve_geom_allowlist,
)
from qdgrasp.dynamic.wrapped_rollout import run_wrapped_contact_rollout
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.scenes.release_recipes import build_release_grasp_recipe

DATASET_ID = "QDGrasp-ContactRich-Active-Tiny"
DATASET_VERSION = "2.0.0"

#: The declared coverage grid, pinned before the run. A cell that does not
#: appear in the result is a hole, not an omission someone can wave through.
ENVIRONMENT_CLASSES: tuple[str, ...] = ("table", "tray", "bin")
CLUTTER_TIERS: tuple[str, ...] = ("sparse", "denser")
GENERATION_MODES: tuple[str, ...] = ("static_seeded", "primitive_sequence", "bounded_cem")

#: Safety budgets, pinned per hand before generation and hashed into the
#: manifest. Simulation limits only: this is not a claim that a physical hand
#: survives them.
BUDGETS: dict[str, ContactSafetyBudget] = {
    hand: ContactSafetyBudget(
        budget_id=f"contactrich-active-{hand}-v1",
        robot_profile=hand,
        peak_normal_force_N=25.0,
        peak_tangential_force_N=15.0,
        normal_impulse_Ns=2.5,
        tangential_impulse_Ns=1.5,
        contact_duration_s=6.0,
        contact_work_J=0.6,
        max_penetration_m=0.003,
        max_wrist_force_N=60.0,
        max_wrist_torque_Nm=8.0,
        max_joint_or_tendon_load=25.0,
        max_non_target_translation_m=0.02,
        max_non_target_rotation_rad=0.30,
        max_non_target_velocity_mps=0.10,
        impulse_window_s=0.1,
        environment_class="table",
    )
    for hand in (*ACTIVE_HANDS, *PAUSED_HANDS)
}


@dataclasses.dataclass(frozen=True)
class NegativeControl:
    """A control, and the failure it is supposed to produce.

    Without the predicate a control is just another run: whatever it produced
    gets filed as a negative, including a pass.
    """

    control_id: str
    description: str
    overrides: dict[str, Any]
    expected_reason_prefixes: tuple[str, ...]

    def matches(self, outcome: DynamicSearchOutcome) -> bool:
        return any(
            outcome.failure_reason.startswith(prefix)
            for prefix in self.expected_reason_prefixes
        )


NEGATIVE_CONTROLS: tuple[NegativeControl, ...] = (
    NegativeControl(
        control_id="no_closure",
        description="fingers never leave the pregrasp pose, so nothing encloses the target",
        overrides={"use_initial_as_closed": True},
        expected_reason_prefixes=("validated_rollout:", "insufficient_enclosure", "no_closure"),
    ),
    NegativeControl(
        control_id="heavy_object",
        description="target thirty times the validated mass, so the grasp cannot hold it",
        overrides={"object_mass": 0.60},
        expected_reason_prefixes=(
            "validated_rollout:",
            "insufficient_lift",
            "support_not_released",
            "perturbation_slip",
        ),
    ),
    NegativeControl(
        control_id="palm_crush",
        description="palm driven into the support, so the target is crushed against the table",
        overrides={"palm_drop_m": 0.025},
        expected_reason_prefixes=("damaging_contact", "safety_budget_violation"),
    ),
    NegativeControl(
        control_id="excessive_perturbation",
        description="perturbation wrench well past the validated envelope, so the grasp slips",
        overrides={"perturbation_wrench": (12.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        expected_reason_prefixes=(
            "validated_rollout:",
            "perturbation_slip",
            "insufficient_lift",
            "damaging_contact",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def _neighbour(object_id: str, position: tuple[float, float, float], size: float = 0.02) -> RolloutSceneObject:
    return RolloutSceneObject(
        object_id=object_id,
        collision_geoms=(
            SubGeomSpec(
                type="box",
                size=(size, size, size),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
        pos=position,
        mass=0.05,
    )


def _wall(object_id: str, position: tuple[float, float, float], size: tuple[float, float, float]) -> RolloutSceneObject:
    """A static-ish rim or wall, heavy enough not to be pushed around."""
    return RolloutSceneObject(
        object_id=object_id,
        collision_geoms=(
            SubGeomSpec(type="box", size=size, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
        ),
        pos=position,
        mass=5.0,
    )


def scene_objects(environment: str, clutter: str) -> tuple[RolloutSceneObject, ...]:
    """Non-target geometry for one environment class and clutter tier.

    The three environment classes differ by what surrounds the target: an open
    plane, a low rim, and higher walls. They are built from the same scene
    builder rather than from three bespoke scenes, so the only thing that
    changes between them is the geometry the plan says should change.
    """
    if environment not in ENVIRONMENT_CLASSES:
        raise ValueError(f"unknown environment class {environment!r}")
    if clutter not in CLUTTER_TIERS:
        raise ValueError(f"unknown clutter tier {clutter!r}")

    # The rim and the walls sit outside the neighbours, so a neighbour is inside
    # the enclosure rather than spawned through it -- a box overlapping a wall is
    # ejected at reset, and the trajectory then fails for the spawn rather than
    # for anything the hand did.
    objects: list[RolloutSceneObject] = []
    if environment == "tray":
        objects += [
            _wall("tray_rim_x", (0.18, 0.0, 0.008), (0.008, 0.18, 0.008)),
            _wall("tray_rim_y", (0.0, 0.18, 0.008), (0.18, 0.008, 0.008)),
        ]
    elif environment == "bin":
        objects += [
            _wall("bin_wall_x", (0.18, 0.0, 0.030), (0.008, 0.18, 0.030)),
            _wall("bin_wall_y", (0.0, 0.18, 0.030), (0.18, 0.008, 0.030)),
            _wall("bin_wall_nx", (-0.18, 0.0, 0.030), (0.008, 0.18, 0.030)),
        ]

    # Far enough out that a neighbour is a neighbour rather than something the
    # hand closes on: at 0.07 m the fingers reached them and every cell came
    # back as a damaging contact, which measured the placement, not the grasp.
    neighbours = 1 if clutter == "sparse" else 3
    offsets = ((0.14, 0.0), (0.0, 0.14), (-0.14, 0.0))
    objects += [
        _neighbour(f"neighbour_{index}", (offsets[index][0], offsets[index][1], 0.02))
        for index in range(neighbours)
    ]
    return tuple(objects)


def roles_builder(*, wrist_link: str, palm_link: str, policy: SelfContactPolicy):
    """A ``roles_from_model`` closure that classifies by identity.

    Every geom is placed by the body it belongs to: the target body, the floor
    and the declared walls as support, the neighbours as non-target, and
    everything else as robot. A geom whose role cannot be determined stays
    unknown, which the observer treats as forbidden rather than harmless.
    """

    def build(model: mujoco.MjModel) -> SceneRoles:
        target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        target: set[int] = set()
        support: set[int] = set()
        non_target: set[int] = set()
        robot: set[int] = set()
        for geom in range(int(model.ngeom)):
            body = int(model.geom_bodyid[geom])
            name = mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_BODY), body) or ""
            if body == target_body:
                target.add(geom)
            elif geom == floor or name.startswith(("tray_rim", "bin_wall")):
                support.add(geom)
            elif name.startswith("neighbour_"):
                non_target.add(geom)
            else:
                robot.add(geom)

        wrist_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_link)
        palm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_link)
        if wrist_body < 0:
            raise ValueError(
                f"wrist body {wrist_link!r} is not in the compiled model; the wrist "
                "force and torque budgets cannot be measured without it"
            )
        return SceneRoles(
            frozenset(target),
            frozenset(support),
            frozenset(non_target),
            frozenset(robot),
            resolve_geom_allowlist(model, policy, frozenset(robot)),
            wrist_body=int(wrist_body),
            palm_body=int(palm_body) if palm_body >= 0 else None,
        )

    return build


def wrist_link_of(spec: RobotSpec) -> str:
    """Where the wrist budget resolves for this hand.

    Neither active hand declares a separate wrist link: both are welded to the
    mocap through the palm, so the external wrench at that body is the load the
    wrist carries. Which body it was is recorded, because a wrist number nobody
    can locate is not a measurement.
    """
    return spec.wrist_link or spec.base_link or spec.palm_link


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Sample:
    """One generated record and everything needed to file it."""

    sample_id: str
    hand: str
    environment: str
    clutter: str
    mode: str
    control_id: str
    trajectory: DynamicGraspTrajectory
    outcome: DynamicSearchOutcome
    group_key: str
    disposition: str


#: Per-mode variation of the validated protocol. The modes differ in what
#: chooses the parameters, not in what physics runs: static-seeded holds the
#: pinned recipe, the primitive sequence walks the declared stage schedule, and
#: the bounded search varies the lift and the squeeze inside a fixed budget.
MODE_OVERRIDES: dict[str, dict[str, Any]] = {
    "static_seeded": {},
    "primitive_sequence": {"squeeze_steps": 180, "lift_steps": 180},
    "bounded_cem": {"squeeze_steps": 210, "lift_steps": 200, "lift_height": 0.06},
}


def generate_one(
    hand: str,
    *,
    environment: str,
    clutter: str,
    mode: str,
    overrides: dict[str, Any] | None = None,
) -> tuple[DynamicGraspTrajectory, DynamicSearchOutcome]:
    spec = RobotSpec.from_config(profile_of_hand(hand), sample_anchors=False)
    recipe = build_release_grasp_recipe(profile_of_hand(hand))
    kwargs = dict(recipe.rollout_kwargs)
    kwargs.update(MODE_OVERRIDES[mode])

    extra = dict(overrides or {})
    if extra.pop("use_initial_as_closed", False):
        kwargs["joint_targets"] = dict(kwargs["initial_joint_targets"])
    drop = float(extra.pop("palm_drop_m", 0.0))
    if drop:
        # Lower the palm into the support so the target is crushed against the
        # table. This changes a physical condition -- where the hand is -- and
        # never a threshold, so the rejection is measured rather than arranged.
        palm = tuple(float(v) for v in kwargs["palm_pos"])
        kwargs["palm_pos"] = (palm[0], palm[1], palm[2] - drop)
    if "perturbation_wrench" in extra:
        extra["perturbation_wrench"] = np.asarray(extra["perturbation_wrench"], dtype=np.float64)
    kwargs.update(extra)
    kwargs["non_target_objects"] = scene_objects(environment, clutter)

    policy = build_self_contact_policy(spec, robot_profile=hand)
    budget = dataclasses.replace(BUDGETS[hand], environment_class=environment)
    trajectory, outcome, _ = run_wrapped_contact_rollout(
        hand_xml_path=resolve_robot_asset(spec.config.source_asset),
        collision_geoms=recipe.target_geoms,
        fingertip_body_names=spec.fingertip_links,
        roles_from_model=roles_builder(
            wrist_link=wrist_link_of(spec), palm_link=spec.palm_link, policy=policy
        ),
        budget=budget,
        rollout_kwargs=kwargs,
        palm_body_name=spec.palm_link,
        robot_profile=hand,
        trajectory_ref=f"{hand}/{environment}/{clutter}/{mode}",
    )
    return trajectory, outcome


def generate_samples(hands: tuple[str, ...]) -> tuple[list[Sample], list[dict[str, Any]]]:
    """Walk the declared grid, then the declared controls."""
    samples: list[Sample] = []
    log: list[dict[str, Any]] = []

    for hand in hands:
        for environment in ENVIRONMENT_CLASSES:
            for clutter in CLUTTER_TIERS:
                for mode in GENERATION_MODES:
                    started = time.perf_counter()
                    trajectory, outcome = generate_one(
                        hand, environment=environment, clutter=clutter, mode=mode
                    )
                    disposition = "positive" if outcome.passed else "negative"
                    sample_id = f"{hand}/{environment}/{clutter}/{mode}"
                    samples.append(
                        Sample(
                            sample_id=sample_id,
                            hand=hand,
                            environment=environment,
                            clutter=clutter,
                            mode=mode,
                            control_id="",
                            trajectory=trajectory,
                            outcome=outcome,
                            # Grouped by the scene template and the hand, so a
                            # split never puts two variants of the same scene on
                            # both sides of it.
                            group_key=f"{hand}:{environment}:{clutter}",
                            disposition=disposition,
                        )
                    )
                    log.append(
                        {
                            "sample_id": sample_id,
                            "passed": outcome.passed,
                            "failure_reason": outcome.failure_reason,
                            "seconds": round(time.perf_counter() - started, 2),
                        }
                    )
                    print(
                        f"  {sample_id:52s} {'pass' if outcome.passed else 'fail':4s} "
                        f"{outcome.failure_reason}",
                        flush=True,
                    )

        for control in NEGATIVE_CONTROLS:
            started = time.perf_counter()
            trajectory, outcome = generate_one(
                hand,
                environment="table",
                clutter="sparse",
                mode="static_seeded",
                overrides=dict(control.overrides),
            )
            if outcome.passed:
                # A control that does not control anything is evidence about the
                # harness, not about the hand. It is never filed as a negative.
                disposition = "unexpected_control_outcome"
            elif control.matches(outcome):
                disposition = "negative"
            else:
                disposition = "unexpected_control_reason"
            sample_id = f"{hand}/control/{control.control_id}"
            samples.append(
                Sample(
                    sample_id=sample_id,
                    hand=hand,
                    environment="table",
                    clutter="sparse",
                    mode="static_seeded",
                    control_id=control.control_id,
                    trajectory=trajectory,
                    outcome=outcome,
                    group_key=f"{hand}:control:{control.control_id}",
                    disposition=disposition,
                )
            )
            log.append(
                {
                    "sample_id": sample_id,
                    "passed": outcome.passed,
                    "failure_reason": outcome.failure_reason,
                    "disposition": disposition,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
            print(
                f"  {sample_id:52s} {'pass' if outcome.passed else 'fail':4s} "
                f"{outcome.failure_reason} -> {disposition}",
                flush=True,
            )

    return samples, log


# ---------------------------------------------------------------------------
# Splitting and manifest
# ---------------------------------------------------------------------------


def split_samples(samples: list[Sample]) -> dict[str, list[Sample]]:
    """Group-disjoint train/val split.

    The group is the scene template, so two runs of the same scene never land on
    opposite sides. Assignment is by a hash of the group rather than by
    position, so adding a sample does not reshuffle the rest.
    """
    groups = sorted({sample.group_key for sample in samples})
    validation = {
        group
        for group in groups
        if int(hashlib.sha256(group.encode("utf-8")).hexdigest(), 16) % 4 == 0
    }
    # Every hand has to appear on both sides, or the split is not a split.
    for hand in sorted({sample.hand for sample in samples}):
        hand_groups = [group for group in groups if group.startswith(f"{hand}:")]
        if not any(group in validation for group in hand_groups):
            validation.add(hand_groups[-1])
        if all(group in validation for group in hand_groups):
            validation.discard(hand_groups[0])

    out: dict[str, list[Sample]] = {"train": [], "val": []}
    for sample in samples:
        out["val" if sample.group_key in validation else "train"].append(sample)
    return out


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def build_manifest(
    *,
    splits: dict[str, list[Sample]],
    shards: list[dict[str, Any]],
    scope,
    generation_log: list[dict[str, Any]],
    release_blocked: bool,
    blocked_reasons: list[str],
) -> dict[str, Any]:
    every = [sample for group in splits.values() for sample in group]
    dispositions: dict[str, int] = {}
    for sample in every:
        dispositions[sample.disposition] = dispositions.get(sample.disposition, 0) + 1

    cells = sorted(
        {
            f"{s.hand}:{s.environment}:{s.clutter}:{s.mode}"
            for s in every
            if not s.control_id
        }
    )
    positives_by_hand: dict[str, int] = {}
    for sample in every:
        if sample.disposition == "positive":
            positives_by_hand[sample.hand] = positives_by_hand.get(sample.hand, 0) + 1

    return {
        "schema": CONTACTRICH_MANIFEST_SCHEMA_V2,
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        # Scope disclosure travels with the artifact, so a reader cannot mistake
        # a two-hand dataset for the three-hand contract.
        "scope": scope.as_disclosure(),
        # Coverage status is metadata about hands that were NOT run. It is kept
        # apart from the samples so it can never be counted as one (blocker B-07).
        "coverage_status": {hand: "paused_by_ADR-0008" for hand in PAUSED_HANDS},
        "coverage": {
            "environment_classes": list(ENVIRONMENT_CLASSES),
            "clutter_tiers": list(CLUTTER_TIERS),
            "generation_modes": list(GENERATION_MODES),
            "declared_cells": len(scope.hands)
            * len(ENVIRONMENT_CLASSES)
            * len(CLUTTER_TIERS)
            * len(GENERATION_MODES),
            "observed_cells": len(cells),
            "cells": cells,
            "positives_by_hand": positives_by_hand,
        },
        "counts": {
            "samples": len(every),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "dispositions": dict(sorted(dispositions.items())),
        },
        "shards": shards,
        "splits": {
            name: {
                "count": len(group),
                "groups": sorted({sample.group_key for sample in group}),
                "sample_ids": sorted(sample.sample_id for sample in group),
            }
            for name, group in splits.items()
        },
        "negative_controls": [
            {
                "control_id": control.control_id,
                "description": control.description,
                "expected_reason_prefixes": list(control.expected_reason_prefixes),
            }
            for control in NEGATIVE_CONTROLS
        ],
        "safety_budgets": {
            hand: {"budget_id": budget.budget_id, "budget_hash": budget.budget_hash}
            for hand, budget in BUDGETS.items()
            if hand in scope.hands
        },
        "generation_log": generation_log,
        "release_blocked": release_blocked,
        "blocked_reasons": blocked_reasons,
        "license": "AGPL-3.0-or-later",
        "limitations": [
            "Simulation-only contact. Nothing here is a hardware safety claim.",
            "Two active hands under ADR-0008; this is not three-hand coverage.",
            "GPU-derived samples require the Kaggle CUDA gate and are absent until it runs.",
        ],
    }


def audit_manifest(manifest: dict[str, Any], root: Path) -> list[str]:
    """Check the manifest against what is actually on disk.

    Every count is recomputed from the shard files rather than trusted, because
    a manifest that agrees with itself is not evidence of anything (B-07).
    """
    problems: list[str] = []
    total = 0
    for shard in manifest["shards"]:
        path = root / shard["path"]
        if not path.is_file():
            problems.append(f"shard {shard['path']} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != shard["sha256"]:
            problems.append(f"shard {shard['path']} hash {digest} != declared {shard['sha256']}")
        payload = json.loads(text)
        if int(payload.get("count", -1)) != len(payload.get("records", [])):
            problems.append(f"shard {shard['path']} header count disagrees with its records")
        if len(payload["records"]) != int(shard["count"]):
            problems.append(f"shard {shard['path']} record count disagrees with the manifest")
        total += len(payload["records"])

    if total != int(manifest["counts"]["samples"]):
        problems.append(
            f"manifest counts {manifest['counts']['samples']} samples but the shards hold {total}"
        )
    if manifest["coverage"]["observed_cells"] != manifest["coverage"]["declared_cells"]:
        problems.append(
            f"coverage grid is incomplete: {manifest['coverage']['observed_cells']} of "
            f"{manifest['coverage']['declared_cells']} declared cells"
        )
    for hand in manifest["scope"]["selected_hands"]:
        if manifest["coverage"]["positives_by_hand"].get(hand, 0) < 1:
            problems.append(f"{hand} has no CPU-confirmed positive")
    if manifest["counts"]["train"] == 0 or manifest["counts"]["val"] == 0:
        problems.append("train and val must both be non-empty")
    train_groups = set(manifest["splits"]["train"]["groups"])
    val_groups = set(manifest["splits"]["val"]["groups"])
    if train_groups & val_groups:
        problems.append(f"split leakage: groups on both sides {sorted(train_groups & val_groups)}")
    unexpected = manifest["counts"]["dispositions"].get("unexpected_control_outcome", 0)
    if unexpected:
        problems.append(
            f"{unexpected} negative control(s) passed; a control that does not control "
            "anything is evidence about the harness, not a negative"
        )
    if manifest["counts"]["dispositions"].get("unexpected_control_reason", 0):
        problems.append("a negative control failed for a reason it did not declare")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "contactrich-active-tiny")
    parser.add_argument("--dry-run", action="store_true", help="print the declared grid and stop")
    args = parser.parse_args()

    scope = resolve_workload_hands()
    require_release_scope(scope.hands)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset_id": DATASET_ID,
                    "hands": list(scope.hands),
                    "environment_classes": list(ENVIRONMENT_CLASSES),
                    "clutter_tiers": list(CLUTTER_TIERS),
                    "generation_modes": list(GENERATION_MODES),
                    "grid_cells": len(scope.hands)
                    * len(ENVIRONMENT_CLASSES)
                    * len(CLUTTER_TIERS)
                    * len(GENERATION_MODES),
                    "negative_controls": len(NEGATIVE_CONTROLS) * len(scope.hands),
                },
                indent=2,
            )
        )
        return 0

    print(f"Generating {DATASET_ID} for {list(scope.hands)}")
    samples, log = generate_samples(scope.hands)
    splits = split_samples(samples)

    root = Path(args.out)
    shards: list[dict[str, Any]] = []
    for name, group in splits.items():
        if not group:
            continue
        path = root / "shards" / f"{name}.json"
        digest = write_trajectory_shard(
            path, [(sample.trajectory, sample.outcome) for sample in group]
        )
        shards.append(
            {
                "path": str(path.relative_to(root)),
                "split": name,
                "count": len(group),
                "sha256": digest,
            }
        )

    blocked_reasons = [
        (
            "G08 (CUDA gate) has no run yet: GPU-derived samples and the GPU/CPU "
            "divergence fixture require the Kaggle T4 harness"
        ),
        "G10 (independent review) has not been issued",
    ]
    manifest = build_manifest(
        splits=splits,
        shards=shards,
        scope=scope,
        generation_log=log,
        release_blocked=True,
        blocked_reasons=blocked_reasons,
    )
    problems = audit_manifest(manifest, root)
    manifest["self_audit"] = {"problems": problems, "clean": not problems}

    manifest_path = root / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print(f"wrote {manifest_path.relative_to(REPO_ROOT)}")
    print(f"samples={manifest['counts']['samples']} "
          f"train={manifest['counts']['train']} val={manifest['counts']['val']}")
    print(f"dispositions={manifest['counts']['dispositions']}")
    print(f"coverage={manifest['coverage']['observed_cells']}/"
          f"{manifest['coverage']['declared_cells']} cells")
    if problems:
        print()
        for problem in problems:
            print(f"  PROBLEM: {problem}", file=sys.stderr)
        return 1
    print(f"{GOVERNING_DECISION}: two active hands, release_blocked=True until the CUDA "
          "gate and the independent review land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
