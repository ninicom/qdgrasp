"""Shared machinery for the characterization suite (``PLAN.md`` §9.3 item 2).

Every test in this directory reproduces one chain from ``PLAN.md`` §9.2 and is
written against the **target** state, not against today's behaviour.  While the
finding is open in :mod:`qdgrasp.corrective.registry` the test is a strict
expected failure; when the gate that closes it lands, the registry entry flips
and the same test becomes an ordinary regression test.

Strictness is the whole point.  A non-strict xfail would let a fix land without
anyone noticing that the finding is closed, and an ordinary assertion would let
the suite go red for months and be ignored.  With ``strict=True`` an unexpected
pass fails the run and says, in effect: this is fixed now, close the entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
import torch

from qdgrasp.corrective import registry

F = TypeVar("F", bound=Callable[..., Any])


def characterization(finding_id: str, *, note: str = "", satisfied_by: str = "") -> Callable[[F], F]:
    """Bind a test to a registered finding and follow the registry's verdict.

    ``satisfied_by`` names the PR that already delivered this part of the chain
    while the chain as a whole is still open -- typically because the rest of it
    waits on data regeneration.  Such a test is an ordinary regression test from
    then on: it may not fail, and it may not be marked satisfied while it does.
    """

    finding = registry.get(finding_id)

    def decorate(function: F) -> F:
        marked = pytest.mark.corrective(finding_id)(function)
        if finding.is_open and not satisfied_by:
            reason = f"{finding.describe()}"
            if note:
                reason = f"{reason} [{note}]"
            marked = pytest.mark.xfail(strict=True, reason=reason)(marked)
        return marked  # type: ignore[return-value]

    return decorate


def refuses(action: Callable[[], Any], *, because: str) -> Exception:
    """Run ``action`` and return the refusal it raised, failing if it raised none.

    A characterization test cares that the path refuses, not which exception
    type it picks: the target implementation is free to choose one, and pinning
    it here would turn a fix into a test failure.
    """

    try:
        action()
    except Exception as error:  # noqa: BLE001 - any refusal is acceptable, silence is not
        return error
    pytest.fail(because)


# -- corpus fixtures -------------------------------------------------------


def sample(robot_name: str = "leap_hand", object_id: str = "prim_box_01", *, joints: int = 16) -> dict[str, Any]:
    """One sample carrying every field the public loader reads."""

    generator = torch.Generator().manual_seed(0)
    return {
        "points": torch.randn(64, 3, generator=generator),
        "palm_pos": torch.zeros(3),
        "palm_rot": torch.eye(3),
        "joint_angles": torch.zeros(joints),
        "fingertip_positions": torch.zeros(4, 3),
        "success": torch.tensor(0.0),
        "quality": torch.tensor(0.0),
        "object_id": object_id,
        "robot_name": robot_name,
        "robot_profile_hash": hashlib.sha256(f"profile:{robot_name}".encode()).hexdigest(),
        "joint_names": tuple(f"{robot_name}:joint_{index}" for index in range(joints)),
        "kinematics_valid": True,
        "pose_target_valid": True,
        "joint_target_valid": True,
        "fk_target_valid": True,
    }


def write_shard(path: Path, samples: list[dict[str, Any]]) -> str:
    """Write a shard and return its SHA-256, the way the generator does."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, path, _use_new_zipfile_serialization=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_document(
    *,
    filename: str,
    sha256: str,
    num_samples: int = 1,
    split: str = "train",
    robot_name: str = "leap_hand",
    dataset_id: str = "characterization-corpus",
    release_blocked: bool = False,
) -> dict[str, Any]:
    """A manifest the current schema accepts, with one shard entry."""

    return {
        "schema": "qdgrasp/dataset-manifest/v3",
        "dataset_id": dataset_id,
        "generator_version": "characterization",
        "generator_commit": "0" * 40,
        "generator_worktree_dirty": False,
        "seed": 0,
        "environment_fingerprint": {"python": "test"},
        "robot_profile_hashes": {robot_name: "0" * 64},
        "object_manifest_hashes": {},
        "generator_source_hashes": {},
        "recipe_id": "characterization",
        "proposal_module": "characterization",
        "solver_module": "characterization",
        "certifier_version": "characterization",
        "dynamic_protocol_version": "characterization",
        "splits": {"train": ["prim_box_01"], "val": ["prim_box_02"]},
        "shards": [
            {
                "filename": filename,
                "sha256": sha256,
                "num_samples": num_samples,
                "positive_samples": 0,
                "robot_name": robot_name,
                "split": split,
                "recipe_id": "characterization",
            }
        ],
        "success_criteria": {"min_contacts": 3.0, "max_penetration": 0.001},
        "license": "CC0-1.0",
        "release_blocked": release_blocked,
        "invalidated": False,
        "invalidation_reason": "",
    }


def write_manifest(root: Path, document: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# -- a payload that proves whether pickle reducers run ---------------------
#
# The effect is deliberately inert: a directory inside the test's own tmp_path.
# What matters is only whether it exists afterwards, which answers "did loading
# this artifact execute code" without a payload anyone has to trust.


class ReducerProbe:
    """An object whose unpickling would create ``marker``."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (os.mkdir, (str(self.marker),))


def write_reducer_payload(path: Path, marker: Path, *, wrap_in_list: bool = True) -> Path:
    """Save an artifact whose load executes a reducer under an unsafe loader."""

    path.parent.mkdir(parents=True, exist_ok=True)
    probe = ReducerProbe(marker)
    torch.save([probe] if wrap_in_list else probe, path)
    return path


def payload_with_reducer(payload: dict[str, Any], marker: Path, *, key: str = "metadata") -> dict[str, Any]:
    """A checkpoint-shaped mapping carrying the probe in one field."""

    document = dict(payload)
    document[key] = ReducerProbe(marker)
    return document
