#!/usr/bin/env python3
"""Generate ``QDGrasp-RL-Env-Tiny`` (P3.5-17).

``ROADMAP-P3.5-001`` §13.8 asks for object-only, loaded-scene and
generated-scene cases, positive scripted, negative and random fixtures, and raw
evidence hashes.  Each case is *measured* here and its outcome recorded as it
came out; the negative case is included precisely because an artifact that only
contains successes cannot show that the predicate is capable of refusing.

The artifact proves the environment is ready, not that a policy is good.  The
positive case is driven by a fitted pinch prior with privileged access to the
target's pose -- a fixture, not a learner -- and the manifest says so.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.objects.generate import generate_box
from qdgrasp.objects.manifest import create_object_asset, save_object_asset
from qdgrasp.rl.envs import ACTIVE_ROBOT_PROFILES, DexAcquireConfig, DexAcquireEnv
from qdgrasp.rl.envs.object_settle import ObjectSettleConfig, ObjectSettleEnv
from qdgrasp.rl.tasks import random_policy_probe
from qdgrasp.rl.tasks.grasp_prior import GraspPriorSpec, run_prior_episode
from qdgrasp.scenes.resolver import resolve_scene
from qdgrasp.scenes.serialize import scene_spec_hash, write_scene_spec
from qdgrasp.scenes.virtual_drop import DropObjectRequest, SpawnRegion, VirtualDropSceneSpec

DATASET_ID = "QDGrasp-RL-Env-Tiny"
MANIFEST_SCHEMA = "qdgrasp/rl-env-tiny-manifest/v1"
DEFAULT_OUTPUT = Path("datasets/qdgrasp-rl-env-tiny")


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_target(directory: Path, name: str, size_range: tuple[float, float], seed: int) -> str:
    rng = np.random.default_rng(seed)
    mesh, geoms, params, mass, inertia = generate_box(rng, size_range=size_range, density=650.0)
    mesh_bytes, manifest = create_object_asset(name, "primitive", "box", mesh, geoms, params, mass, inertia)
    return str(save_object_asset(mesh_bytes, manifest, directory))


def _scene_config(half_extent: float = 0.03) -> VirtualDropSceneSpec:
    return VirtualDropSceneSpec(
        spawn_region=SpawnRegion(half_extents=(half_extent, half_extent, 0.0)),
        drop_height_range_m=(0.02, 0.04),
        object_count_range=(1, 2),
    )


def case_object_only(assets: Path) -> dict[str, Any]:
    """Ingest and settle with no scene and no hand: the debug path on its own."""

    asset_ref = _make_target(assets, "solo", (0.028, 0.034), seed=0)
    scene = _scene_config()
    config = ObjectSettleConfig(
        objects=(DropObjectRequest(object_id="solo", asset_ref=asset_ref),),
        virtual_scene=scene,
        max_steps=60,
    )
    env = ObjectSettleEnv(config)
    observation, info = env.reset(seed=101)
    steps = 0
    while True:
        observation, _reward, terminated, truncated, step_info = env.step(np.zeros(1))
        steps += 1
        if terminated or truncated:
            break
    snapshot = env.certify()
    return {
        "case": "object_only",
        "environment_id": ObjectSettleEnv.environment_id,
        "scene_source": info["scene_source"],
        "scene_signature": info["scene_signature"],
        "steps": steps,
        "settled": bool(step_info["settled"]),
        "terminal_reason": step_info["terminal_reason"].value,
        "settle_outcome": snapshot.outcome.value,
        "snapshot_hash": snapshot.content_hash(),
        "observation_dim": int(observation.shape[0]),
    }


def case_generated_scene(assets: Path, profile: str, spec: GraspPriorSpec, cache: dict) -> dict[str, Any]:
    """A scene the system generated, acquired by the fitted prior."""

    asset_ref = _make_target(assets, "generated_target", (0.028, 0.034), seed=1)
    scene = _scene_config()
    config = DexAcquireConfig(
        robot_profile=profile,
        objects=(DropObjectRequest(object_id="generated_target", asset_ref=asset_ref),),
        target_object_id="generated_target",
        virtual_scene=scene,
        max_steps=spec.total_steps,
        settle_steps=400,
    )
    result = run_prior_episode(DexAcquireEnv(config), seed=21, spec=spec, prior_cache=cache)
    result["case"] = "generated_scene_positive"
    result["environment_id"] = DexAcquireEnv.environment_id
    return result


def case_loaded_scene(
    assets: Path, output: Path, profile: str, spec: GraspPriorSpec, cache: dict
) -> tuple[dict[str, Any], Path]:
    """The same acquire, from a scene loaded off disk rather than generated."""

    asset_ref = _make_target(assets, "loaded_target", (0.028, 0.034), seed=2)
    scene = _scene_config()
    objects = (DropObjectRequest(object_id="loaded_target", asset_ref=asset_ref),)
    generated = resolve_scene(objects=objects, virtual_scene_config=scene, seed=7, scene_id="rl-env-tiny-loaded")
    scene_path = write_scene_spec(output / "scenes" / "loaded_scene.json", generated.spec)

    config = DexAcquireConfig(
        robot_profile=profile,
        objects=objects,
        target_object_id="loaded_target",
        virtual_scene=scene,
        max_steps=spec.total_steps,
        settle_steps=400,
    )
    env = DexAcquireEnv(config, scene_ref=str(scene_path))
    result = run_prior_episode(env, seed=21, spec=spec, prior_cache=cache)
    result["case"] = "loaded_scene_positive"
    result["environment_id"] = DexAcquireEnv.environment_id
    result["scene_ref"] = str(scene_path)
    result["scene_spec_hash"] = scene_spec_hash(generated.spec)
    return result, scene_path


def case_negative(assets: Path, profile: str, spec: GraspPriorSpec, cache: dict) -> dict[str, Any]:
    """A target too wide for the hand's aperture: the predicate must refuse.

    Without a case the environment is *supposed* to fail, a manifest of
    successes says nothing about whether the predicate can say no.
    """

    asset_ref = _make_target(assets, "too_wide", (0.075, 0.080), seed=3)
    scene = _scene_config(half_extent=0.02)
    config = DexAcquireConfig(
        robot_profile=profile,
        objects=(DropObjectRequest(object_id="too_wide", asset_ref=asset_ref),),
        target_object_id="too_wide",
        virtual_scene=scene,
        max_steps=spec.total_steps,
        settle_steps=400,
    )
    result = run_prior_episode(DexAcquireEnv(config), seed=31, spec=spec, prior_cache=cache)
    result["case"] = "negative_out_of_aperture"
    result["environment_id"] = DexAcquireEnv.environment_id
    result["expected"] = "not success"
    result["behaved_as_expected"] = not result["success"]
    return result


def case_random(assets: Path, profile: str) -> dict[str, Any]:
    """A random policy: finite observations, and no success by accident."""

    asset_ref = _make_target(assets, "random_target", (0.028, 0.034), seed=4)
    scene = _scene_config()
    config = DexAcquireConfig(
        robot_profile=profile,
        objects=(DropObjectRequest(object_id="random_target", asset_ref=asset_ref),),
        target_object_id="random_target",
        virtual_scene=scene,
        max_steps=60,
        settle_steps=400,
    )
    probe = random_policy_probe(DexAcquireEnv(config), seed=41, steps=60)
    return {
        "case": "random_policy",
        "environment_id": DexAcquireEnv.environment_id,
        "robot_profile": profile,
        "expected": "no successes, finite observations",
        "behaved_as_expected": bool(probe["observations_finite"] and probe["successes"] == 0),
        **probe,
    }


def build(output: Path) -> dict[str, Any]:
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    spec = GraspPriorSpec()
    cache: dict = {}
    cases: list[dict[str, Any]] = [case_object_only(assets)]
    for profile in ACTIVE_ROBOT_PROFILES:
        cases.append(case_generated_scene(assets, profile, spec, cache))
        loaded, _path = case_loaded_scene(assets, output, profile, spec, cache)
        cases.append(loaded)
        cases.append(case_negative(assets, profile, spec, cache))
        cases.append(case_random(assets, profile))

    positives = [item for item in cases if item["case"].endswith("_positive")]
    negatives = [item for item in cases if item["case"].startswith("negative")]
    randoms = [item for item in cases if item["case"] == "random_policy"]
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "commit": _commit(),
        "active_hands": list(ACTIVE_ROBOT_PROFILES),
        "shadow_hand": "paused_by_ADR-0008",
        "release_class": "experimental_non_release",
        "prior_spec": dataclasses.asdict(spec),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "positive_cases": len(positives),
            "positive_successes": sum(1 for item in positives if item["success"]),
            "negative_cases": len(negatives),
            "negatives_behaved": sum(1 for item in negatives if item["behaved_as_expected"]),
            "random_cases": len(randoms),
            "randoms_behaved": sum(1 for item in randoms if item["behaved_as_expected"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build(args.out)
    manifest_path = args.out / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = sorted(path for path in args.out.rglob("*") if path.is_file() and path.name != "artifact_hashes.json")
    hashes = {
        "schema": "qdgrasp/rl-env-tiny-hashes/v1",
        "artifacts": [
            {"path": str(path.relative_to(args.out)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifacts
        ],
    }
    (args.out / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = manifest["summary"]
    for case in manifest["cases"]:
        label = case["case"]
        detail = case.get("terminal_reason", case.get("settle_outcome", ""))
        print(
            f"{label:28s} {case.get('robot_profile', '-'):20s} {detail:20s} "
            f"success={case.get('success', '-')} expected_ok={case.get('behaved_as_expected', '-')}"
        )
    print(f"\n{json.dumps(summary, sort_keys=True)}")
    print(f"wrote {manifest_path} and {len(artifacts) + 1} artifacts")

    healthy = (
        summary["positive_successes"] == summary["positive_cases"]
        and summary["negatives_behaved"] == summary["negative_cases"]
        and summary["randoms_behaved"] == summary["random_cases"]
    )
    if not healthy:
        print("QDGrasp-RL-Env-Tiny is incomplete: a case did not behave as its class requires")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
