#!/usr/bin/env python3
"""CPU gate for Phase 3.5 (``ROADMAP-P3.5-001`` §12).

The checker walks the work breakdown in §9 and reports each package as
``delivered``, ``open`` or ``blocked``.  It is written to *fail* while packages
remain open: P3.5 closes on a GPU backend decision made from a real run and an
independent review, and a gate that returned zero without them would be
describing a phase that is not finished.

``--profile micro`` runs the live asset → scene → drop → settle → reset/step path
for both active hands.  ``--profile contract`` checks only the static contracts,
for use where MuJoCo cannot run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

STATUS_DELIVERED = "delivered"
STATUS_OPEN = "open"
STATUS_BLOCKED = "blocked"


@dataclasses.dataclass
class PackageResult:
    package: str
    title: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == STATUS_DELIVERED


def _import_checks() -> list[PackageResult]:
    """Every P3.5 module must import and expose its public surface."""

    results: list[PackageResult] = []
    surfaces = {
        "P3.5-01/02": (
            "asset ingest",
            "qdgrasp.objects.ingest",
            ("AssetIngestRequest", "ingest_asset", "normalize_mesh", "IngestErrorCode"),
        ),
        "P3.5-03/04": (
            "public CoACD API",
            "qdgrasp.objects.coacd",
            ("CoACDConfig", "decompose_collision_mesh", "build_collision_asset", "UPSTREAM_ARGUMENT_NAMES"),
        ),
        "P3.5-05": (
            "ObjectAssetManifestV2",
            "qdgrasp.objects.manifest_v2",
            ("ObjectAssetManifestV2", "write_object_asset_manifest_v2", "load_object_asset_manifest_v2"),
        ),
        "P3.5-06": ("scene resolver", "qdgrasp.scenes.resolver", ("resolve_scene", "SceneSource", "SceneLoadError")),
        "P3.5-07": (
            "virtual drop scene",
            "qdgrasp.scenes.virtual_drop",
            ("VirtualDropSceneSpec", "build_virtual_drop_scene", "SpawnRegion", "SettleThresholds"),
        ),
        "P3.5-08": (
            "settle certifier",
            "qdgrasp.scenes.settle",
            ("certify_settle", "SceneSnapshot", "SettleOutcome", "replay_snapshot"),
        ),
        "P3.5-09": (
            "RL contracts",
            "qdgrasp.rl.contracts",
            ("ObservationSchema", "RlActionSpec", "RewardBreakdown", "StepResult", "TerminalReason"),
        ),
        "P3.5-10": (
            "environments",
            "qdgrasp.rl.envs",
            ("ObjectSettleEnv", "DexAcquireEnv", "DexAcquireSceneEnv", "ACTIVE_ROBOT_PROFILES"),
        ),
        "P3.5-11": ("task fixtures", "qdgrasp.rl.tasks", ("run_scripted_episode", "random_policy_probe")),
        "P3.5-12": (
            "randomization",
            "qdgrasp.rl.randomization",
            ("SeedStreams", "DomainRandomization", "scene_signature", "EvaluationSplit"),
        ),
    }
    for package, (title, module_name, names) in surfaces.items():
        try:
            module = __import__(module_name, fromlist=list(names))
        except Exception as error:  # noqa: BLE001 - an import failure is the finding
            results.append(PackageResult(package, title, STATUS_OPEN, f"{module_name} failed to import: {error}"))
            continue
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            results.append(PackageResult(package, title, STATUS_OPEN, f"{module_name} is missing {missing}"))
        else:
            results.append(PackageResult(package, title, STATUS_DELIVERED, f"{module_name} exposes {len(names)} names"))
    return results


def _micro_checks() -> list[PackageResult]:
    """Run the live path once per active hand."""

    import tempfile

    import mujoco
    import numpy as np

    from qdgrasp.objects.generate import generate_box
    from qdgrasp.objects.manifest import create_object_asset, save_object_asset
    from qdgrasp.rl.envs import ACTIVE_ROBOT_PROFILES, DexAcquireConfig, DexAcquireEnv
    from qdgrasp.rl.tasks import ScriptedAcquireSpec, random_policy_probe, run_scripted_episode
    from qdgrasp.scenes.resolver import SceneSource, resolve_scene
    from qdgrasp.scenes.settle import SettleOutcome, certify_settle
    from qdgrasp.scenes.virtual_drop import DropObjectRequest, SpawnRegion, VirtualDropSceneSpec

    results: list[PackageResult] = []
    directory = Path(tempfile.mkdtemp(prefix="qdgrasp-p35-micro-"))
    rng = np.random.default_rng(0)
    mesh, geoms, params, mass, inertia = generate_box(rng, size_range=(0.028, 0.034), density=650.0)
    mesh_bytes, manifest = create_object_asset("target", "primitive", "box", mesh, geoms, params, mass, inertia)
    asset_ref = str(save_object_asset(mesh_bytes, manifest, directory))
    objects = (DropObjectRequest(object_id="target", asset_ref=asset_ref),)
    scene_config = VirtualDropSceneSpec(
        spawn_region=SpawnRegion(half_extents=(0.05, 0.05, 0.0)), drop_height_range_m=(0.02, 0.04)
    )

    resolved = resolve_scene(objects=objects, virtual_scene_config=scene_config, seed=1)
    results.append(
        PackageResult(
            "P3.5-06",
            "scene resolution",
            STATUS_DELIVERED if resolved.source is SceneSource.GENERATED else STATUS_OPEN,
            f"resolved a {resolved.source.value} scene with {len(resolved.spec.objects)} object(s)",
        )
    )

    snapshot = certify_settle(
        resolved.spec,
        resolved.model,
        mujoco.MjData(resolved.model),
        scene_config.settle_thresholds,
        spawn_region=scene_config.spawn_region,
    )
    results.append(
        PackageResult(
            "P3.5-08",
            "settle certification",
            STATUS_DELIVERED if snapshot.outcome is SettleOutcome.SETTLED else STATUS_OPEN,
            f"outcome={snapshot.outcome.value} after {snapshot.steps} steps",
        )
    )

    spec = ScriptedAcquireSpec()
    for profile in ACTIVE_ROBOT_PROFILES:
        config = DexAcquireConfig(
            robot_profile=profile,
            objects=objects,
            target_object_id="target",
            virtual_scene=scene_config,
            max_steps=spec.total_steps,
            settle_steps=400,
        )
        scripted = run_scripted_episode(DexAcquireEnv(config), seed=21, spec=spec)
        probe = random_policy_probe(DexAcquireEnv(config), seed=21, steps=40)
        healthy = (
            scripted["observations_finite"]
            and probe["observations_finite"]
            and probe["successes"] == 0
            and scripted["terminal_reason"] in ("horizon", "success")
        )
        results.append(
            PackageResult(
                "P3.5-10/11",
                f"reset/step for {profile}",
                STATUS_DELIVERED if healthy else STATUS_OPEN,
                (
                    f"scripted={scripted['terminal_reason']} lift={scripted['max_lift_m']:.4f} m; "
                    f"random finite={probe['observations_finite']} successes={probe['successes']}"
                ),
            )
        )
    return results


def _outstanding() -> list[PackageResult]:
    """Packages that cannot be closed from this machine, stated as such."""

    return [
        PackageResult(
            "P3.5-13/14/15",
            "GPU backend compatibility spike and decision record",
            STATUS_BLOCKED,
            "needs a real CUDA run on Kaggle/Colab; §7 forbids choosing a backend without measured 2-hand parity",
        ),
        PackageResult(
            "P3.5-16",
            "Kaggle/Colab harness with checkpoint and resume",
            STATUS_OPEN,
            "not built",
        ),
        PackageResult(
            "P3.5-17",
            "QDGrasp-RL-Env-Tiny artifact",
            STATUS_OPEN,
            "needs a positive scripted fixture; the current fixture runs to the horizon without acquiring",
        ),
        PackageResult(
            "P3.5-18",
            "independent review and roadmap handoff",
            STATUS_BLOCKED,
            "the author of an artifact may not sign its verdict",
        ),
    ]


def run_checks(profile: str) -> list[PackageResult]:
    results = _import_checks()
    if profile == "micro":
        results.extend(_micro_checks())
    results.extend(_outstanding())
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="mujoco-cpu", choices=["mujoco-cpu"])
    parser.add_argument("--profile", default="micro", choices=["micro", "contract"])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run_checks(args.profile)
    for result in results:
        marker = {STATUS_DELIVERED: "PASS", STATUS_OPEN: "OPEN", STATUS_BLOCKED: "BLOCKED"}[result.status]
        print(f"{marker:8s} [{result.package}] {result.title}: {result.detail}")

    delivered = sum(1 for item in results if item.passed)
    print(f"\n{delivered}/{len(results)} packages delivered (backend={args.backend}, profile={args.profile})")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema": "qdgrasp/phase3-5-gate/v0",
            "backend": args.backend,
            "profile": args.profile,
            "delivered": delivered,
            "total": len(results),
            "packages": [dataclasses.asdict(item) for item in results],
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outstanding = [item for item in results if not item.passed]
    if outstanding:
        print("Phase 3.5 status: in_progress -- " + ", ".join(sorted({item.package for item in outstanding})))
        return 1
    print("Phase 3.5 status: complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
