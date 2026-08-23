"""Analyze and generate comprehensive statistics for a QDGrasp dataset release."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from qdgrasp.dataset.manifest import load_dataset_manifest
from qdgrasp.dataset.shards import read_shard_file


def analyze_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_dir).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found at {manifest_path}")

    manifest = load_dataset_manifest(manifest_path)
    shards_dir = dataset_path / "shards"

    total_samples = 0
    positive_samples = 0
    negative_samples = 0

    robot_counts = Counter()
    robot_positive = Counter()
    robot_qualities = defaultdict(list)
    robot_palm_distances = defaultdict(list)
    robot_contact_counts = defaultdict(list)

    object_counts = Counter()
    object_positive = Counter()
    object_families = Counter()

    split_counts = Counter()
    split_positive = Counter()

    stage_pass_counts = Counter()
    qualities = []
    palm_norms = []
    contact_spreads = []

    for shard_meta in manifest.shards:
        candidate_p = dataset_path / shard_meta.filename
        if not candidate_p.exists():
            candidate_p = dataset_path / "shards" / shard_meta.filename
        if not candidate_p.exists():
            continue
        samples = read_shard_file(candidate_p)
        split = shard_meta.split

        for s in samples:
            total_samples += 1
            is_success = bool(s.get("success", torch.tensor(0.0)).item() > 0.5)
            quality = float(s.get("quality", torch.tensor(0.0)).item())
            robot = str(s.get("robot_name", "unknown"))
            obj_id = str(s.get("object_id", "unknown"))

            # Determine object family
            if obj_id.startswith("comp_"):
                family = "compound_convex"
            elif obj_id.startswith("sq_"):
                family = "superquadric"
            elif obj_id.startswith("prim_"):
                family = "primitive"
            else:
                family = "other"

            object_families[family] += 1
            object_counts[obj_id] += 1
            robot_counts[robot] += 1
            split_counts[split] += 1

            if is_success:
                positive_samples += 1
                robot_positive[robot] += 1
                object_positive[obj_id] += 1
                split_positive[split] += 1
            else:
                negative_samples += 1

            qualities.append(quality)
            robot_qualities[robot].append(quality)

            # Kinematic & physical features
            palm_pos = s.get("palm_pos")
            if palm_pos is not None:
                p_norm = float(torch.norm(palm_pos.float()).item())
                palm_norms.append(p_norm)
                robot_palm_distances[robot].append(p_norm)

            tips = s.get("fingertip_positions")
            if tips is not None:
                tips_t = tips.float()
                # Non-zero contact points
                non_zeros = (tips_t.abs().sum(dim=-1) > 1e-4).sum().item()
                robot_contact_counts[robot].append(non_zeros)
                if len(tips_t) > 1:
                    spread = float(torch.std(tips_t, dim=0).mean().item())
                    contact_spreads.append(spread)

            # Stage progression
            if bool(s.get("proposal_valid", False)):
                stage_pass_counts["S1_proposal"] += 1
            if bool(s.get("ik_valid", False)):
                stage_pass_counts["S2_ik"] += 1
            if bool(s.get("collision_valid", False)):
                stage_pass_counts["S3_collision"] += 1
            if bool(s.get("static_force_valid", False)):
                stage_pass_counts["S4_static_force"] += 1
            if bool(s.get("dynamic_valid", False)):
                stage_pass_counts["S5_dynamic_rollout"] += 1

    stats = {
        "dataset_id": manifest.dataset_id,
        "recipe_id": manifest.recipe_id,
        "generator_version": manifest.generator_version,
        "seed": manifest.seed,
        "release_blocked": manifest.release_blocked,
        "total_samples": total_samples,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "success_rate": positive_samples / max(1, total_samples),
        "splits": {
            k: {
                "total": split_counts[k],
                "positive": split_positive[k],
                "success_rate": split_positive[k] / max(1, split_counts[k]),
            }
            for k in split_counts
        },
        "robots": {
            r: {
                "total": robot_counts[r],
                "positive": robot_positive[r],
                "success_rate": robot_positive[r] / max(1, robot_counts[r]),
                "avg_quality": float(np.mean(robot_qualities[r])) if robot_qualities[r] else 0.0,
                "avg_palm_dist_m": float(np.mean(robot_palm_distances[r])) if robot_palm_distances[r] else 0.0,
                "avg_active_fingers": float(np.mean(robot_contact_counts[r])) if robot_contact_counts[r] else 0.0,
            }
            for r in robot_counts
        },
        "objects": {
            obj: {
                "total": object_counts[obj],
                "positive": object_positive[obj],
                "success_rate": object_positive[obj] / max(1, object_counts[obj]),
                "family": (
                    "compound_convex"
                    if obj.startswith("comp_")
                    else "superquadric"
                    if obj.startswith("sq_")
                    else "primitive"
                ),
            }
            for obj in sorted(object_counts)
        },
        "object_families": dict(object_families),
        "total_objects": len(object_counts),
        "quality_metrics": {
            "mean": float(np.mean(qualities)) if qualities else 0.0,
            "std": float(np.std(qualities)) if qualities else 0.0,
            "min": float(np.min(qualities)) if qualities else 0.0,
            "max": float(np.max(qualities)) if qualities else 0.0,
        },
        "stage_progression": {
            k: {
                "passed": v,
                "rate": v / max(1, total_samples),
            }
            for k, v in stage_pass_counts.items()
        },
    }

    return stats


def print_rich_report(stats: dict[str, Any]) -> None:
    console = Console()

    console.print()
    console.print(
        Panel(
            f"[bold cyan]QDGrasp Dataset Statistical Analysis[/bold cyan]\n"
            f"[bold]Dataset ID:[/bold] {stats['dataset_id']} | [bold]Recipe:[/bold] {stats['recipe_id']} | [bold]Seed:[/bold] {stats['seed']}\n"
            f"[bold]Total Samples:[/bold] [green]{stats['total_samples']}[/green] | [bold]Positive Grasps:[/bold] [green]{stats['positive_samples']} ({stats['success_rate']*100:.1f}%)[/green] | [bold]Release Blocked:[/bold] [{'red' if stats['release_blocked'] else 'green'}]{stats['release_blocked']}[/{'red' if stats['release_blocked'] else 'green'}]",
            title="[bold green]Overview[/bold green]",
            border_style="cyan",
        )
    )

    # Robot Breakdown Table
    robot_table = Table(title="Embodiment (Robot) Breakdown", header_style="bold magenta")
    robot_table.add_column("Robot", style="cyan")
    robot_table.add_column("Samples", justify="right")
    robot_table.add_column("Positive", justify="right")
    robot_table.add_column("Success Rate", justify="right")
    robot_table.add_column("Avg Active Fingers", justify="right")
    robot_table.add_column("Avg Palm Dist (m)", justify="right")
    robot_table.add_column("Avg Quality", justify="right")

    for robot, r_data in stats["robots"].items():
        robot_table.add_row(
            robot,
            str(r_data["total"]),
            str(r_data["positive"]),
            f"{r_data['success_rate']*100:.1f}%",
            f"{r_data['avg_active_fingers']:.1f}",
            f"{r_data['avg_palm_dist_m']:.4f}",
            f"{r_data['avg_quality']:.4f}",
        )
    console.print(robot_table)

    # Split Breakdown Table
    split_table = Table(title="Data Split Distribution", header_style="bold blue")
    split_table.add_column("Split", style="cyan")
    split_table.add_column("Samples", justify="right")
    split_table.add_column("Positive", justify="right")
    split_table.add_column("Percentage", justify="right")

    for split, s_data in stats["splits"].items():
        split_table.add_row(
            split.upper(),
            str(s_data["total"]),
            str(s_data["positive"]),
            f"{s_data['total']/max(1, stats['total_samples'])*100:.1f}%",
        )
    console.print(split_table)

    # Object Breakdown Table
    obj_table = Table(title="Object-Level Breakdown (12 Procedural Shapes)", header_style="bold green")
    obj_table.add_column("Object ID", style="cyan")
    obj_table.add_column("Family", style="magenta")
    obj_table.add_column("Samples", justify="right")
    obj_table.add_column("Positive", justify="right")
    obj_table.add_column("Success Rate", justify="right")

    for obj_id, o_data in stats["objects"].items():
        obj_table.add_row(
            obj_id,
            o_data["family"].replace("_", " ").title(),
            str(o_data["total"]),
            str(o_data["positive"]),
            f"{o_data['success_rate']*100:.1f}%",
        )
    console.print(obj_table)

    # Object Family Table
    fam_table = Table(title="Object Shape Families", header_style="bold yellow")
    fam_table.add_column("Shape Family", style="cyan")
    fam_table.add_column("Objects", justify="right")
    fam_table.add_column("Grasp Samples", justify="right")
    fam_table.add_column("Share (%)", justify="right")

    for fam, cnt in stats["object_families"].items():
        objs_in_fam = sum(1 for o, d in stats["objects"].items() if d["family"] == fam)
        fam_table.add_row(
            fam.replace("_", " ").title(),
            str(objs_in_fam),
            str(cnt),
            f"{cnt/max(1, stats['total_samples'])*100:.1f}%",
        )
    console.print(fam_table)

    # Stage Progression Table
    stage_table = Table(title="Pipeline Stage Progression (Verification Gates)", header_style="bold green")
    stage_table.add_column("Pipeline Stage", style="cyan")
    stage_table.add_column("Passed Samples", justify="right")
    stage_table.add_column("Pass Rate", justify="right")

    for stage, data in stats["stage_progression"].items():
        stage_table.add_row(
            stage,
            str(data["passed"]),
            f"{data['rate']*100:.1f}%",
        )
    console.print(stage_table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze QDGrasp dataset statistics.")
    parser.add_argument("--dataset-dir", default="datasets/dgn-open-tiny", help="Path to dataset directory.")
    parser.add_argument("--json-out", default=None, help="Optional path to write stats JSON.")
    args = parser.parse_args()

    stats = analyze_dataset(args.dataset_dir)
    print_rich_report(stats)

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Saved JSON statistics to {out_p}")


if __name__ == "__main__":
    main()
