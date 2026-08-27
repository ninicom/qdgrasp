#!/usr/bin/env python3
"""Render one frame per acquisition stage, for visual QA (S11; C06.6, C06.7).

A stage label is a claim about what the hand was doing. This renders the frame
that each stage's first sample was actually taken at, so a reviewer can look at
the moment the trajectory says the target left its support and see whether it
did.

Two rules, both from the plan. The renderer never decides anything: it produces
images and a metadata record, and the physics verdict stays with the contact
observer and the certifier. And every image points back at the trajectory it
came from by hash, so a picture cannot be filed against a different run.

    MUJOCO_GL=egl python scripts/render_phase3_4_3_stages.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdgrasp.config.active_scope import resolve_workload_hands
from qdgrasp.dataset.dynamic_contracts import (
    REQUIRED_TERMINAL_STAGES,
    TrajectoryStage,
    canonical_hash,
)

RENDER_SCHEMA = "qdgrasp/evidence/phase3.4.3-stage-render/v1"

#: Pinned views. Two, so a stage cannot look right from one angle and wrong from
#: the other without somebody noticing.
VIEWS: tuple[tuple[str, dict[str, float]], ...] = (
    ("front", {"azimuth": 90.0, "elevation": -15.0, "distance": 0.30}),
    ("side", {"azimuth": 0.0, "elevation": -10.0, "distance": 0.30}),
)

#: Stages a positive has to show. Rendered whether or not the run produced one:
#: a missing stage image is evidence too.
RENDERED_STAGES: tuple[TrajectoryStage, ...] = (
    TrajectoryStage.APPROACH,
    *REQUIRED_TERMINAL_STAGES,
    TrajectoryStage.RETAIN,
)

HEIGHT, WIDTH = 360, 480


def _write_png(path: Path, image: np.ndarray) -> str:
    """Write a frame as PNG and return its sha256.

    Uses Pillow when it is available and falls back to a raw ``.npy`` otherwise,
    because an absent optional dependency should degrade the format rather than
    silently drop the evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    except ImportError:
        path = path.with_suffix(".npy")
        np.save(path, np.asarray(image, dtype=np.uint8))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera(settings: dict[str, float], lookat: np.ndarray) -> mujoco.MjvCamera:
    """A pinned free camera aimed at the target.

    Aimed at the target rather than at the scene centroid: the QA question is
    what happened to the object, and a default camera framed on the hand answers
    a different one.
    """
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = settings["azimuth"]
    camera.elevation = settings["elevation"]
    camera.distance = settings["distance"]
    camera.lookat[:] = lookat
    return camera


def render_hand(hand: str, out_root: Path, generate_one) -> dict[str, Any]:
    """Run one rollout, keeping the frames each sample was taken at."""
    frames: dict[int, dict[str, np.ndarray]] = {}
    renderer: dict[str, Any] = {}

    def capture(index: int, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if "r" not in renderer:
            renderer["r"] = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
            renderer["target"] = mujoco.mj_name2id(
                model, int(mujoco.mjtObj.mjOBJ_BODY), "target_object"
            )
        view = renderer["r"]
        target_body = renderer["target"]
        lookat = (
            np.array(data.xpos[target_body], dtype=np.float64)
            if target_body >= 0
            else np.zeros(3)
        )
        captured: dict[str, np.ndarray] = {}
        for name, settings in VIEWS:
            view.update_scene(data, camera=_camera(settings, lookat))
            captured[name] = view.render().copy()
        frames[index] = captured

    trajectory, outcome = generate_one(
        hand,
        environment="table",
        clutter="sparse",
        mode="static_seeded",
        frame_observer=capture,
    )
    if "r" in renderer:
        renderer["r"].close()

    trajectory_hash = canonical_hash(
        {
            "hand": hand,
            "steps": trajectory.num_steps,
            "time": [float(v) for v in trajectory.time],
            "stages": [stage.value for stage in trajectory.stage],
        }
    )

    images: list[dict[str, Any]] = []
    for stage in RENDERED_STAGES:
        first = next(
            (index for index, value in enumerate(trajectory.stage) if value is stage), None
        )
        if first is None or first not in frames:
            images.append(
                {
                    "stage": stage.value,
                    "present": False,
                    "detail": "the trajectory never reached this stage",
                }
            )
            continue
        rendered: dict[str, dict[str, str]] = {}
        for name, _ in VIEWS:
            path = out_root / hand / f"{stage.value}.{name}.png"
            rendered[name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _write_png(path, frames[first][name]),
            }
        images.append(
            {
                "stage": stage.value,
                "present": True,
                "sample_index": first,
                "time_s": float(trajectory.time[first]),
                "views": rendered,
            }
        )

    return {
        "hand": hand,
        "trajectory_hash": trajectory_hash,
        "passed": bool(outcome.passed),
        "failure_reason": outcome.failure_reason,
        "samples": trajectory.num_steps,
        "views": [name for name, _ in VIEWS],
        "stages_reached": sorted({stage.value for stage in trajectory.stage}),
        "images": images,
        "missing_stages": [entry["stage"] for entry in images if not entry["present"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "evidence" / "phase3_4_3" / "s11" / "renders"
    )
    args = parser.parse_args()

    import runpy

    generator = runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_contactrich_active_tiny.py"),
        run_name="render_generator",
    )
    generate_one = generator["generate_one"]

    scope = resolve_workload_hands()
    hands = [render_hand(hand, args.out, generate_one) for hand in scope.hands]

    report = {
        "schema": RENDER_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "renderer": os.environ.get("MUJOCO_GL"),
        "resolution": [HEIGHT, WIDTH],
        "scope": scope.as_disclosure(),
        "required_stages": [stage.value for stage in RENDERED_STAGES],
        "hands": hands,
        "note": (
            "These images are for visual QA of target identity, support release "
            "and orientation. The renderer never decides an outcome: the physics "
            "verdict stays with the contact observer and the certifier."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    target = args.out.parent / "stage-renders.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for entry in hands:
        print(
            f"{entry['hand']:16s} stages rendered: "
            f"{len([i for i in entry['images'] if i['present']])}/{len(RENDERED_STAGES)} "
            f"missing={entry['missing_stages']}"
        )
    print(f"wrote {target.relative_to(REPO_ROOT)}")
    missing = [hand for entry in hands for hand in entry["missing_stages"]]
    if missing:
        print(
            f"stages with no image: {sorted(set(missing))}. A missing stage image is "
            "evidence about the trajectory, not a rendering failure.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
