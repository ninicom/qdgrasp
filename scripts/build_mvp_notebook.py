#!/usr/bin/env python3
"""Build the Kaggle/Colab runner notebook for the temporary Grasp Policy MVP.

``ROADMAP-MVP-001`` MVP-06 wants a fresh cloud runtime to install the exact
public commit, run the pipeline, resume after a runtime is reclaimed, and
evaluate the checkpoint it produced.

One thing the notebook says out loud, because the plan insists on it: the MVP's
physics is MuJoCo **CPU**.  A GPU runtime here trains a small MLP faster and
nothing else.  No cell in this notebook is GPU-physics evidence, and none of it
touches the CUDA gates that remain open in P3.4.3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "mvp_grasp_policy.ipynb"

# Replaced with the immutable implementation commit before notebook publication.
MVP_CODE_REVISION = "c8f2749770fb0ade8552d3b9f576143a63e37a1c"
MENAGERIE_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"


def _code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build(revision: str) -> dict[str, object]:
    if len(revision) != 40:
        raise RuntimeError("the notebook must pin an immutable 40-character commit")

    cells = [
        _markdown(
            """# QDGrasp Grasp Policy MVP — LEAP vertical slice

Runs `ROADMAP-MVP-001` end to end on a fresh Kaggle or Colab runtime: expert
demonstrations, behaviour cloning, residual PPO, and the locked evaluation.

**This notebook is not GPU-physics evidence.** The MVP's simulator is MuJoCo on
CPU. A GPU runtime only makes the small MLP train faster; every episode, every
contact and every verdict in here is CPU physics. It changes nothing about the
CUDA gates that remain open in P3.4.3, and the artifact it produces is
`experimental_non_release`.

Each stage writes into `WORK` and is skipped when its output already exists, so
a reclaimed runtime resumes by re-running the notebook top to bottom."""
        ),
        _code(
            f'''import os, subprocess, sys
from pathlib import Path

CODE_REVISION = "{revision}"
MENAGERIE_REVISION = "{MENAGERIE_REVISION}"
REPO_URL = "https://github.com/ninicom/qdgrasp.git"
REPO_DIR = Path("/tmp/qdgrasp_repo")
ASSETS_DIR = Path("/tmp/robot-assets/mujoco-menagerie")
# Kaggle and Colab both keep /kaggle/working and /content across a restart of
# the same session; WORK is where every resumable artifact lands.
WORK = Path("/kaggle/working/mvp") if Path("/kaggle/working").is_dir() else Path("/content/mvp")
WORK.mkdir(parents=True, exist_ok=True)

assert sys.version_info >= (3, 11), f"Python >=3.11 required, got {{sys.version}}"

def run(*command, cwd=None):
    print("$", " ".join(str(part) for part in command))
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)

if not REPO_DIR.is_dir():
    run("git", "clone", "--filter=blob:none", REPO_URL, REPO_DIR)
run("git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", CODE_REVISION)
run("git", "-C", REPO_DIR, "checkout", "--detach", CODE_REVISION)
head = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
assert head.stdout.strip() == CODE_REVISION, head.stdout
print("pinned at", CODE_REVISION)'''
        ),
        _code(
            """run(sys.executable, "-m", "pip", "install", "--quiet", "mujoco>=3.3.0", "pydantic>=2.10.0", "PyYAML>=6.0.0", "scipy>=1.14.0")
run(sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(REPO_DIR))

if not ASSETS_DIR.is_dir():
    ASSETS_DIR.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--filter=blob:none", "https://github.com/google-deepmind/mujoco_menagerie.git", ASSETS_DIR)
run("git", "-C", ASSETS_DIR, "checkout", "--detach", MENAGERIE_REVISION)
os.environ["QDGRASP_ROBOT_ASSETS_ROOT"] = str(ASSETS_DIR.parent)
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

import mujoco, torch
print("mujoco", mujoco.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("NOTE: the simulator below is MuJoCo CPU regardless of the line above.")"""
        ),
        _markdown(
            """## Stage 0 — the locked scope

Nothing downstream is meaningful if these hashes differ from the ones in the
repository, so they are printed before anything is measured."""
        ),
        _code(
            """from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable
from qdgrasp.mvp.env import environment_fingerprint

scope = load_mvp_scope()
prior = PinchPriorTable.load(DEFAULT_PRIOR_PATH)
for key, value in environment_fingerprint(scope, prior).items():
    print(f"{key:26s} {value}")"""
        ),
        _markdown("""## Stage 1 — expert demonstrations (MVP-02)"""),
        _code(
            """DEMOS = WORK / "demonstrations"
if (DEMOS / "index.json").is_file():
    print("demonstrations already present; skipping")
else:
    run(sys.executable, "scripts/generate_mvp_demos.py", "--out", DEMOS,
        "--train-episodes", 400, "--dev-episodes", 120, "--workers", os.cpu_count() or 2)
print(open(DEMOS / "index.json").read())"""
        ),
        _markdown("""## Stage 2 — behaviour cloning and residual PPO (MVP-03/04)"""),
        _code(
            """POLICY = WORK / "policy"
if (POLICY / "training-report.json").is_file():
    print("training report already present; skipping")
else:
    run(sys.executable, "scripts/train_mvp_policy.py", "--demos", DEMOS, "--out", POLICY,
        "--workers", os.cpu_count() or 2)
import json
report = json.load(open(POLICY / "training-report.json"))
print("candidate:", report["candidate"])
print("bc dev:", report["bc"]["dev"])
if "ppo" in report:
    print("ppo dev:", report["ppo"]["dev"], "promoted:", report["ppo"]["promoted"])"""
        ),
        _markdown(
            """## Stage 3 — locked evaluation (MVP-05)

The seeds come from the immutable evaluation manifest. Run this once per
candidate: §8 allows one locked-eval run per tune round, and re-running it does
not retract the first result."""
        ),
        _code(
            """EVAL = WORK / "evaluation"
run(sys.executable, "scripts/evaluate_mvp.py", "--checkpoint", report["candidate"],
    "--out", EVAL, "--workers", os.cpu_count() or 2)"""
        ),
        _markdown("""## Stage 4 — closure gate (MVP-07)"""),
        _code("""subprocess.run([sys.executable, "scripts/check_mvp.py", "--runs", str(WORK)], check=False)"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=MVP_CODE_REVISION, help="immutable public commit to pin")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    notebook = build(args.revision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} pinned at {args.revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
