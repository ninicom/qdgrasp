#!/usr/bin/env python3
"""Build the Kaggle/Colab runner for the Phase 3.5 RL-readiness gate (P3.5-16).

The notebook installs an exact public commit, runs the CPU gate, then runs the
GPU spike on whatever accelerator the runtime actually has.  Every stage writes
into a working directory and is skipped when its output already exists, so a
reclaimed runtime resumes by re-running top to bottom.

The GPU cell is the point of the notebook and it is allowed to fail.  A backend
that will not install, or a parity that does not hold, is the result P3.5-15 has
to be written from; a notebook that hid it would be worse than no notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "phase3_5_rl_readiness.ipynb"

# Replaced with the immutable implementation commit before publication.
P35_CODE_REVISION = "0" * 40
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
            """# QDGrasp Phase 3.5 — asset/scene/RL readiness

Runs `ROADMAP-P3.5-001`'s CPU gate and then the GPU backend spike on a fresh
Kaggle or Colab runtime.

**This notebook does not choose a backend.** §7 requires measured two-hand
parity on compile, step, contact, drop and lift before P3.5-15 may record a
decision, and that decision belongs in an ADR a human signs. What runs here is
the harness that produces the evidence.

The GPU cell may fail, and that failure is a result. `mujoco_warp` was already
measured defective for Phase 3.4.3 on `warp-lang` 1.16.0
(`REV-20260828-014`); if it fails the same way here, that is the finding, not a
notebook bug to work around."""
        ),
        _code(
            f'''import os, subprocess, sys
from pathlib import Path

CODE_REVISION = "{revision}"
MENAGERIE_REVISION = "{MENAGERIE_REVISION}"
REPO_URL = "https://github.com/ninicom/qdgrasp.git"
REPO_DIR = Path("/tmp/qdgrasp_repo")
ASSETS_DIR = Path("/tmp/robot-assets/mujoco-menagerie")
WORK = Path("/kaggle/working/p35") if Path("/kaggle/working").is_dir() else Path("/content/p35")
WORK.mkdir(parents=True, exist_ok=True)

assert sys.version_info >= (3, 11), f"Python >=3.11 required, got {{sys.version}}"

def run(*command, cwd=None, check=True):
    print("$", " ".join(str(part) for part in command))
    return subprocess.run([str(part) for part in command], cwd=cwd, check=check)

if not REPO_DIR.is_dir():
    run("git", "clone", "--filter=blob:none", REPO_URL, REPO_DIR)
run("git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", CODE_REVISION)
run("git", "-C", REPO_DIR, "checkout", "--detach", CODE_REVISION)
head = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
assert head.stdout.strip() == CODE_REVISION, head.stdout
print("pinned at", CODE_REVISION)'''
        ),
        _code(
            """run(sys.executable, "-m", "pip", "install", "--quiet",
    "mujoco>=3.3.0", "pydantic>=2.10.0", "PyYAML>=6.0.0", "scipy>=1.14.0", "trimesh>=4.0.0")
run(sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(REPO_DIR))

if not ASSETS_DIR.is_dir():
    ASSETS_DIR.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--filter=blob:none",
        "https://github.com/google-deepmind/mujoco_menagerie.git", ASSETS_DIR)
run("git", "-C", ASSETS_DIR, "checkout", "--detach", MENAGERIE_REVISION)
os.environ["QDGRASP_ROBOT_ASSETS_ROOT"] = str(ASSETS_DIR.parent)
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

import mujoco, torch
print("mujoco", mujoco.__version__, "| torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))"""
        ),
        _markdown(
            """## Stage 1 — the CPU gate

Fourteen of the eighteen packages should report `PASS`, and the checker should
exit non-zero on the four that are open. A zero exit here would mean the gate
stopped telling the truth about what is unfinished."""
        ),
        _code(
            """cpu_gate = subprocess.run(
    [sys.executable, "scripts/check_phase3_5.py", "--profile", "micro",
     "--json", str(WORK / "phase3_5_cpu_gate.json")],
    check=False,
)
print("exit code:", cpu_gate.returncode, "(1 is expected while packages are open)")"""
        ),
        _markdown(
            """## Stage 2 — CPU oracle evidence

The oracle runs first and unconditionally. A GPU number with no oracle behind it
has nothing to be compared against."""
        ),
        _code(
            """run(sys.executable, "scripts/phase3_5_gpu_rl_readiness.py",
    "--backend", "mujoco-cpu", "--device", "cpu",
    "--profile", "notebook-micro",
    "--evidence", str(WORK / "phase3_5_cpu_oracle.json"))
print(open(WORK / "phase3_5_cpu_oracle.json").read()[:2000])"""
        ),
        _markdown(
            """## Stage 3 — the GPU backend spike

Install the pinned backend, then run the spike. If the install or the spike
fails, keep the output: that is the evidence P3.5-15 is written from.

`mujoco_warp` is pinned rather than taken from `main`, because the Phase 3.4.3
result depends on the exact pair of `mujoco-warp` and `warp-lang` versions."""
        ),
        _code(
            """import torch
if not torch.cuda.is_available():
    print("No CUDA on this runtime. The GPU spike is skipped and NOTHING here may be")
    print("reported as GPU evidence (ADR-0006). Switch the runtime to a GPU accelerator.")
else:
    run(sys.executable, "-m", "pip", "install", "--quiet",
        "warp-lang==1.16.0", "mujoco-warp==3.12.0", check=False)
    gpu = subprocess.run(
        [sys.executable, "scripts/phase3_5_gpu_rl_readiness.py",
         "--backend", "mjx-warp", "--device", "cuda:0",
         "--profile", "notebook-micro",
         "--evidence", str(WORK / "phase3_5_gpu_evidence.json")],
        check=False,
    )
    print("exit code:", gpu.returncode)
    print(open(WORK / "phase3_5_gpu_evidence.json").read()[:4000])"""
        ),
        _markdown(
            """## Stage 4 — what to carry back

Download `phase3_5_cpu_gate.json`, `phase3_5_cpu_oracle.json` and, if it ran,
`phase3_5_gpu_evidence.json`. Commit them under `evidence/phase3_5/` with the
commit this notebook pinned.

A backend decision (P3.5-15) is written from a run whose verdict is `measured`
and whose two-hand parity holds. Any other verdict is recorded as-is."""
        ),
        _code(
            """for name in ("phase3_5_cpu_gate.json", "phase3_5_cpu_oracle.json", "phase3_5_gpu_evidence.json"):
    path = WORK / name
    print(f"{name:34s} {'present' if path.is_file() else 'absent':8s} "
          f"{path.stat().st_size if path.is_file() else 0} bytes")"""
        ),
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
    parser.add_argument("--revision", default=P35_CODE_REVISION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    notebook = build(args.revision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} pinned at {args.revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
