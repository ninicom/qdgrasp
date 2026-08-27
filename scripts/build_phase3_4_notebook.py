"""Build the Kaggle notebook for the Phase 3.4 CUDA backend decision.

Stage 1 of the harness (P3.4-15). Its job is to answer the question P3.4-04
could not answer on a CPU host: does MuJoCo Warp actually carry tendon
transmission, weld equality and per-contact force for the release hands.

The pinned revision must already exist on a remote, because the notebook
installs the library from GitHub. Pinning a local-only commit produces a
notebook that cannot run, so the builder refuses it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "kaggle-phase3-4"
NOTEBOOK_PATH = NOTEBOOK_DIR / "qdgrasp-phase-3-4-cuda-gate.ipynb"
METADATA_PATH = NOTEBOOK_DIR / "kernel-metadata.json"

MENAGERIE_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"
REPO_URL = "https://github.com/ninicom/qdgrasp.git"


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


def assert_public_revision(revision: str) -> None:
    """A pin the notebook cannot fetch is worse than no notebook."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise SystemExit(f"revision must be a full 40-character commit sha, got {revision!r}")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"revision {revision} is not a commit in this repository") from exc

    remotes = subprocess.check_output(
        ["git", "branch", "-r", "--contains", revision], cwd=ROOT, text=True
    ).strip()
    if not remotes:
        raise SystemExit(
            f"revision {revision} exists only locally. The notebook installs from "
            f"{REPO_URL}, so push the branch before pinning it."
        )


def build(revision: str, kaggle_slug: str) -> None:
    assert_public_revision(revision)

    cells = [
        _markdown(
            """# QDGrasp Phase 3.4 — CUDA backend decision

This notebook resolves one question: **can MuJoCo Warp run the QDGrasp release hands?**

`P3.4-04` measured what the models require, on CPU:

| requirement | why it blocks the phase |
| --- | --- |
| `mjTRN_TENDON` | `shadow_hand` drives 4 of 20 actuators through tendons |
| `equality:mjEQ_WELD` | the `mocap-weld-v3` protocol drives the wrist through a welded mocap body |
| `mocap_body` | same protocol |
| per-contact force + frame | the safety budget is defined on resolved contact force |

If Warp cannot carry all four, Phase 3.4 **stays blocked** and a backend decision
record is written. Substituting a mock CUDA backend, or dropping Shadow from the
gate, is not an accepted resolution.

This notebook does **not** benchmark search throughput: the CUDA backend
(`P3.4-05`) does not exist yet, and reporting a CPU number as CUDA evidence is
exactly what the gate forbids.
"""
        ),
        _code(
            f'''import os
import subprocess
import sys
from pathlib import Path

CODE_REVISION = "{revision}"
MENAGERIE_REVISION = "{MENAGERIE_REVISION}"
REPO_URL = "{REPO_URL}"
REPO_DIR = Path("/tmp/qdgrasp_repo")
ASSETS_DIR = Path("/tmp/robot-assets/mujoco-menagerie")

assert sys.version_info >= (3, 11), f"Python >=3.11 required, got {{sys.version}}"
os.environ.update(
    QDGRASP_ROBOT_ASSETS_ROOT="/tmp/robot-assets",
    MUJOCO_GL="egl",
    PYTHONHASHSEED="0",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
)

subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
    "lightning==2.6.5", "mujoco==3.12.0", "numpy==2.4.6", "scipy==1.17.1",
    "trimesh==4.12.2", "safetensors==0.8.0", "pydantic==2.13.4", "PyYAML==6.0.3",
    "einops==0.8.2", "rich==14.3.4", "typer==0.27.1", "torchmetrics==1.9.0",
    "Pillow==12.1.1", "pytest==9.1.1",
], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--force-reinstall",
    f"git+{{REPO_URL}}@{{CODE_REVISION}}",
], check=True)

for directory, url, revision in (
    (REPO_DIR, REPO_URL, CODE_REVISION),
    (ASSETS_DIR, "https://github.com/google-deepmind/mujoco_menagerie.git", MENAGERIE_REVISION),
):
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(directory)], check=True)
    subprocess.run(["git", "-C", str(directory), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(directory), "checkout", "--detach", revision], check=True)
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    assert actual == revision, (directory, actual, revision)

print("Pinned QDGrasp revision:", CODE_REVISION)
print("Pinned Menagerie revision:", MENAGERIE_REVISION)
'''
        ),
        _markdown(
            """## 1. Refuse a CPU host

A CPU fallback is never admissible as CUDA evidence
(`docs/decisions/0006-cuda-hardware-required.md`). This cell fails the run rather
than continuing on CPU.
"""
        ),
        _code(
            '''import torch

assert torch.cuda.is_available(), "no CUDA device: this notebook must run on a GPU kernel"
props = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", f"{props.major}.{props.minor}")
print("VRAM GiB:", round(props.total_memory / (1024 ** 3), 2))
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
'''
        ),
        _markdown(
            """## 2. Previous CUDA gates still pass

Plan section 10 requires the Phase 1 CUDA smoke and the Phase 2 FK parity to be
re-run before any Phase 3.4 measurement, so a regression in the foundation is
never reported as a Phase 3.4 result.
"""
        ),
        _code(
            '''import subprocess
import sys

for script, out in (
    ("scripts/phase1_cuda_smoke.py", "/tmp/phase1_cuda_evidence.json"),
    ("scripts/phase2_cuda_fk_parity.py", "/tmp/phase2_cuda_evidence.json"),
):
    print("=" * 70)
    print("running", script)
    completed = subprocess.run(
        [sys.executable, script, "--out", out],
        cwd="/tmp/qdgrasp_repo", capture_output=True, text=True,
    )
    print(completed.stdout[-3000:])
    print(completed.stderr[-2000:], file=sys.stderr)
    assert completed.returncode == 0, f"{script} failed with {completed.returncode}"
'''
        ),
        _markdown(
            """## 3. Install MuJoCo Warp

Not pinned in the repository locks yet: this notebook is the spike that decides
whether it earns a pin. The install is reported, not assumed to succeed.
"""
        ),
        _code(
            '''import subprocess
import sys

warp_install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "warp-lang", "mujoco-warp"],
    capture_output=True, text=True,
)
print("install returncode:", warp_install.returncode)
print(warp_install.stdout[-2000:])
print(warp_install.stderr[-2000:], file=sys.stderr)

import importlib.util
for module in ("warp", "mujoco_warp", "mujoco.mjx"):
    print(f"{module:14s}", "available" if importlib.util.find_spec(module) else "NOT installed")
'''
        ),
        _markdown(
            """## 4. Requirement matrix and backend verdict

The spike reports what the models need; the gate script decides. A verdict other
than `supported` exits nonzero and keeps Phase 3.4 blocked.
"""
        ),
        _code(
            '''import json
import subprocess
import sys

spike = subprocess.run(
    [sys.executable, "scripts/phase3_4_backend_spike.py", "--out", "/tmp/phase3_4_requirements.json"],
    cwd="/tmp/qdgrasp_repo", capture_output=True, text=True,
)
print(spike.stdout[-4000:])
assert spike.returncode == 0, spike.stderr[-2000:]

gate = subprocess.run(
    [sys.executable, "scripts/phase3_4_cuda_contact_search.py",
     "--device", "cuda:0", "--profile", "kaggle-t4-micro",
     "--evidence", "/tmp/phase3_4_cuda_evidence.json"],
    cwd="/tmp/qdgrasp_repo", capture_output=True, text=True,
)
print(gate.stdout[-6000:])
print(gate.stderr[-3000:], file=sys.stderr)

evidence = json.loads(open("/tmp/phase3_4_cuda_evidence.json", encoding="utf-8").read())
resolution = evidence["backend_resolution"]
print()
print("VERDICT:", resolution["verdict"])
print("unsupported:", resolution.get("unsupported", []))
print("gate exit:", gate.returncode)
'''
        ),
        _markdown(
            """## 5. What this run does and does not establish

A `supported` verdict unblocks `P3.4-05` (the CUDA backend). It does **not**
close Phase 3.4: throughput, VRAM, CPU/GPU parity fixtures, a CPU-confirmed
finalist per hand and the ContactRich dataset are all still outstanding.

A blocked verdict is a legitimate result. Record it and write the backend
decision record; do not work around it.
"""
        ),
        _code(
            '''import json

evidence = json.loads(open("/tmp/phase3_4_cuda_evidence.json", encoding="utf-8").read())
print(json.dumps(evidence, indent=2, sort_keys=True))
'''
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "id": f"{kaggle_slug}/qdgrasp-phase-3-4-cuda-gate",
                "title": "QDGrasp Phase 3.4 CUDA Backend Decision",
                "code_file": NOTEBOOK_PATH.name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": False,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "keywords": ["gpu"],
                "dataset_sources": [],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} pinned at {revision}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Public 40-char commit to pin.")
    parser.add_argument("--kaggle-slug", default="niniflo", help="Kaggle account slug.")
    args = parser.parse_args()
    build(args.revision, args.kaggle_slug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
