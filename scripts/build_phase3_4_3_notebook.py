"""Build the Kaggle notebook for the Phase 3.4.3 CUDA gate (S10, G08, C07).

The development machine has no NVIDIA device, and that does not make the GPU
gate optional: it makes Kaggle the place it runs. This builder produces the
notebook that runs it, pinned to an exact public commit so the result can be
reproduced by someone who was not there.

What the notebook establishes, in order:

1. the host is a real GPU, not a CPU kernel reporting under a CUDA schema;
2. the Phase 1 CUDA smoke and Phase 2 active-hand FK parity still pass, so a
   regression in the foundation is never reported as a Phase 3.4.3 result;
3. which MuJoCo Warp version, if any, is clean under Compute Sanitizer -- the
   pinned 1.16.0 has an upstream uninitialised-read defect
   (``REV-20260827-010``), and the accepted resolutions are a newer pinned
   version or a blocked gate, never dropping the sanitizer;
4. the Phase 3.4.3 gate itself: capability, three-tier parity, performance.

The pinned revision must already exist on a remote, because the notebook
installs the library from GitHub. Pinning a local-only commit produces a
notebook that cannot run, so the builder refuses it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "kaggle-phase3-4-3"
NOTEBOOK_PATH = NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb"
METADATA_PATH = NOTEBOOK_DIR / "kernel-metadata.json"

MENAGERIE_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"
REPO_URL = "https://github.com/ninicom/qdgrasp.git"

#: The compatibility matrix, over **mujoco-warp** versions.
#:
#: The first attempt pinned ``mujoco-warp==1.16.0/1.17.0/1.18.0`` and all three
#: failed to install, because 1.16.0 is a ``warp-lang`` version and mujoco-warp
#: tracks MuJoCo's own numbering. That distinction matters for more than a pin:
#: REV-20260827-010 names the defective kernel as ``_linesearch_iterative_kernel``
#: in ``solver.py``, which is mujoco-warp's solver, so the version worth varying
#: is mujoco-warp's -- and warp-lang 1.16.0 is the newest there is, so there is
#: no newer runtime to move to.
#:
#: Newest first, matched to the pinned MuJoCo 3.12.0 at the top.
WARP_MATRIX = ("mujoco-warp==3.12.0", "mujoco-warp==3.11.0", "mujoco-warp==3.10.0.3")


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
            """# QDGrasp Phase 3.4.3 — CUDA gate for the two active hands

Scope is **LEAP Hand and Wonik Allegro**. `ADR-0008` pauses the Shadow Hand, so
nothing here is three-hand coverage and a missing Shadow result reads
`paused_by_ADR-0008` — never `pass`, `zero`, `unsupported` or a bare `not_run`.

This notebook produces GPU evidence. It does **not** close the phase: closure
needs the completeness manifest with zero open required items and an independent
review, neither of which a benchmark can supply.

| stage | what a failure means |
| --- | --- |
| host check | a CPU kernel is not CUDA evidence (`ADR-0006`) |
| prior gates | a regression in the foundation, reported as such |
| Warp matrix | no clean version → the GPU gate stays **blocked** |
| capability | the build cannot read the contact fields the budget needs |
| parity | the GPU and the CPU oracle disagree about the same world |
| performance | below 2x, or above the 14 GiB device budget |

Dropping the sanitizer, filtering out bad worlds after a rollout, or lowering a
threshold are all *not* accepted resolutions.
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
# /kaggle/working is the only directory Kaggle persists as a kernel
# output. Writing to /tmp produced a run whose evidence could not be
# downloaded, which is a packet nobody can review.
EVIDENCE_DIR = Path("/kaggle/working/phase3_4_3_evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

assert sys.version_info >= (3, 11), f"Python >=3.11 required, got {{sys.version}}"
os.environ.update(
    QDGRASP_ROBOT_ASSETS_ROOT="/tmp/robot-assets",
    MUJOCO_GL="egl",
    PYTHONHASHSEED="0",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
)

# The full pinned set, not a subset: importing qdgrasp pulls in the engine
# runner, which needs lightning. Trimming the list to what the gate "obviously"
# uses is how the first run died in the prior-gates cell.
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
    "lightning==2.6.5", "mujoco==3.12.0", "numpy==2.4.6", "scipy==1.17.1",
    "trimesh==4.12.2", "safetensors==0.8.0", "pydantic==2.13.4", "PyYAML==6.0.3",
    "einops==0.8.2", "rich==14.3.4", "typer==0.27.1", "torchmetrics==1.9.0",
    "Pillow==12.1.1", "pytest==9.1.1", "nvidia-ml-py",
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
            '''import json
import subprocess
import sys

import torch

assert torch.cuda.is_available(), "no CUDA device: this notebook must run on a GPU kernel"
props = torch.cuda.get_device_properties(0)
FINGERPRINT = {
    "gpu": torch.cuda.get_device_name(0),
    "capability": f"{props.major}.{props.minor}",
    "vram_gib": round(props.total_memory / (1024 ** 3), 3),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "driver": subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip(),
    "python": sys.version.split()[0],
}
print(json.dumps(FINGERPRINT, indent=2, sort_keys=True))
'''
        ),
        _markdown(
            """## 2. The prior CUDA gates still pass

Section 10 of `ROADMAP-P3.4-001` requires the Phase 1 CUDA smoke and the Phase 2
active-hand FK parity to be re-run before any Phase 3.4.3 measurement, so a
regression in the foundation is never reported as a Phase 3.4.3 result.
"""
        ),
        _code(
            '''import subprocess
import sys

for script, out in (
    ("scripts/phase1_cuda_smoke.py", "/kaggle/working/phase3_4_3_evidence/phase1_cuda.json"),
    ("scripts/phase2_cuda_fk_parity.py", "/kaggle/working/phase3_4_3_evidence/phase2_cuda.json"),
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
            f"""## 3. MuJoCo Warp compatibility matrix

`REV-20260827-010` isolated an uninitialised-read defect in MuJoCo Warp 1.16.0
and showed it reproduces with no QDGrasp code in the call path. The plan allows
exactly two resolutions: a newer pinned version that is clean, or a blocked GPU
gate. Removing the sanitizer, or discarding the worlds it flags, is not one of
them.

This cell tries {len(WARP_MATRIX)} versions under `compute-sanitizer --tool
initcheck` on a small reproducer and reports which, if any, is clean. Small on
purpose: the sanitizer costs one to two orders of magnitude, and this is a
diagnostic, never performance evidence.
"""
        ),
        _code(
            f'''import json
import shutil
import subprocess
import sys

WARP_MATRIX = {list(WARP_MATRIX)!r}
sanitizer = shutil.which("compute-sanitizer")
print("compute-sanitizer:", sanitizer or "NOT FOUND")

# The reproducer from REV-20260827-010 V-003, not a toy scene. The defect was
# isolated on the LEAP hand model with its meshes; a three-geom scene may never
# reach the kernel in question, so a clean result there would say nothing about
# the defect this matrix exists to re-test.
PROBE = (
    "import mujoco, mujoco_warp\\n"
    "from qdgrasp.dataset.pipeline.generated_reachable import "
    "build_generated_reachable_object as f\\n"
    "from qdgrasp.dataset.pipeline.validators.mujoco_rollout import "
    "build_rollout_scene_model as b\\n"
    "from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset\\n"
    "s = RobotSpec.from_config('leap_hand.yaml', sample_anchors=False)\\n"
    "x = f('leap_hand')\\n"
    "m = b(resolve_robot_asset(s.config.source_asset), x.collision_geoms,"
    " object_pos=x.object_pos, object_mass=x.mass)\\n"
    "print('model ngeom', m.ngeom, 'nq', m.nq, 'nu', m.nu)\\n"
    "d = mujoco.MjData(m); mujoco.mj_forward(m, d)\\n"
    "wm = mujoco_warp.put_model(m); wd = mujoco_warp.put_data(m, d, nworld=4)\\n"
    "for _ in range(8):\\n"
    "    mujoco_warp.step(wm, wd)\\n"
    "print('stepped 8 times over 4 worlds')\\n"
)

WARP_MATRIX_RESULT = {{}}
for pin in WARP_MATRIX:
    print("=" * 70)
    print("trying", pin)
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "warp-lang", pin],
        capture_output=True, text=True,
    )
    if install.returncode != 0:
        WARP_MATRIX_RESULT[pin] = {{
            "status": "install_failed",
            "detail": (install.stderr or install.stdout)[-600:],
        }}
        print("install failed:", (install.stderr or install.stdout).strip().splitlines()[-1][:200])
        continue
    if not sanitizer:
        WARP_MATRIX_RESULT[pin] = {{"status": "sanitizer_unavailable"}}
        continue
    run = subprocess.run(
        [sanitizer, "--tool", "initcheck", "--error-exitcode", "0", "--print-limit", "8",
         sys.executable, "-c", PROBE],
        cwd="/tmp/qdgrasp_repo", capture_output=True, text=True, timeout=5400,
    )
    text = run.stdout + run.stderr
    records = [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("=========")]
    summary = [ln for ln in records if "ERROR SUMMARY" in ln]
    errors = [ln for ln in records if "error" in ln.lower() and "0 errors" not in ln.lower()]
    stepped = "stepped 8 times over 4 worlds" in run.stdout

    # "Clean" needs positive proof, not the absence of an error line: a probe
    # that died before the first instrumented call also prints no errors, and
    # that is the reading that produced a false clean in the previous run.
    if not stepped:
        status = "probe_did_not_run"
    elif errors:
        status = "uninitialized_reads"
    elif summary and "0 errors" in summary[0]:
        status = "clean"
    else:
        status = "inconclusive_no_error_summary"

    WARP_MATRIX_RESULT[pin] = {{
        "status": status,
        "probe_stepped": stepped,
        "sanitizer_lines": len(records),
        "error_summary": summary[:2],
        "first_errors": errors[:5],
        "verbatim_tail": records[-8:],
        "probe_stdout_tail": run.stdout[-400:],
    }}
    print(status, f"({{len(records)}} sanitizer lines, stepped={{stepped}})")
    for ln in summary[:2]:
        print("   ", ln[:160])

try:
    import warp as _warp
    _warp_version = getattr(_warp, "__version__", "unknown")
except Exception as _exc:
    _warp_version = f"unavailable: {{type(_exc).__name__}}"

# The runtime version goes beside the results, not among them: mixing a scalar
# into a dict of per-pin dicts is what broke the previous run.
WARP_MATRIX_REPORT = {{"warp_lang_version": _warp_version, "pins": WARP_MATRIX_RESULT}}
json.dump(
    WARP_MATRIX_REPORT,
    open("/kaggle/working/phase3_4_3_evidence/warp_matrix.json", "w"),
    indent=2, sort_keys=True,
)
CLEAN = [pin for pin, r in WARP_MATRIX_RESULT.items() if r["status"] == "clean"]
print()
print("clean versions:", CLEAN or "none — the GPU gate stays BLOCKED")
'''
        ),
        _markdown(
            """## 4. Dry run: what the gate will cost

`C07.1` requires the resource estimate to be printed before the run, not
discovered when it OOMs halfway through.
"""
        ),
        _code(
            '''import subprocess
import sys

dry = subprocess.run(
    [sys.executable, "scripts/check_phase3_4_3_cuda.py", "--dry-run", "--worlds", "1024"],
    cwd="/tmp/qdgrasp_repo", capture_output=True, text=True,
)
print(dry.stdout)
assert dry.returncode == 0, dry.stderr
'''
        ),
        _markdown(
            """## 5. The Phase 3.4.3 CUDA gate

Capability, three-tier parity and performance, in that order, with a checkpoint
so a wall-clock kill does not throw away the stages that already finished. The
deadline guard flushes the ledger before the Kaggle session ends.

If the Warp matrix found no clean version, run this anyway and record the
result: a blocked gate with measured evidence is a legitimate outcome, and a
skipped gate is not.
"""
        ),
        _code(
            '''import json
import subprocess
import sys

gate = subprocess.run(
    [sys.executable, "scripts/check_phase3_4_3_cuda.py",
     "--device", "cuda:0", "--worlds", "1024", "--runs", "3",
     "--evidence", "/kaggle/working/phase3_4_3_evidence/cuda-gate.json",
     "--checkpoint", "/kaggle/working/phase3_4_3_evidence/cuda-gate.checkpoint.json",
     "--deadline-seconds", "24000"],
    cwd="/tmp/qdgrasp_repo", capture_output=True, text=True,
)
print(gate.stdout[-8000:])
print(gate.stderr[-3000:], file=sys.stderr)
print("gate exit:", gate.returncode)

# A gate that refused to run still has to leave a readable result. If it exited
# before writing evidence -- a configuration error, a missing dependency -- the
# reason is what matters, and turning that into a KeyError here would hide it.
from pathlib import Path as _Path

_evidence = _Path("/kaggle/working/phase3_4_3_evidence/cuda-gate.json")
if _evidence.is_file():
    GATE = json.loads(_evidence.read_text(encoding="utf-8"))
else:
    GATE = {
        "verdict": "NO_EVIDENCE",
        "gate_exit": gate.returncode,
        "stdout_tail": gate.stdout[-2000:],
        "stderr_tail": gate.stderr[-2000:],
    }
    _evidence.parent.mkdir(parents=True, exist_ok=True)
    _evidence.write_text(json.dumps(GATE, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print("VERDICT:", GATE["verdict"])
'''
        ),
        _markdown(
            """## 6. Sanitizer on the gate's own workload

Zero invalid reads is a gate criterion, not a nice-to-have. Bounded worlds and
horizon because the sanitizer is slow; this is a diagnostic and never
performance evidence.
"""
        ),
        _code(
            '''import shutil
import subprocess
import sys

sanitizer = shutil.which("compute-sanitizer")
SANITIZER_RESULT = {"tool_available": bool(sanitizer)}
if not sanitizer:
    print("compute-sanitizer unavailable; the question stays open rather than guessed at.")
else:
    for tool in ("racecheck", "initcheck"):
        run = subprocess.run(
            [sanitizer, "--tool", tool, "--error-exitcode", "0", "--print-limit", "40",
             sys.executable, "scripts/phase3_4_1_sanitizer.py", "--worlds", "4", "--horizon", "8"],
            cwd="/tmp/qdgrasp_repo", capture_output=True, text=True, timeout=5400,
        )
        text = run.stdout + run.stderr
        records = [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("=========")]
        SANITIZER_RESULT[tool] = {"lines": len(records), "head": [ln[:190] for ln in records[:20]]}
        print("=" * 70)
        print(tool, f"-- {len(records)} report lines")
        for ln in records[:20]:
            print(ln[:190])
'''
        ),
        _markdown(
            """## 7. Environment fingerprint and evidence hashes

Everything a reviewer needs to tell whether two runs are comparable, and nothing
a notebook should not carry: no credentials, no private paths, no mutable
notebook name used as a pin.
"""
        ),
        _code(
            '''import hashlib
import json
from pathlib import Path

PACKET = {
    "schema": "qdgrasp/evidence/phase3.4.3-kaggle/v1",
    "commit": CODE_REVISION,
    "menagerie": MENAGERIE_REVISION,
    "environment": FINGERPRINT,
    "warp_matrix": WARP_MATRIX_REPORT,
    "sanitizer": SANITIZER_RESULT,
    "gate_verdict": GATE["verdict"],
    "artifact_hashes": {},
}
for path in sorted(Path("/kaggle/working/phase3_4_3_evidence").glob("*.json")):
    PACKET["artifact_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

out = Path("/kaggle/working/phase3_4_3_evidence/packet.json")
out.write_text(json.dumps(PACKET, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(PACKET, indent=2, sort_keys=True))
print()
print("packet sha256:", hashlib.sha256(out.read_bytes()).hexdigest())
print()
print("Download /kaggle/working/phase3_4_3_evidence/ and commit it under evidence/phase3_4_3/s10/.")
print("A PASS here is GPU evidence for two active hands. It is not phase closure,")
print("and it is not three-hand coverage.")
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
                # Kaggle derives the real slug from the title, so the two have
                # to agree: a title that resolves elsewhere silently creates a
                # second kernel, and the plan forbids using a mutable notebook
                # name as the pin.
                "id": f"{kaggle_slug}/qdgrasp-phase-3-4-3-cuda-gate-active-hands",
                "title": "QDGrasp Phase 3 4 3 CUDA Gate Active Hands",
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
    parser.add_argument("--kaggle-slug", default="quyndang", help="Kaggle username.")
    args = parser.parse_args()
    build(args.revision, args.kaggle_slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
