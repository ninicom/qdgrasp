"""Build the pinned Kaggle notebook for measured Phase 3.3 release evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Replaced with the immutable implementation commit before notebook publication.
KAGGLE_CODE_REVISION = "d54723ae024561c3f600456f6b3f8fc8c3bb70b5"
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


def build_unified_notebook() -> None:
    if len(KAGGLE_CODE_REVISION) != 40:
        raise RuntimeError("KAGGLE_CODE_REVISION must be an immutable 40-character Git commit")

    cells = [
        _markdown(
            """# QDGrasp Phase 3.3 — measured grasp release gate

This notebook replays the three admitted `qdgrasp-scene-tiny` positive controls through the exact
Phase 3.3 scene validator and renders its live MuJoCo state from four synchronized cameras.

The videos are **measured release-control evidence**, not a generator-quality claim.
Every green result is derived from the replayed validator verdict and checked against the immutable
dataset record (identity hashes, state hashes, lift, finger contact, palm support, and floor support).
"""
        ),
        _code(
            f'''import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_REVISION = "{KAGGLE_CODE_REVISION}"
MENAGERIE_REVISION = "{MENAGERIE_REVISION}"
REPO_URL = "https://github.com/ninicom/qdgrasp.git"
REPO_DIR = Path("/tmp/qdgrasp_repo")
CPU_TORCH_DIR = Path("/tmp/torch-cpu-pin")
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
    "einops==0.8.2", "rich==14.3.4", "typer==0.27.1", "imageio[ffmpeg]==2.37.2",
    "torchmetrics==1.9.0",
    "Pillow==12.1.1", "pytest==9.1.1",
], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--force-reinstall",
    f"git+{{REPO_URL}}@{{CODE_REVISION}}",
], check=True)

# recipe_hash is a SHA-256 over full-precision float64 values whose low bits come from the
# float32 DLS IK solve. Reproducing the release dataset hashes requires the exact torch build
# that generated them, so it is installed side by side and used only for the CPU replay work.
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
    "--target", str(CPU_TORCH_DIR), "torch==2.11.0", "torchvision==0.26.0",
    "--index-url", "https://download.pytorch.org/whl/cpu",
], check=True)

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", CODE_REVISION], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", CODE_REVISION], check=True)
checked_out = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
assert checked_out == CODE_REVISION, (checked_out, CODE_REVISION)

if not ASSETS_DIR.exists():
    ASSETS_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--filter=blob:none", "--no-checkout",
        "https://github.com/google-deepmind/mujoco_menagerie.git", str(ASSETS_DIR),
    ], check=True)
subprocess.run(["git", "-C", str(ASSETS_DIR), "fetch", "--depth", "1", "origin", MENAGERIE_REVISION], check=True)
subprocess.run(["git", "-C", str(ASSETS_DIR), "checkout", "--detach", MENAGERIE_REVISION], check=True)
assets_revision = subprocess.check_output(
    ["git", "-C", str(ASSETS_DIR), "rev-parse", "HEAD"], text=True
).strip()
assert assets_revision == MENAGERIE_REVISION, (assets_revision, MENAGERIE_REVISION)

print("Pinned QDGrasp revision:", checked_out)
print("Pinned MuJoCo Menagerie revision:", assets_revision)
print("Python:", sys.version.split()[0])
'''
        ),
        _markdown(
            """## Recipe identity drift probe

`recipe_hash` is a SHA-256 over full-precision float64 values produced by torch FK and the DLS IK
solver. This cell reports, without failing the run, how the Kaggle-rebuilt recipe payload deviates
from the payload measured on the machine that generated the release dataset.
"""
        ),
        _code(
            '''import os
import subprocess
import sys
from pathlib import Path

RECIPE_REFERENCE_JSON = r"""{"leap_hand.yaml":{"payload":{"recipe_id":"scene_pinch_leap_v1","robot_profile":"leap_hand.yaml","rollout_kwargs":{"approach_steps":100,"expected_fingertip_positions":[[0.013676881790161133,-1.862645149230957e-09,0.019999992102384567],[0.16076520085334778,-0.03235624358057976,0.06339819729328156],[0.16597706079483032,-0.07745367288589478,0.06293082237243652],[-0.01367688924074173,-1.862645149230957e-09,0.019999997690320015]],"fingertip_local_offsets":[[-0.003192000091075897,-0.04983900114893913,0.013322999700903893],[-0.003192000091075897,-0.04983900114893913,0.013322999700903893],[-0.003192000091075897,-0.04983900114893913,0.013322999700903893],[-0.004124999977648258,-0.06197499856352806,-0.016346000134944916]],"initial_joint_targets":{"if_dip":1.7462109327316284,"if_mcp":0.5731149911880493,"if_pip":0.5556632280349731,"if_rot":-0.39795437455177307,"mf_dip":0.0,"mf_mcp":0.0,"mf_pip":0.0,"mf_rot":0.0,"rf_dip":0.0,"rf_mcp":0.0,"rf_pip":0.0,"rf_rot":0.0,"th_axl":0.1407385915517807,"th_cmc":1.2242861986160278,"th_ipl":1.7322335243225098,"th_mcp":-0.20632995665073395},"joint_targets":{"if_dip":1.6490451097488403,"if_mcp":0.6067735552787781,"if_pip":0.6571747064590454,"if_rot":-0.3639439642429352,"mf_dip":0.0,"mf_mcp":0.0,"mf_pip":0.0,"mf_rot":0.0,"rf_dip":0.0,"rf_mcp":0.0,"rf_pip":0.0,"rf_rot":0.0,"th_axl":0.13191194832324982,"th_cmc":1.230963110923767,"th_ipl":1.6148759126663208,"th_mcp":-0.07931005209684372},"object_mass":0.02,"object_pos":[0.0,0.0,0.02],"palm_pos":[0.0316806807301218,-0.011387211292634529,0.10949749041965837],"palm_rot":[[0.9774340436304988,-0.11479870282982955,0.17732441507226462],[0.11479870282982955,0.9933354327473721,0.010294458566607986],[-0.17732441507226462,0.010294458566607986,0.9840986108831268]],"pregrasp_direction":[0.0,0.0,1.0],"pregrasp_distance":0.08,"rollout_protocol":{"actuator_tracking_range_fraction":0.1,"cone_tolerance":1e-06,"contact_window_fraction":0.25,"gains_source":"compiled_mjcf","joint_tracking_range_fraction":0.15,"lift_success_fraction":0.5,"minimum_contact_duty_cycle":0.8,"minimum_contact_impulse_ratio":0.5,"palm_position_tolerance":0.005,"palm_rotation_tolerance":0.08726646259971647,"root_mocap_position_tolerance":0.005,"timestep_source":"compiled_model"},"squeeze_steps":300},"target_geoms":[{"density":1000.0,"pos":[0.0,0.0,0.0],"quat":[1.0,0.0,0.0,0.0],"size":[0.013676886446774006,0.015,0.02],"type":"box"}]},"recipe_hash":"da37dc189e8e256d648aa65be162305d84282fbe84a60798719de809ea41ba25"},"shadow_hand.yaml":{"payload":{"recipe_id":"scene_pinch_shadow_v1","robot_profile":"shadow_hand.yaml","rollout_kwargs":{"approach_steps":150,"expected_fingertip_positions":[[0.02653578855097294,7.450580596923828e-09,0.10000000327825546],[-0.053611576557159424,0.04909365251660347,0.12700334370136263],[-0.055151522159576416,0.051290664821863174,0.14920248270034792],[-0.061148494482040405,0.05390365794301033,0.17186211973428728],[-0.026535747572779655,1.1175870895385742e-08,0.09999999769032002]],"fingertip_local_offsets":[[-3.999999989900971e-06,0.0026459998916834593,0.03172300010919571],[-3.999999989900971e-06,0.0026459998916834593,0.03172300010919571],[-3.999999989900971e-06,0.0026459998916834593,0.03172300010919571],[-3.999999989900971e-06,0.0026459998916834593,0.03172300010919571],[-5.500000042957254e-05,0.005553000140935183,0.03474099934101105]],"initial_joint_targets":{"rh_FFJ1":0.44999998807907104,"rh_FFJ2":0.44999998807907104,"rh_FFJ3":0.550000011920929,"rh_FFJ4":0.0,"rh_LFJ1":1.2000000476837158,"rh_LFJ2":1.2000000476837158,"rh_LFJ3":1.399999976158142,"rh_LFJ4":0.0,"rh_LFJ5":0.0,"rh_MFJ1":1.2000000476837158,"rh_MFJ2":1.2000000476837158,"rh_MFJ3":1.399999976158142,"rh_MFJ4":0.0,"rh_RFJ1":1.2000000476837158,"rh_RFJ2":1.2000000476837158,"rh_RFJ3":1.399999976158142,"rh_RFJ4":0.0,"rh_THJ1":0.44999998807907104,"rh_THJ2":0.44999998807907104,"rh_THJ3":0.0,"rh_THJ4":0.949999988079071,"rh_THJ5":0.0,"rh_WRJ1":0.0,"rh_WRJ2":0.0},"joint_targets":{"rh_FFJ1":0.6000000238418579,"rh_FFJ2":0.6000000238418579,"rh_FFJ3":0.7200000286102295,"rh_FFJ4":0.0,"rh_LFJ1":1.2000000476837158,"rh_LFJ2":1.2000000476837158,"rh_LFJ3":1.399999976158142,"rh_LFJ4":0.0,"rh_LFJ5":0.0,"rh_MFJ1":1.2000000476837158,"rh_MFJ2":1.2000000476837158,"rh_MFJ3":1.399999976158142,"rh_MFJ4":0.0,"rh_RFJ1":1.2000000476837158,"rh_RFJ2":1.2000000476837158,"rh_RFJ3":1.399999976158142,"rh_RFJ4":0.0,"rh_THJ1":0.6000000238418579,"rh_THJ2":0.6000000238418579,"rh_THJ3":0.0,"rh_THJ4":1.100000023841858,"rh_THJ5":0.0,"rh_WRJ1":0.0,"rh_WRJ2":0.0},"lift_height":0.05,"lift_steps":150,"object_mass":0.02,"object_pos":[0.0,0.0,0.1],"palm_pos":[-0.10704224240082154,0.09502237955613341,0.1401925049768188],"palm_rot":[[-0.11008375944040366,0.08305691228590359,0.9904459173670204],[-0.08305691228590359,0.992248202286846,-0.09243946332629592],[-0.9904459173670204,-0.09243946332629592,-0.10233196172724984]],"perturbation_steps":40,"perturbation_wrench":[0.02,0.02,0.0,0.002,0.002,0.002],"pregrasp_direction":[0.0,0.0,1.0],"pregrasp_distance":0.12,"rollout_protocol":{"actuator_tracking_range_fraction":0.1,"cone_tolerance":1e-06,"contact_window_fraction":0.25,"gains_source":"compiled_mjcf","joint_tracking_range_fraction":0.15,"lift_success_fraction":0.5,"minimum_contact_duty_cycle":0.8,"minimum_contact_impulse_ratio":0.5,"palm_position_tolerance":0.005,"palm_rotation_tolerance":0.08726646259971647,"root_mocap_position_tolerance":0.005,"timestep_source":"compiled_model"},"squeeze_steps":250},"target_geoms":[{"density":1000.0,"pos":[0.0,0.0,0.0],"quat":[1.0,0.0,0.0,0.0],"size":[0.019035764336586,0.015,0.1],"type":"box"}]},"recipe_hash":"7703bd826a107f1b051dc89ee2d21a19daf32a47313acfe99b3143674c39b1c8"},"wonik_allegro.yaml":{"payload":{"recipe_id":"scene_pinch_allegro_v1","robot_profile":"wonik_allegro.yaml","rollout_kwargs":{"approach_steps":100,"expected_fingertip_positions":[[0.031550366431474686,-4.190951585769653e-09,0.019999992102384567],[0.055993981659412384,-0.05414988473057747,0.0015403181314468384],[0.008777178823947906,-0.10373803228139877,0.045259516686201096],[-0.03155036270618439,1.862645149230957e-09,0.019999999552965164]],"fingertip_local_offsets":[[0.0,0.0,0.04100000113248825],[0.0,0.0,0.04100000113248825],[0.0,0.0,0.04100000113248825],[0.0,0.0,0.054999999701976776]],"initial_joint_targets":{"ffj0":-0.14100635051727295,"ffj1":0.6320668458938599,"ffj2":0.4346795082092285,"ffj3":1.5279934406280518,"mfj0":-0.18294981122016907,"mfj1":0.7104878425598145,"mfj2":0.46372127532958984,"mfj3":0.6895720362663269,"rfj0":-0.3722456991672516,"rfj1":0.4500102400779724,"rfj2":1.2411249876022339,"rfj3":1.3361222743988037,"thj0":0.9744022488594055,"thj1":0.7169086933135986,"thj2":0.1245168149471283,"thj3":1.6876024007797241},"joint_targets":{"ffj0":-0.14100636541843414,"ffj1":0.7589393854141235,"ffj2":0.29052919149398804,"ffj3":1.6104965209960938,"mfj0":-0.18294981122016907,"mfj1":0.7104878425598145,"mfj2":0.46372127532958984,"mfj3":0.6895720362663269,"rfj0":-0.3722456991672516,"rfj1":0.4500102400779724,"rfj2":1.2411249876022339,"rfj3":1.3361222743988037,"thj0":1.0663591623306274,"thj1":0.5970826745033264,"thj2":0.10715543478727341,"thj3":1.6771000623703003},"object_mass":0.02,"object_pos":[0.0,0.0,0.02],"palm_pos":[-0.010646260003985417,-0.034806324986692304,0.1063505889381681],"palm_rot":[[0.05200149866123227,-0.003908675911283051,0.9986393575208251],[0.003908675911283051,0.999985477447134,0.0037104106845553073],[-0.9986393575208251,0.0037104106845553073,0.05201602121409832]],"perturbation_wrench":[0.15,0.15,0.0,0.01,0.01,0.01],"pregrasp_direction":[0.0,0.0,1.0],"pregrasp_distance":0.08,"rollout_protocol":{"actuator_tracking_range_fraction":0.1,"cone_tolerance":1e-06,"contact_window_fraction":0.25,"gains_source":"compiled_mjcf","joint_tracking_range_fraction":0.15,"lift_success_fraction":0.5,"minimum_contact_duty_cycle":0.8,"minimum_contact_impulse_ratio":0.5,"palm_position_tolerance":0.005,"palm_rotation_tolerance":0.08726646259971647,"root_mocap_position_tolerance":0.005,"timestep_source":"compiled_model"},"squeeze_steps":500},"target_geoms":[{"density":1000.0,"pos":[0.0,0.0,0.0],"quat":[1.0,0.0,0.0,0.0],"size":[0.031550366431474686,0.015,0.02],"type":"box"}]},"recipe_hash":"9b20e399d2379a550964c95baa9615447472aba27ea0d4990900b8590fc84aa0"}}"""
Path("/tmp/recipe_ref.json").write_text(RECIPE_REFERENCE_JSON, encoding="utf-8")

drift_probe_code = r"""
import json
import sys

sys.path.insert(0, "/tmp/qdgrasp_repo")

import numpy
import scipy
import torch

print("torch", torch.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__)
print("torch threads", torch.get_num_threads(), "| interop", torch.get_num_interop_threads())

from qdgrasp.scenes.release_recipes import _hash, _json_value, build_release_grasp_recipe

reference = json.loads(open("/tmp/recipe_ref.json", encoding="utf-8").read())


def walk(prefix, actual, expected, out):
    if isinstance(expected, dict):
        for key in sorted(expected):
            walk(f"{prefix}.{key}", actual.get(key) if isinstance(actual, dict) else None, expected[key], out)
    elif isinstance(expected, list):
        for index, item in enumerate(expected):
            child = actual[index] if isinstance(actual, list) and index < len(actual) else None
            walk(f"{prefix}[{index}]", child, item, out)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)):
        delta = abs(float(actual) - float(expected))
        if delta:
            out.append((delta, prefix, actual, expected))
    elif actual != expected:
        out.append((float("inf"), prefix, actual, expected))


for profile in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
    recipe = build_release_grasp_recipe(profile)
    payload = {
        "recipe_id": recipe.recipe_id,
        "robot_profile": recipe.robot_profile,
        "target_geoms": [geom.model_dump(mode="json") for geom in recipe.target_geoms],
        "rollout_kwargs": recipe.rollout_kwargs,
    }
    actual = _json_value(payload)
    expected = reference[profile]
    print("")
    print(profile, "recipe_hash match:", recipe.recipe_hash == expected["recipe_hash"])
    print("  kaggle:", recipe.recipe_hash)
    print("  local :", expected["recipe_hash"])
    for key in sorted(expected["payload"]["rollout_kwargs"]):
        left = _hash(actual["rollout_kwargs"][key])
        right = _hash(expected["payload"]["rollout_kwargs"][key])
        if left != right:
            print("  drifting key:", key)
    diffs = []
    walk("payload", actual, expected["payload"], diffs)
    diffs.sort(key=lambda item: -item[0])
    print("  differing scalars:", len(diffs))
    for delta, path, left, right in diffs[:10]:
        print(f"   {path}: kaggle={left!r} local={right!r} absdiff={delta:.3e}")
"""
probe = subprocess.run(
    [sys.executable, "-c", drift_probe_code],
    env=dict(os.environ, PYTHONPATH="/tmp/torch-cpu-pin"),
    check=False,
)
print("drift probe exit code:", probe.returncode)
'''
        ),
        _markdown(
            """## CUDA framework smoke and Phase 3.3 release audit

The CUDA check below remains a bounded framework smoke test. It is not presented as grasp-model
quality evidence. The grasp evidence comes from the release audit and exact validator replays.
"""
        ),
        _code(
            '''import os
import subprocess
import sys

# Every heavy import runs in a fresh interpreter. The Kaggle kernel already has the
# base-image numpy loaded before cell 1 upgrades it on disk, so importing torch/qdgrasp
# in-process mixes new numpy .py files with the old in-memory _multiarray_umath.
cuda_smoke_code = r"""
import os

os.environ.update(
    QDGRASP_ROBOT_ASSETS_ROOT="/tmp/robot-assets",
    PYTHONHASHSEED="0",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
)

import torch
from qdgrasp import require_cuda
from qdgrasp.api import QDGrasp

assert torch.cuda.is_available(), "Kaggle GPU gate requires an NVIDIA GPU"
require_cuda(expected_runtime=torch.version.cuda)
smoke = QDGrasp().train(
    "dummy-tiny.yaml", device="cuda:0", amp=True, max_steps=8, batch_size=4,
    run_name="phase1-cuda-framework-smoke",
)
assert smoke.runtime["effective"]["precision"] == "16-mixed"
print("CUDA framework smoke: PASS", torch.cuda.get_device_name(0))
"""
subprocess.run([sys.executable, "-c", cuda_smoke_code], env=dict(os.environ), check=True)

dataset_root = "/tmp/qdgrasp_repo/datasets/qdgrasp-scene-tiny"
subprocess.run([
    sys.executable, "/tmp/qdgrasp_repo/scripts/check_phase3_3.py",
    "--profile", "release", "--dataset-root", dataset_root,
], env=dict(os.environ), check=True)
print("Phase 3.3 immutable release audit: PASS")
'''
        ),
        _markdown(
            """## Measured four-view validator rollouts

Exactly one admitted positive record is replayed for each embodiment: LEAP Hand, Wonik Allegro,
and Shadow Hand. Rendering observes the same `MjData` used by the validator; it does not inject a
separate hand trajectory, move the floor, or infer success from video category names.
"""
        ),
        _code(
            '''import base64
import json
import subprocess
import sys
from pathlib import Path
from IPython.display import HTML, display

video_runner_code = r"""
import json
import os
import sys
from pathlib import Path

os.environ.update(
    QDGRASP_ROBOT_ASSETS_ROOT="/tmp/robot-assets",
    MUJOCO_GL="egl",
    PYTHONHASHSEED="0",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
)
sys.path.insert(0, "/tmp/qdgrasp_repo")

from scripts.render_4view_rollout import run_kaggle_video_suite

results = run_kaggle_video_suite(
    output_dir="/kaggle/working/videos",
    robot_assets_root="/tmp/robot-assets",
    dataset_root="/tmp/qdgrasp_repo/datasets/qdgrasp-scene-tiny",
)
assert len(results) == 3
assert {item["robot_profile"] for item in results} == {
    "leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"
}
for item in results:
    assert item["actual_outcome"] == "PASS"
    assert item["final_active_fingers"] >= 2
    assert not item["has_palm_contact"]
    assert not item["floor_support"]

Path("/kaggle/working/video_manifest.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
"""
subprocess.run(
    [sys.executable, "-c", video_runner_code],
    env=dict(os.environ, PYTHONPATH="/tmp/torch-cpu-pin"),
    check=True,
)

results = json.loads(Path("/kaggle/working/video_manifest.json").read_text())
for result in results:
    # Card state is derived only from the replayed validator outcome.
    passed = result["actual_outcome"] == "PASS"
    color = "#22c55e" if passed else "#ef4444"
    badge = "MEASURED VALIDATOR PASS" if passed else "VALIDATOR FAIL"
    video_path = Path(result["video_path"])
    payload = base64.b64encode(video_path.read_bytes()).decode("ascii")
    metrics = (
        f"lift={result['measured_target_lift']:.4f} m · "
        f"active fingers={result['final_active_fingers']} · "
        f"palm support={int(result['has_palm_contact'])} · "
        f"floor support={int(result['floor_support'])}"
    )
    display(HTML(f"""
    <div style="margin:18px 0;padding:14px;border:1px solid {color};border-radius:9px;font-family:sans-serif">
      <h3 style="color:{color};margin:0 0 7px">{badge}: {result['robot_profile']}</h3>
      <div><code>{result['scenario']}</code></div>
      <div>{result['source_class']} · {metrics}</div>
      <video controls loop muted style="width:100%;max-width:960px;margin-top:10px">
        <source src="data:video/mp4;base64,{payload}" type="video/mp4">
      </video>
    </div>
    """))

print("All three measured Phase 3.3 release-control rollouts: PASS")
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
    for path in (
        ROOT / "kaggle-phase3" / "qdgrasp-phase-3-cuda-gate.ipynb",
        ROOT / "kaggle-phase1" / "qdgrasp-phase-1-cuda-framework-gate.ipynb",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(f"Built pinned notebook {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    build_unified_notebook()
