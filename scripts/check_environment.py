#!/usr/bin/env python3
"""Validate the declared QDGrasp research environment without importing project code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import sys
from importlib import metadata
from pathlib import Path


REQUIRED = {
    "torch": "2.11.0",
    "lightning": "2.6.5",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "mujoco": "3.12.0",
    "trimesh": "4.12.2",
    "safetensors": "0.8.0",
    "pydantic": "2.13.4",
    "PyYAML": "6.0.3",
    "einops": "0.8.2",
    "rich": "14.3.4",
    "typer": "0.27.1",
}


def check_lock_manifest(problems: list[str]) -> None:
    try:
        yaml = importlib.import_module("yaml")
        manifest = yaml.safe_load(Path("environments/environment.lock.yaml").read_text())
    except Exception as exc:
        problems.append(f"không đọc được environment lock manifest: {exc}")
        return
    for name, profile in manifest.get("profiles", {}).items():
        lock_name = profile.get("lock")
        expected = profile.get("sha256")
        if not lock_name or not expected:
            problems.append(f"profile {name} thiếu lock hoặc sha256")
            continue
        path = Path(lock_name)
        if not path.is_file():
            problems.append(f"profile {name} thiếu file {path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"profile {name} hash sai: cần {expected}, hiện tại {actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    problems: list[str] = []
    versions: dict[str, str | None] = {}
    check_lock_manifest(problems)
    if sys.version_info < (3, 11):
        problems.append(f"Python phải >=3.11, hiện tại là {platform.python_version()}")

    for package, expected in REQUIRED.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = None
            problems.append(f"thiếu package {package}=={expected}")
        else:
            version_matches = actual == expected or (package == "torch" and actual.startswith(f"{expected}+"))
            if not version_matches:
                problems.append(f"{package}: cần {expected}, hiện tại {actual}")
        versions[package] = actual

    torch_info: dict[str, object] = {}
    if versions["torch"] is not None:
        torch = importlib.import_module("torch")
        torch_info = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if args.profile == "cpu" and "+cpu" not in torch.__version__:
            problems.append(f"CPU profile cần torch wheel +cpu, hiện tại {torch.__version__}")
        if args.profile == "cuda":
            if torch.version.cuda != "12.8":
                problems.append(f"CUDA profile cần torch cu128, hiện tại CUDA build={torch.version.cuda}")
            if not torch.cuda.is_available():
                problems.append("CUDA profile không nhìn thấy GPU/driver NVIDIA")
        try:
            device = torch.device("cuda" if args.profile == "cuda" else "cpu")
            value = (torch.eye(3, device=device) @ torch.ones(3, device=device)).sum().item()
            if not math.isclose(value, 3.0):
                problems.append(f"Torch smoke test trả về {value}, cần 3.0")
        except Exception as exc:  # pragma: no cover - depends on runtime/driver
            problems.append(f"Torch smoke test lỗi: {exc}")

    if versions["lightning"] is not None:
        try:
            importlib.import_module("lightning.fabric")
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            problems.append(f"Lightning Fabric import lỗi: {exc}")

    if versions["mujoco"] is not None:
        try:
            mujoco = importlib.import_module("mujoco")
            model = mujoco.MjModel.from_xml_string(
                "<mujoco><worldbody><body><joint/><geom type='sphere' size='.01'/></body></worldbody></mujoco>"
            )
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
        except Exception as exc:  # pragma: no cover - native runtime specific
            problems.append(f"MuJoCo smoke test lỗi: {exc}")

    report = {
        "ok": not problems,
        "profile": args.profile,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch": torch_info,
        "problems": problems,
    }
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"QDGrasp environment: {'PASS' if report['ok'] else 'FAIL'} ({args.profile})")
        for problem in problems:
            print(f"- {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
