from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT_ROOT / "scripts" / "check_static_core.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_static_core_under_test", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_static_boundary_covers_every_corrective_runtime_chain_without_legacy() -> None:
    checker = _load_checker()
    targets = set(checker.active_ruff_targets(PROJECT_ROOT))

    assert {
        "qdgrasp/api/facade.py",
        "qdgrasp/dataset/artifact.py",
        "qdgrasp/dataset/pipeline/orchestrator.py",
        "qdgrasp/engine/checkpoint.py",
        "qdgrasp/models/flow.py",
        "qdgrasp/mvp/policy.py",
        "qdgrasp/robot/spec.py",
        "scripts/check_dataset_manifest.py",
        "scripts/check_phase4.py",
        "scripts/check_wheel.py",
    } <= targets
    assert not any(name.startswith(checker.FORBIDDEN_PREFIXES) for name in targets)


def test_mypy_boundary_pins_the_public_artifact_and_identity_contracts() -> None:
    checker = _load_checker()
    targets = set(checker.MYPY_TARGETS)

    assert len(targets) == 32
    assert {
        "qdgrasp/api/facade.py",
        "qdgrasp/corrective/gate.py",
        "qdgrasp/dataset/artifact.py",
        "qdgrasp/engine/checkpoint.py",
        "qdgrasp/models/protocol.py",
        "qdgrasp/mvp/evaluate.py",
        "qdgrasp/robot/schema.py",
    } <= targets
