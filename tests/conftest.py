"""Fixtures shared across the suite.

The only one here is a corpus whose provenance is self-consistent.  The dataset
in the repository is too, since R8 regenerated it -- but it stops being so the
moment anyone edits a file the manifest records as a generator source, which is
a normal thing to do halfway through a change.

That drift is a release finding, not a reason for a dozen unrelated tests to go
red, so tests that only need "a corpus the loader accepts" take this copy.  The
tests that are *about* provenance read the artifact in the tree directly, and
they are the ones that must notice.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_DATASET = REPO_ROOT / "datasets" / "dgn-open-tiny"


@pytest.fixture(scope="session")
def verified_corpus(tmp_path_factory) -> Path:
    """The shipped shards, copied out with their generator hashes refreshed."""

    if not SHIPPED_DATASET.is_dir():
        pytest.skip(f"no corpus at {SHIPPED_DATASET}")

    root = tmp_path_factory.mktemp("verified-corpus")
    dataset = root / "datasets" / SHIPPED_DATASET.name
    shutil.copytree(SHIPPED_DATASET, dataset)

    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refreshed: dict[str, str] = {}
    for name in manifest["generator_source_hashes"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
        refreshed[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["generator_source_hashes"] = refreshed
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset
