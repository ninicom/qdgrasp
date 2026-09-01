"""P5's input gate: is there enough signal in the data to train on at all?

The gate exists because the answer is currently no, and because "no" is the
kind of finding that otherwise surfaces after a week of runs, in the shape of a
model that reproduces the proposal distribution and a quality head that has
learned the prior.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "datasets/dgn-open-tiny"
PROTOCOL = REPO_ROOT / "configs/phase5/protocol-v2.yaml"


@pytest.fixture(scope="module")
def gate():
    path = REPO_ROOT / "scripts/check_phase5_inputs.py"
    spec = importlib.util.spec_from_file_location("_qdgrasp_phase5_inputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_shipped_corpus_is_measured_through_the_locked_view(gate, capsys) -> None:
    assert gate.main([]) == 1
    output = capsys.readouterr().out
    assert "Not enough successful grasps" in output or "could not be measured" in output


def test_the_gate_reports_insufficient_and_exits_nonzero(gate, verified_corpus, capsys) -> None:
    assert gate.main(["--dataset", str(verified_corpus), "--protocol", str(PROTOCOL)]) == 1
    assert "Not enough successful grasps" in capsys.readouterr().out


def test_the_measurement_names_every_active_hand_and_split(gate, verified_corpus) -> None:
    report = gate.measure(verified_corpus, PROTOCOL)
    pairs = {(row["split"], row["robot"]) for row in report["rows"]}
    assert pairs == {
        ("train", "leap_hand"),
        ("train", "wonik_allegro"),
        ("val", "leap_hand"),
        ("val", "wonik_allegro"),
    }
    assert report["sufficient"] is False
    assert report["train_positives_total"] < gate.MINIMUM_POSITIVES_PER_HAND


def test_the_floor_is_per_hand_not_across_hands(gate, verified_corpus) -> None:
    """The held-out embodiment is evaluated, never required to have train labels."""

    report = gate.measure(verified_corpus, PROTOCOL)
    train = [row for row in report["rows"] if row["split"] == "train" and row["admitted_to_training"]]
    held_out = [row for row in report["rows"] if row["split"] == "train" and not row["admitted_to_training"]]
    assert [row["robot"] for row in train] == ["leap_hand"]
    assert len(held_out) == 1 and held_out[0]["samples"] == 0
    assert report["sufficient"] == all(row["positives"] >= gate.MINIMUM_POSITIVES_PER_HAND for row in train)


def test_the_report_carries_the_protocol_hash(gate, verified_corpus) -> None:
    """A count is about a split, so it has to name which split."""

    report = gate.measure(verified_corpus, PROTOCOL)
    assert len(report["protocol_hash"]) == 64
    assert report["dataset_id"] == "dgn-open-tiny-v1"
