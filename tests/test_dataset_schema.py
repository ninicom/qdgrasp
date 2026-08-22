from __future__ import annotations

import pytest
import yaml

from qdgrasp.config import ConfigError, dump_document, parse_document
from qdgrasp.config.schema import DATA_SCHEMA_V1, DataConfig
from qdgrasp.dataset.schema import DATA_SCHEMA_V2, DataConfigV2


def test_data_schema_v1_legacy_still_valid() -> None:
    doc = {
        "schema": "qdgrasp/data/v1",
        "name": "dummy_data",
        "type": "dummy",
        "params": {
            "num_samples": 32,
            "point_count": 512,
            "batch_size": 8,
        },
    }
    cfg = parse_document(doc, DataConfig, origin="test")
    assert cfg.schema_version == DATA_SCHEMA_V1
    assert cfg.name == "dummy_data"
    assert cfg.type == "dummy"


def test_data_schema_v2_round_trip() -> None:
    doc = {
        "schema": "qdgrasp/data/v2",
        "name": "dgn_open_test",
        "dataset_root": "datasets/dgn-open-tiny",
        "manifest_file": "dataset_manifest.json",
        "point_count": 1024,
        "batch_size": 16,
        "num_workers": 2,
        "pin_memory": True,
        "drop_last": False,
        "seed": 42,
        "robot_profiles": ["leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"],
    }
    cfg = parse_document(doc, DataConfigV2, origin="test")
    assert cfg.schema_version == DATA_SCHEMA_V2
    assert cfg.name == "dgn_open_test"
    assert cfg.batch_size == 16

    dumped = dump_document(cfg)
    reparsed = parse_document(yaml.safe_load(dumped), DataConfigV2, origin="round_trip")
    assert reparsed == cfg
    assert reparsed.content_hash() == cfg.content_hash()


def test_data_schema_v2_rejects_extra_keys() -> None:
    doc = {
        "schema": "qdgrasp/data/v2",
        "name": "bad_data",
        "dataset_root": "datasets/test",
        "unknown_extra_field": 123,
    }
    with pytest.raises(ConfigError):
        parse_document(doc, DataConfigV2, origin="test")
