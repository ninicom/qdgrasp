from __future__ import annotations

from unittest import mock

import pytest
import torch
import yaml

from qdgrasp.config import (
    ConfigError,
    DataConfig,
    ModelConfig,
    RegistryError,
    RobotConfig,
    RunConfig,
    dump_document,
    get_model_builder,
    load_data_config,
    load_model_config,
    load_robot_config,
    parse_document,
    preset_names,
    registered_datasets,
    registered_models,
    resolve_runtime,
)


def test_presets_are_packaged() -> None:
    assert set(preset_names()) >= {"dummy-hand.yaml", "dummy-tiny.yaml", "qdgrasp-dummy-n.yaml"}


def test_documents_round_trip_through_yaml() -> None:
    loaders = (
        ("qdgrasp-dummy-n.yaml", load_model_config, ModelConfig),
        ("dummy-hand.yaml", load_robot_config, RobotConfig),
        ("dummy-tiny.yaml", load_data_config, DataConfig),
    )
    for reference, loader, model in loaders:
        original = loader(reference)
        reparsed = model.model_validate(yaml.safe_load(dump_document(original)))
        assert reparsed == original
        assert reparsed.content_hash() == original.content_hash()


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ConfigError, match="epochs"):
        parse_document({"schema": "qdgrasp/run/v1", "epochs": 3}, RunConfig, origin="test")


def test_unknown_schema_version_is_an_error() -> None:
    with pytest.raises(ConfigError, match="schema"):
        parse_document({"schema": "qdgrasp/run/v2"}, RunConfig, origin="test")


def test_dead_model_param_is_an_error() -> None:
    from qdgrasp.dummy import build_dummy_grasp

    model = ModelConfig.model_validate(
        {"schema": "qdgrasp/model/v1", "name": "x", "type": "dummy_grasp", "params": {"width": 4}}
    )
    with pytest.raises(ConfigError, match="unknown params"):
        build_dummy_grasp(model, load_robot_config("dummy-hand.yaml"))


def test_robot_profile_requires_finite_limits_for_every_joint() -> None:
    with pytest.raises(ConfigError, match="finite joint limits"):
        parse_document(
            {
                "schema": "qdgrasp/robot/v1",
                "name": "broken",
                "palm_link": "palm",
                "joints": ["a", "b"],
                "joint_limits": {"a": [0.0, 1.0]},
            },
            RobotConfig,
            origin="test",
        )


def test_registry_rejects_unregistered_names() -> None:
    assert "dummy_grasp" in registered_models()
    assert "dummy_points" in registered_datasets()
    with pytest.raises(RegistryError, match="unknown model type"):
        get_model_builder("arbitrary.module:Class")


def test_missing_document_is_rejected() -> None:
    with pytest.raises(ConfigError, match="neither a file nor a packaged preset"):
        load_model_config("does-not-exist.yaml")


def test_absolute_project_dir_is_rejected() -> None:
    with pytest.raises(ConfigError, match="relative path"):
        parse_document({"schema": "qdgrasp/run/v1", "project_dir": "/tmp/runs"}, RunConfig, origin="test")


def test_cpu_forces_amp_off_and_records_the_adjustment() -> None:
    runtime = resolve_runtime(RunConfig(device="cpu", amp=True))
    assert runtime.amp is False
    assert runtime.precision == "32-true"
    record = runtime.to_dict()
    assert record["requested"]["amp"] is True
    assert record["effective"]["amp"] is False
    assert [item["field"] for item in record["adjustments"]] == ["amp"]


def test_cuda_request_never_falls_back_to_cpu() -> None:
    with (
        mock.patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="CPU fallback is forbidden"),
    ):
        resolve_runtime(RunConfig(device="cuda:0"))


def test_cuda_index_beyond_device_count_is_rejected() -> None:
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.device_count", return_value=1),
        mock.patch("torch.cuda.get_device_name", return_value="Mock GPU"),
        mock.patch.object(torch.version, "cuda", "12.8"),pytest.raises(ConfigError, match="only 1 CUDA device")
    ):
        resolve_runtime(RunConfig(device="cuda:3"))


def test_data_config_loads_and_hashes_stably() -> None:
    first = load_data_config("dummy-tiny.yaml")
    second = load_data_config("dummy-tiny.yaml")
    assert first.content_hash() == second.content_hash()
