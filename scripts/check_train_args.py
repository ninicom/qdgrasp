#!/usr/bin/env python3
"""Audit the pinned Ultralytics configuration surface against QDGrasp's registry.

The checker intentionally uses only the Python standard library. ``--registry-only``
is suitable for hooks and source archives that do not contain ``.references``.
The full mode additionally verifies the pinned Git HEAD, source hashes, all 115
canonical defaults/groups/type sets, two custom keys, nine legacy names and the
``trainer`` API control directly from the cloned Ultralytics source.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_REPOSITORY = "https://github.com/ultralytics/ultralytics.git"
PINNED_COMMIT = "329682a29d27203582ba30e519340f95abccc6a6"
PINNED_DEFAULT_SHA256 = "eb5e9ab6825a5d55076f8b38aed00953dec722ed5d5368a6584df35f50f32839"
PINNED_CFG_SHA256 = "ea9a98b498686f44561e9ce9892aa0e7de52226429496b08a20475ebbfa4eed8"
PINNED_MODEL_API_SHA256 = "69cb4c9c6f572c50bd35328e5154e6b596ed50dc10d218584da8aed3f7c9b32a"
PINNED_TRAINER_SHA256 = "d02bfd82d2af38fb58a6fe7903ef0cb9a93633a75b3db778768896d6c57a32e6"
# These two values are updated together after the schema-v2 registry and
# normative Markdown body have reached their final reviewed form.  Keeping the
# checker-side pins prevents a coordinated registry/document edit from silently
# redefining the accepted contract.
PINNED_MANIFEST_FINGERPRINT = "0efcc5be32bd033afc6db2475b93f96b40e4378be3077e300bbb7c456b915ab4"
PINNED_MARKDOWN_BODY_SHA256 = "8e9e9702d8391eb5cfb6be9100df00ac9885e4c369a7143da4c63e529c459856"
EXPECTED_MARKDOWN_METADATA = {
    "document_id": "TRAIN-ARGS-ULTRALYTICS-8.4.125",
    "document_type": "registry",
    "title": "Registry đầy đủ tham số train tương thích Ultralytics v8.4.125",
    "version": "1.4.0",
    "status": "active",
    "date": "2026-08-22",
    "revises": "none",
    "related_plan": "PLAN-V2",
    "source_commit": PINNED_COMMIT,
    "source_sha256": PINNED_DEFAULT_SHA256,
    "latest_revision_record": (
        "docs/revisions/REV-20260822-003-raw-scalar-fsmonitor-hardening.md"
    ),
}
EXPECTED_MARKDOWN_REVISION_ID = "REV-20260822-003"
EXPECTED_MARKDOWN_SESSION_ID = "SESSION-20260822-003"
ALLOWED_MARKDOWN_REVISION_STATUSES = {"draft", "in_review", "complete"}
PINNED_COUNTS = {
    "canonical_key_count": 115,
    "extra_config_key_count": 2,
    "legacy_key_count": 9,
    "api_control_count": 1,
    "total_public_names": 127,
}
EXPECTED_EXTRA_KEYS = {"augmentations", "save_dir"}
EXPECTED_LEGACY_KEYS = {
    "boxes",
    "hide_labels",
    "hide_conf",
    "line_thickness",
    "half",
    "int8",
    "label_smoothing",
    "save_hybrid",
    "crop_fraction",
}
EXPECTED_API_CONTROLS = {"trainer"}
EXPECTED_TOP_FIELDS = {
    "schema_version",
    "registry_id",
    "status",
    "generated_at",
    "related_plan",
}
EXPECTED_UPSTREAM_FIELDS = {
    "repository",
    "tag",
    "commit",
    "default_config_path",
    "default_config_sha256",
    "config_validator_path",
    "config_validator_sha256",
    "model_api_path",
    "model_api_sha256",
    "trainer_path",
    "trainer_sha256",
    "canonical_key_count",
    "extra_config_key_count",
    "legacy_key_count",
    "api_control_count",
    "total_public_names",
    "canonical_disposition_counts",
    "public_disposition_counts",
    "documentation_body_sha256",
    "manifest_fingerprint",
}
PINNED_UPSTREAM_VALUES = {
    "repository": PINNED_REPOSITORY,
    "tag": "v8.4.125",
    "commit": PINNED_COMMIT,
    "default_config_path": "ultralytics/cfg/default.yaml",
    "default_config_sha256": PINNED_DEFAULT_SHA256,
    "config_validator_path": "ultralytics/cfg/__init__.py",
    "config_validator_sha256": PINNED_CFG_SHA256,
    "model_api_path": "ultralytics/engine/model.py",
    "model_api_sha256": PINNED_MODEL_API_SHA256,
    "trainer_path": "ultralytics/engine/trainer.py",
    "trainer_sha256": PINNED_TRAINER_SHA256,
    "canonical_disposition_counts": "retain=31,adapt=27,defer=8,reject=49",
    "public_disposition_counts": "retain=32,adapt=34,defer=8,reject=53",
    "documentation_body_sha256": PINNED_MARKDOWN_BODY_SHA256,
}
EXPECTED_EXTRA_CONTRACTS = {
    "augmentations": {
        "source": "allowed_custom_keys",
        "type": "python_list_or_named_yaml_list",
        "disposition": "adapt",
        "dexgrasp_key": "augmentations",
        "train_role": "augmentation",
    },
    "save_dir": {
        "source": "allowed_custom_keys",
        "type": "path",
        "disposition": "retain",
        "dexgrasp_key": "save_dir",
        "train_role": "train_core",
    },
}
EXPECTED_LEGACY_CONTRACTS = {
    "boxes": {"source": "deprecated_mapping", "target": "show_boxes", "behavior": "alias", "disposition": "reject"},
    "hide_labels": {"source": "deprecated_mapping", "target": "show_labels", "behavior": "inverted_alias", "disposition": "adapt"},
    "hide_conf": {"source": "deprecated_mapping", "target": "show_conf", "behavior": "inverted_alias", "disposition": "adapt"},
    "line_thickness": {"source": "deprecated_mapping", "target": "line_width", "behavior": "alias", "disposition": "adapt"},
    "half": {"source": "precision_alias", "target": "quantize", "behavior": "value_16", "disposition": "adapt"},
    "int8": {"source": "precision_alias", "target": "quantize", "behavior": "value_8", "disposition": "adapt"},
    "label_smoothing": {"source": "removed_keys", "target": "null", "behavior": "warn_and_remove", "disposition": "reject"},
    "save_hybrid": {"source": "removed_keys", "target": "null", "behavior": "warn_and_remove", "disposition": "reject"},
    "crop_fraction": {"source": "removed_keys", "target": "null", "behavior": "warn_and_remove", "disposition": "reject"},
}
EXPECTED_API_CONTRACTS = {
    "trainer": {
        "source": "Model.train_signature",
        "type": "trainer_or_null",
        "disposition": "adapt",
        "dexgrasp_key": "trainer",
    }
}
EXPECTED_EXTENSION_CONTRACTS = {
    "robot": {"status": "required_design", "type": "path_or_profile", "purpose": "robot_profile"},
    "sim": {"status": "required_design", "type": "enum", "purpose": "validation_backend"},
    "max_steps": {"status": "required_design", "type": "integer_or_null", "purpose": "step_budget"},
    "points_per_scene": {"status": "required_design", "type": "integer", "purpose": "raw_point_budget"},
    "voxel_size": {"status": "required_design", "type": "positive_number", "purpose": "tokenizer_resolution"},
    "max_tokens": {"status": "required_design", "type": "integer", "purpose": "active_token_budget"},
    "flow_steps": {"status": "required_design", "type": "positive_integer", "purpose": "wrist_solver_steps"},
    "topk": {"status": "required_design", "type": "positive_integer", "purpose": "returned_grasps"},
    "radius": {"status": "required_design", "type": "positive_number", "purpose": "seed_suppression_radius"},
    "rotation_aug_deg": {"status": "required_design", "type": "nonnegative_number", "purpose": "se3_augmentation"},
    "translation_aug_m": {"status": "required_design", "type": "nonnegative_number", "purpose": "se3_augmentation"},
    "point_jitter_std": {"status": "required_design", "type": "nonnegative_number", "purpose": "point_augmentation"},
    "point_dropout": {"status": "required_design", "type": "fraction", "purpose": "point_augmentation"},
    "loss_objectness": {"status": "required_design", "type": "nonnegative_number", "purpose": "loss_gain"},
    "loss_graspness": {"status": "required_design", "type": "nonnegative_number", "purpose": "loss_gain"},
    "loss_flow": {"status": "required_design", "type": "nonnegative_number", "purpose": "loss_gain"},
    "loss_joint": {"status": "required_design", "type": "nonnegative_number", "purpose": "loss_gain"},
    "loss_quality": {"status": "required_design", "type": "nonnegative_number", "purpose": "loss_gain"},
    "quality_negatives": {"status": "required_design", "type": "enum", "purpose": "quality_sampling"},
}
EXPECTED_MERGE_CONTRACT = {
    "get_cfg_priority": "base_then_overrides",
    "model_train_without_cfg": "self_overrides_then_method_defaults_data_model_task_then_kwargs_then_forced_mode_train",
    "model_train_with_cfg": "loaded_cfg_replaces_self_overrides_then_method_defaults_data_model_task_then_kwargs_then_forced_mode_train",
    "cfg_data_special_case": "loaded_cfg_data_is_method_default_unless_kwargs_data",
    "model_task_authority": "current_model_and_task_override_cfg_values",
    "unknown_key_policy": "reject_with_suggestion",
    "typed_key_count": "93",
    "centrally_untyped_key_count": "22",
}
EXPECTED_CONSTRAINTS = {
    "float_keys": "integer_or_float",
    "fraction_keys": "closed_0_1",
    "dataset_fraction": "open_0_closed_1",
    "int_keys": "integer_only",
    "typed_none": "rejected_when_pinned_default_is_nonnull_except_auto_augment",
    "int_minimums": "mask_ratio_1,max_det_1,nbs_1,seed_0,vid_stride_1",
    "scale": "scalar_closed_0_1_or_exactly_two_numbers_pair_range_unchecked",
    "compile": "boolean_or_string_mode_not_centrally_enumerated",
    "quantize": "null_or_8_16_32_int8_fp16_fp32_w8a8_w16a16_w32a32_w8a16_w8a32",
    "quantize_aliases": "8=8,16=16,32=32,int8=8,fp16=16,fp32=32,w8a8=8,w16a16=16,w32a32=32,w8a16=w8a16,w8a32=w8a32",
    "auto_augment": "string_or_null_downstream_randaugment_augmix_autoaugment",
    "copy_paste_mode": "downstream_flip_or_mixup",
    "optimizer": "downstream_SGD_MuSGD_Adam_Adamax_AdamW_NAdam_RAdam_RMSProp_auto",
    "batch_runtime": "integer_or_cuda_single_device_fraction",
}
PINNED_RESUME_MUTABLE_KEYS = {
    "augmentations", "batch", "cache", "close_mosaic", "device", "distill_model",
    "freeze", "imgsz", "patience", "plots", "save_dir", "save_period", "time",
    "val", "workers",
}

STRUCTURED_SECTIONS = {
    "canonical_arguments",
    "extra_config_arguments",
    "legacy_arguments",
    "api_controls",
    "dexgrasp_extensions",
}
SCALAR_SECTIONS = {
    "upstream",
    "upstream_validation",
    "upstream_merge_contract",
    "upstream_constraints",
}
CANONICAL_FIELDS = {
    "default",
    "group",
    "type",
    "train_role",
    "disposition",
    "dexgrasp_key",
    "device",
}
GROUPS = {
    "global",
    "train",
    "segmentation",
    "classification",
    "validation",
    "predict",
    "visualize",
    "export",
    "hyperparameters",
    "custom",
    "tracker",
}
TRAIN_ROLES = {
    "routing",
    "train_core",
    "validation",
    "optimization",
    "augmentation",
    "task_specific",
    "lifecycle_only",
    "export_only",
    "compatibility_only",
}
DISPOSITIONS = {"retain", "adapt", "defer", "reject"}
DEVICE_POLICIES = {
    "portable",
    "cpu_adjust",
    "portable_guarded",
    "mode_specific",
    "not_applicable",
}
DISPOSITION_LETTERS = {"retain": "R", "adapt": "A", "defer": "D", "reject": "X"}
VALIDATION_FIELDS = {
    "cfg_float_keys",
    "cfg_fraction_keys",
    "cfg_int_keys",
    "cfg_int_min",
    "cfg_bool_keys",
    "cfg_str_keys",
    "special_keys",
    "centrally_untyped_keys",
    "resume_mutable_keys",
}
VALIDATION_SET_FIELDS = VALIDATION_FIELDS - {"cfg_int_min"}
SOURCE_GROUPS = {
    "Train settings": "train",
    "Segmentation": "segmentation",
    "Classification": "classification",
    "Val/Test settings": "validation",
    "Predict settings": "predict",
    "Visualize settings": "visualize",
    "Export settings": "export",
    "Hyperparameters": "hyperparameters",
    "Custom config.yaml": "custom",
    "Tracker settings": "tracker",
}
VALIDATOR_NAMES = {
    "CFG_FLOAT_KEYS": "cfg_float_keys",
    "CFG_FRACTION_KEYS": "cfg_fraction_keys",
    "CFG_INT_KEYS": "cfg_int_keys",
    "CFG_INT_MIN": "cfg_int_min",
    "CFG_BOOL_KEYS": "cfg_bool_keys",
    "CFG_STR_KEYS": "cfg_str_keys",
}
EXPECTED_TYPES_BY_VALIDATOR = {
    "cfg_float_keys": "number",
    "cfg_fraction_keys": "fraction",
    "cfg_int_keys": "integer",
    "cfg_bool_keys": "boolean",
    "cfg_str_keys": "string",
}
SPECIAL_TYPES = {
    "scale": "number_or_pair",
    "compile": "boolean_or_string",
    "quantize": "precision_or_null",
}
MARKDOWN_TYPE_LABELS = {
    "boolean": "boolean",
    "boolean_or_enum": "bool/enum",
    "boolean_or_null": "bool/null",
    "boolean_or_path": "bool/path",
    "boolean_or_string": "bool/string",
    "device_selector": "selector",
    "enum": "enum",
    "fraction_or_null": "`[0,1]`/null",
    "integer_or_list": "int/list",
    "integer_or_list_or_null": "int/list/null",
    "integer_or_null": "int/null",
    "list_or_null": "list/null",
    "number": "number",
    "number_or_null": "number/null",
    "number_or_pair": "number/pair",
    "path_or_null": "path/null",
    "path_or_string": "path/string",
    "precision_or_null": "precision/null",
    "source_or_null": "source/null",
    "string": "string",
    "string_or_null": "string/null",
}
# TRAIN_ARGUMENTS.yaml deliberately uses a much smaller language than general
# YAML.  Exact separators make the parser agree with YAML mapping semantics
# without implementing YAML's quoted strings, escapes, implicit tags or flow
# collection corner cases.  Source default.yaml and Markdown front matter keep
# separate, intentionally flexible parsers below.
REGISTRY_TOP_SECTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):$")
REGISTRY_TOP_SCALAR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*): (.+)$")
REGISTRY_ENTRY_RE = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*): (\{.*\})$")
REGISTRY_NESTED_SCALAR_RE = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*): (.+)$")
REGISTRY_INLINE_FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*): (.+)$")
SOURCE_TOP_LEVEL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:[ \t]*(.*))?$")
MARKDOWN_METADATA_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*): ([^\r\n]+)$")
MARKDOWN_NESTED_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):$")
SCALAR_TOKEN_RE = re.compile(r"^[A-Za-z0-9~-][A-Za-z0-9._:/=,+|()~-]*$")
INTEGER_SCALAR_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
FLOAT_SCALAR_RE = re.compile(
    r"^-?(?:(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][+-]?[0-9]+)?|"
    r"(?:0|[1-9][0-9]*)[eE][+-]?[0-9]+)$"
)


@dataclass(frozen=True)
class Problem:
    message: str
    line: int = 1


@dataclass(frozen=True)
class ScalarSignature:
    """YAML-relevant scalar kind plus its exact source lexeme."""

    kind: str
    lexeme: str


@dataclass
class Registry:
    path: Path
    top: dict[str, str] = field(default_factory=dict)
    scalars: dict[str, dict[str, str]] = field(default_factory=dict)
    entries: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    entry_lines: dict[tuple[str, str], int] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)


@dataclass(frozen=True)
class SourceArgument:
    default: str
    group: str
    line: int


def scalar_signature(value: str) -> ScalarSignature:
    """Classify a strict plain scalar without converting away its lexeme."""

    if value == "null":
        kind = "null"
    elif value in {"True", "False"}:
        kind = "boolean"
    elif INTEGER_SCALAR_RE.fullmatch(value):
        kind = "integer"
    elif FLOAT_SCALAR_RE.fullmatch(value):
        kind = "float"
    else:
        kind = "string"
    return ScalarSignature(kind=kind, lexeme=value)


def scalar_problem(value: str, context: str, line: int) -> Problem | None:
    """Validate the deliberately restricted scalar subset used by this registry."""

    if not value:
        return Problem(f"{context} không được rỗng", line)
    if value != value.strip():
        return Problem(f"{context} có whitespace bao quanh scalar", line)
    if value.startswith(("'", '"')) or value.endswith(("'", '"')):
        return Problem(
            f"{context} không được dùng quoted scalar vì dấu nháy đổi kiểu YAML: {value!r}",
            line,
        )
    if not SCALAR_TOKEN_RE.fullmatch(value):
        return Problem(
            f"{context} không thuộc YAML scalar subset an toàn: {value!r}", line
        )
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_inline_mapping(raw: str, line: int) -> tuple[dict[str, str], list[Problem]]:
    problems: list[Problem] = []
    if not raw.startswith("{") or not raw.endswith("}"):
        return {}, [Problem("entry phải là inline mapping {...}", line)]
    body = raw[1:-1]
    if not body:
        return {}, [Problem("inline mapping không được rỗng", line)]
    if body != body.strip():
        return {}, [Problem("inline mapping không được có whitespace sát dấu ngoặc", line)]

    values: dict[str, str] = {}
    for item in body.split(", "):
        match = REGISTRY_INLINE_FIELD_RE.fullmatch(item)
        if not match:
            problems.append(
                Problem(
                    "field inline phải dùng chính xác 'key: value' và phân cách ', ': "
                    f"{item!r}",
                    line,
                )
            )
            continue
        key, value = match.groups()
        if key in values:
            problems.append(Problem(f"field trùng trong entry: {key}", line))
        else:
            values[key] = value
            if problem := scalar_problem(value, f"field '{key}'", line):
                problems.append(problem)
    return values, problems


def parse_registry(path: Path) -> Registry:
    registry = Registry(path=path)
    current_section: str | None = None
    seen_sections: set[str] = set()

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        registry.problems.append(Problem(f"không đọc được registry UTF-8: {exc}"))
        return registry

    for line_number, raw_line in enumerate(lines, start=1):
        if "\t" in raw_line:
            registry.problems.append(
                Problem("registry dialect không cho phép tab", line_number)
            )
            continue
        if raw_line != raw_line.rstrip(" "):
            registry.problems.append(
                Problem("registry dialect không cho phép whitespace cuối dòng", line_number)
            )
            continue
        if not raw_line or raw_line.lstrip(" ").startswith("#"):
            continue

        section_match = REGISTRY_TOP_SECTION_RE.fullmatch(raw_line)
        if section_match:
            key = section_match.group(1)
            if key in seen_sections:
                registry.problems.append(Problem(f"section trùng: {key}", line_number))
            seen_sections.add(key)
            current_section = key
            if key in STRUCTURED_SECTIONS:
                registry.entries.setdefault(key, {})
            elif key in SCALAR_SECTIONS:
                registry.scalars.setdefault(key, {})
            else:
                registry.problems.append(Problem(f"section không được hỗ trợ: {key}", line_number))
            continue

        top_match = REGISTRY_TOP_SCALAR_RE.fullmatch(raw_line)
        if top_match:
            key, value = top_match.groups()
            if key in registry.top:
                registry.problems.append(Problem(f"metadata trùng: {key}", line_number))
            else:
                registry.top[key] = value
                if problem := scalar_problem(value, f"metadata '{key}'", line_number):
                    registry.problems.append(problem)
            current_section = None
            continue

        if current_section in STRUCTURED_SECTIONS:
            entry_match = REGISTRY_ENTRY_RE.fullmatch(raw_line)
            if not entry_match:
                registry.problems.append(
                    Problem(f"dòng entry không hợp lệ trong {current_section}", line_number)
                )
                continue
            key, raw_mapping = entry_match.groups()
            section_entries = registry.entries[current_section]
            if key in section_entries:
                registry.problems.append(
                    Problem(f"entry trùng trong {current_section}: {key}", line_number)
                )
                continue
            mapping, mapping_problems = parse_inline_mapping(raw_mapping, line_number)
            section_entries[key] = mapping
            registry.entry_lines[(current_section, key)] = line_number
            registry.problems.extend(mapping_problems)
            continue

        if current_section in SCALAR_SECTIONS:
            scalar_match = REGISTRY_NESTED_SCALAR_RE.fullmatch(raw_line)
            if not scalar_match:
                registry.problems.append(
                    Problem(f"dòng scalar không hợp lệ trong {current_section}", line_number)
                )
                continue
            key, value = scalar_match.groups()
            section_values = registry.scalars[current_section]
            if key in section_values:
                registry.problems.append(
                    Problem(f"field trùng trong {current_section}: {key}", line_number)
                )
            elif not value:
                registry.problems.append(
                    Problem(f"field '{current_section}.{key}' không được rỗng", line_number)
                )
            else:
                section_values[key] = value
                if problem := scalar_problem(
                    value, f"field '{current_section}.{key}'", line_number
                ):
                    registry.problems.append(problem)
            continue

        registry.problems.append(Problem("nội dung nằm ngoài section hợp lệ", line_number))

    return registry


def manifest_fingerprint(registry: Registry) -> str:
    """Hash all registry semantics with explicit scalar kinds (schema v2)."""

    def typed_fields(values: dict[str, str]) -> dict[str, dict[str, str]]:
        return {
            key: {
                "kind": scalar_signature(value).kind,
                "lexeme": scalar_signature(value).lexeme,
            }
            for key, value in sorted(values.items())
        }

    upstream = {
        key: value
        for key, value in registry.scalars.get("upstream", {}).items()
        if key != "manifest_fingerprint"
    }
    payload = {
        "fingerprint_schema": 2,
        "top": typed_fields(registry.top),
        "entries": {
            section: {
                key: typed_fields(fields)
                for key, fields in sorted(registry.entries.get(section, {}).items())
            }
            for section in sorted(STRUCTURED_SECTIONS)
        },
        "scalars": {
            section: typed_fields(
                upstream if section == "upstream" else registry.scalars.get(section, {})
            )
            for section in sorted(SCALAR_SECTIONS)
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_markdown_front_matter(
    text: str,
    *,
    context: str,
    allow_nested_values: bool = False,
) -> tuple[dict[str, str], list[Problem]]:
    """Parse top-level front matter without hiding malformed or duplicate fields."""

    lines = text.splitlines()
    problems: list[Problem] = []
    if not lines or lines[0] != "---":
        return {}, [Problem(f"{context} thiếu dòng mở front matter chính xác '---'")]
    try:
        closing = next(index for index, line in enumerate(lines[1:], 1) if line == "---")
    except StopIteration:
        return {}, [Problem(f"{context} thiếu dòng đóng front matter chính xác '---'")]

    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:closing], 2):
        if not line:
            continue
        if allow_nested_values and line[0].isspace():
            continue
        match = MARKDOWN_METADATA_RE.fullmatch(line)
        if match:
            key, value = match.groups()
        else:
            nested_match = MARKDOWN_NESTED_KEY_RE.fullmatch(line)
            if allow_nested_values and nested_match:
                key = nested_match.group(1)
                value = ""
            else:
                problems.append(
                    Problem(f"{context} metadata malformed: {line!r}", index)
                )
                continue
        if key in metadata:
            problems.append(Problem(f"{context} metadata trùng: '{key}'", index))
            continue
        metadata[key] = value
    return metadata, problems


def parse_markdown_metadata(text: str) -> dict[str, str]:
    """Compatibility wrapper returning top-level Markdown front-matter values."""

    metadata, _ = parse_markdown_front_matter(text, context="Markdown")
    return metadata


def validate_revision_pointer(metadata: dict[str, str]) -> list[Problem]:
    """Resolve and validate the revision record named by registry front matter."""

    pointer = metadata.get("latest_revision_record")
    if not pointer:
        return []

    project_root = PROJECT_ROOT.resolve()
    revision_path = (project_root / pointer).resolve()
    try:
        revision_path.relative_to(project_root)
    except ValueError:
        return [Problem(f"latest_revision_record thoát project root: {pointer!r}")]
    if not revision_path.is_file():
        return [Problem(f"latest_revision_record không tồn tại: {pointer!r}")]
    try:
        revision_text = revision_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [Problem(f"không đọc được latest_revision_record {pointer!r}: {exc}")]

    revision_metadata, problems = parse_markdown_front_matter(
        revision_text,
        context="Revision record",
        allow_nested_values=True,
    )
    expected_identity = {
        "document_id": EXPECTED_MARKDOWN_REVISION_ID,
        "record_id": EXPECTED_MARKDOWN_REVISION_ID,
        "document_type": "revision_record",
        "session_id": EXPECTED_MARKDOWN_SESSION_ID,
    }
    for key, expected in expected_identity.items():
        actual = revision_metadata.get(key)
        if actual != expected:
            problems.append(
                Problem(
                    f"Revision record metadata '{key}' phải là {expected!r}, "
                    f"nhận {actual!r}"
                )
            )
    status = revision_metadata.get("status")
    if status not in ALLOWED_MARKDOWN_REVISION_STATUSES:
        problems.append(
            Problem(
                "Revision record metadata 'status' phải thuộc "
                f"{sorted(ALLOWED_MARKDOWN_REVISION_STATUSES)}, nhận {status!r}"
            )
        )
    return problems


def normalized_markdown_body(text: str) -> str | None:
    """Return a platform-stable but otherwise exact body after front matter."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        closing = next(
            index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"
        )
    except StopIteration:
        return None
    body_lines = lines[closing + 1 :]
    while body_lines and body_lines[0] == "":
        body_lines.pop(0)
    while body_lines and body_lines[-1] == "":
        body_lines.pop()
    return "\n".join(body_lines) + "\n"


def markdown_body_sha256(text: str) -> str | None:
    body = normalized_markdown_body(text)
    if body is None:
        return None
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def expected_markdown_type(
    key: str, fields: dict[str, str], registry: Registry
) -> str | None:
    """Derive the exact human-table type label from the machine contract."""

    type_name = fields.get("type")
    if type_name == "fraction":
        return "`(0,1]`" if key == "fraction" else "`[0,1]`"
    if type_name == "integer":
        try:
            minimums = key_value_map(
                registry.scalars.get("upstream_validation", {}).get("cfg_int_min", "")
            )
        except (TypeError, ValueError):
            minimums = {}
        minimum = minimums.get(key)
        return f"integer ≥{minimum}" if minimum is not None else "integer"
    return MARKDOWN_TYPE_LABELS.get(type_name or "")


def validate_documentation(registry: Registry, path: Path) -> tuple[list[Problem], bool]:
    """Keep the exhaustive human tables synchronized with the machine registry."""

    if not path.is_file():
        return [Problem(f"train-argument documentation không tồn tại: {path}")], True
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [Problem(f"không đọc được train-argument documentation: {exc}")], True

    metadata, problems = parse_markdown_front_matter(
        text, context="Markdown"
    )
    actual_fields = set(metadata)
    expected_fields = set(EXPECTED_MARKDOWN_METADATA)
    if actual_fields != expected_fields:
        problems.append(
            Problem(
                "Markdown metadata sai schema: "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"extra={sorted(actual_fields - expected_fields)}"
            )
        )
    for key, expected in EXPECTED_MARKDOWN_METADATA.items():
        actual = metadata.get(key)
        if actual != expected:
            problems.append(
                Problem(
                    f"Markdown metadata '{key}' phải là {expected!r}, nhận {actual!r}"
                )
            )
    problems.extend(validate_revision_pointer(metadata))

    computed_body_hash = markdown_body_sha256(text)
    recorded_body_hash = registry.scalars.get("upstream", {}).get(
        "documentation_body_sha256"
    )
    if computed_body_hash is None:
        problems.append(Problem("Markdown không có front matter hoàn chỉnh để hash body"))
    elif computed_body_hash != PINNED_MARKDOWN_BODY_SHA256:
        problems.append(
            Problem(
                "Markdown normative body SHA-256 sai: "
                f"expected={PINNED_MARKDOWN_BODY_SHA256}, actual={computed_body_hash}"
            )
        )
    if recorded_body_hash != PINNED_MARKDOWN_BODY_SHA256:
        problems.append(
            Problem(
                "upstream.documentation_body_sha256 không khớp checker pin: "
                f"expected={PINNED_MARKDOWN_BODY_SHA256}, actual={recorded_body_hash}"
            )
        )
    if (
        computed_body_hash is not None
        and recorded_body_hash is not None
        and computed_body_hash != recorded_body_hash
    ):
        problems.append(
            Problem(
                "Markdown normative body không khớp hash ghi trong registry: "
                f"registry={recorded_body_hash}, actual={computed_body_hash}"
            )
        )

    canonical_rows: dict[str, tuple[str, str, str]] = {}
    row_pattern = re.compile(
        r"^\| `([A-Za-z_][A-Za-z0-9_]*)` \| `([^`]*)` \| ([^|]*?) \| ([RADX]) \|",
        re.MULTILINE,
    )
    duplicates: set[str] = set()
    for key, default, type_label, disposition in row_pattern.findall(text):
        if key in canonical_rows:
            duplicates.add(key)
        canonical_rows[key] = (default, type_label.strip(), disposition)
    canonical = registry.entries.get("canonical_arguments", {})
    if duplicates:
        problems.append(Problem(f"Markdown canonical rows trùng: {sorted(duplicates)}"))
    if set(canonical_rows) != set(canonical):
        problems.append(
            Problem(
                "Markdown canonical rows sai set: "
                f"missing={sorted(set(canonical) - set(canonical_rows))}, "
                f"extra={sorted(set(canonical_rows) - set(canonical))}"
            )
        )
    for key in sorted(set(canonical_rows) & set(canonical)):
        documented_default, documented_type, documented_disposition = canonical_rows[key]
        fields = canonical[key]
        if documented_default != fields.get("default"):
            problems.append(
                Problem(
                    f"Markdown default '{key}' sai: {documented_default!r} != {fields.get('default')!r}"
                )
            )
        expected_type = expected_markdown_type(key, fields, registry)
        if expected_type is None:
            problems.append(
                Problem(
                    f"Markdown type '{key}' không có mapping cho {fields.get('type')!r}"
                )
            )
        elif documented_type != expected_type:
            problems.append(
                Problem(
                    f"Markdown type '{key}' sai: {documented_type!r} != {expected_type!r}"
                )
            )
        expected_letter = DISPOSITION_LETTERS.get(fields.get("disposition", ""))
        if documented_disposition != expected_letter:
            problems.append(
                Problem(
                    f"Markdown disposition '{key}' sai: {documented_disposition!r} != {expected_letter!r}"
                )
            )

    section_match = re.search(r"^## 8\..*?(?=^## 9\.)", text, re.MULTILINE | re.DOTALL)
    if not section_match:
        problems.append(Problem("Markdown thiếu mục 8 cho extra/legacy/API names"))
    else:
        section = section_match.group(0)
        surface_rows: dict[str, str] = {}
        surface_duplicates: set[str] = set()
        for line in section.splitlines():
            match = re.match(
                r"^\| `([A-Za-z_][A-Za-z0-9_]*)` \| [^|]* \| ([RADX])(?:\b|:)", line
            )
            if not match:
                continue
            key, letter = match.groups()
            if key in surface_rows:
                surface_duplicates.add(key)
            surface_rows[key] = letter
        expected_surface = {
            **registry.entries.get("extra_config_arguments", {}),
            **registry.entries.get("legacy_arguments", {}),
            **registry.entries.get("api_controls", {}),
        }
        if surface_duplicates:
            problems.append(Problem(f"Markdown surface rows trùng: {sorted(surface_duplicates)}"))
        if set(surface_rows) != set(expected_surface):
            problems.append(
                Problem(
                    "Markdown extra/legacy/API rows sai set: "
                    f"missing={sorted(set(expected_surface) - set(surface_rows))}, "
                    f"extra={sorted(set(surface_rows) - set(expected_surface))}"
                )
            )
        for key in sorted(set(surface_rows) & set(expected_surface)):
            expected_letter = DISPOSITION_LETTERS[expected_surface[key]["disposition"]]
            if surface_rows[key] != expected_letter:
                problems.append(
                    Problem(
                        f"Markdown surface disposition '{key}' sai: "
                        f"{surface_rows[key]} != {expected_letter}"
                    )
                )

    normalized = " ".join(text.split())
    summary_fragments = (
        "Riêng 115 canonical key: 31 `R`, 27 `A`, 8 `D`, 49 `X`.",
        "toàn bộ 127 public names là 32 `R`, 34 `A`, 8 `D`, 53 `X`.",
    )
    for fragment in summary_fragments:
        if fragment not in normalized:
            problems.append(Problem(f"Markdown thiếu disposition summary: {fragment}"))
    return problems, False


def csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def key_value_map(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        key, separator, raw_number = item.strip().partition("=")
        if not separator:
            raise ValueError(f"thiếu '=' trong {item!r}")
        result[key] = int(raw_number)
    return result


def integer_field(values: dict[str, str], key: str, problems: list[Problem]) -> int | None:
    try:
        return int(values[key])
    except KeyError:
        problems.append(Problem(f"thiếu field: {key}"))
    except ValueError:
        problems.append(Problem(f"field '{key}' phải là integer"))
    return None


def validate_registry(registry: Registry) -> list[Problem]:
    problems = list(registry.problems)
    required_top = {
        "schema_version": "2",
        "registry_id": "DEXGRASP-TRAIN-ARGS-ULTRALYTICS-8.4.125",
        "status": "active",
        "generated_at": "2026-08-22",
        "related_plan": "PLAN-V2",
    }
    for key, expected in required_top.items():
        actual = registry.top.get(key)
        if actual is None:
            problems.append(Problem(f"thiếu metadata: {key}"))
        elif actual != expected:
            problems.append(Problem(f"metadata '{key}' phải là {expected!r}, nhận {actual!r}"))

    if set(registry.top) != EXPECTED_TOP_FIELDS:
        problems.append(
            Problem(
                "top-level metadata sai schema: "
                f"missing={sorted(EXPECTED_TOP_FIELDS - set(registry.top))}, "
                f"extra={sorted(set(registry.top) - EXPECTED_TOP_FIELDS)}"
            )
        )
    actual_sections = set(registry.entries) | set(registry.scalars)
    expected_sections = STRUCTURED_SECTIONS | SCALAR_SECTIONS
    if actual_sections != expected_sections:
        problems.append(
            Problem(
                "top-level sections sai schema: "
                f"missing={sorted(expected_sections - actual_sections)}, "
                f"extra={sorted(actual_sections - expected_sections)}"
            )
        )

    upstream = registry.scalars.get("upstream", {})
    if set(upstream) != EXPECTED_UPSTREAM_FIELDS:
        problems.append(
            Problem(
                "upstream fields sai schema: "
                f"missing={sorted(EXPECTED_UPSTREAM_FIELDS - set(upstream))}, "
                f"extra={sorted(set(upstream) - EXPECTED_UPSTREAM_FIELDS)}"
            )
        )
    for key, expected in PINNED_UPSTREAM_VALUES.items():
        actual = upstream.get(key)
        if actual != expected:
            problems.append(
                Problem(f"upstream.{key} phải pin {expected!r}, nhận {actual!r}")
            )
    for key, expected in PINNED_COUNTS.items():
        value = integer_field(upstream, key, problems)
        if value is not None and value != expected:
            problems.append(Problem(f"upstream.{key} phải là {expected}, nhận {value}"))

    computed_fingerprint = manifest_fingerprint(registry)
    recorded_fingerprint = upstream.get("manifest_fingerprint")
    if recorded_fingerprint != PINNED_MANIFEST_FINGERPRINT:
        problems.append(
            Problem(
                "upstream.manifest_fingerprint không khớp checker pin: "
                f"expected={PINNED_MANIFEST_FINGERPRINT}, actual={recorded_fingerprint}"
            )
        )
    if computed_fingerprint != PINNED_MANIFEST_FINGERPRINT:
        problems.append(
            Problem(
                "semantic manifest fingerprint sai: "
                f"expected={PINNED_MANIFEST_FINGERPRINT}, actual={computed_fingerprint}"
            )
        )

    canonical = registry.entries.get("canonical_arguments", {})
    extra = registry.entries.get("extra_config_arguments", {})
    legacy = registry.entries.get("legacy_arguments", {})
    api_controls = registry.entries.get("api_controls", {})
    extensions = registry.entries.get("dexgrasp_extensions", {})
    section_counts = {
        "canonical_arguments": (len(canonical), PINNED_COUNTS["canonical_key_count"]),
        "extra_config_arguments": (len(extra), PINNED_COUNTS["extra_config_key_count"]),
        "legacy_arguments": (len(legacy), PINNED_COUNTS["legacy_key_count"]),
        "api_controls": (len(api_controls), PINNED_COUNTS["api_control_count"]),
    }
    for section, (actual, expected) in section_counts.items():
        if actual != expected:
            problems.append(Problem(f"{section} có {actual} entry; yêu cầu {expected}"))

    if set(extra) != EXPECTED_EXTRA_KEYS:
        problems.append(
            Problem(
                "extra_config_arguments sai set: "
                f"missing={sorted(EXPECTED_EXTRA_KEYS - set(extra))}, "
                f"extra={sorted(set(extra) - EXPECTED_EXTRA_KEYS)}"
            )
        )
    if set(legacy) != EXPECTED_LEGACY_KEYS:
        problems.append(
            Problem(
                "legacy_arguments sai set: "
                f"missing={sorted(EXPECTED_LEGACY_KEYS - set(legacy))}, "
                f"extra={sorted(set(legacy) - EXPECTED_LEGACY_KEYS)}"
            )
        )
    if set(api_controls) != EXPECTED_API_CONTROLS:
        problems.append(
            Problem(
                "api_controls sai set: "
                f"missing={sorted(EXPECTED_API_CONTROLS - set(api_controls))}, "
                f"extra={sorted(set(api_controls) - EXPECTED_API_CONTROLS)}"
            )
        )

    exact_contracts = {
        "extra_config_arguments": (extra, EXPECTED_EXTRA_CONTRACTS),
        "legacy_arguments": (legacy, EXPECTED_LEGACY_CONTRACTS),
        "api_controls": (api_controls, EXPECTED_API_CONTRACTS),
        "dexgrasp_extensions": (extensions, EXPECTED_EXTENSION_CONTRACTS),
    }
    for section, (actual_contract, expected_contract) in exact_contracts.items():
        if actual_contract != expected_contract:
            problems.append(
                Problem(f"{section} contract khác pinned manifest")
            )

    for key, fields in canonical.items():
        line = registry.entry_lines.get(("canonical_arguments", key), 1)
        missing = CANONICAL_FIELDS - set(fields)
        extra_fields = set(fields) - CANONICAL_FIELDS
        if missing:
            problems.append(Problem(f"canonical '{key}' thiếu field: {sorted(missing)}", line))
        if extra_fields:
            problems.append(Problem(f"canonical '{key}' có field lạ: {sorted(extra_fields)}", line))
        if fields.get("group") not in GROUPS:
            problems.append(Problem(f"canonical '{key}' có group không hợp lệ", line))
        if fields.get("train_role") not in TRAIN_ROLES:
            problems.append(Problem(f"canonical '{key}' có train_role không hợp lệ", line))
        disposition = fields.get("disposition")
        if disposition not in DISPOSITIONS:
            problems.append(Problem(f"canonical '{key}' có disposition không hợp lệ", line))
        if fields.get("device") not in DEVICE_POLICIES:
            problems.append(Problem(f"canonical '{key}' có device policy không hợp lệ", line))
        dexgrasp_key = fields.get("dexgrasp_key")
        if disposition == "reject" and dexgrasp_key != "null":
            problems.append(Problem(f"canonical '{key}' reject phải có dexgrasp_key=null", line))
        if disposition in {"retain", "adapt", "defer"} and dexgrasp_key == "null":
            problems.append(Problem(f"canonical '{key}' {disposition} cần dexgrasp_key", line))

    total = len(canonical) + len(extra) + len(legacy) + len(api_controls)
    if total != PINNED_COUNTS["total_public_names"]:
        problems.append(Problem(f"tổng public names là {total}; yêu cầu 127"))
    overlap = (set(canonical) | set(extra) | set(legacy) | set(api_controls)) & set(extensions)
    if overlap:
        problems.append(Problem(f"QDGrasp extension trùng tên upstream: {sorted(overlap)}"))
    for key, fields in extensions.items():
        missing = {"status", "type", "purpose"} - set(fields)
        if missing:
            line = registry.entry_lines.get(("dexgrasp_extensions", key), 1)
            problems.append(Problem(f"extension '{key}' thiếu field: {sorted(missing)}", line))

    canonical_dispositions = {
        name: sum(
            fields.get("disposition") == name for fields in canonical.values()
        )
        for name in sorted(DISPOSITIONS)
    }
    public_entries = list(canonical.values()) + list(extra.values()) + list(legacy.values()) + list(api_controls.values())
    public_dispositions = {
        name: sum(fields.get("disposition") == name for fields in public_entries)
        for name in sorted(DISPOSITIONS)
    }
    expected_canonical_dispositions = {"retain": 31, "adapt": 27, "defer": 8, "reject": 49}
    expected_public_dispositions = {"retain": 32, "adapt": 34, "defer": 8, "reject": 53}
    if canonical_dispositions != expected_canonical_dispositions:
        problems.append(
            Problem(f"canonical disposition counts sai: {canonical_dispositions}")
        )
    if public_dispositions != expected_public_dispositions:
        problems.append(Problem(f"public disposition counts sai: {public_dispositions}"))

    validation = registry.scalars.get("upstream_validation", {})
    missing_validation = VALIDATION_FIELDS - set(validation)
    if missing_validation:
        problems.append(Problem(f"upstream_validation thiếu field: {sorted(missing_validation)}"))
    else:
        validation_sets = {key: csv_set(validation[key]) for key in VALIDATION_SET_FIELDS}
        typed_fields = {
            key: validation_sets[key]
            for key in EXPECTED_TYPES_BY_VALIDATOR
        }
        membership: dict[str, str] = {}
        for set_name, keys in typed_fields.items():
            unknown = keys - set(canonical)
            if unknown:
                problems.append(Problem(f"{set_name} chứa key ngoài canonical: {sorted(unknown)}"))
            for key in keys:
                if key in membership:
                    problems.append(
                        Problem(f"key '{key}' nằm trong cả {membership[key]} và {set_name}")
                    )
                membership[key] = set_name
                expected_type = EXPECTED_TYPES_BY_VALIDATOR[set_name]
                if canonical.get(key, {}).get("default") == "null":
                    expected_type = f"{expected_type}_or_null"
                if key == "auto_augment":
                    expected_type = "string_or_null"
                if canonical.get(key, {}).get("type") != expected_type:
                    problems.append(
                        Problem(
                            f"canonical '{key}' type phải là {expected_type} theo {set_name}"
                        )
                    )
        special = validation_sets["special_keys"]
        if special != set(SPECIAL_TYPES):
            problems.append(
                Problem(f"special_keys phải là {sorted(SPECIAL_TYPES)}, nhận {sorted(special)}")
            )
        for key, expected_type in SPECIAL_TYPES.items():
            if canonical.get(key, {}).get("type") != expected_type:
                problems.append(Problem(f"canonical '{key}' type phải là {expected_type}"))

        expected_untyped = set(canonical) - set(membership) - special
        if validation_sets["centrally_untyped_keys"] != expected_untyped:
            problems.append(
                Problem(
                    "centrally_untyped_keys không khớp canonical trừ typed/special: "
                    f"missing={sorted(expected_untyped - validation_sets['centrally_untyped_keys'])}, "
                    f"extra={sorted(validation_sets['centrally_untyped_keys'] - expected_untyped)}"
                )
            )
        resume_keys = validation_sets["resume_mutable_keys"]
        allowed_resume = set(canonical) | set(extra)
        if not resume_keys or not resume_keys <= allowed_resume:
            problems.append(
                Problem(f"resume_mutable_keys chứa key lạ: {sorted(resume_keys - allowed_resume)}")
            )
        if resume_keys != PINNED_RESUME_MUTABLE_KEYS:
            problems.append(
                Problem(
                    "resume_mutable_keys khác pinned source: "
                    f"missing={sorted(PINNED_RESUME_MUTABLE_KEYS - resume_keys)}, "
                    f"extra={sorted(resume_keys - PINNED_RESUME_MUTABLE_KEYS)}"
                )
            )
        try:
            minimums = key_value_map(validation["cfg_int_min"])
        except (ValueError, TypeError) as exc:
            problems.append(Problem(f"cfg_int_min không hợp lệ: {exc}"))
        else:
            if not set(minimums) <= validation_sets["cfg_int_keys"]:
                problems.append(Problem("cfg_int_min chứa key không nằm trong cfg_int_keys"))

        merge_contract = registry.scalars.get("upstream_merge_contract", {})
        if merge_contract != EXPECTED_MERGE_CONTRACT:
            problems.append(Problem("upstream_merge_contract khác pinned contract"))
        typed_count = integer_field(merge_contract, "typed_key_count", problems)
        untyped_count = integer_field(merge_contract, "centrally_untyped_key_count", problems)
        calculated_typed = len(membership) + len(special)
        if typed_count is not None and typed_count != calculated_typed:
            problems.append(
                Problem(f"typed_key_count={typed_count}, nhưng registry tính được {calculated_typed}")
            )
        if untyped_count is not None and untyped_count != len(expected_untyped):
            problems.append(
                Problem(
                    f"centrally_untyped_key_count={untyped_count}, "
                    f"nhưng registry tính được {len(expected_untyped)}"
                )
            )

        constraints = registry.scalars.get("upstream_constraints", {})
        if constraints != EXPECTED_CONSTRAINTS:
            problems.append(Problem("upstream_constraints khác pinned contract"))

    return problems


def parse_default_yaml(path: Path) -> tuple[dict[str, SourceArgument], list[Problem]]:
    entries: dict[str, SourceArgument] = {}
    problems: list[Problem] = []
    group = "global"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return {}, [Problem(f"không đọc được upstream default.yaml: {exc}")]

    for line_number, raw_line in enumerate(lines, start=1):
        if raw_line.startswith("# "):
            heading = raw_line[2:].split("---", 1)[0].strip()
            if heading in SOURCE_GROUPS:
                group = SOURCE_GROUPS[heading]
            continue
        match = SOURCE_TOP_LEVEL_RE.match(raw_line)
        if not match:
            continue
        key = match.group(1)
        payload = (match.group(2) or "").strip()
        if payload.startswith("#"):
            payload = ""
        elif " #" in payload:
            payload = payload.split(" #", 1)[0].rstrip()
        default = payload or "null"
        if key in entries:
            problems.append(Problem(f"upstream default.yaml trùng key '{key}'", line_number))
        else:
            entries[key] = SourceArgument(default=default, group=group, line=line_number)
    return entries, problems


def literal_assignment(tree: ast.AST, target_name: str):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == target_name for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "frozenset":
            value = value.args[0]
        try:
            return ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            return None
    return None


def mapping_keys(tree: ast.AST, target_name: str) -> set[str]:
    """Return static string keys even when mapping values contain lambdas."""

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == target_name for target in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            return set()
        return {
            str(key.value)
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return set()


def parse_cfg_source(path: Path) -> tuple[dict[str, set[str] | dict[str, int]], set[str], set[str], list[Problem]]:
    problems: list[Problem] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return {}, set(), set(), [Problem(f"không parse được cfg source: {exc}")]

    validation: dict[str, set[str] | dict[str, int]] = {}
    for source_name, registry_name in VALIDATOR_NAMES.items():
        value = literal_assignment(tree, source_name)
        if isinstance(value, dict):
            validation[registry_name] = {str(key): int(number) for key, number in value.items()}
        elif isinstance(value, (set, frozenset, list, tuple)):
            validation[registry_name] = {str(item) for item in value}
        else:
            problems.append(Problem(f"không trích được {source_name} từ cfg source"))

    allowed = literal_assignment(tree, "allowed_custom_keys")
    allowed_keys = set(allowed) if isinstance(allowed, (set, frozenset, list, tuple)) else set()
    if not allowed_keys:
        problems.append(Problem("không trích được allowed_custom_keys từ cfg source"))

    deprecated_keys = mapping_keys(tree, "deprecated_mappings")
    removed = literal_assignment(tree, "removed_keys")
    legacy_keys = set(deprecated_keys)
    if isinstance(removed, (set, frozenset, list, tuple)):
        legacy_keys.update(str(item) for item in removed)
    handle_function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_handle_deprecation"
        ),
        None,
    )
    if handle_function:
        for node in ast.walk(handle_function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                legacy_keys.add(node.args[0].value)
    if not legacy_keys:
        problems.append(Problem("không trích được legacy keys từ cfg source"))
    return validation, allowed_keys, legacy_keys, problems


def model_train_controls(path: Path) -> tuple[set[str], list[Problem]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return set(), [Problem(f"không parse được model API source: {exc}")]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "train":
                    names = {argument.arg for argument in child.args.args if argument.arg != "self"}
                    return names, []
    return set(), [Problem("không tìm thấy Model.train trong model API source")]


def compare_canonical(
    registry_entries: dict[str, dict[str, str]], source_entries: dict[str, SourceArgument]
) -> list[Problem]:
    problems: list[Problem] = []
    registry_keys, source_keys = set(registry_entries), set(source_entries)
    missing = source_keys - registry_keys
    extra = registry_keys - source_keys
    if missing:
        problems.append(Problem(f"registry thiếu upstream keys: {sorted(missing)}"))
    if extra:
        problems.append(Problem(f"registry có canonical keys ngoài upstream: {sorted(extra)}"))
    for key in sorted(registry_keys & source_keys):
        registered = registry_entries[key]
        source = source_entries[key]
        if registered.get("default") != source.default:
            problems.append(
                Problem(
                    f"default '{key}' sai: registry={registered.get('default')!r}, "
                    f"upstream={source.default!r}"
                )
            )
        if registered.get("group") != source.group:
            problems.append(
                Problem(
                    f"group '{key}' sai: registry={registered.get('group')!r}, "
                    f"upstream={source.group!r}"
                )
            )
    return problems


def git_value(source_root: Path, *args: str) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return None, str(exc)
    if result.returncode:
        return None, result.stderr.strip() or result.stdout.strip()
    return result.stdout.strip(), None


def validate_source_checkout(source_root: Path) -> list[Problem]:
    """Pin standalone full mode to the exact clean upstream checkout."""

    problems: list[Problem] = []
    head, error = git_value(source_root, "rev-parse", "HEAD")
    if error:
        problems.append(Problem(f"không đọc được Git HEAD của source: {error}"))
    elif head != PINNED_COMMIT:
        problems.append(Problem(f"source HEAD sai: expected={PINNED_COMMIT}, actual={head}"))

    origin, error = git_value(source_root, "remote", "get-url", "origin")
    if error:
        problems.append(Problem(f"không đọc được Git origin của source: {error}"))
    elif origin != PINNED_REPOSITORY:
        problems.append(
            Problem(f"source origin sai: expected={PINNED_REPOSITORY}, actual={origin}")
        )

    for flag, flag_name in (("-v", "assume/skip-worktree"), ("-f", "fsmonitor")):
        index_entries, error = git_value(source_root, "ls-files", flag)
        if error:
            problems.append(
                Problem(
                    f"không đọc được Git index flags {flag_name} của source: {error}"
                )
            )
        elif index_entries:
            concealed_entries = [
                line for line in index_entries.splitlines() if not line.startswith("H ")
            ]
            if concealed_entries:
                problems.append(
                    Problem(
                        f"source Git index {flag_name} có prefix che giấu/không chuẩn; "
                        "chỉ prefix 'H ' được phép: "
                        f"{concealed_entries!r}"
                    )
                )

    status, error = git_value(
        source_root, "status", "--porcelain", "--untracked-files=all"
    )
    if error:
        problems.append(Problem(f"không đọc được Git status của source: {error}"))
    elif status:
        problems.append(
            Problem(f"source checkout không sạch (tracked/untracked): {status!r}")
        )
    return problems


def validate_source(registry: Registry, source_root: Path) -> tuple[list[Problem], bool]:
    if not source_root.is_dir():
        return [Problem(f"source clone không tồn tại: {source_root}")], True

    upstream = registry.scalars.get("upstream", {})
    required_paths = {
        "default_config_path": PINNED_DEFAULT_SHA256,
        "config_validator_path": PINNED_CFG_SHA256,
        "model_api_path": PINNED_MODEL_API_SHA256,
        "trainer_path": PINNED_TRAINER_SHA256,
    }
    paths: dict[str, Path] = {}
    missing_files: list[str] = []
    for field_name in required_paths:
        relative = upstream.get(field_name)
        if not relative:
            missing_files.append(f"registry thiếu upstream.{field_name}")
            continue
        target = source_root / relative
        paths[field_name] = target
        if not target.is_file():
            missing_files.append(f"source thiếu file: {target}")
    if missing_files:
        return [Problem(message) for message in missing_files], True

    problems = validate_source_checkout(source_root)

    for field_name, expected_hash in required_paths.items():
        actual_hash = sha256(paths[field_name])
        if actual_hash != expected_hash:
            problems.append(
                Problem(
                    f"SHA-256 {field_name} sai: expected={expected_hash}, actual={actual_hash}"
                )
            )

    source_entries, source_problems = parse_default_yaml(paths["default_config_path"])
    problems.extend(source_problems)
    problems.extend(compare_canonical(registry.entries.get("canonical_arguments", {}), source_entries))

    source_validation, custom_keys, legacy_keys, cfg_problems = parse_cfg_source(
        paths["config_validator_path"]
    )
    problems.extend(cfg_problems)
    registered_validation = registry.scalars.get("upstream_validation", {})
    for source_name in EXPECTED_TYPES_BY_VALIDATOR:
        registered_set = csv_set(registered_validation.get(source_name, ""))
        actual_set = source_validation.get(source_name, set())
        if registered_set != actual_set:
            problems.append(
                Problem(
                    f"{source_name} sai source: "
                    f"missing={sorted(set(actual_set) - registered_set)}, "
                    f"extra={sorted(registered_set - set(actual_set))}"
                )
            )
    try:
        registered_minimums = key_value_map(registered_validation.get("cfg_int_min", ""))
    except (ValueError, TypeError) as exc:
        problems.append(Problem(f"cfg_int_min không parse được: {exc}"))
    else:
        if registered_minimums != source_validation.get("cfg_int_min", {}):
            problems.append(
                Problem(
                    f"cfg_int_min sai source: registry={registered_minimums}, "
                    f"source={source_validation.get('cfg_int_min', {})}"
                )
            )

    registered_extra = set(registry.entries.get("extra_config_arguments", {}))
    if registered_extra != custom_keys:
        problems.append(
            Problem(
                "extra config keys sai source: "
                f"missing={sorted(custom_keys - registered_extra)}, "
                f"extra={sorted(registered_extra - custom_keys)}"
            )
        )
    registered_legacy = set(registry.entries.get("legacy_arguments", {}))
    if registered_legacy != legacy_keys:
        problems.append(
            Problem(
                "legacy keys sai source: "
                f"missing={sorted(legacy_keys - registered_legacy)}, "
                f"extra={sorted(registered_legacy - legacy_keys)}"
            )
        )
    controls, control_problems = model_train_controls(paths["model_api_path"])
    problems.extend(control_problems)
    registered_controls = set(registry.entries.get("api_controls", {}))
    if registered_controls != controls:
        problems.append(
            Problem(
                "Model.train controls sai source: "
                f"missing={sorted(controls - registered_controls)}, "
                f"extra={sorted(registered_controls - controls)}"
            )
        )
    return problems, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kiểm tra registry tham số train với Ultralytics đã pin."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("docs/configuration/TRAIN_ARGUMENTS.yaml"),
        help="Đường dẫn registry YAML.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Root clone Ultralytics; mặc định .references/ultralytics ở full mode.",
    )
    parser.add_argument(
        "--documentation",
        type=Path,
        default=Path("docs/configuration/TRAIN_ARGUMENTS.md"),
        help="Bảng tra cứu Markdown phải khớp registry YAML.",
    )
    parser.add_argument(
        "--registry-only",
        action="store_true",
        help="Chỉ kiểm cấu trúc/coverage đã pin, không yêu cầu clone source.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.registry_only and args.source is not None:
        print("ERROR: không dùng đồng thời --registry-only và --source", file=sys.stderr)
        return 2

    registry_path = args.registry.expanduser().resolve()
    if not registry_path.is_file():
        print(f"ERROR: registry không tồn tại: {registry_path}", file=sys.stderr)
        return 2
    registry = parse_registry(registry_path)
    problems = validate_registry(registry)
    invocation_error = False

    documentation_path = args.documentation.expanduser().resolve()
    documentation_problems, documentation_invocation_error = validate_documentation(
        registry, documentation_path
    )
    problems.extend(documentation_problems)
    invocation_error |= documentation_invocation_error

    if not args.registry_only:
        source_root = (args.source or Path(".references/ultralytics")).expanduser().resolve()
        source_problems, source_invocation_error = validate_source(registry, source_root)
        problems.extend(source_problems)
        invocation_error |= source_invocation_error

    if problems:
        for problem in problems:
            print(f"{registry_path}:{problem.line}: ERROR: {problem.message}")
        print(f"Train argument registry check failed: {len(problems)} lỗi.")
        return 2 if invocation_error else 1

    mode = "registry-only" if args.registry_only else f"source={PINNED_COMMIT[:12]}"
    print(
        "Train argument registry check passed: "
        "115 canonical + 2 extra + 9 legacy + 1 API = 127 names; "
        f"exact semantic and Markdown manifest; zero missing/extra ({mode})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
