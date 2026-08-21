#!/usr/bin/env python3
"""Audit the pinned Ultralytics configuration surface against DexGrasp's registry.

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
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


PINNED_COMMIT = "329682a29d27203582ba30e519340f95abccc6a6"
PINNED_DEFAULT_SHA256 = "eb5e9ab6825a5d55076f8b38aed00953dec722ed5d5368a6584df35f50f32839"
PINNED_CFG_SHA256 = "ea9a98b498686f44561e9ce9892aa0e7de52226429496b08a20475ebbfa4eed8"
PINNED_MODEL_API_SHA256 = "69cb4c9c6f572c50bd35328e5154e6b596ed50dc10d218584da8aed3f7c9b32a"
PINNED_TRAINER_SHA256 = "d02bfd82d2af38fb58a6fe7903ef0cb9a93633a75b3db778768896d6c57a32e6"
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

STRUCTURED_SECTIONS = {
    "canonical_arguments",
    "extra_config_arguments",
    "legacy_arguments",
    "api_controls",
    "dexgrasp_extensions",
}
SCALAR_SECTIONS = {"upstream", "upstream_validation", "upstream_merge_contract"}
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
    "quantize": "precision",
}
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TOP_LEVEL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:[ \t]*(.*))?$")
ENTRY_RE = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*):[ \t]*(\{.*\})[ \t]*$")
NESTED_SCALAR_RE = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*):[ \t]*(.*)$")


@dataclass(frozen=True)
class Problem:
    message: str
    line: int = 1


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


def clean_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


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
    body = raw[1:-1].strip()
    if not body:
        return {}, [Problem("inline mapping không được rỗng", line)]

    values: dict[str, str] = {}
    for item in body.split(","):
        if ":" not in item:
            problems.append(Problem(f"field không có dấu ':': {item.strip()}", line))
            continue
        key, value = item.split(":", 1)
        key, value = key.strip(), clean_scalar(value)
        if not KEY_RE.fullmatch(key):
            problems.append(Problem(f"tên field không hợp lệ: {key}", line))
        elif key in values:
            problems.append(Problem(f"field trùng trong entry: {key}", line))
        elif not value:
            problems.append(Problem(f"field '{key}' không được rỗng", line))
        else:
            values[key] = value
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
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        top_match = TOP_LEVEL_RE.match(raw_line)
        if top_match:
            key, raw_value = top_match.group(1), (top_match.group(2) or "").strip()
            if not raw_value:
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
            if key in registry.top:
                registry.problems.append(Problem(f"metadata trùng: {key}", line_number))
            else:
                registry.top[key] = clean_scalar(raw_value)
            current_section = None
            continue

        if current_section in STRUCTURED_SECTIONS:
            entry_match = ENTRY_RE.match(raw_line)
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
            scalar_match = NESTED_SCALAR_RE.match(raw_line)
            if not scalar_match:
                registry.problems.append(
                    Problem(f"dòng scalar không hợp lệ trong {current_section}", line_number)
                )
                continue
            key, value = scalar_match.group(1), clean_scalar(scalar_match.group(2))
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
            continue

        registry.problems.append(Problem("nội dung nằm ngoài section hợp lệ", line_number))

    return registry


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
        "schema_version": "1",
        "registry_id": "DEXGRASP-TRAIN-ARGS-ULTRALYTICS-8.4.125",
        "status": "active",
        "generated_at": "2026-08-21",
        "related_plan": "PLAN-V2",
    }
    for key, expected in required_top.items():
        actual = registry.top.get(key)
        if actual is None:
            problems.append(Problem(f"thiếu metadata: {key}"))
        elif actual != expected:
            problems.append(Problem(f"metadata '{key}' phải là {expected!r}, nhận {actual!r}"))

    upstream = registry.scalars.get("upstream", {})
    pinned_fields = {
        "commit": PINNED_COMMIT,
        "default_config_sha256": PINNED_DEFAULT_SHA256,
        "config_validator_sha256": PINNED_CFG_SHA256,
        "model_api_sha256": PINNED_MODEL_API_SHA256,
        "trainer_sha256": PINNED_TRAINER_SHA256,
    }
    for key, expected in pinned_fields.items():
        actual = upstream.get(key)
        if actual != expected:
            problems.append(
                Problem(f"upstream.{key} phải pin {expected!r}, nhận {actual!r}")
            )
    for key, expected in PINNED_COUNTS.items():
        value = integer_field(upstream, key, problems)
        if value is not None and value != expected:
            problems.append(Problem(f"upstream.{key} phải là {expected}, nhận {value}"))

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
        problems.append(Problem(f"DexGrasp extension trùng tên upstream: {sorted(overlap)}"))
    for key, fields in extensions.items():
        missing = {"status", "type", "purpose"} - set(fields)
        if missing:
            line = registry.entry_lines.get(("dexgrasp_extensions", key), 1)
            problems.append(Problem(f"extension '{key}' thiếu field: {sorted(missing)}", line))

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
        try:
            minimums = key_value_map(validation["cfg_int_min"])
        except (ValueError, TypeError) as exc:
            problems.append(Problem(f"cfg_int_min không hợp lệ: {exc}"))
        else:
            if not set(minimums) <= validation_sets["cfg_int_keys"]:
                problems.append(Problem("cfg_int_min chứa key không nằm trong cfg_int_keys"))

        merge_contract = registry.scalars.get("upstream_merge_contract", {})
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
        match = TOP_LEVEL_RE.match(raw_line)
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


def git_head(source_root: Path) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return None, str(exc)
    if result.returncode:
        return None, result.stderr.strip() or result.stdout.strip()
    return result.stdout.strip(), None


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

    problems: list[Problem] = []
    head, head_error = git_head(source_root)
    if head_error:
        problems.append(Problem(f"không đọc được Git HEAD của source: {head_error}"))
    elif head != PINNED_COMMIT:
        problems.append(Problem(f"source HEAD sai: expected={PINNED_COMMIT}, actual={head}"))

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

    if not args.registry_only:
        source_root = (args.source or Path(".references/ultralytics")).expanduser().resolve()
        source_problems, invocation_error = validate_source(registry, source_root)
        problems.extend(source_problems)

    if problems:
        for problem in problems:
            print(f"{registry_path}:{problem.line}: ERROR: {problem.message}")
        print(f"Train argument registry check failed: {len(problems)} lỗi.")
        return 2 if invocation_error else 1

    mode = "registry-only" if args.registry_only else f"source={PINNED_COMMIT[:12]}"
    print(
        "Train argument registry check passed: "
        "115 canonical + 2 extra + 9 legacy + 1 API = 127 names; "
        f"zero missing/extra ({mode})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
