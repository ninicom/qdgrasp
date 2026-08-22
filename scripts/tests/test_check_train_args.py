from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT_ROOT / "scripts" / "check_train_args.py"
DOCS_CHECKER = PROJECT_ROOT / "scripts" / "check_docs.py"
REGISTRY = PROJECT_ROOT / "docs" / "configuration" / "TRAIN_ARGUMENTS.yaml"
DOCUMENTATION = PROJECT_ROOT / "docs" / "configuration" / "TRAIN_ARGUMENTS.md"


def load_checker_module():
    spec = importlib.util.spec_from_file_location("check_train_args_under_test", CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Không thể load check_train_args.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER_MODULE = load_checker_module()


def run_git(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def create_source_checkout_fixture(parent: Path) -> tuple[Path, Path]:
    source = parent / "source"
    subprocess.run(
        ["git", "init", "--quiet", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    for key, value in (("user.name", "Test"), ("user.email", "test@example.invalid")):
        run_git(source, "config", key, value)
    tracked = source / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    run_git(source, "add", "tracked.txt")
    run_git(source, "commit", "--quiet", "-m", "fixture")
    run_git(source, "remote", "add", "origin", CHECKER_MODULE.PINNED_REPOSITORY)
    return source, tracked


def run_registry_checker(registry: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--registry",
            str(registry),
            "--registry-only",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def run_full_checker(registry: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--registry",
            str(registry),
            "--source",
            str(PROJECT_ROOT / ".references" / "ultralytics"),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def run_mutation(old: str, new: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        target = Path(temporary_directory) / "TRAIN_ARGUMENTS.yaml"
        source = REGISTRY.read_text(encoding="utf-8")
        if old not in source:
            raise AssertionError(f"Mutation source không tồn tại: {old}")
        target.write_text(source.replace(old, new, 1), encoding="utf-8")
        return run_registry_checker(target)


def validate_markdown_mutation(old: str, new: str) -> str:
    with tempfile.TemporaryDirectory() as temporary_directory:
        target = Path(temporary_directory) / "TRAIN_ARGUMENTS.md"
        source = DOCUMENTATION.read_text(encoding="utf-8")
        if old not in source:
            raise AssertionError(f"Markdown mutation source không tồn tại: {old}")
        target.write_text(source.replace(old, new, 1), encoding="utf-8")
        registry = CHECKER_MODULE.parse_registry(REGISTRY)
        problems, invocation_error = CHECKER_MODULE.validate_documentation(
            registry, target
        )
        if invocation_error:
            raise AssertionError("Markdown mutation unexpectedly caused invocation error")
        return "\n".join(problem.message for problem in problems)


class TrainArgumentRegistryTests(unittest.TestCase):
    def test_current_registry_has_complete_pinned_surface(self) -> None:
        result = run_registry_checker(REGISTRY)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("115 canonical + 2 extra + 9 legacy + 1 API = 127", result.stdout)
        self.assertIn("zero missing/extra", result.stdout)

    def test_missing_canonical_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "TRAIN_ARGUMENTS.yaml"
            text = REGISTRY.read_text(encoding="utf-8")
            text = "\n".join(
                line
                for line in text.splitlines()
                if not line.startswith("  epochs: {default:")
            ) + "\n"
            target.write_text(text, encoding="utf-8")

            result = run_registry_checker(target)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("canonical_arguments có 114 entry", result.stdout)

    def test_missing_disposition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "TRAIN_ARGUMENTS.yaml"
            text = REGISTRY.read_text(encoding="utf-8").replace(
                ", disposition: retain, dexgrasp_key: mode",
                ", dexgrasp_key: mode",
                1,
            )
            target.write_text(text, encoding="utf-8")

            result = run_registry_checker(target)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("thiếu field: ['disposition']", result.stdout)

    def test_source_set_comparison_names_omitted_key(self) -> None:
        registry = CHECKER_MODULE.parse_registry(REGISTRY)
        canonical = registry.entries["canonical_arguments"]
        source = {
            key: CHECKER_MODULE.SourceArgument(
                default=fields["default"], group=fields["group"], line=1
            )
            for key, fields in canonical.items()
            if key != "epochs"
        }

        problems = CHECKER_MODULE.compare_canonical(canonical, source)
        messages = "\n".join(problem.message for problem in problems)

        self.assertIn("canonical keys ngoài upstream", messages)
        self.assertIn("epochs", messages)

    def test_invalid_yaml_like_unknown_metadata_is_rejected(self) -> None:
        result = run_mutation(
            "related_plan: PLAN-V2",
            "related_plan: PLAN-V2\nunvalidated_metadata: [",
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("YAML scalar subset an toàn", result.stdout)
        self.assertIn("top-level metadata sai schema", result.stdout)

    def test_registry_dialect_rejects_type_erasure_and_invalid_spacing(self) -> None:
        cases = (
            (
                "epochs: {default: 100, group: train",
                'epochs: {default: "100", group: train',
                "quoted scalar",
            ),
            (
                "save: {default: True, group: train",
                'save: {default: "True", group: train',
                "quoted scalar",
            ),
            (
                "model: {default: null, group: train",
                'model: {default: "null", group: train',
                "quoted scalar",
            ),
            ("schema_version: 2", 'schema_version: " 2 "', "quoted scalar"),
            (
                "epochs: {default: 100, group: train",
                "epochs: {default:100, group: train",
                "chính xác 'key: value'",
            ),
            (
                "epochs: {default: 100, group: train",
                "epochs: {default:\t100, group: train",
                "không cho phép tab",
            ),
            ("schema_version: 2\n", "schema_version: 2 \n", "whitespace cuối dòng"),
        )
        for old, new, expected_message in cases:
            with self.subTest(mutation=new):
                result = run_mutation(old, new)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(expected_message, result.stdout)

    def test_scalar_signatures_preserve_kind_and_lexeme(self) -> None:
        expected = {
            "null": ("null", "null"),
            "True": ("boolean", "True"),
            "False": ("boolean", "False"),
            "100": ("integer", "100"),
            "-1": ("integer", "-1"),
            "1.0": ("float", "1.0"),
            "1.00": ("float", "1.00"),
            "auto": ("string", "auto"),
        }
        for lexeme, (kind, preserved) in expected.items():
            with self.subTest(lexeme=lexeme):
                signature = CHECKER_MODULE.scalar_signature(lexeme)
                self.assertEqual(signature.kind, kind)
                self.assertEqual(signature.lexeme, preserved)

    def test_manifest_fingerprint_v2_excludes_only_its_self_reference(self) -> None:
        registry = CHECKER_MODULE.parse_registry(REGISTRY)
        baseline = CHECKER_MODULE.manifest_fingerprint(registry)

        registry.scalars["upstream"]["manifest_fingerprint"] = "self-reference-change"
        self.assertEqual(CHECKER_MODULE.manifest_fingerprint(registry), baseline)

        registry.top["status"] = "changed"
        self.assertNotEqual(CHECKER_MODULE.manifest_fingerprint(registry), baseline)

    @unittest.skipUnless(
        (PROJECT_ROOT / ".references" / "ultralytics").is_dir(),
        "full source clone không có trong source archive",
    )
    def test_full_mode_rejects_quoted_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "TRAIN_ARGUMENTS.yaml"
            target.write_text(
                REGISTRY.read_text(encoding="utf-8").replace(
                    "epochs: {default: 100, group: train",
                    'epochs: {default: "100", group: train',
                    1,
                ),
                encoding="utf-8",
            )
            result = run_full_checker(target)

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("quoted scalar", result.stdout)

    def test_registry_only_rejects_default_and_group_corruption(self) -> None:
        default_result = run_mutation(
            "epochs: {default: 100, group: train",
            "epochs: {default: 999, group: train",
        )
        group_result = run_mutation(
            "epochs: {default: 100, group: train",
            "epochs: {default: 100, group: export",
        )

        self.assertEqual(default_result.returncode, 1, default_result.stdout)
        self.assertEqual(group_result.returncode, 1, group_result.stdout)
        self.assertIn("semantic manifest fingerprint sai", default_result.stdout)
        self.assertIn("semantic manifest fingerprint sai", group_result.stdout)

    def test_registry_only_rejects_legacy_semantic_corruption(self) -> None:
        result = run_mutation(
            "half: {source: precision_alias, target: quantize",
            "half: {source: precision_alias, target: show_boxes",
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("legacy_arguments contract khác pinned manifest", result.stdout)

    def test_registry_only_rejects_resume_and_merge_corruption(self) -> None:
        resume_result = run_mutation(
            "resume_mutable_keys: augmentations,batch,cache",
            "resume_mutable_keys: epochs",
        )
        merge_result = run_mutation(
            "get_cfg_priority: base_then_overrides",
            "get_cfg_priority: overrides_then_base",
        )

        self.assertEqual(resume_result.returncode, 1, resume_result.stdout)
        self.assertEqual(merge_result.returncode, 1, merge_result.stdout)
        self.assertIn("resume_mutable_keys khác pinned source", resume_result.stdout)
        self.assertIn("upstream_merge_contract khác pinned contract", merge_result.stdout)

    def test_registry_only_rejects_extension_removal(self) -> None:
        result = run_mutation(
            "  quality_negatives: {status: required_design, type: enum, purpose: quality_sampling}\n",
            "",
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("dexgrasp_extensions contract khác pinned manifest", result.stdout)

    def test_registry_only_rejects_source_identity_corruption(self) -> None:
        repository_result = run_mutation(
            "repository: https://github.com/ultralytics/ultralytics.git",
            "repository: https://example.invalid/ultralytics.git",
        )
        tag_result = run_mutation("tag: v8.4.125", "tag: v0.0.0")

        self.assertEqual(repository_result.returncode, 1, repository_result.stdout)
        self.assertEqual(tag_result.returncode, 1, tag_result.stdout)
        self.assertIn("upstream.repository phải pin", repository_result.stdout)
        self.assertIn("upstream.tag phải pin", tag_result.stdout)

    def test_registry_only_rejects_extra_api_and_untyped_type_corruption(self) -> None:
        extra_result = run_mutation(
            "augmentations: {source: allowed_custom_keys, type: python_list_or_named_yaml_list",
            "augmentations: {source: allowed_custom_keys, type: string",
        )
        api_result = run_mutation(
            "trainer: {source: Model.train_signature, type: trainer_or_null",
            "trainer: {source: Model.train_signature, type: string",
        )
        untyped_result = run_mutation(
            "amp: {default: True, group: train, type: boolean",
            "amp: {default: True, group: train, type: string",
        )

        self.assertEqual(extra_result.returncode, 1, extra_result.stdout)
        self.assertEqual(api_result.returncode, 1, api_result.stdout)
        self.assertEqual(untyped_result.returncode, 1, untyped_result.stdout)
        self.assertIn("extra_config_arguments contract khác", extra_result.stdout)
        self.assertIn("api_controls contract khác", api_result.stdout)
        self.assertIn("semantic manifest fingerprint sai", untyped_result.stdout)

    def test_markdown_default_and_disposition_drift_is_rejected(self) -> None:
        registry = CHECKER_MODULE.parse_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as temporary_directory:
            default_doc = Path(temporary_directory) / "default.md"
            default_doc.write_text(
                DOCUMENTATION.read_text(encoding="utf-8")
                .replace("| `epochs` | `100`", "| `epochs` | `999`", 1),
                encoding="utf-8",
            )
            disposition_doc = Path(temporary_directory) / "disposition.md"
            disposition_doc.write_text(
                DOCUMENTATION.read_text(encoding="utf-8")
                .replace("| `epochs` | `100` | integer | R |", "| `epochs` | `100` | integer | X |", 1),
                encoding="utf-8",
            )

            default_problems, _ = CHECKER_MODULE.validate_documentation(registry, default_doc)
            disposition_problems, _ = CHECKER_MODULE.validate_documentation(
                registry, disposition_doc
            )

        self.assertIn(
            "Markdown default 'epochs' sai",
            "\n".join(problem.message for problem in default_problems),
        )
        self.assertIn(
            "Markdown disposition 'epochs' sai",
            "\n".join(problem.message for problem in disposition_problems),
        )

    def test_markdown_normative_type_behavior_and_prose_drift_is_rejected(self) -> None:
        cases = (
            (
                "| `epochs` | `100` | integer | R |",
                "| `epochs` | `100` | banana/string | R |",
                "Markdown type 'epochs' sai",
            ),
            (
                "| `quantize` | `null` | precision/null | A |",
                "| `quantize` | `null` | boolean | A |",
                "Markdown type 'quantize' sai",
            ),
            (
                "| `half` | Alias → `quantize=16` |",
                "| `half` | Alias → `show_boxes` |",
                "Markdown normative body SHA-256 sai",
            ),
            (
                "Scalar upstream bị chặn `[0,1]`; pair cần đúng hai số",
                "Scalar upstream không có range; pair nhận mọi độ dài",
                "Markdown normative body SHA-256 sai",
            ),
            (
                "loaded cfg thay self.overrides",
                "self.overrides rồi loaded cfg",
                "Markdown normative body SHA-256 sai",
            ),
        )
        for old, new, expected_message in cases:
            with self.subTest(mutation=new):
                messages = validate_markdown_mutation(old, new)
                self.assertIn(expected_message, messages)

    def test_markdown_front_matter_tpr_002_mutations_are_rejected(self) -> None:
        pointer = (
            "latest_revision_record: "
            "docs/revisions/REV-20260822-003-raw-scalar-fsmonitor-hardening.md"
        )
        cases = (
            (pointer + "\n", "", "latest_revision_record"),
            (
                pointer,
                "latest_revision_record: "
                "docs/revisions/REV-20260822-002-source-metadata-hardening.md",
                "latest_revision_record",
            ),
            ("date: 2026-08-22", "date: never", "Markdown metadata 'date'"),
            (
                "title: Registry đầy đủ tham số train tương thích Ultralytics v8.4.125",
                "title: Registry runtime đã triển khai hoàn chỉnh",
                "Markdown metadata 'title'",
            ),
            (
                pointer,
                pointer + "\nruntime_implemented: True",
                "extra=['runtime_implemented']",
            ),
        )
        for old, new, expected_message in cases:
            with self.subTest(mutation=new):
                messages = validate_markdown_mutation(old, new)
                self.assertIn(expected_message, messages)

    def test_markdown_front_matter_values_use_exact_raw_lexemes(self) -> None:
        mutation_styles = {
            "quoted_and_padded": lambda key, value: f'{key}: " {value} "',
            "leading_padding": lambda key, value: f"{key}:  {value}",
            "trailing_padding": lambda key, value: f"{key}: {value} ",
            "tab_separator": lambda key, value: f"{key}:\t{value}",
        }
        for key, value in CHECKER_MODULE.EXPECTED_MARKDOWN_METADATA.items():
            original = f"{key}: {value}"
            for style, mutation in mutation_styles.items():
                with self.subTest(field=key, style=style):
                    messages = validate_markdown_mutation(
                        original, mutation(key, value)
                    )
                    if style == "tab_separator":
                        self.assertIn("Markdown metadata malformed", messages)
                    else:
                        self.assertIn(f"Markdown metadata '{key}'", messages)

    def test_markdown_front_matter_duplicate_and_malformed_fields_are_rejected(self) -> None:
        cases = (
            (
                "status: active",
                "status: active\nstatus: active",
                "Markdown metadata trùng: 'status'",
            ),
            (
                "status: active",
                "status active",
                "Markdown metadata malformed",
            ),
        )
        for old, new, expected_message in cases:
            with self.subTest(mutation=new):
                messages = validate_markdown_mutation(old, new)
                self.assertIn(expected_message, messages)

    def test_markdown_revision_pointer_requires_current_record_identity(self) -> None:
        metadata = dict(CHECKER_MODULE.EXPECTED_MARKDOWN_METADATA)
        original_root = CHECKER_MODULE.PROJECT_ROOT
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / metadata["latest_revision_record"]
            target.parent.mkdir(parents=True)
            target.write_text(
                """---
document_id: REV-WRONG
document_type: revision_record
status: superseded
record_id: REV-WRONG
session_id: SESSION-WRONG
revises:
  - record_id: REV-OLD
---
""",
                encoding="utf-8",
            )
            CHECKER_MODULE.PROJECT_ROOT = root
            try:
                messages = "\n".join(
                    problem.message
                    for problem in CHECKER_MODULE.validate_revision_pointer(metadata)
                )
            finally:
                CHECKER_MODULE.PROJECT_ROOT = original_root

        self.assertIn("Revision record metadata 'document_id'", messages)
        self.assertIn("Revision record metadata 'record_id'", messages)
        self.assertIn("Revision record metadata 'session_id'", messages)
        self.assertIn("Revision record metadata 'status'", messages)

    def test_markdown_revision_pointer_identity_uses_exact_raw_lexemes(self) -> None:
        expected = {
            "document_id": CHECKER_MODULE.EXPECTED_MARKDOWN_REVISION_ID,
            "record_id": CHECKER_MODULE.EXPECTED_MARKDOWN_REVISION_ID,
            "document_type": "revision_record",
            "session_id": CHECKER_MODULE.EXPECTED_MARKDOWN_SESSION_ID,
            "status": "in_review",
        }
        base_lines = [
            "---",
            f"document_id: {expected['document_id']}",
            f"document_type: {expected['document_type']}",
            f"status: {expected['status']}",
            f"record_id: {expected['record_id']}",
            f"session_id: {expected['session_id']}",
            "revises:",
            "  - record_id: REV-OLD",
            "---",
            "",
        ]
        mutation_styles = {
            "quoted_and_padded": lambda key, value: f'{key}: " {value} "',
            "leading_padding": lambda key, value: f"{key}:  {value}",
            "trailing_padding": lambda key, value: f"{key}: {value} ",
            "tab_separator": lambda key, value: f"{key}:\t{value}",
        }
        metadata = dict(CHECKER_MODULE.EXPECTED_MARKDOWN_METADATA)
        original_root = CHECKER_MODULE.PROJECT_ROOT
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / metadata["latest_revision_record"]
            target.parent.mkdir(parents=True)
            CHECKER_MODULE.PROJECT_ROOT = root
            try:
                for key, value in expected.items():
                    original = f"{key}: {value}"
                    for style, mutation in mutation_styles.items():
                        with self.subTest(field=key, style=style):
                            mutated_lines = [
                                mutation(key, value) if line == original else line
                                for line in base_lines
                            ]
                            target.write_text("\n".join(mutated_lines), encoding="utf-8")
                            messages = "\n".join(
                                problem.message
                                for problem in CHECKER_MODULE.validate_revision_pointer(
                                    metadata
                                )
                            )
                            self.assertIn(
                                f"Revision record metadata '{key}'", messages
                            )
            finally:
                CHECKER_MODULE.PROJECT_ROOT = original_root

    def test_markdown_revision_pointer_requires_existing_target(self) -> None:
        metadata = dict(CHECKER_MODULE.EXPECTED_MARKDOWN_METADATA)
        original_root = CHECKER_MODULE.PROJECT_ROOT
        with tempfile.TemporaryDirectory() as temporary_directory:
            CHECKER_MODULE.PROJECT_ROOT = Path(temporary_directory)
            try:
                messages = "\n".join(
                    problem.message
                    for problem in CHECKER_MODULE.validate_revision_pointer(metadata)
                )
            finally:
                CHECKER_MODULE.PROJECT_ROOT = original_root

        self.assertIn("latest_revision_record không tồn tại", messages)

    def test_nullable_special_types_are_explicit(self) -> None:
        registry = CHECKER_MODULE.parse_registry(REGISTRY)
        canonical = registry.entries["canonical_arguments"]

        self.assertEqual(canonical["auto_augment"]["type"], "string_or_null")
        self.assertEqual(canonical["quantize"]["type"], "precision_or_null")

    @unittest.skipUnless(
        (PROJECT_ROOT / ".references" / "ultralytics").is_dir(),
        "full source clone không có trong source archive",
    )
    def test_full_cli_passes_against_pinned_clone(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--source",
                str(PROJECT_ROOT / ".references" / "ultralytics"),
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("exact semantic and Markdown manifest", result.stdout)

    def test_full_check_reports_unpinned_source_and_hashes(self) -> None:
        registry = CHECKER_MODULE.parse_registry(REGISTRY)
        upstream = registry.scalars["upstream"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory)
            for field in (
                "default_config_path",
                "config_validator_path",
                "model_api_path",
                "trainer_path",
            ):
                target = source / upstream[field]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# synthetic unpinned fixture\n", encoding="utf-8")

            problems, invocation_error = CHECKER_MODULE.validate_source(registry, source)
            messages = "\n".join(problem.message for problem in problems)

            self.assertFalse(invocation_error)
            self.assertIn("Git HEAD", messages)
            self.assertIn("SHA-256", messages)

    def test_source_checkout_helper_rejects_tracked_and_untracked_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, tracked = create_source_checkout_fixture(Path(temporary_directory))
            tracked.write_text("dirty\n", encoding="utf-8")
            (source / "untracked.txt").write_text("untracked\n", encoding="utf-8")

            problems = CHECKER_MODULE.validate_source_checkout(source)
            messages = "\n".join(problem.message for problem in problems)

        self.assertIn("source checkout không sạch (tracked/untracked)", messages)
        self.assertIn("tracked.txt", messages)
        self.assertIn("untracked.txt", messages)

    def test_source_checkout_helper_rejects_concealed_index_changes(self) -> None:
        cases = (
            ("--assume-unchanged", "h tracked.txt"),
            ("--skip-worktree", "S tracked.txt"),
        )
        for flag, expected_entry in cases:
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary_directory:
                source, tracked = create_source_checkout_fixture(
                    Path(temporary_directory)
                )
                run_git(source, "update-index", flag, "tracked.txt")
                tracked.write_text("concealed change\n", encoding="utf-8")

                porcelain = run_git(
                    source, "status", "--porcelain", "--untracked-files=all"
                ).stdout.strip()
                self.assertEqual(porcelain, "")

                problems = CHECKER_MODULE.validate_source_checkout(source)
                messages = "\n".join(problem.message for problem in problems)
                self.assertIn("source Git index assume/skip-worktree", messages)
                self.assertIn(expected_entry, messages)

    def test_source_checkout_helper_rejects_fsmonitor_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, tracked = create_source_checkout_fixture(Path(temporary_directory))
            fake_monitor = source / "fake-fsmonitor.sh"
            fake_monitor.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_monitor.chmod(0o755)
            run_git(source, "config", "core.fsmonitor", str(fake_monitor))
            run_git(source, "config", "core.fsmonitorHookVersion", "1")
            run_git(source, "update-index", "--fsmonitor-valid", "tracked.txt")
            tracked.write_text("fsmonitor-concealed change\n", encoding="utf-8")

            verbose_view = run_git(source, "ls-files", "-v").stdout.strip()
            fsmonitor_view = run_git(source, "ls-files", "-f").stdout.strip()
            self.assertEqual(verbose_view, "H tracked.txt")
            self.assertEqual(fsmonitor_view, "h tracked.txt")

            problems = CHECKER_MODULE.validate_source_checkout(source)
            messages = "\n".join(problem.message for problem in problems)
            self.assertIn("source Git index fsmonitor", messages)
            self.assertIn("h tracked.txt", messages)

    def test_docs_checker_discovers_configuration_registry(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            configuration = root / "docs" / "configuration"
            configuration.mkdir(parents=True)
            (configuration / "BROKEN.md").write_text(
                "# Registry không có metadata\n", encoding="utf-8"
            )
            result = subprocess.run(
                [sys.executable, str(DOCS_CHECKER), "--root", str(root)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("docs/configuration/BROKEN.md", result.stdout)


if __name__ == "__main__":
    unittest.main()
