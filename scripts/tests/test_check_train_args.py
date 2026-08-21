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


def load_checker_module():
    spec = importlib.util.spec_from_file_location("check_train_args_under_test", CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Không thể load check_train_args.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER_MODULE = load_checker_module()


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


def run_mutation(old: str, new: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        target = Path(temporary_directory) / "TRAIN_ARGUMENTS.yaml"
        source = REGISTRY.read_text(encoding="utf-8")
        if old not in source:
            raise AssertionError(f"Mutation source không tồn tại: {old}")
        target.write_text(source.replace(old, new, 1), encoding="utf-8")
        return run_registry_checker(target)


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
                (PROJECT_ROOT / "docs" / "configuration" / "TRAIN_ARGUMENTS.md")
                .read_text(encoding="utf-8")
                .replace("| `epochs` | `100`", "| `epochs` | `999`", 1),
                encoding="utf-8",
            )
            disposition_doc = Path(temporary_directory) / "disposition.md"
            disposition_doc.write_text(
                (PROJECT_ROOT / "docs" / "configuration" / "TRAIN_ARGUMENTS.md")
                .read_text(encoding="utf-8")
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
