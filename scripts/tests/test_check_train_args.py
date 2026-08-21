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
