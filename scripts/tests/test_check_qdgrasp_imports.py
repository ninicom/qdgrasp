from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT_ROOT / "scripts" / "check_qdgrasp_imports.py"


def load_checker_module():
    spec = importlib.util.spec_from_file_location("check_qdgrasp_imports_under_test", CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Không thể load check_qdgrasp_imports.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER_MODULE = load_checker_module()


def run_checker(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(CHECKER), *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class QDGraspImportCheckerTests(unittest.TestCase):
    def test_resolves_valid_absolute_and_relative_internal_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "qdgrasp"
            write(root / "__init__.py", "")
            write(root / "utils" / "__init__.py", "LOGGER = None\n")
            write(
                root / "engine" / "model.py",
                "from ultralytics.utils import LOGGER\nfrom . import utils\nfrom ..utils import LOGGER as L2\n",
            )
            write(root / "engine" / "__init__.py", "")

            result = run_checker("--root", str(root))

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("0 unresolved", result.stdout)

    def test_rejects_unresolved_absolute_internal_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "qdgrasp"
            write(root / "__init__.py", "")
            write(root / "cfg" / "__init__.py", "from ultralytics.solutions.config import SolutionConfig\n")

            result = run_checker("--root", str(root))

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("không resolve được: ultralytics.solutions.config", result.stdout)

    def test_rejects_unresolved_relative_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "qdgrasp"
            write(root / "__init__.py", "")
            write(root / "models" / "__init__.py", "from .yolo import YOLO\n")

            result = run_checker("--root", str(root))

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("relative import không resolve được: .yolo", result.stdout)

    def test_relative_import_beyond_package_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "qdgrasp"
            write(root / "__init__.py", "")
            write(root / "utils" / "__init__.py", "from .... import impossible\n")

            result = run_checker("--root", str(root))

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("vượt quá package root", result.stdout)

    def test_third_party_and_stdlib_imports_are_not_resolved_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "qdgrasp"
            write(root / "__init__.py", "")
            write(root / "utils" / "__init__.py", "import os\nimport sys\nimport torch\nimport numpy as np\n")

            result = run_checker("--root", str(root))

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nested_import_inside_function_body_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "qdgrasp"
            write(root / "__init__.py", "")
            write(
                root / "engine" / "exporter.py",
                "def export_coreml(self):\n"
                "    from ultralytics.utils.export.coreml import convert\n"
                "    return convert\n",
            )

            result = run_checker("--root", str(root))

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("ultralytics.utils.export.coreml", result.stdout)

    def test_ultralytics_token_count_is_case_sensitive_lowercase_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "qdgrasp"
            write(root / "__init__.py", "")
            write(
                root / "utils" / "__init__.py",
                '"""Ultralytics utilities. import ultralytics.utils as u. See ultralytics.com."""\n',
            )

            result = run_checker("--root", str(root), "--summary")

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # "Ultralytics" (capitalized, prose) is not counted; only the two
            # lowercase hits ("ultralytics.utils", "ultralytics.com") are.
            self.assertIn("ultralytics_tokens=2", result.stdout)

    def test_missing_root_is_reported_as_invocation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "does-not-exist"

            result = run_checker("--root", str(missing))

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("package root không tồn tại", result.stdout)

    def test_resolve_internal_helper_handles_package_and_module_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "qdgrasp"
            write(root / "__init__.py", "")
            write(root / "utils" / "__init__.py", "")
            write(root / "engine" / "model.py", "")

            self.assertTrue(CHECKER_MODULE.resolve_internal(("qdgrasp",), root))
            self.assertTrue(CHECKER_MODULE.resolve_internal(("qdgrasp", "utils"), root))
            self.assertTrue(CHECKER_MODULE.resolve_internal(("ultralytics", "engine", "model"), root))
            self.assertFalse(CHECKER_MODULE.resolve_internal(("qdgrasp", "missing"), root))
            self.assertFalse(CHECKER_MODULE.resolve_internal(("numpy",), root))


if __name__ == "__main__":
    unittest.main()
