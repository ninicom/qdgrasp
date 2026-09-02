from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT_ROOT / "scripts" / "check_references.py"
LOCK = PROJECT_ROOT / "references.lock.yaml"


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


class ReferenceSourceCheckerTests(unittest.TestCase):
    def test_current_lock_is_immutable_and_valid(self) -> None:
        result = run_checker("--lock-only")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("immutable lock", result.stdout)

    def test_modified_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "references.lock.yaml"
            target.write_text(
                LOCK.read_text(encoding="utf-8").replace("schema_version: 1", "schema_version: 2", 1),
                encoding="utf-8",
            )
            result = run_checker("--lock", str(target), "--lock-only")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("SHA-256 sai", result.stdout)

    @unittest.skipUnless(
        (PROJECT_ROOT / ".references").is_dir(),
        "reference clones không có trong source archive",
    )
    def test_required_local_checkouts_match_lock(self) -> None:
        result = run_checker("--source-root", str(PROJECT_ROOT / ".references"))

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("clean pinned HEAD/hash/origin", result.stdout)


if __name__ == "__main__":
    unittest.main()
