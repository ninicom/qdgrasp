from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT_ROOT / "scripts" / "check_docs.py"
SESSION_TEMPLATE = PROJECT_ROOT / "docs" / "templates" / "SESSION_REPORT.md"


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


class DocumentationValidatorTests(unittest.TestCase):
    def test_complete_review_allows_no_open_severity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            review_directory = root / "docs" / "reviews"
            review_directory.mkdir(parents=True)
            (review_directory / "TPR-20260821-999-valid.md").write_text(
                """---
document_id: TPR-20260821-999
document_type: third_party_review
title: Review fixture không có finding mở
status: complete
date: 2026-08-21
revises: none
report_id: TPR-20260821-999
reviewed_session: SESSION-20260821-999
reviewed_revision: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
reviewer: validator-test-reviewer
reviewer_organization: validator-test-team
review_date: 2026-08-21
independence: internal_independent
verdict: pass
max_severity: NONE
---

# TPR-20260821-999 — Review fixture

## Bằng chứng

Fixture có metadata cụ thể.

## Phát hiện

Không có phát hiện mở.

## Kết luận

Verdict pass và severity cao nhất là NONE.
""",
                encoding="utf-8",
            )

            result = run_checker(root)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_completed_template_with_placeholders_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session_directory = root / "docs" / "sessions"
            session_directory.mkdir(parents=True)
            target = session_directory / "SESSION-20260821-999-template-copy.md"
            shutil.copy2(SESSION_TEMPLATE, target)
            target.write_text(
                target.read_text(encoding="utf-8").replace(
                    "status: draft", "status: complete", 1
                ),
                encoding="utf-8",
            )

            result = run_checker(root)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("placeholder", result.stdout)

    def test_completed_session_with_concrete_metadata_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session_directory = root / "docs" / "sessions"
            session_directory.mkdir(parents=True)
            (session_directory / "SESSION-20260821-999-valid.md").write_text(
                """---
document_id: SESSION-20260821-999
document_type: session_report
title: Kiểm tra fixture hợp lệ
status: complete
date: 2026-08-21
session_id: SESSION-20260821-999
author: validator-test
revises: none
related_plan: PLAN-V2
---

# SESSION-20260821-999 — Fixture hợp lệ

## Mục tiêu phiên

Xác minh validator.

## Việc đã hoàn tất

Validator đã được chạy.

## Bằng chứng

Exit code của tiến trình kiểm tra.

## Kiểm tra đã chạy

Lệnh validator hiện tại.

## Việc chưa hoàn tất

Không.

## Sửa đổi phiên trước

Không.

## Bàn giao

Fixture kết thúc tại đây.
""",
                encoding="utf-8",
            )

            result = run_checker(root)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_managed_index_without_revises_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_directory = root / "docs"
            docs_directory.mkdir(parents=True)
            (docs_directory / "README.md").write_text(
                """---
document_id: TEST-INDEX
document_type: index
title: Chỉ mục thử nghiệm
status: active
date: 2026-08-21
---

# Chỉ mục thử nghiệm
""",
                encoding="utf-8",
            )

            result = run_checker(root)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("revises", result.stdout)

    def test_revision_schema_two_requires_all_template_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            revisions = root / "docs" / "revisions"
            revisions.mkdir(parents=True)
            (revisions / "REV-20260821-999-incomplete.md").write_text(
                """---
document_id: REV-20260821-999
document_type: revision_record
revision_schema: 2
title: Revision schema two fixture
status: complete
date: 2026-08-21
record_id: REV-20260821-999
session_id: SESSION-20260821-999
created_at: 2026-08-21T23:00:00+07:00
author: validator-test
revises: SESSION-20260821-998
reason: Fixture kiểm tra section bắt buộc.
necessity: N2
impact: Chỉ ảnh hưởng fixture.
---

# Revision fixture

## Lý do chỉnh sửa

Fixture.

## Mức độ cần thiết

N2.

## Phạm vi và tác động

Fixture.

## Xác minh

Fixture.
""",
                encoding="utf-8",
            )

            result = run_checker(root)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("Liên kết truy vết", result.stdout)
            self.assertIn("Xác nhận đóng hồ sơ", result.stdout)


if __name__ == "__main__":
    unittest.main()
