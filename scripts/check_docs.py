#!/usr/bin/env python3
"""Validate the project's auditable Markdown documentation contracts.

The checker intentionally uses only the Python standard library.  It validates
published records under ``docs/reports``, ``docs/sessions``, ``docs/reviews``,
``docs/revisions`` and ``docs/metrics`` plus normative configuration registries,
while leaving templates alone.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


RECORD_DIRECTORIES = ("reports", "sessions", "reviews", "revisions", "metrics")
MANAGED_INDEX_DIRECTORIES = (
    "archive",
    "configuration",
    "reports",
    "reviews",
    "revisions",
    "sessions",
)
NORMATIVE_CATEGORIES = {
    "plan",
    "governance",
    "decision",
    "schema",
    "configuration",
    "index",
}
EXPECTED_NORMATIVE_TYPES = {
    "plan": "plan",
    "governance": "policy",
    "decision": "decision",
    "schema": "policy",
    "configuration": "registry",
    "index": "index",
}

COMMON_FIELDS = (
    ("document_id",),
    ("document_type",),
    ("title",),
    ("status",),
    ("date",),
    ("revises",),
)

SESSION_FIELDS = (
    ("session_id",),
    ("author",),
    ("related_plan",),
)
SESSION_SECTIONS = (
    "Mục tiêu phiên",
    "Việc đã hoàn tất",
    "Bằng chứng",
    "Kiểm tra đã chạy",
    "Việc chưa hoàn tất",
    "Sửa đổi phiên trước",
    "Bàn giao",
)

THIRD_PARTY_REVIEW_FIELDS = (
    ("report_id",),
    ("reviewed_session",),
    ("reviewed_revision",),
    ("reviewer",),
    ("reviewer_organization",),
    ("review_date",),
    ("independence",),
    ("verdict",),
    ("max_severity",),
)
THIRD_PARTY_REVIEW_SECTIONS = ("Bằng chứng", "Phát hiện", "Kết luận")

REVISION_RECORD_FIELDS = (
    ("record_id",),
    ("session_id",),
    ("created_at",),
    ("author",),
    ("reason",),
    ("necessity",),
    ("impact",),
)
REVISION_RECORD_SECTIONS = (
    "Lý do chỉnh sửa",
    "Mức độ cần thiết",
    "Phạm vi và tác động",
    "Xác minh",
)
REVISION_RECORD_V2_SECTIONS = (
    "Liên kết truy vết",
    "Lý do chỉnh sửa",
    "Mức độ cần thiết",
    "Phạm vi và tác động",
    "Nội dung thay đổi đã hoàn tất trong phiên này",
    "Xác minh",
    "Ảnh hưởng tới báo cáo và quyết định cũ",
    "Xác nhận đóng hồ sơ",
)

GENERIC_REPORT_FIELDS = (
    ("author",),
)
GENERIC_REPORT_SECTIONS = ("Tóm tắt", "Bằng chứng", "Kết luận")

METRICS_REGISTRY_SECTIONS = (
    "Định danh và vòng đời metric",
    "Trường bắt buộc của một metric",
    "Quy ước protocol, môi trường và artifact",
    "Quy tắc độ bất định",
    "Khả năng so sánh và vô hiệu hóa",
    "Registry khởi tạo",
    "Bảng kết quả bắt buộc",
    "Kiểm tra độc lập",
)
METRICS_REGISTRY_TERMS = (
    "Metric ID",
    "Protocol ID",
    "Sample unit",
    "Độ bất định",
    "Artifact thô",
    "SHA-256",
    "COMPARABLE",
    "QUALIFIED",
    "NOT_COMPARABLE",
    "VALID",
    "INVALID",
    "người kiểm tra",
)

METRICS_REPORT_SECTIONS = (
    "Thông tin báo cáo",
    "Việc hoàn thành trong phiên",
    "Chỉnh sửa công việc từ phiên trước",
    "Protocol và phạm vi mẫu",
    "Môi trường thực thi",
    "Kết quả metric",
    "Manifest artifact",
    "Vô hiệu hóa và thay thế",
    "Kiểm tra độc lập",
    "Lịch sử chỉnh sửa báo cáo",
)
METRICS_REPORT_FIELDS = (
    ("run_id",),
    ("author",),
    ("reviewer",),
)
METRICS_REPORT_TERMS = (
    "Report ID",
    "Metric ID",
    "Protocol ID",
    "Environment ID",
    " n ",
    "95% CI",
    "Raw artifact",
    "SHA-256",
    "COMPARABLE",
    "QUALIFIED",
    "NOT_COMPARABLE",
    "VALID",
    "INVALID",
    "Reviewer",
)

RECORD_STATUSES = {
    "draft", "in_review", "complete", "accepted", "superseded", "invalidated"
}
NORMATIVE_STATUSES = RECORD_STATUSES | {"active"}
NONE_VALUES = {
    "none",
    "null",
    "~",
    "n/a",
    "na",
    "no",
    "false",
    "không",
    "không có",
    "[]",
}
PLACEHOLDER_VALUES = {
    "",
    "tbd",
    "todo",
    "n/a",
    "na",
    "null",
    "unknown",
    "unassigned",
    "chưa có",
    "tên ngắn của phiên",
    "tên phiên",
    "tên-người-thực-hiện",
    "họ tên",
    "họ tên hoặc agent id",
    "tổ chức/nhóm",
    "tên phiên hoặc chủ đề",
}
PLACEHOLDER_TEXT_FRAGMENTS = (
    "tên phiên hoặc chủ đề",
    "tên ngắn của phiên",
    "tên-người-thực-hiện",
    "họ tên hoặc agent id",
)

PLACEHOLDER_TOKEN_RE = re.compile(
    r"(?:"
    r"YYYY(?:MMDD)?|YYYY-MM-DD|NNN|RUNID|AUTHOR-ID|SESSION-ID|"
    r"CURRENT-SESSION-ID|PREVIOUS-SESSION-ID|COMMIT-OR-CHECKSUM(?:-BEFORE)?|"
    r"REPORT-YYYY|RUN-YYYY|ENV-YYYY|<[^>]+>|\{[^}]+\}"
    r")",
    re.IGNORECASE,
)

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(?:[ \t]*(.*))?$")
NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?[ \t]+")


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    message: str


@dataclass
class MarkdownDocument:
    path: Path
    category: str
    text: str
    metadata: dict[str, str]
    metadata_lines: dict[str, int]
    headings: list[tuple[int, str, int]]
    front_matter_error: str | None = None

    @property
    def document_type(self) -> str:
        return clean_scalar(self.metadata.get("document_type", "")).casefold()

    @property
    def status(self) -> str:
        return clean_scalar(self.metadata.get("status", "")).casefold()


def clean_scalar(value: str) -> str:
    """Remove YAML-like scalar quoting without trying to implement YAML."""

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def normalized_heading(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = value.replace("`", "").replace("**", "").replace("__", "")
    value = NUMBERED_HEADING_RE.sub("", value)
    return value.rstrip(" :.\t").casefold()


def parse_front_matter(
    lines: Sequence[str],
) -> tuple[dict[str, str], dict[str, int], int, str | None]:
    """Parse top-level keys and raw continuation blocks from YAML front matter.

    Full YAML support is deliberately unnecessary: the documentation schemas
    use scalar top-level values plus the nested ``revises`` list.  Preserving a
    continuation block is enough to distinguish an empty/``none`` revision from
    an actual list without adding PyYAML as a bootstrap dependency.
    """

    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return {}, {}, 0, "thiếu YAML front matter mở đầu bằng '---'"

    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        return {}, {}, 0, "YAML front matter không có dấu '---' kết thúc"

    metadata: dict[str, str] = {}
    key_lines: dict[str, int] = {}
    current_key: str | None = None
    duplicates: list[str] = []

    for zero_index, raw_line in enumerate(lines[1:closing], start=1):
        match = TOP_LEVEL_KEY_RE.match(raw_line)
        if match:
            key = match.group(1).casefold()
            value = (match.group(2) or "").strip()
            if key in metadata:
                duplicates.append(key)
            else:
                metadata[key] = value
                key_lines[key] = zero_index + 1
            current_key = key
            continue

        if current_key and (raw_line.startswith((" ", "\t")) or not raw_line.strip()):
            continuation = raw_line.strip()
            if continuation:
                previous = metadata[current_key]
                metadata[current_key] = f"{previous}\n{continuation}".strip()

    error = None
    if duplicates:
        error = "metadata trùng khóa: " + ", ".join(sorted(set(duplicates)))
    return metadata, key_lines, closing + 1, error


def parse_headings(lines: Sequence[str], start: int = 0) -> list[tuple[int, str, int]]:
    headings: list[tuple[int, str, int]] = []
    fence: str | None = None

    for zero_index, line in enumerate(lines[start:], start=start):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence is not None:
            continue
        match = HEADING_RE.match(line)
        if match:
            headings.append((len(match.group(1)), match.group(2).strip(), zero_index + 1))
    return headings


def load_document(path: Path, category: str) -> MarkdownDocument:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    metadata, metadata_lines, body_start, front_matter_error = parse_front_matter(lines)
    return MarkdownDocument(
        path=path,
        category=category,
        text=text,
        metadata=metadata,
        metadata_lines=metadata_lines,
        headings=parse_headings(lines, body_start),
        front_matter_error=front_matter_error,
    )


def first_h1(document: MarkdownDocument) -> str:
    return next((title for level, title, _line in document.headings if level == 1), "")


def is_placeholder(value: str) -> bool:
    value = clean_scalar(value)
    folded = value.casefold()
    return (
        folded in PLACEHOLDER_VALUES
        or any(fragment in folded for fragment in PLACEHOLDER_TEXT_FRAGMENTS)
        or PLACEHOLDER_TOKEN_RE.search(value) is not None
        or (value.startswith("<") and value.endswith(">"))
        or (value.startswith("{") and value.endswith("}"))
    )


def has_revision(value: str) -> bool:
    return clean_scalar(value).casefold() not in NONE_VALUES


def require_metadata(
    document: MarkdownDocument,
    groups: Sequence[Sequence[str]],
    issues: list[Issue],
) -> None:
    for alternatives in groups:
        present_key = next((key for key in alternatives if key in document.metadata), None)
        label = " | ".join(alternatives)
        if present_key is None:
            issues.append(Issue(document.path, 1, f"thiếu metadata bắt buộc: {label}"))
            continue
        if not clean_scalar(document.metadata[present_key]):
            issues.append(
                Issue(
                    document.path,
                    document.metadata_lines.get(present_key, 1),
                    f"metadata '{present_key}' không được để trống",
                )
            )
        elif (
            present_key != "revises"
            and bool(document.status)
            and document.status != "draft"
            and is_placeholder(document.metadata[present_key])
        ):
            issues.append(
                Issue(
                    document.path,
                    document.metadata_lines.get(present_key, 1),
                    f"metadata '{present_key}' còn là placeholder trong tài liệu đã hoàn tất",
                )
            )


def require_sections(
    document: MarkdownDocument, required: Iterable[str], issues: list[Issue]
) -> None:
    actual = {
        normalized_heading(title)
        for level, title, _line in document.headings
        if level == 2
    }
    missing = [title for title in required if normalized_heading(title) not in actual]
    if missing:
        issues.append(
            Issue(document.path, 1, "thiếu mục H2 bắt buộc: " + "; ".join(missing))
        )


def require_terms(
    document: MarkdownDocument, required: Iterable[str], issues: list[Issue]
) -> None:
    haystack = unicodedata.normalize("NFKC", document.text).casefold()
    missing = [term for term in required if term.casefold() not in haystack]
    if missing:
        issues.append(
            Issue(document.path, 1, "thiếu trường/giá trị bắt buộc: " + "; ".join(missing))
        )


def require_choice(
    document: MarkdownDocument,
    key: str,
    allowed: set[str],
    issues: list[Issue],
) -> None:
    if key not in document.metadata:
        return
    value = clean_scalar(document.metadata[key])
    if value.casefold() not in {item.casefold() for item in allowed}:
        issues.append(
            Issue(
                document.path,
                document.metadata_lines.get(key, 1),
                f"metadata '{key}' phải là một trong: {', '.join(sorted(allowed))}",
            )
        )


def validate_revision_fields(
    document: MarkdownDocument,
    issues: list[Issue],
    *,
    reason_keys: Sequence[str],
    necessity_values: set[str],
) -> None:
    revises = document.metadata.get("revises")
    if revises is None or not has_revision(revises):
        return

    required = (tuple(reason_keys), ("necessity",), ("impact",))
    before = len(issues)
    require_metadata(document, required, issues)

    # Revision accountability is mandatory even for drafts: a record may be
    # incomplete, but it may not claim to revise previous work without saying
    # why, how necessary the change is, and what it affects.
    for alternatives in required:
        key = next((candidate for candidate in alternatives if candidate in document.metadata), None)
        if key and is_placeholder(document.metadata[key]):
            issue = Issue(
                document.path,
                document.metadata_lines.get(key, 1),
                f"metadata '{key}' phải có giá trị cụ thể khi 'revises' khác none",
            )
            if issue not in issues[before:]:
                issues.append(issue)
    require_choice(document, "necessity", necessity_values, issues)


def validate_session(document: MarkdownDocument, issues: list[Issue]) -> None:
    require_metadata(document, SESSION_FIELDS, issues)
    require_sections(document, SESSION_SECTIONS, issues)
    if document.document_type and document.document_type != "session_report":
        issues.append(
            Issue(document.path, document.metadata_lines.get("document_type", 1),
                  "document_type trong docs/sessions phải là 'session_report'")
        )


def validate_third_party_review(document: MarkdownDocument, issues: list[Issue]) -> None:
    require_metadata(document, THIRD_PARTY_REVIEW_FIELDS, issues)
    require_sections(document, THIRD_PARTY_REVIEW_SECTIONS, issues)
    require_choice(document, "independence", {"external", "internal_independent"}, issues)
    require_choice(
        document,
        "verdict",
        {"pass", "conditional_pass", "fail", "inconclusive"},
        issues,
    )
    require_choice(
        document,
        "max_severity",
        {"S0", "S1", "S2", "S3", "INFO", "NONE"},
        issues,
    )


def validate_revision_record(document: MarkdownDocument, issues: list[Issue]) -> None:
    require_metadata(document, REVISION_RECORD_FIELDS, issues)
    revision_schema = clean_scalar(document.metadata.get("revision_schema", "1"))
    if revision_schema not in {"1", "2"}:
        issues.append(
            Issue(
                document.path,
                document.metadata_lines.get("revision_schema", 1),
                "revision_schema phải là '1' hoặc '2'",
            )
        )
    require_sections(
        document,
        REVISION_RECORD_V2_SECTIONS if revision_schema == "2" else REVISION_RECORD_SECTIONS,
        issues,
    )
    if "revises" in document.metadata and not has_revision(document.metadata["revises"]):
        issues.append(
            Issue(
                document.path,
                document.metadata_lines.get("revises", 1),
                "revision_record phải liên kết ít nhất một phiên/artifact trong 'revises'",
            )
        )


def validate_metrics_registry(document: MarkdownDocument, issues: list[Issue]) -> None:
    if document.document_type and document.document_type != "registry":
        issues.append(
            Issue(
                document.path,
                document.metadata_lines.get("document_type", 1),
                "Metrics Registry phải có document_type 'registry'",
            )
        )
    require_sections(document, METRICS_REGISTRY_SECTIONS, issues)
    require_terms(document, METRICS_REGISTRY_TERMS, issues)


def validate_metrics_report(document: MarkdownDocument, issues: list[Issue]) -> None:
    require_metadata(document, METRICS_REPORT_FIELDS, issues)
    if document.document_type and document.document_type != "metrics_report":
        issues.append(
            Issue(
                document.path,
                document.metadata_lines.get("document_type", 1),
                "báo cáo số liệu phải có document_type 'metrics_report'",
            )
        )
    require_sections(document, METRICS_REPORT_SECTIONS, issues)
    require_terms(document, METRICS_REPORT_TERMS, issues)


def validate_generic_report(document: MarkdownDocument, issues: list[Issue]) -> None:
    require_metadata(document, GENERIC_REPORT_FIELDS, issues)
    require_sections(document, GENERIC_REPORT_SECTIONS, issues)


def validate_common_metadata(
    document: MarkdownDocument,
    issues: list[Issue],
    *,
    normative: bool,
) -> None:
    require_metadata(document, COMMON_FIELDS, issues)
    require_choice(
        document,
        "status",
        NORMATIVE_STATUSES if normative else RECORD_STATUSES,
        issues,
    )

    expected_type = EXPECTED_NORMATIVE_TYPES.get(document.category)
    if expected_type and document.document_type and document.document_type != expected_type:
        issues.append(
            Issue(
                document.path,
                document.metadata_lines.get("document_type", 1),
                f"tài liệu nhóm '{document.category}' phải có document_type '{expected_type}'",
            )
        )

    if document.document_type == "session_report":
        reason_keys = ("revision_reason",)
    elif document.document_type == "revision_record":
        reason_keys = ("reason",)
    else:
        reason_keys = ("revision_reason", "reason")
    validate_revision_fields(
        document,
        issues,
        reason_keys=reason_keys,
        necessity_values={"N0", "N1", "N2", "N3"},
    )


def validate_document(document: MarkdownDocument) -> list[Issue]:
    issues: list[Issue] = []
    h1 = normalized_heading(first_h1(document))
    metrics_registry = document.category == "metrics" and h1.startswith(
        "sổ đăng ký chỉ số"
    )

    if document.front_matter_error:
        issues.append(Issue(document.path, 1, document.front_matter_error))
        return issues

    normative = document.category in NORMATIVE_CATEGORIES or metrics_registry
    validate_common_metadata(document, issues, normative=normative)

    if document.category in NORMATIVE_CATEGORIES:
        return issues
    if metrics_registry:
        validate_metrics_registry(document, issues)
    elif h1.startswith("báo cáo số liệu") or document.document_type == "metrics_report":
        validate_metrics_report(document, issues)
    elif document.category == "sessions":
        validate_session(document, issues)
    elif document.document_type == "third_party_review":
        validate_third_party_review(document, issues)
    elif document.document_type == "revision_record":
        validate_revision_record(document, issues)
    elif document.category == "reviews":
        issues.append(
            Issue(
                document.path,
                document.metadata_lines.get("document_type", 1),
                "docs/reviews chỉ nhận document_type 'third_party_review' hoặc 'revision_record'",
            )
        )
    elif document.category == "revisions":
        issues.append(
            Issue(
                document.path,
                document.metadata_lines.get("document_type", 1),
                "docs/revisions chỉ nhận document_type 'revision_record'",
            )
        )
    else:
        validate_generic_report(document, issues)
    return issues


def discover_documents(root: Path) -> list[tuple[Path, str]]:
    docs_root = root / "docs"
    discovered: dict[Path, str] = {}

    def add(path: Path, category: str) -> None:
        if path.is_file() and not path.name.startswith(("_", ".")):
            discovered[path] = category

    add(root / "PLAN.md", "plan")
    add(docs_root / "README.md", "index")
    add(docs_root / "schemas" / "README.md", "schema")

    for path in sorted((docs_root / "governance").glob("*.md")):
        add(path, "governance")
    for path in sorted((docs_root / "decisions").glob("*.md")):
        add(path, "decision")
    for path in sorted((docs_root / "configuration").glob("*.md")):
        if path.name.casefold() != "readme.md":
            add(path, "configuration")
    for directory in MANAGED_INDEX_DIRECTORIES:
        add(docs_root / directory / "README.md", "index")

    for category in RECORD_DIRECTORIES:
        category_root = docs_root / category
        if not category_root.is_dir():
            continue
        for path in sorted(category_root.rglob("*.md")):
            relative_parts = path.relative_to(category_root).parts
            if (
                path.name.casefold() == "readme.md"
                or path.name.startswith(("_", "."))
                or any(part.casefold() == "templates" for part in relative_parts[:-1])
            ):
                continue
            add(path, category)

    return [(path, discovered[path]) for path in sorted(discovered)]


def validate_root(root: Path) -> tuple[list[Issue], int]:
    issues: list[Issue] = []
    documents = discover_documents(root)
    for path, category in documents:
        try:
            document = load_document(path, category)
        except (OSError, UnicodeError) as exc:
            issues.append(Issue(path, 1, f"không đọc được UTF-8 Markdown: {exc}"))
            continue
        issues.extend(validate_document(document))
    return issues, len(documents)


def relative_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Kiểm tra metadata, section, metrics evidence và revision accountability "
            "trong bộ tài liệu dự án."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Thư mục gốc dự án chứa docs/ (mặc định: thư mục hiện tại).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: --root không phải thư mục: {root}", file=sys.stderr)
        return 2

    issues, checked = validate_root(root)
    if issues:
        for issue in sorted(issues, key=lambda item: (str(item.path), item.line, item.message)):
            print(f"{relative_display(issue.path, root)}:{issue.line}: ERROR: {issue.message}")
        print(f"Documentation check failed: {len(issues)} lỗi trong {checked} file.")
        return 1

    print(f"Documentation check passed: {checked} file đã kiểm tra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
