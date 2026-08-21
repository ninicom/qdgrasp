---
document_id: REV-20260821-003
document_type: revision_record
title: Chuẩn hóa dòng trống cuối ADR-0001
status: complete
date: 2026-08-21
record_id: REV-20260821-003
session_id: SESSION-20260821-003
created_at: 2026-08-21T23:20:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260821-001
    artifact: docs/decisions/0001-documentation-and-traceability.md
    revision: d404dad820edf87ab540a7cbf9481b72e5205ae55cfc1a8f37c896b4e248f6b7
reason: Git diff check phát hiện một dòng trống thừa ở cuối file trước commit feature tài liệu.
necessity: N0
impact: Chỉ thay byte kết thúc file; không đổi câu chữ, quyết định, trạng thái, số liệu hoặc hành vi.
---

# REV-20260821-003 — Chuẩn hóa EOF của ADR-0001

## Lý do chỉnh sửa

`git diff --cached --check` báo `new blank line at EOF` cho ADR-0001. Dòng trống
thừa được bỏ để feature commit vượt kiểm tra whitespace.

## Mức độ cần thiết

Mức `N0`: thay đổi định dạng thuần túy, không đổi nội dung hay kết luận.

## Phạm vi và tác động

- Trước sửa: SHA-256
  `d404dad820edf87ab540a7cbf9481b72e5205ae55cfc1a8f37c896b4e248f6b7`.
- Sau sửa: SHA-256
  `9d667fee0458a8f6c55fb02b11e20f262a8d90a3df3cba14d16ce32f1666a9bf`.
- Tác động semantic/runtime/API/data/metric: không.
- Review snapshot V2 tiếp tục áp dụng cho nội dung semantic; checksum mới được
  ghi rõ ở record này thay vì sửa im lặng.

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact |
|---|---|---|
| CH-001 | Bỏ một dòng trống thừa ở EOF | `docs/decisions/0001-documentation-and-traceability.md` |

## Xác minh

| Verification ID | Lệnh | Kết quả |
|---|---|---|
| V-001 | `git diff --cached --check` | exit 0 |
| V-002 | `python3 scripts/check_docs.py --root .` | phải pass trước commit |

## Ảnh hưởng tới báo cáo và quyết định cũ

Không đổi verdict hoặc quyết định. Đây là delta N0 có hash trước/sau; không cần
chạy lại model, metric hoặc full review.

## Xác nhận đóng hồ sơ

Thay đổi định dạng và hash đã được ghi; feature gate tiếp tục chạy trên working
tree hiện tại.
