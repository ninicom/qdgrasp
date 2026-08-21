---
document_id: REV-20260821-002
document_type: revision_record
title: Cho phép max_severity NONE trong review hoàn tất
status: complete
date: 2026-08-21
record_id: REV-20260821-002
session_id: SESSION-20260821-002
created_at: 2026-08-21T23:10:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260821-001
    artifact: scripts/check_docs.py
    revision: 2f87c77d70025412ae44f4665071f4b646c4d907fa81a2177ee2bf5f7d6895c3
  - session_id: SESSION-20260821-001
    artifact: scripts/tests/test_check_docs.py
    revision: fd298aaa25cea581f773b71501169de3f5ac1327b663a93dae51736b543eb6f8
reason: Delta-review tạo report hợp lệ với max_severity NONE nhưng validator đồng thời coi NONE là placeholder và trả exit code 1.
necessity: N2
impact: Sửa contract validator cho enum hợp lệ của third-party review và thêm regression test; không thay đổi verdict, dữ liệu, model hoặc runtime.
---

# REV-20260821-002 — Sửa xung đột enum `NONE`

## Liên kết truy vết

- Phiên bị sửa: `SESSION-20260821-001`.
- Phiên sửa: `SESSION-20260821-002`.
- Finding nguồn: blocker sau khi tạo
  `docs/reviews/TPR-20260821-002-documentation-delta.md`.
- Checker trước sửa:
  `2f87c77d70025412ae44f4665071f4b646c4d907fa81a2177ee2bf5f7d6895c3`.
- Checker sau sửa:
  `b1df42bff93c453baec4a67e88c2a4edf86820c2a5ed07d3c2835d770dc0564a`.
- Test trước sửa:
  `fd298aaa25cea581f773b71501169de3f5ac1327b663a93dae51736b543eb6f8`.
- Test sau sửa:
  `b85effd3e6a8425750d0c89b748c2996118351c9f5a44a0e0a0610ccded49425`.

## Lý do chỉnh sửa

`NONE` là giá trị hợp lệ của `max_severity` khi review không còn finding mở.
Validator cũ đặt chuỗi `none` trong tập placeholder chung nên một report
`status: complete`, `verdict: pass`, `max_severity: NONE` không thể vượt chính
validator, dù enum review cho phép giá trị đó.

## Mức độ cần thiết

- Mức: `N2` — lỗi contract làm từ chối một record hợp lệ.
- Nếu không sửa: mọi review pass không có severity mở phải dùng metadata sai
  hoặc chấp nhận release gate thất bại.
- Phạm vi: validator tài liệu và test; không đổi policy N0–N3 hay nội dung model.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Tác động |
|---|---|---|---|
| Placeholder metadata | `none` luôn bị từ chối ở record hoàn tất | `NONE` được dùng hợp lệ cho `max_severity`; `revises: none` vẫn được xử lý riêng | Contract nhất quán |
| Review validation | Report pass/không finding không thể pass checker | Report hợp lệ pass checker | Đóng release gate giả |
| Regression | Không có fixture cho enum `NONE` | Có positive review fixture | Ngăn tái diễn |

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact | Evidence |
|---|---|---|---|
| CH-001 | Loại `none` khỏi danh sách placeholder chung | `scripts/check_docs.py` | hash sau sửa ở trên |
| CH-002 | Thêm test review hoàn tất với `max_severity: NONE` | `scripts/tests/test_check_docs.py` | hash sau sửa ở trên |

## Xác minh

| Verification ID | Lệnh | Kết quả | Trạng thái |
|---|---|---|---|
| V-001 | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` | 4/4 test pass | pass |
| V-002 | `PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_docs.py --root .` | `Documentation check passed: 17 file đã kiểm tra.` | pass |

Không chạy model/runtime test vì thay đổi chỉ nằm trong validator tài liệu.

## Ảnh hưởng tới báo cáo và quyết định cũ

- `TPR-20260821-002` vẫn áp dụng đúng cho snapshot V2 mà nó review.
- Checker mới là revision kế tiếp; phải nằm trong release snapshot và được chạy
  lại trước khi merge release vào `main`.
- Loại review lại: delta/release review; không cần fact-check model cho thay đổi
  validator này.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-21 (Asia/Bangkok).
- Kết luận: sửa đổi và regression test đã hoàn tất; release gate phải chạy lại
  trên commit tích hợp.
