---
document_id: SESSION-20260822-005
document_type: session_report
title: Sửa căn cứ pháp lý sang public AGPL-3.0, chuẩn bị vendor import
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-005
author: codex-primary-agent
revises:
  - SESSION-20260822-004
  - REV-20260822-004
revision_reason: Maintainer đảo ngược tiền đề "internal-only" của REV-004 ngay sau khi ghi — dự án thật ra sẽ public cho cộng đồng dưới AGPL-3.0. Căn cứ pháp lý cũ sai với hành vi thật sắp diễn ra, phải sửa trước khi commit vendor import.
necessity: N3
impact: Thay căn cứ pháp lý cho phép fork Ultralytics; không đổi kết luận "được phép fork" nhưng đổi hoàn toàn lý do; DGN2 CC-BY-NC-4.0 vẫn là gate riêng chưa giải quyết.
related_plan: PLAN-V2
---

# SESSION-20260822-005 — Sửa căn cứ pháp lý, chuẩn bị vendor import

## Mục tiêu phiên

Sửa đúng quy trình (revision record supersede, không tự ý sửa PLAN.md chay)
cho việc maintainer đổi ý về mục đích dự án: public cho cộng đồng dưới
AGPL-3.0, không phải nội bộ như REV-004 đã ghi. Chuẩn bị sẵn vendor import
(365 file từ `.references/ultralytics` tại commit `329682a`) nhưng KHÔNG commit
cho tới khi căn cứ pháp lý đúng tồn tại.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Phát hiện maintainer đảo ngược tiền đề, không tự suy đoán hay im lặng chấp nhận | chat 2026-08-22 | verified |
| W-02 | Hỏi lại license public cụ thể bằng `AskUserQuestion` thay vì tự chọn | "AGPL-3.0 (copyleft, giống Ultralytics)" | verified |
| W-03 | Viết `REV-20260822-005` supersede `REV-20260822-004` theo template (8 mục) | `docs/revisions/REV-20260822-005-agpl-public-release-basis.md` | `check_docs.py` pass |
| W-04 | Sửa `PLAN.md` §1 + M0: ghi rõ public AGPL-3.0 và DGN2 CC-BY-NC-4.0 còn mở, bump `2.2.0 → 2.3.0` | `PLAN.md` | `check_docs.py` pass |
| W-05 | Unstage vendor import đã chuẩn bị trước (365 file), giữ nguyên trên đĩa, chưa commit | `vendor/ultralytics-329682a/` (untracked) | `git status` |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | maintainer decision (verbal/chat) | "public cho cộng đồng... AGPL-3.0" | tier E4 — yếu, đủ ghi quyết định phạm vi, không đủ cho tuyên bố pháp lý cuối cùng |
| E-02 | PLAN.md trước sửa (REV-004 state) | commit trên `feature/ultralytics-fork-import` trước phiên này | `edc98c5eba522eda5d67e9b957ebf54676213a14538b99c78601c3a15efe1ca2` |
| E-03 | PLAN.md sau sửa | working tree | `7ef0353f2233f8ef407381b0260a5d65bc6264edf09c19703074b6b9f254b922` |
| E-04 | revision record | `REV-20260822-005` | supersedes REV-004, ghi rõ nguyên nhân đảo ngược |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `sha256sum PLAN.md` (trước và sau edit) | 0 | khớp E-02/E-03 |
| T-02 | `python3 scripts/check_docs.py --root .` | 0 | `44 file đã kiểm tra` |
| T-03 | `git status --short` | 0 | xác nhận `vendor/ultralytics-329682a/` untracked, chưa `git add` |

## Việc chưa hoàn tất

- Commit `PLAN.md`/`REV-005`/`SESSION-005` (đang chuẩn bị ngay sau phiên này).
- Commit vendor import (365 file, đã chuẩn bị sẵn từ phiên trước, chờ commit
  riêng sau khi căn cứ pháp lý này được commit).
- `references.lock.yaml` còn `distribution_status: blocked_pending_commercial_license`
  / `blocked_noncommercial_terms_and_pending_legal_review` — mâu thuẫn với
  quyết định public AGPL-3.0 (theo hướng khác REV-004). KHÔNG tự sửa; cần
  maintainer quyết định giá trị mới vì field này bị hash-lock bởi
  `check_references.py`.
- Đánh giá pháp lý DexGraspNet2 (CC-BY-NC-4.0) — chưa làm, vẫn là gate riêng.
- Đánh giá pháp lý chính thức bằng văn bản cho lựa chọn AGPL-3.0 — khuyến nghị,
  chưa bắt buộc theo quyết định hiện tại.

## Sửa đổi phiên trước

Có, mức `N3`. `REV-20260822-005` supersede `REV-20260822-004` — không xóa,
chỉ ghi đè căn cứ pháp lý. `SESSION-20260822-004` được giữ nguyên làm hồ sơ
lịch sử của lần đảo tiền đề đầu tiên.

## Bàn giao

Căn cứ pháp lý đúng (public AGPL-3.0) đã ghi nhận. Bước tiếp theo: commit các
file tài liệu của phiên này, sau đó commit vendor import riêng
(`SESSION-20260822-006`) trích dẫn rõ REV-005 làm căn cứ. Không public/release
thật cho tới khi có đánh giá pháp lý bằng văn bản và DGN2 được xử lý riêng.
