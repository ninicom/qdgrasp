---
document_id: SESSION-20260822-004
document_type: session_report
title: Ghi nhận quyết định phạm vi nội bộ, bỏ gate license PLAN.md
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-004
author: codex-primary-agent
revises:
  - SESSION-20260821-007
  - REV-20260821-006
revision_reason: Maintainer quyết định dự án chỉ dùng nội bộ, không phát hành/phân phối, nên câu release gate "văn bản cấp phép thương mại" trong PLAN.md §1 không còn áp dụng cho phạm vi hiện tại; PLAN.md đã bị sửa trực tiếp trên đĩa trước khi có revision record.
necessity: N3
impact: Mở khóa bắt đầu fork/sửa code Ultralytics vào dexgrasp/ trong nội bộ; không giải quyết AGPL-3.0 network-use clause hay CC-BY-NC-4.0 của DGN2; không tự cho phép phát hành.
related_plan: PLAN-V2
---

# SESSION-20260822-004 — Quyết định phạm vi nội bộ, bỏ gate license

## Mục tiêu phiên

Ghi nhận đúng quy trình (revision record + session report) cho việc maintainer
xóa câu license-gate khỏi `PLAN.md` §1 và tuyên bố lý do (dự án chỉ dùng nội
bộ, không phân phối). Không tự sửa code Ultralytics/DGN2 trong phiên này;
không tự sửa `references.lock.yaml`.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Phát hiện `PLAN.md` bị sửa trực tiếp trên đĩa (working tree), chưa commit, chưa có revision record | `git diff PLAN.md` | verified |
| W-02 | Hỏi maintainer lý do cụ thể thay vì tự suy đoán/bịa | câu trả lời trong chat: "Dự án chỉ dùng nội bộ, không phát hành/phân phối" | verified |
| W-03 | Viết `REV-20260822-004` theo đúng template `REVISION_RECORD.md` (revision_schema 2, đủ 8 mục) | `docs/revisions/REV-20260822-004-plan-internal-only-scope.md` | `check_docs.py` pass |
| W-04 | Viết session report này theo đúng template `SESSION_REPORT.md` (đủ 7 mục) | `docs/sessions/SESSION-20260822-004-plan-internal-only-scope.md` | `check_docs.py` pass |
| W-05 | Bump front matter `PLAN.md` (`version → 2.2.0`, `latest_revision_record → REV-004`), giữ nguyên phần còn lại theo tiền lệ REV-006 | `PLAN.md` | `check_docs.py` pass |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | maintainer decision (verbal/chat, không văn bản) | chat 2026-08-22, "đã cập nhật plan, không cần lience" | tier E4 — yếu, không đủ cho tuyên bố định lượng, chỉ đủ ghi quyết định phạm vi |
| E-02 | PLAN.md trước sửa | commit `5d86a426ff7f250bc7a77e80e241610d10a9958a` | `3bb353e6db014bc8742f912c5e33de8ea4167762d917f43338a2b1e4df24b084` |
| E-03 | PLAN.md sau sửa (working tree, chưa bump front matter) | `PLAN.md` | `f3a172a7963f0c53204322e6357ae4ae984f6ec291c25497ff2aae69933de7cd` |
| E-04 | revision record | `REV-20260822-004` | ghi nguyên văn câu bị xóa và lý do |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `git show HEAD:PLAN.md \| sha256sum` | 0 | khớp E-02 |
| T-02 | `sha256sum PLAN.md` | 0 | khớp E-03 |
| T-03 | `sha256sum references.lock.yaml` | 0 | `921e798163abace643c8a825559425e1bbd16d19476e1464436615fa00b193b7`, KHÔNG đổi — vẫn khớp `LOCK_SHA256` pin trong `scripts/check_references.py`, chưa sửa field `distribution_status` |
| T-04 | `python3 scripts/check_docs.py --root .` | 0 | `44 file đã kiểm tra` (sau khi bump front matter `PLAN.md`) |
| T-05 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | 0 | `38/38 pass`, không hồi quy |

## Việc chưa hoàn tất

- `references.lock.yaml` còn `distribution_status: blocked_pending_commercial_license`
  (ultralytics) / `blocked_noncommercial_terms_and_pending_legal_review`
  (dexgraspnet2) — MÂU THUẪN với quyết định mới. KHÔNG tự sửa field này; cần
  maintainer quyết định rõ ràng vì đây là field bằng chứng pháp lý bị hash-lock
  bởi `check_references.py` (`LOCK_SHA256`), sửa sai cách sẽ làm gate hỏng.
- Chưa bắt đầu fork/sửa code Ultralytics vào `dexgrasp/`; cần lệnh xác nhận rõ
  ràng riêng từ maintainer, không suy đoán từ revision này.
- Đánh giá AGPL-3.0 §13 (network use) và CC-BY-NC-4.0 (DGN2) cho use case cụ
  thể — chưa làm, đã ghi caveat trong `REV-20260822-004`.

## Sửa đổi phiên trước

Có, mức `N3`. `REV-20260822-004` ghi rõ nguyên nhân (quyết định phạm vi của
maintainer, không phải sửa lỗi kỹ thuật), before/after hash của `PLAN.md` và
tác động. `SESSION-20260821-007`/`REV-20260821-006` được giữ nguyên, không sửa
lại kết luận cũ của các phiên đó.

## Bàn giao

Quyết định đã được ghi nhận đúng quy trình nhưng CHƯA hoàn tất: còn front-matter
bump, `check_docs.py` replay, và quan trọng nhất — vẫn giữ gate "không đụng
code Ultralytics/DGN2 tới khi có lệnh xác nhận riêng của maintainer". Không
dùng revision này để tự suy ra rằng fork đã được phép bắt đầu.
