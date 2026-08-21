---
document_id: REV-20260822-001
document_type: revision_record
revision_schema: 2
title: Thay train-argument registry parser contract bằng dialect v2
status: in_review
date: 2026-08-22
record_id: REV-20260822-001
session_id: SESSION-20260822-001
created_at: 2026-08-22T00:16:53+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260821-007
    artifact: scripts/check_train_args.py
    revision: 34d8b1593a0c5fcfbea4117a746ab5da44acf2ee7b28be488932a401fabd85ad
  - session_id: SESSION-20260821-007
    artifact: scripts/tests/test_check_train_args.py
    revision: f50b5328d523f1043cd483bddd7f023f857c60b6ba87d5df17e0713579d16e58
  - session_id: SESSION-20260821-007
    artifact: docs/configuration/TRAIN_ARGUMENTS.yaml
    revision: b584afaccfb749450aac3a3954ce2b03e1a675b1fc76e3b6a0ed17a560223805
  - session_id: SESSION-20260821-007
    artifact: docs/configuration/TRAIN_ARGUMENTS.md
    revision: 3444e2738eefbf82a7cef4da60525d00799d4898084b0bc3647844cb3cd4dd8d
  - session_id: SESSION-20260821-007
    artifact: docs/revisions/REV-20260821-006-audit-remediation.md
    revision: 00500d75b087e1cfd51f3c6f4fca44e46c07097d802c56eb991a3db2b67e6a2b
reason: Independent delta audit TPR-20260822-001 reproduced typed scalar, invalid YAML-like syntax, normative Markdown and dirty-source false-negatives at fdb60eb.
necessity: N2
impact: Parser/fingerprint/document pin/source checkout guarantees change materially; inventory names/defaults and upstream commit do not change.
supersedes: REV-20260821-006
---

# REV-20260822-001 — Registry dialect v2

## Liên kết truy vết

- Phiên sửa: `SESSION-20260822-001`.
- Snapshot lỗi: `fdb60eb51e4793ee5764b272acc3a40960608846`, tree
  `2b26aed81b7f8647e85d2bc64b2a6ecf2da979f5`.
- Finding source: `TPR-20260822-001`, F-001/F-002/F-004/F-005/F-010.
- Revision trước: `REV-20260821-006` đã bị invalidated và giữ nguyên.
- Revision sau ở working tree:
  - checker `a77e57e26b277b777585dddb99aea7e0f8bb5b7b8764cb744ca440c85bf73c8d`;
  - tests `71c6ede6799c7e6011a342ffc6747820d28d07f0714317afbc402ce824321b4d`;
  - YAML `5fe682f1d84351564edf858c36bd1f6ac07005295234ad9b23f97d3081c48626`;
  - Markdown `a77ab9466ddf44fecca5d8d59410ab315b7423652f7ef5f42e4f07b55dad5480`.

## Lý do chỉnh sửa

Checker cũ unquote/strip scalar trước khi fingerprint, chấp nhận separator sai,
chỉ kiểm vài cột Markdown và không yêu cầu source checkout sạch khi chạy độc lập.
Do đó test xanh không chứng minh contract “zero omission/drift”. Đây là lỗi bằng
chứng và kết luận kỹ thuật, không phải biên tập.

## Mức độ cần thiết

- Mức: `N2`.
- Nếu không sửa: config khác kiểu hoặc source/docs khác nghĩa vẫn được phát hành.
- Thời hạn: trước merge feature hoặc alpha.2.
- Giảm thiểu: cấm merge mọi commit đang chịu verdict fail/S1.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa dự kiến | Tác động | Hành động |
|---|---|---|---|---|
| Runtime/model | Chưa có | Chưa có | Không | Không claim |
| Registry schema | Scalar chuẩn hóa text | Kind + lexeme, grammar nghiêm | Cao | Schema v2 |
| Markdown | Một phần row fields | Normative body hash | Cao | Version 1.2.0 |
| Source full mode | HEAD + 4 hashes | Origin + HEAD + clean tree + hashes | Cao | Dirty regression |
| Inventory | 127 tên | 127 tên | Không | Count/source tests |

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact | Evidence |
|---|---|---|---|
| CH-001 | Cấm quote/tab/trailing whitespace/sai separator | checker schema v2 | mutation tests |
| CH-002 | Fingerprint kind + lexeme cho toàn manifest | `a43bf2…7a10` | registry-only |
| CH-003 | Pin normalized normative Markdown body | `b10f58…13ae` | 5 doc probes |
| CH-004 | Khóa canonical `quantize_aliases` | YAML/Markdown | exact contract check |
| CH-005 | Full source kiểm origin/HEAD/clean status | checker | dirty fixture + real clone |

## Xác minh

| ID | Phương pháp | Mong đợi | Thực tế hiện tại | Trạng thái | Evidence |
|---|---|---|---|---|---|
| V-001 | Reviewer mutation suite | Mọi corruption fail | 7 dialect + 5 Markdown probes pass | pass | 30-test suite |
| V-002 | Full project gate | Zero failure | reference/registry/docs/30 tests/shell/diff pass | pass | console; replay sau commit |
| V-003 | PyYAML semantic spot check | int/bool/null đúng kiểu | pass | pass | local diagnostic |
| V-004 | Delta review | Pass/NONE | Chưa chạy | not_run | Chờ exact commit |

- Regression đã chạy lại: 30/30 cùng full source/reference/docs/shell/diff.
- Kiểm tra chưa chạy: independent review trên exact committed snapshot.
- Rollback: không merge; giữ commit thất bại chỉ làm audit evidence.

## Ảnh hưởng tới báo cáo và quyết định cũ

- `TPR-20260822-001` vẫn đúng cho `fdb60eb` và không bị sửa.
- Verdict cũ không có hiệu lực cho revision mới; bắt buộc delta review.
- Không có metric/model cần đính chính.
- Không có ngoại lệ/risk acceptance.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22 Asia/Bangkok.
- Người kiểm tra: sẽ là independent delta reviewer, không tham gia implementation.
- Kết luận: `in_review`; implementation/local gates pass, còn exact
  commit/evidence và independent verdict.
- Bản ghi phiên: `SESSION-20260822-001`.
