---
document_id: REV-20260822-002
document_type: revision_record
revision_schema: 2
title: Khóa Git index flags và train registry front matter
status: in_review
date: 2026-08-22
record_id: REV-20260822-002
session_id: SESSION-20260822-002
created_at: 2026-08-22T00:37:08+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-001
    artifact: scripts/check_train_args.py
    revision: a77e57e26b277b777585dddb99aea7e0f8bb5b7b8764cb744ca440c85bf73c8d
  - session_id: SESSION-20260822-001
    artifact: scripts/tests/test_check_train_args.py
    revision: 71c6ede6799c7e6011a342ffc6747820d28d07f0714317afbc402ce824321b4d
  - session_id: SESSION-20260822-001
    artifact: docs/configuration/TRAIN_ARGUMENTS.md
    revision: a77ab9466ddf44fecca5d8d59410ab315b7423652f7ef5f42e4f07b55dad5480
  - session_id: SESSION-20260822-001
    artifact: docs/revisions/REV-20260822-001-registry-dialect-v2.md
    revision: 6282371d09e0186e83b4998a6ee91c4a633596d366cfba6c8e3f1e9415e3ef03
reason: TPR-20260822-002 reproduced source drift hidden by Git index flags and unvalidated governance-significant Markdown front matter at 363a671.
necessity: N2
impact: Standalone source and documentation identity guarantees change materially; machine registry inventory/semantics and runtime claims remain unchanged.
supersedes: REV-20260822-001
---

# REV-20260822-002 — Source/metadata hardening

## Liên kết truy vết

- Phiên sửa: `SESSION-20260822-002`.
- Snapshot lỗi: `363a671734ea4d2c89d399cec000fd97d4750870`, tree
  `dfb454f1d0fb5af3252a423ff3a563d57131a2a6`.
- Finding source: `TPR-20260822-002`, F-005/F-010/F-011.
- Revision trước: REV-001 invalidated và giữ nguyên.
- Revision sau ở working tree:
  - checker `99b1b8dde330e011cf0e9e3fb0693ac25aa7c6fd672e69249b72baa1c78a2f29`;
  - tests `76f3480f483963b6d318fbb60ce72e25aa98123259867ecc3b8900e3e3c8fdd9`;
  - Markdown `6911f367e8d068200ecaa4de4410922712614805a3f83971c7c98c88de841b17`.

## Lý do chỉnh sửa

Git porcelain bỏ qua file mang `assume-unchanged`/`skip-worktree`, nên clean
claim có thể sai dù origin/HEAD đúng. Markdown body hash không bao phủ front
matter, trong khi checker chỉ so bảy field và generic validator không bắt pointer
thiếu/giả, ngày/title sai hoặc unknown claim. Hai lỗi làm acceptance guarantee
không đầy đủ dù nội dung inventory hiện tại đúng.

## Mức độ cần thiết

- Mức: `N2`.
- Nếu không sửa: clone bị đổi semantic và claim runtime giả vẫn có thể pass gate.
- Thời hạn: trước feature merge/alpha.2.
- Giảm thiểu: cấm merge `363a671` và mọi descendant chưa đóng findings.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa dự kiến | Tác động | Hành động |
|---|---|---|---|---|
| Source identity | Porcelain clean | Reject non-H index flags + porcelain | Cao | 2 flag regressions |
| Markdown metadata | 7 field values | Exact field set/values + pointer target | Cao | Version 1.3.0 |
| Normative body | Hash pinned | Không đổi | Không | Body hash giữ nguyên |
| Registry inventory | 127 names | 127 names | Không | Existing source tests |
| Runtime/model | Chưa có | Chưa có | Không | Không claim |

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact | Evidence |
|---|---|---|---|
| CH-001 | Reject non-H Git index entries | checker | h/S negative tests |
| CH-002 | Exact 11 metadata fields/values; duplicate/malformed fail | checker | combined probes |
| CH-003 | Resolve pointer trong project, xác minh revision identity/status | checker + REV-002 | pointer tests |
| CH-004 | Bump Markdown 1.3.0/pointer, body không đổi | Markdown | registry-only pass |

## Xác minh

| ID | Phương pháp | Mong đợi | Thực tế hiện tại | Trạng thái | Evidence |
|---|---|---|---|---|---|
| V-001 | assume-unchanged/skip-worktree probes | Cả hai reject | status rỗng nhưng helper reject h/S | pass | unit suite |
| V-002 | Front-matter combined probes | Missing/extra/wrong/pointer fail | 7 adversarial groups pass | pass | unit suite |
| V-003 | Positive project gate | Zero failure | refs/registry/docs/35 tests/shell/diff pass | pass | console; replay sau commit |
| V-004 | Independent delta review | Pass/NONE | Chưa chạy | not_run | Chờ commit |

- Regression đã chạy lại: 35/35 và full positive gate.
- Kiểm tra chưa chạy: exact committed replay và independent review.
- Rollback: không merge; failed snapshots chỉ giữ làm audit evidence.

## Ảnh hưởng tới báo cáo và quyết định cũ

- `TPR-20260822-002` vẫn đúng cho `363a671` và không bị sửa.
- Verdict cũ không áp dụng revision mới; bắt buộc delta review.
- Không có model metric hoặc dataset result cần đính chính.
- Không có risk acceptance/ngoại lệ.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22 Asia/Bangkok.
- Người kiểm tra: independent reviewer không tham gia sửa revision này.
- Kết luận: `in_review`; implementation/local gate pass, còn commit/evidence và
  independent verdict.
- Bản ghi phiên: `SESSION-20260822-002`.
