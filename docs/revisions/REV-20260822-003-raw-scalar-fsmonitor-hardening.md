---
document_id: REV-20260822-003
document_type: revision_record
revision_schema: 2
title: Khóa fsmonitor-valid và raw front-matter scalar
status: in_review
date: 2026-08-22
record_id: REV-20260822-003
session_id: SESSION-20260822-003
created_at: 2026-08-22T03:41:30+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-002
    artifact: scripts/check_train_args.py
    revision: 99b1b8dde330e011cf0e9e3fb0693ac25aa7c6fd672e69249b72baa1c78a2f29
  - session_id: SESSION-20260822-002
    artifact: scripts/tests/test_check_train_args.py
    revision: 76f3480f483963b6d318fbb60ce72e25aa98123259867ecc3b8900e3e3c8fdd9
  - session_id: SESSION-20260822-002
    artifact: docs/configuration/TRAIN_ARGUMENTS.md
    revision: 6911f367e8d068200ecaa4de4410922712614805a3f83971c7c98c88de841b17
  - session_id: SESSION-20260822-002
    artifact: docs/revisions/REV-20260822-002-source-metadata-hardening.md
    revision: 70f498522420ee0f0c4247a262623919dbeff35e78971b1d9e50fb0c4675374d
reason: TPR-20260822-003 reproduced source drift hidden only in ls-files -f and 16 governance scalars accepted after quote/whitespace normalization at 42914f5.
necessity: N2
impact: Exact source cleanliness and documentation identity guarantees change materially; registry inventory/body and runtime claims remain unchanged.
supersedes: REV-20260822-002
---

# REV-20260822-003 — Raw scalar/fsmonitor hardening

## Liên kết truy vết

- Phiên sửa: `SESSION-20260822-003`.
- Snapshot lỗi: `42914f550929789fe5d0df45bafedaa0b9ae469b`, tree
  `e6186cdcbfbe4bcf71571bbafbf33982118c4784`.
- Finding source: `TPR-20260822-003`, F-005/F-010/F-011.
- Revision trước: REV-002 không đạt acceptance và được giữ nguyên.
- Baseline checker/tests/Markdown hashes nằm trong front matter của record này.
- Revision sau ở working tree:
  - checker `975edb8ea49577a8ef4f7f9369de18059706622ac4bfa44a6008c16bc2ecd62a`;
  - tests `c3005835daae23357e2921a903150e3ca0ce633ff773baaa49fe6866bb7aa685`;
  - Markdown `3baf998fdc21c9c8d7a718fe08432d01a7edfa4458d146f401d9e0be48777356`.

## Lý do chỉnh sửa

`git ls-files -v` không hiển thị `fsmonitor-valid`; view `-f` mới đổi prefix
thành chữ thường. Đồng thời parser front matter đã `strip` và bỏ quote trước khi
so sánh, khiến YAML semantic value khác contract vẫn pass. Hai lỗi được reviewer
tái hiện độc lập sau khi mọi finding S1 đã đóng.

## Mức độ cần thiết

- Mức: `N2`.
- Nếu không sửa: source đã đổi và identity/pointer khác semantic có thể vượt gate.
- Thời hạn: trước feature merge/alpha.2.
- Giảm thiểu: cấm merge `42914f5` và descendant chưa qua delta review mới.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Tác động | Hành động |
|---|---|---|---|---|
| Source identity | Chỉ `ls-files -v` | Cả `-v` và `-f`, chỉ prefix H | Cao | fsmonitor regression |
| Markdown metadata | Giá trị đã normalize | Raw scalar lexeme exact | Cao | 11-field matrix |
| Revision identity | Giá trị đã normalize | Raw scalar lexeme exact | Cao | 5-field matrix |
| Normative body | Hash pinned | Không đổi | Không | giữ body hash |
| Registry inventory | 127 names | 127 names | Không | existing source tests |
| Runtime/model | Chưa có | Chưa có | Không | không claim |

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact | Evidence |
|---|---|---|---|
| CH-001 | Kiểm `ls-files -f` ngoài `-v` | checker | fsmonitor negative test |
| CH-002 | Parser giữ raw scalar và separator strict | checker | 11 + 5 field matrix |
| CH-003 | Bump Markdown 1.4.0/pointer REV-003 | Markdown | body hash giữ nguyên |
| CH-004 | Giữ nguyên failed verdict trước sửa | TPR-003 | commit `3b1c0d0` |

## Xác minh

| ID | Phương pháp | Mong đợi | Thực tế hiện tại | Trạng thái | Evidence |
|---|---|---|---|---|---|
| V-001 | fsmonitor-valid probe | `-f` prefix h bị reject | `h tracked.txt` reject, message đúng | pass | unit suite |
| V-002 | 11 registry scalar × 4 styles | tất cả reject | 44 probe pass | pass | unit suite |
| V-003 | 5 revision scalar × 4 styles | tất cả reject | 20 probe pass | pass | unit suite |
| V-004 | Full positive project gate | zero failure | refs/registry/docs/38 tests/shell/diff pass | pass | console; replay sau commit |
| V-005 | Independent delta review | pass/NONE | Chưa chạy | not_run | Chờ commit |

- Regression đã chạy lại: 38/38 và full positive gate (references lock-only,
  references source-root, train-args registry-only, train-args full source,
  check_docs, `bash -n`, `git diff --check`).
- Kiểm tra chưa chạy: exact committed replay và independent review.
- Rollback: không merge; failed snapshots/reports giữ làm immutable evidence.

## Ảnh hưởng tới báo cáo và quyết định cũ

- `TPR-20260822-003` vẫn đúng cho `42914f5` và không bị sửa.
- SESSION-002/REV-002 không được đổi thành complete hồi tố.
- Không có model metric, dataset result hoặc license decision cần đính chính.
- Không có risk acceptance/ngoại lệ.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22 Asia/Bangkok.
- Người kiểm tra: independent reviewer không tham gia sửa revision này.
- Kết luận: `in_review`; implementation/local gate pass, còn commit/evidence và
  independent verdict.
- Bản ghi phiên: `SESSION-20260822-003`.
