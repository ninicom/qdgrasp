---
document_id: RELEASE-X.Y.Z
document_type: release_report
title: Báo cáo release X.Y.Z
status: draft
date: YYYY-MM-DD
author: AUTHOR-ID
revises: none
version: X.Y.Z
source_branch: release/X.Y.Z
source_develop_commit: COMMIT-SHA
target_tag: vX.Y.Z
---

# Báo cáo release X.Y.Z

## Tóm tắt

- Phạm vi release.
- Những tuyên bố được phép và không được phép.

## Feature đã tích hợp

| Feature branch | Feature commit | Merge commit vào develop | Session/review |
|---|---|---|---|
| `feature/<slug>` | `<sha>` | `<sha>` | `<document IDs>` |

## Bằng chứng

| Evidence ID | Lệnh/artifact | Exit code hoặc SHA-256 | Kết quả |
|---|---|---|---|
| E-01 | `scripts/release_gate.sh X.Y.Z` | 0 | PASS |

## Release gates

| Gate | Kết quả | Evidence |
|---|---|---|
| Documentation validator | PASS/FAIL | E-... |
| Unit/regression tests | PASS/FAIL | E-... |
| Metrics/runtime | PASS/FAIL/N/A có lý do | E-... |
| Independent review | PASS/FAIL/N/A có lý do | report ID |

## Giới hạn và rủi ro còn lại

- Liệt kê phần chưa triển khai/chưa kiểm; không để trống.

## Rollback

- Tag/commit trước release.
- Artifact hoặc migration cần phục hồi.

## Kết luận

- `release / do_not_release`.
- Người chấp nhận, ngày và phạm vi quyết định.
