---
document_id: REV-20260821-004
document_type: revision_record
title: Mở rộng quản trị tài liệu cho GitFlow và release report
status: complete
date: 2026-08-21
record_id: REV-20260821-004
session_id: SESSION-20260821-004
created_at: 2026-08-21T23:35:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260821-001
    artifact: docs/README.md
    revision: e20a9ff6a3f047841e9860d77f2a2c66220032893ef94917e60afe4b25e16b5d
  - session_id: SESSION-20260821-001
    artifact: docs/governance/DOCUMENTATION_POLICY.md
    revision: 20f6162da1e82b8f38decc4dfe1bbc79c081d64dc59ed510f30d835d3f5cf10f
  - session_id: SESSION-20260821-001
    artifact: docs/reports/README.md
    revision: ebe5ea06b04f0c53feda4ace9c15026ef9db45eebc59a432192a16fa5ea70ac3
  - session_id: SESSION-20260821-001
    artifact: docs/schemas/README.md
    revision: 249f48ab134f7e8df372c08f7e0cfc785f54e817b007b9770121a05c424e3579
reason: Người dùng bổ sung yêu cầu Git nhiều nhánh, commit khi hoàn tất, merge feature vào develop và chỉ đưa qua release sau kiểm tra.
necessity: N1
impact: Mở rộng governance/index/schema cho GitFlow và release report; không thay đổi plan kiến trúc, metric, model hay runtime.
---

# REV-20260821-004 — Mở rộng quản trị cho GitFlow

## Lý do chỉnh sửa

Bộ tài liệu trước chưa quy định branch, commit, develop integration và release
lifecycle. Yêu cầu mới cần thêm policy Git riêng, release report type và liên kết
từ các index/schema hiện có.

## Mức độ cần thiết

Mức `N1`: mở rộng/làm rõ contract quản trị để tránh hiểu khác nhau về “hoàn
tất”, không sửa kết luận kỹ thuật hoặc số liệu trước đó.

## Phạm vi và tác động

| Artifact | SHA-256 trước | SHA-256 sau | Tác động |
|---|---|---|---|
| `docs/README.md` | `e20a9f…e16b5d` | `4b0468…54fb` | Thêm link workflow/template release |
| `DOCUMENTATION_POLICY.md` | `20f616…cf10f` | `cce811…7afd` | Thêm `release_report` vào contract |
| `docs/reports/README.md` | `ebe5ea…0ac3` | `01a28d…ec92` | Quy định release report |
| `docs/schemas/README.md` | `249f48…3579` | `9f2163…0b50` | Mô tả validation của release report |

Không thay đổi raw archive, PLAN-V2, metrics registry hoặc các verdict audit.

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Output |
|---|---|---|
| CH-001 | Thêm policy GitFlow | `docs/governance/GIT_WORKFLOW.md` |
| CH-002 | Thêm release report template | `docs/templates/RELEASE_REPORT.md` |
| CH-003 | Cập nhật index/policy/schema liên quan | Bốn artifact ở bảng trên |

## Xác minh

| Verification ID | Lệnh | Tiêu chí |
|---|---|---|
| V-001 | `python3 scripts/check_docs.py --root .` | Exit 0 |
| V-002 | `bash -n .githooks/* scripts/git/*.sh scripts/release_gate.sh` | Exit 0 |
| V-003 | `git diff --check` | Exit 0 |

## Ảnh hưởng tới báo cáo và quyết định cũ

Không đổi verdict cũ. Release sau thời điểm này phải dùng workflow/report mới;
release bootstrap không được mô tả như model hoàn chỉnh.

## Xác nhận đóng hồ sơ

Revision được đóng khi các gate V-001 đến V-003 pass và feature được commit.
