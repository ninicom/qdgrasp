---
document_id: TPR-20260821-003
document_type: third_party_review
title: Independent audit train-argument snapshot 7fb01a7
status: complete
date: 2026-08-21
revises: none
report_id: TPR-20260821-003
reviewed_session: SESSION-20260821-006
reviewed_revision: 7fb01a702deebaf9f2b03ae97bd67ad88e837c43
reviewer: codex-agent-third-party-train-args
reviewer_organization: codex-multi-agent-internal-review
review_date: 2026-08-21
independence: internal_independent
verdict: fail
max_severity: S1
---

# TPR-20260821-003 — Audit snapshot train arguments

## Bằng chứng

Reviewer read-only kiểm commit `7fb01a702deebaf9f2b03ae97bd67ad88e837c43`,
tree `8a6b6cc1ac53603249ac8a13ed07c1866cd45a94`, base `57b3ff3` và
upstream clean HEAD `329682a29d27203582ba30e519340f95abccc6a6`.

Independent AST/text extraction xác nhận 115 canonical + 2 extra + 9 legacy +
1 API = 127 unique names; mọi canonical default/group và sáu validation set
khớp source. Bốn source hash khớp registry. Reviewer chạy registry/full checker,
30-doc validator, 10/10 tests, shell syntax, `git diff --check` và `git fsck`;
các lệnh hiện có đều pass nhưng mutation probes chứng minh false-negative.

## Phát hiện

| ID | Severity | Finding | Disposition yêu cầu |
|---|---|---|---|
| F-001 | S1 | Checker nhận YAML sai cú pháp/metadata lạ | Exact schema và scalar grammar; negative test |
| F-002 | S1 | Legacy/extra/API/untyped/merge/resume/extensions có thể bị sửa mà checker vẫn pass | Khóa toàn semantic manifest và Markdown; mutation tests |
| F-003 | S2 | Merge priority thiếu nhánh đặc biệt khi có `cfg=` | Ghi/check cả hai nhánh và data/model/task authority |
| F-004 | S2 | `auto_augment`, `quantize`, `scale` thiếu nullable/range/alias contract | Structured constraints và type đúng |
| F-005 | S2 | Registry-only claim/test cao hơn guarantee thực | Pinned semantic fingerprint và full CLI test |
| F-006 | S2 | N2 active-doc versioning và REV-005 không khớp policy/template | Quy định living contract; revision schema 2 đủ tám mục |
| F-007 | S2 | Evidence không replayable/final; lock/license chưa được gate | Evidence mới pin commit/time; reference checker; DGN2 README license hash |
| F-008 | S3 | Disposition subtotal 115 dễ bị hiểu là total 127 | Ghi cả hai total rõ ràng |
| F-009 | S3 | Automated gate không chạy full check khi clone tồn tại | Conditional full source/reference checks |

Không có finding về sai tên/default/group hiện tại; failure là do cổng kiểm có
thể bỏ lọt corruption và governance/evidence chưa đủ.

## Kết luận

Verdict `fail`, max severity `S1`. Commit `7fb01a7` không được merge/release.
Inventory 127 tên được xác nhận đúng về nội dung, nhưng chỉ được chấp nhận sau
khi F-001–F-009 được sửa bằng revision/session mới và delta review trên exact
remediation commit. Báo cáo này phải được giữ nguyên dù delta review sau pass.
