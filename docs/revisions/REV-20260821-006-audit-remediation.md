---
document_id: REV-20260821-006
document_type: revision_record
revision_schema: 2
title: Sửa false-negative và evidence gaps của train-argument registry
status: in_review
date: 2026-08-21
record_id: REV-20260821-006
session_id: SESSION-20260821-007
created_at: 2026-08-21T23:53:37+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260821-006
    artifact: docs/configuration/TRAIN_ARGUMENTS.yaml
    revision: 6c47ee19d392fa30d612ef8b5ae26b06db587fc51c90da5b9d49641efc21cd66
  - session_id: SESSION-20260821-006
    artifact: docs/configuration/TRAIN_ARGUMENTS.md
    revision: 44baa8a0cbb4e9402980b60beddd5688acaaf1e2c9224adf8c50c47f8f6c1be1
  - session_id: SESSION-20260821-006
    artifact: scripts/check_train_args.py
    revision: 6b6c2642230705d15f324e78ddfb1a405bc0ac4e9e818279e2a4ff2afeba87c1
  - session_id: SESSION-20260821-006
    artifact: references.lock.yaml
    revision: 7e9a8005205a7d2c3f2ecee322a4bd6781170153e90f3d6706fc91ce4fb6a7f6
  - session_id: SESSION-20260821-006
    artifact: docs/revisions/REV-20260821-005-train-argument-contract.md
    revision: 127e829dfc247b77364bfbd258fe6aee3d97b84ec94da1eddcfa5e718dd64cc5
  - session_id: SESSION-20260821-004
    artifact: docs/governance/DOCUMENTATION_POLICY.md
    revision: cce81162005db126150ce01be76669710e48b314914dae6d7d48a3195e777afd
  - session_id: SESSION-20260821-004
    artifact: docs/governance/GIT_WORKFLOW.md
    revision: 4950b764095d52f8720c0bd7197c4f0532a1a44401e829b9710baa01a3c0ea4d
  - session_id: SESSION-20260821-001
    artifact: PLAN.md
    revision: 7fc224bd200dd3034ec8b0576201b216c5c8b8e60552a4b400c31a19acc642b0
reason: Independent audit TPR-20260821-003 found S1 false-negatives and S2/S3 source, policy and evidence gaps in completed snapshot 7fb01a7.
necessity: N2
impact: Replaces the checker guarantee and accountability record for this feature; 127-name inventory remains unchanged and no model/runtime result is altered.
supersedes: REV-20260821-005
---

# REV-20260821-006 — Audit remediation

## Liên kết truy vết

- Phiên sửa: `SESSION-20260821-007`.
- Snapshot bị sửa: commit `7fb01a702deebaf9f2b03ae97bd67ad88e837c43`.
- Independent finding source: `TPR-20260821-003`, F-001–F-009.
- Revision trước: `REV-20260821-005`; record cũ được giữ nguyên.
- Revision sau: commit `5d86a426ff7f250bc7a77e80e241610d10a9958a`,
  tree `9020e17aae346d71233f501db658314c28b3a065`.

## Lý do chỉnh sửa

Mutation probes của reviewer chứng minh checker pass cho YAML hỏng và nhiều thay
đổi semantic. Evidence cũ không pin final tree, reference lock chưa có gate và
N2 active-doc/version/revision template chưa nhất quán. Vì vậy claim “chống bỏ
sót/source drift” của SESSION-006 chưa đủ dù inventory hiện tại đúng 127 tên.

## Mức độ cần thiết

- Mức: `N2`.
- Hậu quả nếu không sửa: release gate có thể pass config sai nghĩa hoặc source
  provenance/license evidence sai.
- Thời hạn: trước merge feature vào `develop`.
- Giảm thiểu khi trì hoãn: commit `7fb01a7` bị TPR-003 cấm merge/release.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Tác động | Hành động |
|---|---|---|---|---|
| Runtime/model | Chưa có | Chưa có | Không | Không claim smoke |
| Config schema | Name/count + một phần field | Exact semantic fingerprint + Markdown | Cao | 24 tests |
| Source provenance | Manual full check | Immutable lock + conditional full gate | Cao | Check 2 clones |
| Policy | N2 wording xung đột living docs | Stable ID + Git hash/version/review rule | Vừa | Revision schema 2 |
| Số liệu | Không có model metric | Không đổi | Không | Không rerun metric |

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact | Evidence |
|---|---|---|---|
| CH-001 | Khóa exact registry schema/semantics/Markdown | YAML + checker | Mutation tests |
| CH-002 | Ghi merge/special constraints đúng source | YAML + Markdown | Full source check |
| CH-003 | Thêm reference/license evidence checker | lock + script | Full reference check |
| CH-004 | Nối full checks vào automated gates | hooks/common | Shell syntax + gate |
| CH-005 | Sửa living-contract/revision schema policy | policy/template/validator | Docs tests |

## Xác minh

| ID | Phương pháp | Mong đợi | Thực tế hiện tại | Trạng thái | Evidence |
|---|---|---|---|---|---|
| V-001 | Registry-only mutations | Tất cả corruption fail | Pass test | pass | unit suite |
| V-002 | Full source/reference checks | Exact clean HEAD/hash | Pass | pass | console; log sau commit |
| V-003 | Docs validator | Schema 2 đủ tám mục | Pass | pass | unit suite |
| V-004 | Full regression | Zero failure | 24/24 | pass | replay log |

Replay log bất biến: `docs/reports/evidence/TRAIN-ARGS-20260821-audit-remediation.txt`,
4.178 byte, SHA-256
`fd7888018bfa8e691dd8a6e236ffbca9fbfb6e864f483ed97522abec0ced037f`.

- Kiểm tra chưa chạy: delta review trên remediation commit.
- Rollback: không merge; quay về immutable commit `7fb01a7` chỉ để audit, không
  dùng nó làm release candidate vì verdict fail.

## Ảnh hưởng tới báo cáo và quyết định cũ

- `TPR-20260821-003` vẫn có hiệu lực cho commit `7fb01a7` và không bị sửa.
- Verdict REV-005/SESSION-006 không đủ cho revision mới; REV-006 supersede phần
  accountability, không xóa record cũ.
- Loại kiểm tra lại: `delta_review` độc lập, xuất TPR-004.
- Không có số liệu model cần đính chính.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-21 Asia/Bangkok.
- Người kiểm tra dự kiến: independent delta reviewer, phạm vi F-001–F-009.
- Kết luận hiện tại: `in_review`; implementation, commit và replay evidence đã
  khóa, còn delta review độc lập.
- Bản ghi phiên: `SESSION-20260821-007`.
