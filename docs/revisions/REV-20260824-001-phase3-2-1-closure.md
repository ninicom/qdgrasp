---
document_id: REV-20260824-001
document_type: revision_record
revision_schema: 2
title: Đóng Phase 3.2.1 full-pipeline correctness và mở release Shadow theo scope
status: complete
date: 2026-08-24
record_id: REV-20260824-001
session_id: SESSION-20260824-P3.2.1
created_at: 2026-08-24T14:00:00+07:00
author: codex-primary-agent
revises:
  - artifact: docs/roadmap/PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md
    revision: 22b40b19d2bda252443620e9795c46503709aef9bea9b8d8d315a6bf26f3f0e5
  - artifact: qdgrasp/presets/robots/shadow_hand.yaml
    revision: 57f8672b6c52bb9771f51f9c68e3c8baf2566173cfa5dfb8d44543c6edfc3395
  - artifact: evidence/phase3_2_1/README.md
    revision: e5812449fd83e0d422b460f4c7222d8340130e4e9362331459db51557f399ad0
  - artifact: docs/roadmap/PHASE3_2_UNDERACTUATED_CONTROL_PLAN.md
    revision: 36802c835edb435315112a5f5a526251d673a4657727ebbf0f8751fd47da1a41
  - artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 4669880ea7144d1dac52e0aee6f80e6b2cb3610d97d6d1653a854ba43c4f99e6
  - artifact: docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md
    revision: 712b8e7eeb461459fc4564be7dd6a35b3c553075d070f87441074701f9844298
reason: "P00-P13 đã qua full-flow/mutation/determinism gates; independent review đóng finding active-count fail-closed và chấp thuận claim scope trong vòng review thứ hai."
necessity: N1
impact: "P3.2.1 chuyển complete và Shadow release_blocked=false cho generated-reachable correctness; canonical-independent yield vẫn 0/12 và P3 tổng vẫn pending."
---

# REV-20260824-001 — Closure Phase 3.2.1

## 1. Liên kết truy vết

- Plan: `docs/roadmap/PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md`.
- Review: `docs/reviews/TPR-20260824-001-phase3-2-1-full-pipeline.md`.
- Gate: `scripts/check_phase3_2_1.py`.
- Evidence: `evidence/phase3_2_1/p10-p13-release-gate/`.

## 2. Lý do chỉnh sửa

P3.2.1 đã hoàn tất mọi work package, đóng finding review và đủ điều kiện đưa
trạng thái tài liệu/profile từ corrective block sang closure theo claim hẹp.

## 3. Mức độ cần thiết

Mức `N1`: trạng thái release phải khớp với gate đã chấp thuận và vẫn giữ rõ
limitation canonical để không tạo claim vượt bằng chứng.

## 4. Phạm vi và tác động

Đóng các work package P3.2.1-00 đến P3.2.1-14 và chuyển Shadow Hand sang
`release_blocked: false`. Quyết định dựa trên provenance liên tục qua proposal,
palm hypothesis, IK, exact collision, static certificate, task-space command và
measured dynamic rollout cho generated-reachable fixtures của cả ba hand.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| ID | Artifact | Thay đổi |
| --- | --- | --- |
| CH-001 | `PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md` | version 1.2.0, status complete, P14 complete và hash evidence mới |
| CH-002 | `evidence/phase3_2_1/README.md` | 100 contract tests, 366-test regression, hash payload/file và review outcome |
| CH-003 | `shadow_hand.yaml` | `release_blocked: false`; provenance ghi rõ scope và limitation 0/12 |
| CH-004 | `TPR-20260824-001` | lưu finding vòng đầu, remediation và verdict vòng hai |
| CH-005 | `PHASE3_2_UNDERACTUATED_CONTROL_PLAN.md` | version 1.2.0, status complete và liên kết closure P3.2.1 |
| CH-006 | `PROJECT_PHASES.md` | giữ P3 pending nhưng đóng blocker P3.2.1 và ghi limitation canonical |
| CH-007 | `PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md` | ghi nhận P3.1-11/12 complete; P3.1-13 ready nhưng chưa chạy |

## 6. Xác minh

| Gate | Kết quả |
| --- | --- |
| Environment CPU | PASS |
| Documentation | PASS, 93 files gồm review và closure records |
| Phase 2 robot layer | PASS |
| Phase 3.2 component/transmission | PASS; Shadow rank 20/24 |
| Phase 3.2.1 release gate | PASS; 100 tests; generated 3/3; deterministic |
| Full tests | 366 passed, 1 skipped, 22 deprecation warnings |
| Independent review | PASS sau một P2 được sửa và review lại |

Generated payload SHA-256 là
`5a34f9d8e7f6568c7dc28e1b5f70c0421b1910ef9fe038d75e33efec55b3c563`;
canonical payload SHA-256 là
`20450c16e2bc74ca6079a7407b9d2573c05784335dcdac3c88f0a6531c3a8eaa`.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

`REV-20260823-009` hoàn tất vai trò re-block/corrective opening. Claim component
của `REV-20260823-005` tiếp tục hợp lệ; full-pipeline claim nay chỉ được phục hồi
trong phạm vi generated-reachable đã review.

## 8. Giới hạn còn mở

Canonical-independent matrix có đủ 12 cells nhưng `0/12` accepted. Vì vậy
revision này không tuyên bố canonical yield, generalization, dataset chính thức,
controlled ablation hay hoàn tất P3. Các bước P3.1-13/P3.1-14 và P4 vẫn phải tuân
theo gate riêng.

## 9. Xác nhận đóng hồ sơ

- Review: `docs/reviews/TPR-20260824-001-phase3-2-1-full-pipeline.md`.
- Tác giả: codex-primary-agent, 2026-08-24 Asia/Bangkok.
- Kết luận: P3.2.1 complete trong claim scope đã review; Shadow release unblock
  không mở rộng sang canonical/generalization claim.
