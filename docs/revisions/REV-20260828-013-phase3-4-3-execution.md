---
document_id: REV-20260828-013
document_type: revision_record
revision_schema: 2
title: Thi công Phase 3.4.3 phần CPU và dựng gate CUDA cho Kaggle
status: complete
date: 2026-08-28
record_id: REV-20260828-013
session_id: SESSION-20260828-P3.4.3-EXECUTION
created_at: 2026-08-28T02:30:00+07:00
author: claude-implementation-agent
revises:
  - session_id: PLAN-V2
    artifact: PLAN.md
    revision: 9832e2a2f5a99ef3e90e8dbed22857f631eab64587b4bd89070ab8a6dcf9148c
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 3678a863bfca796bccfab4257ece2899812066d7f3769a79bdd0334ac3c705e6
  - session_id: DOCS-INDEX
    artifact: docs/README.md
    revision: cd23752f1efe35e85ccf1ad975db15ba0594a81592bf77fcc9a2f7f345c2aed8
reason: "Contract P3.4.3 đã đóng kín phạm vi nhưng chưa có code, test hay evidence; hai mươi blocker vẫn mở và không có gate nào chạy được."
necessity: N3
impact: "Thi công S0–S9 và S11 với code/test/evidence trên CPU; dựng gate CUDA và notebook Kaggle cho S10; tạo QDGrasp-ContactRich-Active-Tiny ở trạng thái release_blocked. Không đổi ADR-0008, không đổi P3.4 historical verdict, không mở P4 contact-rich input."
---

# REV-20260828-013 — Thi công Phase 3.4.3

## 1. Liên kết truy vết

- Contract: `docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md`
  (`ROADMAP-P3.4.3-001@1.1.0`).
- Thứ tự thi công: `docs/roadmap/PHASE3_4_3_EXECUTION_BREAKDOWN.md`
  (`ROADMAP-P3.4.3-002`).
- Điều kiện bàn giao: `docs/roadmap/PHASE3_4_3_HANDOFF.md`
  (`ROADMAP-P3.4.3-003`).
- Manifest requirement: `docs/roadmap/phase3_4_3_requirements.yaml`.
- Evidence: `evidence/phase3_4_3/`.
- Artifact: `datasets/contactrich-active-tiny/`.

## 2. Lý do chỉnh sửa

`REV-20260827-012` đóng kín phạm vi contract nhưng mọi requirement vẫn
`pending`: không có code, không có test, không có evidence. Bản ghi này ghi lại
việc thi công thật, không phải việc lập kế hoạch.

## 3. Mức độ cần thiết

`N3`. Các blocker được sửa gồm safety verdict bỏ sáu giới hạn đã khai báo,
impulse window không phải window, GPU finalist không mang control sequence, và
certifier có thể chứng nhận hai matching failure. Đây là lỗi an toàn mô phỏng
và release provenance.

## 4. Phạm vi và tác động

- Thi công S0–S9 và S11 với code, test và evidence trên CPU.
- Dựng gate CUDA `scripts/check_phase3_4_3_cuda.py` và notebook
  `kaggle-phase3-4-3/` cho S10; **chưa** có run thật trên T4.
- Tạo `QDGrasp-ContactRich-Active-Tiny` với `release_blocked=true`.
- Không đổi `ADR-0008`; không đổi P3.4 historical three-hand verdict;
  không mở P4 contact-rich input.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Sub-phase | Nội dung | Blocker đóng |
| --- | --- | --- |
| S0 | requirement ledger, closure checker, exit-code semantics | B-09 |
| S1 | active-scope registry cho ADR-0008 | B-10 |
| S2 | typed contract v2 fail-closed, schema có version | B-11 |
| S3 | SafetyBudget v2 đủ mười ba sensor, rolling impulse, contact episode | B-01, B-02 |
| S4 | contact pair taxonomy, self-contact policy, terminal/timeout semantics | B-12, B-15 |
| S5 | trajectory v2 timebase/palm pose, ReplayCapsule v1 | B-04, B-06 |
| S6 | certification và release invariants, stage-denominator ledger | B-05 |
| S7 | objective/CEM/refinement fail-closed, capability matrix | B-16 |
| S8 | backend hydration, scene signature đầy đủ, CPU summary v2 | B-13, B-14 |
| S9 | GPU contact summary fail-closed, finalist export bằng capsule | B-03 |
| S10 | gate CUDA + notebook Kaggle; harness cũ hết crash ở nhánh pass | B-08, B-19 |
| S11 | ContactRich-Active-Tiny, loader public, ablation, stage render | B-07, B-17, B-18, B-20 |
| S12 | review packet, closure runner, handoff | — |

Ba phát hiện phụ trong lúc thi công, ghi lại vì chúng đổi kết quả:

1. Contact event lưu view vào buffer sống của MuJoCo thay vì copy, nên event đã
   ghi vẫn đổi theo rollout.
2. `ContactSafetyBudget` bị áp cho cả contact mà robot không tham gia, nên trọng
   lượng của vành khay bị tính vào ngân sách của bàn tay.
3. Bốn negative control đầu tiên không control được gì: hai cái pass. Chúng được
   thay bằng control vật lý thật, không phải bằng cách nới ngưỡng.

## 6. Xác minh

| Verification ID | Lệnh | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `pytest tests/dynamic_grasp tests/contactrich_active -q` | toàn bộ pass | 439 passed | pass | log phiên |
| `V-002` | `check_phase3_4_3.py --skip-tests` trên commit sạch | không thoát 0 khi còn required mở | `INCOMPLETE`, exit 3 | pass | `evidence/phase3_4_3/s0/result.json` |
| `V-003` | `check_phase3_4.py --skip-tests` | `PARTIAL` không thoát 0 | exit 3 | pass | `evidence/phase3_4_3/s0/result.json` |
| `V-004` | `generate_contactrich_active_tiny.py` | 36/36 cell, control có predicate | 44 sample, 36 positive, 8 negative, 0 unexpected | pass | `datasets/contactrich-active-tiny/dataset_manifest.json` |
| `V-005` | `check_contactrich_active.py DATASET_ROOT` | `BLOCKED` với lý do đo được | `BLOCKED`, exit 3 | pass | như trên |
| `V-006` | `phase3_4_3_ablation.py` | verdict có thể là no_measured_difference | `no_measured_difference` | pass | `evidence/phase3_4_3/s11/static-vs-dynamic.json` |
| `V-007` | `render_phase3_4_3_stages.py` | đủ sáu stage mỗi hand | 6/6 cả hai hand | pass | `evidence/phase3_4_3/s11/stage-renders.json` |
| `V-008` | `check_phase3_4_3_cuda.py --device cuda:0` trên host CPU | fail-closed | exit 1, "not CUDA evidence" | pass | log phiên |
| `V-009` | `check_docs.py --root .` | pass | 115 file | pass | log phiên |

`V-008` chỉ chứng minh gate từ chối host CPU. Nó **không** là bằng chứng CUDA.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- P3.4 ba-hand giữ `paused_by_ADR-0008`. Không có gì ở đây đổi verdict đó.
- `QDGrasp-ContactRich-Tiny` v1 vẫn release-blocked; artifact mới là successor
  có ID và schema riêng, không ghi đè.
- `P3.4-10` (MPPI) chuyển sang `deferred_not_claimed` và `required: false`,
  đúng như contract cho phép với package optional.
- Ablation cho `no_measured_difference` trên scene của release recipe. Đây là
  kết quả, không phải lỗi; không threshold nào bị đổi để tránh nó.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-implementation-agent, 2026-08-28 Asia/Bangkok.
- Người kiểm tra: **chưa có**. Tác giả không tự review công việc của mình.
- Kết luận: phần CPU của P3.4.3 đã thi công và có evidence. Phase **chưa đóng**.
  Còn thiếu đúng hai thứ: một run CUDA thật trên Kaggle T4 với verdict `PASS`
  trên commit ứng viên, và một reviewer độc lập ký `PASS` trên exact packet
  hash. Cho tới khi đủ cả hai, `release_blocked` giữ `true` và P4 contact-rich
  input giữ đóng.
- Giới hạn còn lại: MuJoCo Warp 1.16.0 vẫn mang defect upstream
  (`REV-20260827-010`); compatibility matrix chạy trong notebook, và nếu không
  có bản sạch thì GPU gate giữ `blocked`.
- Liên kết bản ghi hoàn tất: `REV-20260828-013`.
