---
document_id: REV-20260825-005
document_type: revision_record
revision_schema: 2
title: Đóng renderer, target crop và multi-object rollout Phase 3.3
status: complete
date: 2026-08-25
record_id: REV-20260825-005
session_id: SESSION-20260825-P3.3-RENDER-ROLLOUT
created_at: 2026-08-25T15:00:00+07:00
author: codex-primary-agent
revises:
  - session_id: ROADMAP-P3.3-001
    artifact: docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md
    revision: e171dd345ca837ea07ea3cdd152fed8e6624c970de247f14ae63a33bfeecb602
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 062fb0429d9d44faf8845bef1627c915ac692d50985468deb8fe2f52b1707730
reason: "Renderer mock visibility, thiếu target-local crop và validator chưa nối vào rollout là ba blocker runtime cuối trước release generation; implementation và evidence mới làm thay đổi completion matrix."
necessity: N2
impact: "P3.3-08, P3.3-09 và P3.3-11 chuyển complete. P3.3 tiếp tục active tại bounded generation/QA P3.3-14 và final gate P3.3-15."
---

# REV-20260825-005 — Renderer và multi-object rollout

## 1. Liên kết truy vết

- Phiên thực hiện: `SESSION-20260825-P3.3-RENDER-ROLLOUT`.
- Baseline tài liệu: `133b89e`.
- Runtime commits: `2cd1e65`, `9465af0`, `3e47ca2`.
- Plan nguồn: `ROADMAP-P3.3-001` v1.4.0.
- Artifacts và checksum trước sửa: hai mục `revises` trong metadata.

## 2. Lý do chỉnh sửa

Visibility cũ xuất raw geom ID và observation chưa pack được vào shard. Dynamic
validator cũ nhận evidence do caller truyền nhưng rollout chưa đo stage/contact/
impulse. Cả hai blocker đã được thay bằng measured runtime path; target-local
crop cũng được bổ sung để hoàn tất output contract cho model/data loader.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: thay đổi verdict ba work package và mở đường cho release generation.
- Thời hạn/độ khẩn: trước P3.3-14.
- Biện pháp nếu trì hoãn: không sinh `QDGrasp-Scene-Tiny`.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Rendering | raw geom visibility mock | canonical object mask/visibility | cao | render 12 scenes |
| Observation | arrays tạm | atomic packed refs + shard record | cao | manifest generation |
| Target input | chỉ có selector | deterministic object-frame crop | vừa | loader parity |
| Scene rollout | caller-supplied evidence | measured stage/contact/impulse hooks | cao | three-hand release run |

Tóm tắt tác động: runtime path 08/09/11 đã đóng; artifact và review vẫn mở.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Canonical segmentation và packed observation | `2cd1e65` | render/pack/fail-closed tests |
| `CH-002` | Physical non-target compile và rollout collector | `9465af0` | stage/contact/impulse tests |
| `CH-003` | Genuine LEAP multi-object sparse-scene pass | `9465af0` | known pinch + far obstacle |
| `CH-004` | Target-local point crop | `3e47ca2` | frame/unprojection/determinism tests |
| `CH-005` | Cập nhật completion matrix | P3.3 plan v1.5.0 | release blockers giữ mở |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | pytest `tests/scenes` | scene runtime xanh | 40 passed | pass | bounded one-thread run |
| `V-002` | focused scene/physics/dataset/no-bypass gate | integration xanh | 82 passed trong 7.10 s | pass | bounded one-thread run |
| `V-003` | genuine LEAP scene rollout | measured positive | pass với obstacle vật lý | pass | `test_known_leap_*` |
| `V-004` | wrong-object fixture | reject obstacle contact | `wrong_object_contact` | pass | collector integration test |
| `V-005` | ruff/diff/docs/import hooks | repository gates xanh | pass | pass | commit hooks |

- Tất cả test chạy tuần tự với một BLAS/OpenMP thread; không OOM.
- Chưa chạy: bounded 12-scene generation, three-hand rendered QA, external real
  source smoke và independent final review.
- Khả năng rollback: revert ba runtime commits độc lập.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `REV-20260825-004` vẫn đúng cho adapter checkpoint.
- Không thay closure P3.2/P3.2.1 hoặc object-level P3.1 release gate.
- P3.3-11 complete nghĩa là integration đo được evidence; không đồng nghĩa
  three-hand release artifact đã tồn tại.
- Loại kiểm tra lại cần thiết: `full_review` cho P3.3-14/15.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-25 Asia/Bangkok.
- Người kiểm tra: codex-primary-agent qua genuine rollout và repository gates.
- Kết luận: runtime blockers đã đóng; Phase 3.3 chuyển sang release generation.
- Liên kết bản ghi hoàn tất: `REV-20260825-005`.
