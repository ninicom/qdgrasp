---
document_id: REV-20260825-004
document_type: revision_record
revision_schema: 2
title: Hoàn thiện implementation adapter Phase 3.3
status: complete
date: 2026-08-25
record_id: REV-20260825-004
session_id: SESSION-20260825-P3.3-ADAPTERS
created_at: 2026-08-25T12:00:00+07:00
author: codex-primary-agent
revises:
  - session_id: ROADMAP-P3.3-001
    artifact: docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md
    revision: 69f8ea1a33bced94b3f042f66519118db1e072bf9ee97556a3998c87b5256db6
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 523e28af3b51584e1528315332745220b326d41d725d10e21b0c4c7835579e11
reason: "Bốn adapter P3.3 đã chuyển từ skeleton/remediation sang implementation có pose, observation, external-label boundary và audit fail-closed; roadmap cần bỏ blocker cũ nhưng vẫn giữ source-scale smoke và release gate mở."
necessity: N2
impact: "P3.3-02 complete; P3.3-03…05 implementation complete với source-scale audit pending. Blocker kế tiếp là renderer 08 và multi-object rollout 11; chưa phát hành QDGrasp-Scene-Tiny."
---

# REV-20260825-004 — Hoàn thiện adapter Phase 3.3

## 1. Liên kết truy vết

- Phiên thực hiện: `SESSION-20260825-P3.3-ADAPTERS`.
- Baseline tài liệu: `ff61eeb`; commit runtime: `a6d067e`.
- Plan nguồn: `ROADMAP-P3.3-001` v1.3.0.
- Artifacts và checksum trước sửa: hai mục `revises` trong metadata.

## 2. Lý do chỉnh sửa

Checkpoint trước ghi chính xác adapters là skeleton. Runtime mới đã đọc layout
native/GraspNet-1Billion/DexGraspNet2/GraspClutter6D, chuẩn hóa frame và giữ
external labels ngoài boundary `dynamic_valid`. Trạng thái cần được nâng theo
evidence nhưng không được suy diễn fixture nhỏ thành full-source validation.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: thay đổi verdict bốn work package và dependency của release path.
- Thời hạn/độ khẩn: trước remediation renderer và rollout.
- Biện pháp nếu trì hoãn: giữ phase active và không sinh release artifact.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Native adapter | chưa load scene/observation | verified manifest/shards/references | cao | dùng cho generation path |
| External adapters | probe/index skeleton | scene, pose, camera, mask và label import | cao | one-scene source smoke |
| Provenance | audit complete giả | pinned manifest/license/file hash, fail closed | cao | license/source review |
| Environment | thiếu image runtime dependency | Pillow pin trong bốn lock | vừa | giữ lock integrity |

Tóm tắt tác động: adapter path đã dùng được ở mức implementation và fixture;
renderer, rollout và release evidence vẫn là blocker thực tế.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Native scene adapter từ verified shards | `a6d067e` | hash/calibration/reference negative tests |
| `CH-002` | GraspNet-1Billion và DexGraspNet2 import | `a6d067e` | world-frame pose, mask và external labels |
| `CH-003` | GraspClutter6D BOP import | `a6d067e` | w2c inversion, m2c pose, visible masks |
| `CH-004` | Pin Pillow và tái lập locks | environment locks | dry-run resolve pass |
| `CH-005` | Cập nhật completion matrix | P3.3 plan v1.4.0 | blocker không bị overclaim |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | pytest `tests/scenes/test_adapters.py` | adapter fixtures xanh | 9 passed | pass | bounded one-thread run |
| `V-002` | pytest `tests/scenes` | scene integration xanh | 35 passed trong 4.45 s | pass | bounded one-thread run |
| `V-003` | ruff check/format + diff check | source sạch | pass | pass | commit hooks/local run |
| `V-004` | uv lock dry-run | dependency graph resolve | chỉ local package/tomli delta | pass | `dev-cpu.lock` dry-run |
| `V-005` | documentation/import/reference checks | repository gates xanh | 50 tests + 96 docs | pass | commit `a6d067e` hooks |

- Regression đã chạy tuần tự với `OMP/MKL/OPENBLAS_NUM_THREADS=1`; không OOM.
- Kiểm tra bắt buộc chưa chạy: source-scale one-scene smoke trên ba external
  dataset, renderer remediation và multi-object rollout.
- Khả năng rollback: revert `a6d067e` và revision này độc lập.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `REV-20260825-003` vẫn đúng tại checkpoint cũ.
- Không thay verdict closure P3.2/P3.2.1 hoặc P3.1 release gate.
- Đính chính blocker: adapter implementation không còn skeleton; source-scale
  audit vẫn pending và không đồng nghĩa dataset được phép phân phối.
- Loại kiểm tra lại cần thiết: `delta_review` sau renderer/rollout integration.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-25 Asia/Bangkok.
- Người kiểm tra: codex-primary-agent qua runtime, lock và documentation gates.
- Kết luận: checkpoint adapter được ghi nhận; Phase 3.3 tiếp tục active.
- Liên kết bản ghi hoàn tất: `REV-20260825-004`.
