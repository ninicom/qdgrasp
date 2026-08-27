---
document_id: REV-20260827-005
document_type: revision_record
revision_schema: 2
title: Triển khai Phase 3.4 tới 14/18 work package và ghi nhận blocker P3.4-16
status: complete
date: 2026-08-27
record_id: REV-20260827-005
session_id: SESSION-20260827-P3.4
created_at: 2026-08-27T17:40:00+07:00
author: claude-primary-agent
revises:
  - session_id: ROADMAP-P3.4-001
    artifact: docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md
    revision: 4a6132a5be065cdbe25b9ea7c86896558068ec80fb7f80f1e5297a415e7152b7
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 7a7612e432b461682b3547a732a551f9eaf33dff4edab3eaafbaa328cdd86135
reason: "P3.4 chuyển từ 0 dòng code sang 14/18 work package có test; đồng thời phát hiện rollout không điều khiển được tay thật nên P3.4-16 bị chặn, cần ghi lại chẩn đoán để lần sau không lặp."
necessity: N2
impact: "P3.4 có contracts, hai backend, observer, primitives, objective, CEM, refine, certifier, storage và ablation, tất cả pass CPU gate. P3.4-16 blocked, P3.4-17 không thể tự phát hành, P3.4-05/15 cần GPU. Phase 3.4 KHÔNG đóng."
---

# REV-20260827-005 — Trạng thái triển khai Phase 3.4

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260827-P3.4`.
- Bản ghi hoàn tất: tài liệu này.
- Artifact bị sửa và checksum trước sửa: hai mục `revises` ở metadata.
- Revision code: `ee6db09242c233b53f7a8d5f121a8485f881ed9d`.
- Evidence: `evidence/phase3_4/p04-backend-spike/`,
  `evidence/phase3_4/p15-cuda-backend-decision/`,
  `evidence/phase3_4/p14-ablation/`,
  `evidence/phase3_4/p16-dataset-blocked/`.

## 2. Lý do chỉnh sửa

`ROADMAP-P3.4-001` mô tả 18 work package chưa có dòng code nào. Phiên này
triển khai 14 trong số đó, giải quyết câu hỏi backend bằng số đo trên phần cứng
thật, và phát hiện một blocker chưa từng được nêu trong kế hoạch.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: thêm một tầng runtime mới và ghi nhận blocker làm thay đổi đường đi
  của phase; không sửa kết luận đã công bố nào.
- Thời hạn/độ khẩn: trước khi có ai bắt đầu P3.4-16.
- Biện pháp nếu trì hoãn: lần thử tiếp theo lặp lại ba vòng chẩn đoán đã tốn.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | không có `qdgrasp/dynamic` | 8 module mới + backend batched | cao | wrap validated rollout |
| API/schema/config | không có dynamic contract | `qdgrasp/dynamic-trajectory/v1` | trung bình | giữ ổn định |
| Dữ liệu/checkpoint | không đổi | không có dataset mới | không | P3.4-16 |
| Tương thích CPU/GPU | không có backend | CPU oracle + CUDA backend | cao | GPU throughput |
| Số liệu/kết luận đã công bố | không có | ablation `no_measured_difference` | trung bình | không claim hypothesis |

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

- `CH-001` P3.4-00/01 typed contracts; `passed=True` không dựng được nếu thiếu
  `cpu_replay_evidence`.
- `CH-002` P3.4-02/03 backend protocol, scene bucketing và MuJoCo CPU oracle.
- `CH-003` P3.4-04 requirement matrix đo từ model thật.
- `CH-004` P3.4-05 MJWarp CUDA backend, không có nhánh nào rơi về CPU.
- `CH-005` P3.4-06 contact observer và safety budget đa đại lượng.
- `CH-006` P3.4-07 primitive controller; primitive không mang object pose.
- `CH-007` P3.4-08 static-seeded rollout.
- `CH-008` P3.4-09 CEM cùng objective khai báo và reason ledger có mẫu số.
- `CH-009` P3.4-11/12 local refine và CPU replay certifier.
- `CH-010` P3.4-13 trajectory storage sparse, deterministic.
- `CH-011` P3.4-14 ablation; verdict `no_measured_difference`.
- `CH-012` P3.4-15 stage 1 harness; verdict backend `supported` trên Tesla T4.
- `CH-013` Ghi nhận blocker P3.4-16 kèm chẩn đoán và hướng sửa.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `pytest tests/dynamic_grasp/ -q` | pass | 117 passed | pass | log phiên |
| `V-002` | `check_phase3_4.py --backend cpu --profile micro` | exit 0, PARTIAL | exit 0, 12 done | pass | log phiên |
| `V-003` | `check_phase3_4.py --backend cuda` | fail-closed | exit 1 | pass | log phiên |
| `V-004` | Kaggle T4, MuJoCo Warp 1.16.0 | verdict backend | `supported`, 4/4 capability | pass | `p15-cuda-backend-decision/` |
| `V-005` | ablation hai nhánh, 6 candidate mỗi nhánh | có verdict | `no_measured_difference`, yield 0/0 | pass | `p14-ablation/` |
| `V-006` | probe rollout ba tay thật, ba vòng | tìm positive | **0/9 hand-iterations** | fail | `p16-dataset-blocked/` |

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Kế hoạch `ROADMAP-P3.4-001` không lường trước rằng rollout của P3.4 phải dùng
  lại protocol `mocap-weld-v3` thay vì tự viết. Đây là bổ sung, không phải đính
  chính: kế hoạch không khẳng định điều ngược lại.
- Ablation `no_measured_difference` **không** xác nhận giả thuyết P3.4. Scene
  micro pusher không có ngón nên không nhánh nào sinh positive được. Không được
  trích kết quả này như bằng chứng ủng hộ hay bác bỏ giả thuyết.
- Verdict backend `supported` chỉ nói Warp nhận model và đọc được contact
  stream. Nó không đo throughput, VRAM hay parity; những mục đó vẫn trống.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-primary-agent, 2026-08-27 Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer.
- Kết luận: **Phase 3.4 không đóng.** 14/18 work package hoàn tất trên CPU;
  P3.4-16 blocked, P3.4-17 không thể do tác giả tự phát hành, P3.4-05/15 còn
  thiếu số đo throughput/parity trên GPU. P3 tổng vẫn `pending`.
- Liên kết bản ghi hoàn tất: `REV-20260827-005`.
