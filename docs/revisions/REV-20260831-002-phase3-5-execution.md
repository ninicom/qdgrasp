---
document_id: REV-20260831-002
document_type: revision_record
revision_schema: 2
title: Thi công Phase 3.5 và ghi trạng thái đo được của từng work package
status: complete
date: 2026-08-31
record_id: REV-20260831-002
session_id: SESSION-20260831-002
created_at: 2026-08-31T12:00:00+07:00
author: claude-agent
revises:
  - session_id: ROADMAP-P3.5-001
    artifact: docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md
    revision: ROADMAP-P3.5-001@1.2.0
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: ROADMAP-001@1.29.0
reason: "P3.5 được thi công tới 16/19 mục của cổng sau khi ROADMAP-MVP-001 đóng; trạng thái execution_priority cũ và mô tả 'chưa thi công' không còn đúng."
necessity: N2
impact: "P3.5 chuyển từ deferred sang in_progress với cổng CPU chạy được; P3.5-15/17/18 còn mở và không gate nào của P3.3, P3.4, P3.4.3 hay ADR-0008 bị đổi."
---

# REV-20260831-002 — Thi công Phase 3.5

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260831-002`.
- Bản ghi hoàn tất: `docs/sessions/SESSION-20260831-002-phase3-5-ingest-and-rl-contracts.md`.
- Artifact bị sửa: `docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md`
  (`ROADMAP-P3.5-001@1.2.0` → `@1.3.0`), `docs/roadmap/PROJECT_PHASES.md`
  (`ROADMAP-001@1.29.0` → `@1.30.0`).
- Cổng mới: `scripts/check_phase3_5.py`.
- Harness cho phần chưa đóng được: `scripts/phase3_5_gpu_rl_readiness.py`,
  `notebooks/phase3_5_rl_readiness.ipynb`.

## 2. Lý do chỉnh sửa

`ROADMAP-P3.5-001@1.2.0` ghi `execution_priority: deferred_by_ROADMAP-MVP-001`
và mô tả toàn bộ implementation là việc tương lai. MVP đã đóng
(`REV-20260831-001`), P3.5 quay lại hàng đợi, và mười sáu trong mười chín mục
của cổng nay có artifact chạy được kèm test. Giữ nguyên tài liệu cũ sẽ mô tả sai
trạng thái theo cả hai hướng: che phần đã làm, và không nêu rõ ba mục còn lại
không thể đóng từ máy phát triển.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: thay đổi trạng thái thực thi của một phase và bổ sung một cổng mới
  là thay đổi kết luận quy phạm về việc phase đang ở đâu. Không phải `N3` vì
  không gate release, license hay quyết định an toàn nào phụ thuộc: P3.5 vẫn
  chưa đóng và không artifact nào của nó được phát hành.
- Thời hạn: đã hoàn tất trong phiên này.
- Biện pháp giảm thiểu nếu trì hoãn: không áp dụng.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | không có ingest, resolver, settle certifier hay RL environment | mười module mới với test | cao | Chạy `scripts/check_phase3_5.py` |
| API/schema/config | — | `qdgrasp/objects/{ingest,coacd,manifest_v2}`, `qdgrasp/scenes/{resolver,serialize,virtual_drop,settle}`, `qdgrasp/rl/*` | cao | Public API mới, chưa có consumer ngoài test |
| Dữ liệu/checkpoint | — | chưa có artifact; `QDGrasp-RL-Env-Tiny` còn mở | không | P3.5-17 |
| Tương thích CPU/GPU/export | — | CPU oracle chạy cho hai active hand; GPU chưa chạy | vừa | P3.5-15 cần CUDA run thật |
| Số liệu/kết luận đã công bố | không có số P3.5 nào | 16/19 mục delivered | vừa | Không có claim release |

Tóm tắt tác động: P3.5 `in_progress`. Không dataset hay checkpoint nào được gỡ
`release_blocked`; `QDGrasp-ContactRich-Active-Tiny` giữ nguyên; Shadow vẫn
`paused_by_ADR-0008` và bị `qdgrasp/rl/envs/hand_scene.py` từ chối load.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Raw mesh ingest với typed refusal và scale áp đúng một lần | `qdgrasp/objects/ingest.py` | `tests/assets_ingest/test_ingest.py` |
| `CH-002` | Public CoACD API đủ tham số, cache, typed error | `qdgrasp/objects/coacd.py` | `tests/assets_ingest/test_coacd_api.py` |
| `CH-003` | `ObjectAssetManifestV2` bất biến | `qdgrasp/objects/manifest_v2.py` | `tests/assets_ingest/test_ingest.py` |
| `CH-004` | Scene resolver và serializer canonical | `qdgrasp/scenes/{resolver,serialize}.py` | `tests/rl_env/test_scene_and_envs.py` |
| `CH-005` | Virtual drop scene và settle certifier | `qdgrasp/scenes/{virtual_drop,settle}.py` | `tests/rl_env/test_scene_and_envs.py` |
| `CH-006` | RL contract, ba environment, fixture, randomization | `qdgrasp/rl/` | `tests/rl_env/` |
| `CH-007` | Actuator khớp theo transmission target thay vì tên | `qdgrasp/rl/envs/hand_scene.py` | Allegro `ffa0` ↔ `ffj0` |
| `CH-008` | Hướng bàn tay ban đầu lấy từ palm-to-fingertip reach của profile | `qdgrasp/rl/envs/dex_acquire.py` | reset không chạm cho cả hai tay |
| `CH-009` | Hai ngân sách xuyên thấu: trạng thái ổn định và trần tức thời | `qdgrasp/scenes/settle.py`, `qdgrasp/rl/envs/dex_acquire.py` | 7.7 mm lúc chạm đất → 0.07 mm khi nghỉ |
| `CH-010` | Descent của fixture kết thúc khi chạm | `qdgrasp/rl/tasks/scripted.py` | cả hai tay chạy hết horizon không vi phạm |
| `CH-011` | Cổng CPU và harness GPU/cloud | `scripts/check_phase3_5.py`, `scripts/phase3_5_gpu_rl_readiness.py`, `notebooks/phase3_5_rl_readiness.ipynb` | `tests/rl_env/test_gpu_harness.py` |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `python -m pytest tests/assets_ingest -q` | pass | 47 passed | `pass` | log phiên |
| `V-002` | `python -m pytest tests/rl_env -q` | pass | 51 passed | `pass` | log phiên |
| `V-003` | `python scripts/check_phase3_5.py --profile micro` | báo đúng trạng thái | 16/19 delivered, exit 1 | `pass` | log phiên |
| `V-004` | `python scripts/phase3_5_gpu_rl_readiness.py --backend mujoco-cpu --device cpu` | CPU oracle pass | pass cho hai hand | `pass` | log phiên |
| `V-005` | `python scripts/phase3_5_gpu_rl_readiness.py --backend mjx-warp --device cuda:0` | từ chối | `verdict=refused` | `pass` | log phiên |
| `V-006` | `python -m pytest tests scripts/tests -q` | không regression | 1168 passed, 1 skipped | `pass` | log phiên |
| `V-007` | `python scripts/check_docs.py --root .` | tài liệu hợp lệ | 142 file pass | `pass` | log phiên |

- Regression đã chạy lại: toàn bộ `tests/` và `scripts/tests/`.
- Kiểm tra chưa chạy: GPU spike (`P3.5-15`) vì không có NVIDIA GPU; parity
  output-class của CoACD vì backend không được cài; independent review vì tác
  giả không được tự ký.
- Khả năng rollback: các module mới là bổ sung thuần; gỡ chúng không đổi hành vi
  của P3.1–P3.4.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có.
- Verdict cũ còn hiệu lực: có. P3.3 `complete`, P3.4 `paused_by_ADR-0008`,
  P3.4.3 chờ CUDA gate và review — không mục nào bị đụng tới.
- Loại kiểm tra lại cần thiết: `none` cho phase cũ; `full_review` cho chính P3.5
  khi nó đóng.
- Đính chính số liệu cần phát hành: không có.
- Người chấp nhận rủi ro/ngoại lệ: không áp dụng.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-agent, 2026-08-31, UTC+07.
- Người kiểm tra: chưa có; P3.5-18 vẫn mở.
- Kết luận: đã hoàn tất phần thi công của phiên. P3.5 chưa đóng và không được
  ghi `complete`; ba mục còn lại được nêu tên cùng lý do trong `§9` của plan.
- Liên kết bản ghi hoàn tất phiên hiện tại:
  `docs/sessions/SESSION-20260831-002-phase3-5-ingest-and-rl-contracts.md`.
