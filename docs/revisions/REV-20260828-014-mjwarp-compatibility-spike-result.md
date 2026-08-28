---
document_id: REV-20260828-014
document_type: revision_record
revision_schema: 2
title: Kết quả compatibility spike MJWarp và verdict CUDA gate đo trên Kaggle T4
status: complete
date: 2026-08-28
record_id: REV-20260828-014
session_id: SESSION-20260828-P3.4.3-EXECUTION
created_at: 2026-08-28T09:40:00+07:00
author: claude-implementation-agent
revises:
  - session_id: SESSION-20260827-002
    artifact: docs/revisions/REV-20260827-010-mjwarp-upstream-defect.md
    revision: pending-spike-outcome
  - session_id: SESSION-20260828-P3.4.3-EXECUTION
    artifact: docs/revisions/REV-20260828-013-phase3-4-3-execution.md
    revision: cuda-gate-not-yet-run
reason: "REV-20260827-010 để hàng `MJWarp newer pinned version` của §3.7 ở trạng thái mở và gọi compatibility spike là bước tiếp theo; REV-20260828-013 ghi CUDA gate chưa có run thật. Spike đã chạy trên Kaggle T4 và cho câu trả lời đo được."
necessity: N2
impact: "Đóng hàng `MJWarp newer pinned version` bằng số đo: không version mujoco-warp nào trong 3.10.0.3–3.12.0 sạch trên warp-lang 1.16.0, và 1.16.0 là bản mới nhất tồn tại. CUDA gate trả FAIL với parity/performance đầy đủ. Không đổi ADR-0008, không đổi P3.4 historical verdict, không mở P4 contact-rich input."
---

# REV-20260828-014 — Kết quả compatibility spike MJWarp

## 1. Liên kết truy vết

- Bản ghi defect gốc: `REV-20260827-010`.
- Bản ghi thi công: `REV-20260828-013`.
- Gate: `scripts/check_phase3_4_3_cuda.py`; harness:
  `kaggle-phase3-4-3/`, kernel `qdgrasp-phase-3-4-3-cuda-gate-active-hands`.
- Evidence: `evidence/phase3_4_3/s10/kaggle-run-v5/`, `kaggle-run-v6/`,
  `kaggle-run-v7/`.

## 2. Lý do chỉnh sửa

`REV-20260827-010` kết luận defect nằm ngoài QDGrasp và chuyển `P3.4.1-02` từ
sửa code sang compatibility spike theo `§3.7`, để ngỏ hàng
`MJWarp newer pinned version`. Bản ghi này ghi kết quả của spike đó.

## 3. Mức độ cần thiết

`N2`. Đóng một hàng trong cây quyết định backend bằng số đo, và chuyển verdict
CUDA gate từ `blocked` sang `failed` với lý do đo được. Không đổi code contract,
không đổi threshold, không đổi trạng thái phase.

## 4. Phạm vi và tác động

- `§3.7` hàng `MJWarp newer pinned version`: **đóng**. Không tồn tại bản mới
  hơn nào sạch.
- `G08`/`C07`/`P3.4-15`: từ `blocked` sang `failed`, kèm số đo.
- `P3.4-04` (compatibility spike): `passed` — spike đã chạy và trả lời được.
- Không đổi `ADR-0008`; P3.4 ba-hand vẫn `paused_by_ADR-0008`; P4 contact-rich
  input vẫn đóng.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

Compatibility matrix chạy trên đúng reproducer `V-003` của `REV-20260827-010`
(LEAP hand kèm mesh), dưới `compute-sanitizer --tool initcheck`, trên
`warp-lang` 1.16.0:

| mujoco-warp | run v5 | run v6 |
| --- | --- | --- |
| 3.12.0 | 68224 errors | 65467 errors |
| 3.11.0 | 66181 errors | 67030 errors |
| 3.10.0.3 | 65248 errors | 69724 errors |

Cùng kernel `_linesearch_iterative_kernel`, cùng thread `(25,0,0)` như bản ghi
gốc. `racecheck` giữ `0 hazards` — đúng thứ phân biệt uninitialized read với
race. `warp-lang` 1.16.0 là bản mới nhất trên PyPI.

Ba defect trong harness của chính phiên này được sửa trước khi số đo được nhận:
matrix pin nhầm sang số hiệu `warp-lang`; `clean` được suy từ **vắng mặt** dòng
lỗi thay vì bằng chứng dương; và probe dùng scene ba geom thay vì reproducer.
Chi tiết trong README của từng thư mục evidence.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | Phase 1 CUDA smoke trên T4 | pass | pass, ba lượt liên tiếp | pass | `kaggle-run-v7/phase1_cuda.json` |
| `V-002` | Phase 2 active-hand FK parity trên T4 | pass | pass | pass | `kaggle-run-v7/phase2_cuda.json` |
| `V-003` | capability probe | mọi contact field đọc được | `supported`, `missing_contact_fields: []` | pass | `kaggle-run-v7/cuda-gate.json` |
| `V-004` | parity tier 1, không contact | delta `<=1e-4` | `5.75e-10` | pass | như trên |
| `V-005` | parity tier 2, một contact pinned | delta `<=2mm` | `8.39mm` | **fail** | như trên |
| `V-006` | parity tier 3, outcome class | đồng ý, mọi survivor có capsule | 4/4 đồng ý, 4/4 capsule | pass | như trên |
| `V-007` | speed, median 3 run | `>=2x` | LEAP `5.47x`, Allegro `14.04x` | pass | như trên |
| `V-008` | device VRAM | `<=14 GiB` | `0.037` / `0.062 GiB` | pass | như trên |
| `V-009` | non-finite worlds | zero | 84/1024 trên LEAP, 0 trên Allegro | **fail** | như trên |
| `V-010` | contact buffer overflow | zero | zero, cả hai hand | pass | như trên |
| `V-011` | `initcheck` | zero invalid read | uninitialized reads, `_linesearch_iterative_kernel` | **fail** | `kaggle-run-v7/sanitizer.json` |
| `V-012` | `racecheck` | zero hazard | `0 hazards displayed` | pass | như trên |

`V-005`, `V-009` và `V-011` là **một** phát hiện nhìn từ ba phía: integrator
khớp CPU tới mười chữ số khi không có gì chạm nhau, và lệch bốn lần dung sai
ngay khi có contact; world chết; sanitizer gọi đúng tên kernel.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- **`§3.7` hàng `MJWarp newer pinned version` đóng.** Điều kiện kích hoạt của nó
  — "upstream 1.16 bug đã cô lập" — đã thoả từ `REV-20260827-010`, spike đã
  chạy, và không có bản nào sạch. Hai hàng còn lại là fallback backend đã qua
  capability/parity, hoặc GPU gate đóng.
- **Tốc độ không mua lại được đúng đắn.** `§10` đòi `>=2x` **và** zero
  non-finite **và** zero sanitizer error. Backend nhanh — 5.47x và 14.04x — và
  không đáng tin. Không threshold nào bị đổi để verdict khác đi.
- **Đính chính bản ghi trước.** `REV-20260828-013` §8 viết CUDA gate "chưa có
  run thật trên T4". Câu đó đúng lúc viết và sai từ lúc này; bản ghi này thay
  thế nó.
- `REV-20260827-010` không bị thu hồi: kết luận nguyên nhân gốc nằm ngoài
  QDGrasp được tái hiện độc lập trên T4 mới, ba version, hai lượt.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-implementation-agent, 2026-08-28 Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer.
- Kết luận: compatibility spike hoàn tất và trả lời được. CUDA gate `FAIL` với
  số đo đầy đủ, không phải `blocked` vì thiếu phần cứng. P3.4.3 **chưa đóng**.
- Giới hạn còn lại: chưa thử fallback backend nào; chưa báo cáo defect lên
  upstream; chưa có reviewer độc lập.
- Liên kết bản ghi hoàn tất: `REV-20260828-014`.
