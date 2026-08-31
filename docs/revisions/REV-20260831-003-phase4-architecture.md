---
document_id: REV-20260831-003
document_type: revision_record
revision_schema: 2
title: Ghi trạng thái thi công Phase 4 và bổ sung execution plan cho P4
status: complete
date: 2026-08-31
record_id: REV-20260831-003
session_id: SESSION-20260831-003
created_at: 2026-08-31T18:00:00+07:00
author: claude-agent
revises:
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: ROADMAP-001@1.31.0
  - session_id: ROADMAP-P4-001
    artifact: docs/roadmap/PHASE4_EXECUTION_PLAN.md
    revision: ROADMAP-P4-001@1.0.0
reason: "P4 được thi công 12/14 mục của cổng sau khi P3.5 cạn phần việc làm được trên máy phát triển; trạng thái `pending` và việc P4 không có execution plan không còn đúng."
necessity: N2
impact: "P4 chuyển từ pending sang in_progress với kiến trúc đo được là học được trên CPU; cổng CUDA của P4 giữ nguyên và chưa mục nào của nó được đánh dấu đạt."
---

# REV-20260831-003 — Thi công Phase 4

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260831-003`.
- Bản ghi hoàn tất: `docs/sessions/SESSION-20260831-003-phase4-qdgrasp-flow.md`.
- Artifact bị sửa: `docs/roadmap/PROJECT_PHASES.md`,
  `docs/roadmap/PHASE4_EXECUTION_PLAN.md`.
- Revision trước sửa: `ROADMAP-001@1.31.0`, `ROADMAP-P4-001@1.0.0`.
- Revision sau sửa: `ROADMAP-001@1.32.0`, `ROADMAP-P4-001@1.1.0`.
- Artifact mới: `docs/roadmap/PHASE4_EXECUTION_PLAN.md` (`ROADMAP-P4-001@1.0.0`),
  `qdgrasp/models/{tokenizer,encoder,hand_graph,flow,losses}.py`,
  `tests/model_flow/test_model_flow.py`, `scripts/overfit_qdgrasp_flow.py`,
  `evidence/phase4/overfit-leap-cpu.json`; commit `ff5d816`.

## 2. Lý do chỉnh sửa

- `PROJECT_PHASES.md` ghi P4 là `pending`. Sau phiên này P4 có 12/14 mục của
  cổng đã giao kèm test, nên `pending` mô tả sai trạng thái.
- P4 là phase duy nhất được mô tả trong `PLAN.md` §M3 và `PROJECT_PHASES.md`
  nhưng không có execution plan, trong khi mọi phase trước đều có. Thiếu nó thì
  không có scope khóa, không có test matrix và không có cổng đóng để đối chiếu.
- Việc ghi nhận này **không** nới bất kỳ cổng nào: cổng P4 đòi CUDA
  forward/backward, gradient coverage, joint/rotation hữu hạn và tiny overfit
  pass. Chỉ hai mục sau đo được trên CPU; hai mục đầu vẫn chờ P4-11.

## 3. Mức độ cần thiết

- Mức: `N2`.
- Lý do: trạng thái phase trong roadmap là thứ phiên sau đọc để chọn việc. Ghi
  sai thì hoặc P4 bị làm lại từ đầu, hoặc bị coi là xong khi cổng CUDA chưa chạy.
- Phương án đã cân nhắc và bị loại: để nguyên `pending` và chỉ ghi trong session
  report. Bị loại vì `PROJECT_PHASES.md` là bảng trạng thái duy nhất được đọc ở
  cấp phase; một sự thật chỉ nằm trong session report thì không ai thấy.

## 4. Phạm vi và tác động

- Tài liệu bị đụng: `docs/roadmap/PROJECT_PHASES.md` (dòng trạng thái P4 và mục
  `### P4 — Model MVP`); `docs/roadmap/PHASE4_EXECUTION_PLAN.md` (cột output của
  §4, không đụng phạm vi, contract hay cổng).
- Tài liệu được thêm: `docs/roadmap/PHASE4_EXECUTION_PLAN.md`.
- Không đụng: `PLAN.md`, mọi ADR, mọi gate của P0–P3.5 và P5–P7.
- `ADR-0006` giữ nguyên và đang chặn P4-11: chưa có bằng chứng CUDA nào cho P4,
  và một lần chạy CPU không được trình bày như CUDA. `ADR-0008` giữ nguyên:
  model chỉ được đo trên LEAP và Allegro.
- Release: không. `release_class` của `ROADMAP-P4-001` là `not_a_release`.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Thay đổi | Trước | Sau |
| --- | --- | --- |
| Trạng thái P4 trong bảng phase | `pending` | `in_progress — 12/14 mục; kiến trúc overfit được trên CPU cho cả hai active hand, cổng CUDA chưa chạy` |
| Execution plan của P4 | không có | `docs/roadmap/PHASE4_EXECUTION_PLAN.md` |
| Model code | `qdgrasp/models/` chỉ có `__init__.py` | tokenizer, encoder, hand graph, flow head, losses, config/registry |
| Cổng của P4 | chỉ là văn bản trong §6 của plan | `scripts/check_phase4.py`, đọc được bằng máy |
| Harness CUDA | không có | `scripts/phase4_cuda_gate.py` + notebook pin commit |
| Packet review | không có | `scripts/phase4_review_packet.py` + `PHASE4_REVIEWER_GUIDE.md` |
| §4 của plan | đoán trước `conditioning.py`, `fk_head.py`, `quality.py` | tên file thật; ba head chung một forward pass nên chung file |
| Bằng chứng học được | không có | `evidence/phase4/overfit-{leap,allegro}-cpu.json` |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `python -m pytest tests/model_flow -q` | pass | 63 passed | `pass` | log phiên |
| `V-002` | `python -m pytest -q` | không regression | 1252 passed, 1 skipped | `pass` | log phiên |
| `V-003` | `python scripts/overfit_qdgrasp_flow.py` | hội tụ dưới bốn ngưỡng pose | palm 0.040 m, rot 0.021 rad, joint 0.053 rad, tip 0.041 m | `pass` | `evidence/phase4/overfit-leap-cpu.json` |
| `V-004` | gradient coverage sau backward | mọi tham số có gradient hữu hạn | 186/186 | `pass` | evidence như trên |
| `V-005` | `python scripts/check_docs.py` | tài liệu hợp lệ | 145 file pass | `pass` | log phiên |
| `V-006` | `ruff check` + `ruff format --check` | sạch | sạch | `pass` | log phiên |
| `V-007` | `python scripts/check_phase4.py --profile micro` | báo đúng trạng thái | 12/14 delivered, exit 1 | `pass` | log phiên |
| `V-008` | `python scripts/phase4_cuda_gate.py --device cpu` | từ chối | `verdict=refused` | `pass` | log phiên |
| `V-009` | `python scripts/overfit_qdgrasp_flow.py --robot wonik_allegro.yaml` | hội tụ | palm 0.046 m, rot 0.027 rad | `pass` | `evidence/phase4/overfit-allegro-cpu.json` |

- Kiểm tra chưa chạy: toàn bộ cổng CUDA của P4 (`P4-11b`) vì không có NVIDIA
  GPU trên máy phát triển; independent review (`P4-12`).
- Khả năng rollback: các module trong `qdgrasp/models/` là bổ sung thuần. Điểm
  duy nhất chúng chạm vào code cũ là một dòng import trong `qdgrasp/api/__init__.py`
  để đăng ký builder `qdgrasp_flow`; gỡ dòng đó và thư mục `models/` đưa hành vi
  của P0–P3.5 về nguyên trạng.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có.
- Verdict cũ còn hiệu lực: có. MVP-T `complete` với kết luận âm về phần học,
  P3.5 `in_progress` 17/19, P3.4 `paused_by_ADR-0008` — không mục nào bị đụng.
  Cổng closure của P3.4.3 được sửa cho chặt hơn (`audit_closure` nay kiểm tồn
  tại của `test_ids`), nhưng trên ledger thật nó cho 0 violation, nên verdict
  `FAIL` và `release_blocked=true` giữ nguyên vì đúng lý do cũ.
- Loại kiểm tra lại cần thiết: `none` cho phase cũ; `full_review` cho chính P4
  khi nó đóng.
- Đính chính số liệu cần phát hành: không có.
- Người chấp nhận rủi ro/ngoại lệ: không áp dụng.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-agent, 2026-08-31, UTC+07.
- Người kiểm tra: chưa có; `P4-12` vẫn mở.
- Kết luận: phần thi công của phiên đã hoàn tất và được đo. P4 **không** được
  ghi `complete`: hai mục còn lại được nêu tên cùng lý do, và cổng nặng nhất của
  phase — forward/backward trên CUDA thật — chưa chạy một lần nào. Cả hai đều
  cần thứ không tồn tại trên máy phát triển: một GPU NVIDIA thật và một người
  kiểm tra không phải tác giả.
- Liên kết bản ghi hoàn tất phiên hiện tại:
  `docs/sessions/SESSION-20260831-003-phase4-qdgrasp-flow.md`.
