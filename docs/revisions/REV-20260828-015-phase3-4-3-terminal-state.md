---
document_id: REV-20260828-015
document_type: revision_record
revision_schema: 2
title: Chốt trạng thái Phase 3.4.3 là blocked theo chính điều khoản Definition of Done
status: complete
date: 2026-08-28
record_id: REV-20260828-015
session_id: SESSION-20260828-P3.4.3-EXECUTION
created_at: 2026-08-28T10:05:00+07:00
author: claude-implementation-agent
revises:
  - session_id: SESSION-20260828-P3.4.3-EXECUTION
    artifact: docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md
    revision: status-active-to-blocked
  - session_id: SESSION-20260828-P3.4.3-EXECUTION
    artifact: docs/roadmap/phase3_4_3_requirements.yaml
    revision: c05-and-p3.4-09-mislabelled
reason: "Plan để `status: active`, hàm ý còn việc đo được chưa đo. Không còn: mọi requirement đã rời `pending`, gate CUDA đã chạy trọn vẹn trên phần cứng thật, và câu hỏi paired-evidence đã hỏi ba cách. Ba mục DoD không đạt vì ba lý do khác hẳn nhau, và §12 quy định trạng thái đúng khi đó là `blocked`."
necessity: N2
impact: "Đổi ROADMAP-P3.4.3-001 từ active sang blocked kèm bảng lý do; sửa hai nhãn ledger sai (C05 gán nhầm lý do phần cứng, P3.4-09 còn BLOCKED sau khi gate đã đo xong); thêm nhánh thứ ba cho ablation. Không đổi verdict, không đổi ADR-0008, không mở P4 contact-rich input, không đụng P3.4 historical three-hand."
---

# REV-20260828-015 — Trạng thái chốt của Phase 3.4.3

## 1. Liên kết truy vết

- Plan: `ROADMAP-P3.4.3-001@1.2.0` §12.1.
- Ledger: `docs/roadmap/phase3_4_3_requirements.yaml`.
- Kết quả GPU: `REV-20260828-014`, `evidence/phase3_4_3/s10/kaggle-run-v7/`.
- Ablation: `evidence/phase3_4_3/s11/static-vs-dynamic.json`.

## 2. Lý do chỉnh sửa

`status: active` nói rằng còn việc đo được mà chưa đo. Điều đó đã hết đúng.
Không requirement nào còn `pending`; gate CUDA chạy hết mọi stage trên T4; câu
hỏi §16.3 đã hỏi ba cách độc lập. Ba mục DoD không đạt, nhưng không mục nào
trong số đó đóng được bằng cách chạy thêm trong repo này.

Đồng thời hai nhãn ledger nói sai điều chúng biết. `C05` mang lý do "requires a
real NVIDIA device" — nhưng C05 là dataset coverage và controlled ablation, cả
hai chạy trên CPU và cả hai đã xong; lý do đó bị gán nhầm gate. `P3.4-09` còn
`blocked` với nghĩa "gate dừng trước khi đo", trong khi gate đã đo xong.

## 3. Mức độ cần thiết

N2. Không sửa thì plan tự mô tả sai trạng thái của chính nó, và hai nhãn sai
khiến người đọc tưởng còn thiếu phần cứng — dẫn tới chạy lại một phép đo đã có
kết quả.

## 4. Phạm vi và tác động

Trong phạm vi: frontmatter plan, §12.1 mới, ba entry ledger, nhánh thứ ba của
ablation. Ngoài phạm vi: ADR-0008, P3.4 historical verdict, mọi gate đã pass.
`release_blocked=true` giữ nguyên và vẫn đúng.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

1. `ROADMAP-P3.4.3-001` 1.1.0 → 1.2.0, `active` → `blocked`, thêm §12.1 kèm
   bảng ba mục DoD không đạt và bản chất từng mục.
2. `C05`: `blocked` → `failed`. Coverage đạt 36/36 cell với đủ fixture dương và
   âm; cái không đạt là tiêu chí paired mà plan xếp dưới C05.
3. `P3.4-09`: `blocked` → `failed`, kèm số đo của run thứ bảy.
4. `R-DOD-03`: giữ `failed`, viết lại lý do theo ba nhánh đo.
5. `scripts/phase3_4_3_ablation.py`: thêm `measured_contact_arm` — force closure
   trên contact **đo được** ở sample enclosure cuối, thay vì contact recipe dự
   kiến. Nhánh này loại bỏ phản biện hiển nhiên với hai nhánh cũ: trước đó nhánh
   tĩnh chứng nhận grasp dự kiến còn nhánh động chạy grasp thực tế, nên bất đồng
   giữa chúng có thể chỉ là bất đồng về việc đang nói tới grasp nào.

## 6. Xác minh

| ID | Kiểm chứng | Kết quả |
|----|-----------|---------|
| V-001 | `audit_closure` trên cây sạch | `FAIL`, 0 violation |
| V-002 | Phân bố trạng thái | 65 passed / 14 failed / 5 blocked / 1 deferred / **0 pending** |
| V-003 | Năm mục `blocked` còn lại | C08, G10, G11, P3.4-17, R-DOD-08 — tất cả là reviewer độc lập |
| V-004 | Nhánh contact đo được, LEAP | 2 contact, static `True`, dynamic `True` |
| V-005 | Nhánh contact đo được, Allegro | 2 contact, static `True`, dynamic `True` |
| V-006 | `static_fail_dynamic_pass_cases` sau ba nhánh | `[]` |
| V-007 | Chiều ngược lại | 4 case static-pass/dynamic-fail |
| V-008 | `tests/contactrich_active` | 350 passed |
| V-009 | `scripts/check_docs.py` | 117 file, pass |
| V-010 | Ablation `parameter_hash` | không đổi ngoài nhánh mới, vẫn immutable |

V-004 và V-005 là kết quả quyết định: pinch hai điểm đối đỉnh thỏa mãn
certifier ở friction đã pin. Predicate tĩnh **rộng hơn** predicate động, ngược
với giả định của §16.3.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

`REV-20260828-013` mô tả C05 là chờ phần cứng; điều đó sai ngay khi viết và bản
ghi này đính chính. `REV-20260828-014` không bị ảnh hưởng. Kết luận GPU giữ
nguyên: hai đường còn lại là fallback backend hoặc bản vá upstream, cả hai là
công việc mới cần plan riêng, và §G07.5 cho phép giữ GPU gate không pass.

Không báo cáo nào trước đây từng claim Phase 3.4.3 `complete`; bản ghi này không
đảo ngược một claim nào mà chỉ đặt tên đúng cho trạng thái đã có.

## 8. Xác nhận đóng hồ sơ

Đóng ở mức bản ghi: mọi thay đổi trên đã commit và push. Không đóng Phase 3.4.3
— trạng thái là `blocked`, và ba mục DoD không đạt được liệt kê kèm điều kiện mở
lại từng mục. Reviewer độc lập vẫn là điều kiện bắt buộc và chưa có.
