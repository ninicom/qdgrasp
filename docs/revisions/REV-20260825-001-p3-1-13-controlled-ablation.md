---
document_id: REV-20260825-001
document_type: revision_record
revision_schema: 2
title: Đóng P3.1-13 controlled ablation và chọn region opposition theo scope
status: complete
date: 2026-08-25
record_id: REV-20260825-001
session_id: SESSION-20260825-P3.1-13
created_at: 2026-08-25T00:30:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260823-001
    artifact: docs/revisions/REV-20260823-006-ablation-study-v2.md
    revision: a5ba55197382b70b46516d0286b2cf38cee40435ea34f790918d65368b5c89e3
  - session_id: ROADMAP-P3.1-001
    artifact: docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md
    revision: 2ab14db2d647c563c45b57871615cbaff193c5562b6bb9a76cb55838c7d77dfa
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: ff0e4843ca807b3891e20f2e57a925689919742264a0c497d0b2060769b32183
reason: "Kết quả ablation cũ không có dynamic evidence để chọn recipe; rerun fail-closed 84 candidates nay cung cấp measured three-hand evidence và tách rõ positive-control selection khỏi canonical generalization."
necessity: N2
impact: "P3.1-13 chuyển complete, region_opposition_v1 được chọn cho regeneration trong validated envelope, P3.1-14 chuyển ready; canonical generalization vẫn chưa được claim."
---

# REV-20260825-001 — P3.1-13 controlled ablation

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260825-P3.1-13`.
- Bản ghi hoàn tất: tài liệu này.
- Artifacts bị sửa và checksum trước sửa: đúng theo ba mục `revises` ở metadata.
- Revision code chạy official evidence:
  `79a997b7ba2a568f1ea2f01812823cedb85ee0a4`.
- Evidence: `evidence/phase3_1/p13-controlled-ablation/`.
- Finding nguồn: P3.1-13 trong `ROADMAP-P3.1-001` và invalidated decision trong
  `REV-20260823-006`.

## 2. Lý do chỉnh sửa

Hai ablation record cũ đều không có measured dynamic positive và vì vậy không
thể chọn release recipe. Protocol mới khóa trước criterion fail-closed, rate có
mẫu số theo từng stage, provenance source/profile/object/protocol và hard cap 96
candidates. Canonical 72-candidate run vẫn không phân biệt được recipe, nên một
tier positive-control riêng được chạy trên đúng generated-reachable fixtures đã
qua P3.2.1 review; hai scope không bị nhập làm một.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: quyết định recipe và trạng thái mở khóa regeneration thay đổi dựa
  trên measured evidence; đây không phải chỉnh sửa biên tập.
- Thời hạn/độ khẩn: phải hoàn tất trước P3.1-14.
- Biện pháp giảm thiểu nếu trì hoãn: P3.1-14 tiếp tục fail-closed, không sinh
  dataset release.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | Wrench chỉ thử một opposing finger | Region và wrench dùng cùng lịch morphology | vừa | giữ regression ba hand |
| API/schema/config | Report v1 thiếu matrix scope và diagnostics | Report v2 ghi matrix, provenance, per-cell diagnostics | vừa | consumer đọc schema v2 |
| Dữ liệu/checkpoint | Chưa có recipe hợp lệ để regenerate | Chọn region cho validated envelope | cao | chạy P3.1-14 staging |
| CPU/GPU/export | Máy từng tràn RAM | CPU tuần tự, một thread, timeout hữu hạn | vừa | giữ resource limits |
| Kết luận công bố | Ablation cũ inconclusive | Region thắng positive-control; canonical vẫn inconclusive | cao | không claim generalization |

Tóm tắt tác động: P3.1-13 được đóng trong scope có bằng chứng; P3 tổng chưa đóng
và canonical limitation không thay đổi.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Harden report accounting, diagnostics, provenance và selection rule | `scripts/ablate_recipes.py` @ `79a997b` | 7 accounting tests pass |
| `CH-002` | Cho wrench-guided enumerate opposing fingers công bằng với region | pipeline @ `79a997b` | proposal tests pass |
| `CH-003` | Chạy official positive-control matrix 84/84 | `report-positive-control.json` | SHA-256 `f2831dc9db276c039c77c8503f4ec7e62cdc02b5f33e454ded9d425aead59186` |
| `CH-004` | Chọn `region_opposition_v1` theo criterion khóa trước | report decision | accepted 3/28, ba hand, ba contact signatures |
| `CH-005` | Cập nhật roadmap, giữ canonical limitation | P3.1 plan và project phases | documentation gate |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | pytest generated-reachable từng hand, một thread | 3 hand pass | LEAP, Allegro, Shadow pass | pass | terminal log phiên |
| `V-002` | pytest orchestrator/region/wrench/accounting | focused regression pass | 14 passed | pass | commit preflight |
| `V-003` | `ablate_recipes.py --execute --matrix positive-control` | 84 accounted, bounded | 84/84, không timeout/OOM | pass | official report |
| `V-004` | pre-registered selection rule | unique winner hoặc inconclusive | unique `region_opposition_v1` | pass | report decision |
| `V-005` | Git commit hooks | docs/source registries pass | 50 checks pass | pass | commit `79a997b` |

- Regression đã chạy lại: generated-reachable full flow cả ba hand và 14 focused
  pipeline/accounting tests.
- Kiểm tra bắt buộc chưa chạy: không có; full repository suite không thuộc gate
  P3.1-13 và không được dùng để mở rộng claim.
- Khả năng rollback: revert commit `79a997b` và revision/evidence commit; P3.1-14
  tự động trở lại blocked vì không còn selection evidence.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo bị ảnh hưởng: `REV-20260823-006` giữ trạng thái invalidated.
- Verdict cũ còn hiệu lực: kết luận “không chọn recipe từ zero-yield data” vẫn
  đúng; quyết định wrench cũ vẫn vô hiệu.
- Loại kiểm tra lại cần thiết: `delta_review` cho protocol/report v2 và scope.
- Đính chính: roadmap ghi region chỉ được chọn trong validated positive-control
  envelope; canonical run vẫn inconclusive.
- Người chấp nhận rủi ro/ngoại lệ: không có ngoại lệ; selection rule fail-closed.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-25 Asia/Bangkok.
- Người kiểm tra: codex-primary-agent, source hash/accounting/regression scope.
- Kết luận: P3.1-13 hoàn tất; P3.1-14 ready, P3.1-15 và P3 tổng còn pending.
- Liên kết bản ghi hoàn tất: `REV-20260825-001`.
