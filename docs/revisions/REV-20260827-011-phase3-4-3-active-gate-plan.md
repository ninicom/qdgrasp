---
document_id: REV-20260827-011
document_type: revision_record
revision_schema: 2
title: Lập corrective plan hoàn tất contact-rich gate cho active hands
status: complete
date: 2026-08-27
record_id: REV-20260827-011
session_id: SESSION-20260827-P3.4-STATIC-LOGIC-AUDIT
created_at: 2026-08-27T23:59:00+07:00
author: codex-primary-agent
revises:
  - session_id: PLAN-V2
    artifact: PLAN.md
    revision: 900d78227abe5f2abc8279ebe73060ef6bd441ef2a6013d9bc78cf765859f3eb
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: c56dd7ad3ee71723bc04fe4d272a7d10e1adff3ef165ff88176f792f739339c0
  - session_id: DOCS-INDEX
    artifact: docs/README.md
    revision: 238f20628b09e3b43c2358a523f2b594091ce9bf9f59b405092057de42edcef8
reason: "Static code review phát hiện các release-blocking gap trong safety accounting, GPU finalist replay, certification, trajectory time/frame, dataset manifest và gate scripts; cần một successor plan active-two-hand thay vì diễn giải P3.4 paused thành pass."
necessity: N3
impact: "Dừng dùng ContactRich v1 làm release/P4 contact-rich evidence; thêm ROADMAP-P3.4.3-001 với gate fail-closed cho LEAP+Allegro, giữ P3.4 ba-hand paused và Shadow ngoài default workload theo ADR-0008."
---

# REV-20260827-011 — Active contact-rich corrective gate plan

## 1. Liên kết truy vết

- Plan tạo mới: `docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md`
  (`ROADMAP-P3.4.3-001`).
- Plan cha: `ROADMAP-P3.4-001`; quyết định phạm vi: `ADR-0008`.
- Artifact bị sửa: `PLAN.md`, `docs/roadmap/PROJECT_PHASES.md`, `docs/README.md`.
- Session: `SESSION-20260827-P3.4-STATIC-LOGIC-AUDIT`.

## 2. Lý do chỉnh sửa

- Trước: roadmap đã ghi P3.4 ba-hand paused nhưng chưa có execution plan active
  xử lý đầy đủ các defect logic còn lại để tạo contact-rich input hai-hand.
- Sau: `ROADMAP-P3.4.3-001` là plan active cho correctness/release gate
  LEAP+Allegro; P3.4 ba-hand vẫn `paused_by_ADR-0008`.

## 3. Mức độ cần thiết

`N3`. Audit phát hiện đường có thể bỏ lọt safety-budget violation, GPU finalist
không có exact replay payload và manifest/evidence có count/split sai. Đây là
vấn đề an toàn mô phỏng và release provenance, nên không được xử lý như chỉnh
tài liệu N0/N1.

## 4. Phạm vi và tác động

- Thêm plan mới, không sửa raw evidence hoặc closed session report.
- Cập nhật project roadmap và docs index để trỏ đúng execution plan.
- Không sửa runtime trong revision này; mọi blocker vẫn open cho tới khi work
  package tương ứng có code và evidence.
- ContactRich v1 không còn được dùng làm release/P4 contact-rich evidence.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

- Tạo `ROADMAP-P3.4.3-001` với blocker B-01–B-10 và gate G00–G11.
- Cập nhật `PLAN.md`, `PROJECT_PHASES.md`, `docs/README.md` trỏ tới plan mới.

Checksum sau sửa:

- `PLAN.md`:
  `a52708dacbf94002112f3c2e16251a526256b377dec9d92b6616535758e5c6dd`;
- `docs/roadmap/PROJECT_PHASES.md`:
  `bed3894b688ebc55af6bced2d2b5b6781bd00589208716365f63a1a5e6018793`;
- `docs/README.md`:
  `3811d10dfe62d897778a4d09bdcda975846409980496121ab79d3bcbf36249e1`;
- `docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md`:
  `92a76da494208595ff4542a579d8982ff4183bce1791a8e3fdcab1b877a66746`.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | static review thủ công trên commit `a3eb016` | liệt kê blocker release-blocking | mười nhóm blocker B-01–B-10 | pass | `ROADMAP-P3.4.3-001` §2 |
| `V-002` | đối chiếu plan với `ADR-0008` | không claim ba-hand mới | verdict giới hạn `P3.4.3-ACTIVE-PASS` 2/2 active | pass | `ROADMAP-P3.4.3-001` §1 |

Không chạy test, simulation hay benchmark trong revision lập plan này.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- P3.4 ba-hand giữ nguyên `paused_by_ADR-0008`; revision này không đổi verdict
  lịch sử và không tái sử dụng tên release ba-hand.
- `QDGrasp-ContactRich-Tiny` v1 chuyển sang trạng thái không dùng làm release
  hoặc P4 contact-rich evidence.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-27 Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer.
- Kết luận: revision này hoàn tất việc lập kế hoạch, không hoàn tất các
  technical gate. P4 contact-rich input chỉ được mở bởi `P3.4.3-ACTIVE-PASS`
  sau independent review.
- Liên kết bản ghi hoàn tất: `REV-20260827-011`.
