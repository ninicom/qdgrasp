---
document_id: DOCS-INDEX
document_type: index
title: Chỉ mục tài liệu QDGrasp
version: 1.9.0
status: active
date: 2026-08-31
revises: DOCS-INDEX@1.8.0
latest_revision_record: docs/revisions/REV-20260831-001-grasp-policy-mvp-closure.md
revision_reason: Grasp Policy MVP đã đóng; chỉ mục trỏ tới model card và evidence của nó.
necessity: N2
impact: Chỉ mục dẫn tới artifact experimental_non_release của MVP; các gate correctness/release cũ giữ nguyên.
---

# Bộ tài liệu kiểm chứng QDGrasp

Mục tiêu của thư mục này là làm cho mọi tuyên bố “đã xong” và mọi số liệu đều
có thể truy ngược đến code, cấu hình, môi trường, artifact thô và người kiểm tra.

## Luồng bằng chứng

```text
PLAN.md
  → decision / protocol
  → session report
  → raw artifact + SHA-256
  → metrics report
  → third-party review
  → revision record (nếu phải sửa kết quả cũ)
```

## Chỉ mục

- `governance/DOCUMENTATION_POLICY.md`: quy tắc chung, trạng thái tài liệu và
  mức cần thiết khi sửa lịch sử.
- `governance/SESSION_RULES.md`: cách ghi đúng việc hoàn tất trong từng phiên.
- `governance/THIRD_PARTY_REVIEW.md`: quy trình kiểm tra độc lập.
- `governance/GIT_WORKFLOW.md`: branch feature/develop/release, commit và gate.
- `governance/REFERENCE_INTAKE.md`: chỉ bổ sung code tham chiếu khi có issue cụ
  thể, kèm exact pin và license/provenance gate.
- `installation/ENVIRONMENT.md`: Python/PyTorch CPU-CUDA profiles, uv lock và
  environment smoke gate, cùng contract cài QDGrasp như thư viện.
- `roadmap/PROJECT_PHASES.md`: roadmap cấp cao gồm tám phase cân bằng từ nền
  dự án đến release/paper/community.
- `roadmap/GRASP_POLICY_MVP_TEMPORARY_PLAN.md`: đường thực thi ưu tiên tạm thời
  để ra checkpoint LEAP state-based gắp một object family trước; artifact là
  experimental/non-release và không đóng giả P3.4.3, P3.5, P4 hoặc P5.
  Đã `complete` từ 2026-08-31; kết quả và giới hạn nằm ở
  `reports/MVP-GRASP-POLICY-MODEL-CARD.md`, artifact thô ở `evidence/mvp/`.
- `configuration/TRAIN_ARGUMENTS.md`: bảng đủ 127 tên trong public train/config
  surface Ultralytics đã pin, disposition và quy tắc CPU/CUDA.
- `configuration/TRAIN_ARGUMENTS.yaml`: registry máy đọc được; checker từ chối
  thiếu/thừa key, sai default/type/hash/commit.
- `references.lock.yaml`: URL/commit/hash/license evidence của source read-only;
  `scripts/check_references.py` khóa file và xác minh checkout sạch.
- `metrics/METRICS_REGISTRY.md`: định nghĩa metric và điều kiện so sánh.
- `templates/`: mẫu bắt buộc cho session, metrics, review, revision và release.
- `sessions/`: báo cáo từng phiên làm việc.
- `metrics/reports/`: báo cáo số liệu đã chạy.
- `reviews/`: báo cáo kiểm tra của người thứ ba.
- `revisions/`: hồ sơ sửa tài liệu hoặc kết luận của phiên trước.
- `decisions/`: quyết định kiến trúc/quy trình có ảnh hưởng dài hạn.
- `decisions/0007-agpl-community-library.md`: quyết định AGPL-3.0, library-first,
  DGN2 paper-only và loại RH56E2; supersede quyết định Apache trước đó.
- `decisions/0008-temporary-shadow-hand-pause.md`: active corpus tạm thời chỉ
  gồm LEAP/Allegro; giữ Shadow preset/evidence nhưng dừng dùng trong workload,
  release gate và RL pipeline mới cho tới quyết định mở lại.
- `roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md`: corrective plan hiện
  hành để hoàn tất safety, trajectory, GPU-to-CPU replay, CUDA và dataset gate
  cho LEAP/Allegro mà không đổi P3.4 ba-hand paused thành pass.
- `roadmap/phase3_4_3_requirements.yaml`: inventory 85 requirement dùng
  zero-unmapped completeness rule; mọi mục giữ `pending` cho tới khi có
  implementation, test, evidence và review tương ứng.
- `roadmap/PHASE3_4_3_EXECUTION_BREAKDOWN.md`: thứ tự thi công S0–S12 của
  corrective plan, kèm quy tắc chuyển trạng thái bốn phần
  (implemented/tested/evidenced/reviewed).
- `roadmap/PHASE3_4_3_HANDOFF.md`: interface freeze và ba điều kiện mở
  contact-rich input cho P4; static/offline không bị chặn.
- `archive/`: tài liệu đã bị thay thế nhưng không bị xóa; `archive/README.md`
  là sidecar trạng thái/hash cho raw archive bất biến.

## Công cụ kiểm tra

Chạy trước khi kết thúc phiên hoặc gửi review:

```bash
python3 scripts/check_references.py --lock-only
python3 scripts/check_train_args.py --registry-only
python3 scripts/check_docs.py --root .
```

Khi có clone chuẩn, chạy thêm `python3 scripts/check_references.py --source-root
.references` và `python3 scripts/check_train_args.py --source
.references/ultralytics`. Các validator không thay thế review nội dung hoặc xác
nhận số liệu.
