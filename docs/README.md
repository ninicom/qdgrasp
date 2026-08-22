---
document_id: DOCS-INDEX
document_type: index
title: Chỉ mục tài liệu QDGrasp
version: 1.3.0
status: active
date: 2026-08-21
revises: none
latest_revision_record: docs/revisions/REV-20260821-006-audit-remediation.md
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
