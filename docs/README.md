---
document_id: DOCS-INDEX
document_type: index
title: Chỉ mục tài liệu DexGrasp
status: active
date: 2026-08-21
revises: none
---

# Bộ tài liệu kiểm chứng DexGrasp

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
- `configuration/TRAIN_ARGUMENTS.md`: bảng đủ 127 tên trong public train/config
  surface Ultralytics đã pin, disposition và quy tắc CPU/CUDA.
- `configuration/TRAIN_ARGUMENTS.yaml`: registry máy đọc được; checker từ chối
  thiếu/thừa key, sai default/type/hash/commit.
- `metrics/METRICS_REGISTRY.md`: định nghĩa metric và điều kiện so sánh.
- `templates/`: mẫu bắt buộc cho session, metrics, review, revision và release.
- `sessions/`: báo cáo từng phiên làm việc.
- `metrics/reports/`: báo cáo số liệu đã chạy.
- `reviews/`: báo cáo kiểm tra của người thứ ba.
- `revisions/`: hồ sơ sửa tài liệu hoặc kết luận của phiên trước.
- `decisions/`: quyết định kiến trúc/quy trình có ảnh hưởng dài hạn.
- `archive/`: tài liệu đã bị thay thế nhưng không bị xóa; `archive/README.md`
  là sidecar trạng thái/hash cho raw archive bất biến.

## Công cụ kiểm tra

Chạy trước khi kết thúc phiên hoặc gửi review:

```bash
python3 scripts/check_train_args.py --registry-only
python3 scripts/check_docs.py --root .
```

Khi có clone chuẩn, chạy thêm `python3 scripts/check_train_args.py --source
.references/ultralytics`. Các validator không thay thế review nội dung hoặc xác
nhận số liệu.
