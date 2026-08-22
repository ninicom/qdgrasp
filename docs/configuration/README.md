---
document_id: CONFIG-INDEX
document_type: index
title: Chỉ mục hợp đồng cấu hình QDGrasp
version: 1.1.0
status: active
date: 2026-08-21
revises: none
latest_revision_record: docs/revisions/REV-20260821-006-audit-remediation.md
---

# Hợp đồng cấu hình QDGrasp

Thư mục này chứa các registry có thể kiểm tra bằng máy cho public configuration
surface của thư viện.

- `TRAIN_ARGUMENTS.yaml`: nguồn sự thật máy đọc được cho toàn bộ tham số cấu
  hình Ultralytics đã pin và disposition tương ứng của QDGrasp.
- `TRAIN_ARGUMENTS.md`: bảng tra cứu đầy đủ, quy tắc CPU/CUDA và cách nâng phiên
  bản upstream.

YAML dùng subset scalar/inline-map cố định để checker standard-library có thể
validate exact schema. Semantic fingerprint khóa cả canonical/default/group,
custom/legacy/API, merge/resume rules, constraints và 19 extension placeholders;
Markdown được đối chiếu lại trong cùng checker.

Registry không cấp quyền sao chép hoặc phân phối code upstream. Provenance và
trạng thái giấy phép nằm trong `references.lock.yaml`.
