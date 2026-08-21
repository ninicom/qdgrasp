---
document_id: REPORTS-INDEX
document_type: index
title: Chỉ mục báo cáo
status: active
date: 2026-08-21
revises: none
---

# Báo cáo

Số liệu chạy nằm ở `docs/metrics/reports/`; kiểm tra độc lập nằm ở
`docs/reviews/`; sửa kết luận cũ nằm ở `docs/revisions/`. Thư mục này dành cho
báo cáo tổng hợp phát hành, không thay thế ba loại hồ sơ gốc đó.

Mỗi release dùng `docs/templates/RELEASE_REPORT.md`, trỏ feature/merge commits,
tag mục tiêu, gate và rollback. Release report không được biến target trong
`PLAN.md` thành kết quả đã đo.

Một báo cáo tổng hợp chỉ được đánh dấu `accepted` khi mọi metric được trích dẫn
có metrics report và third-party review hợp lệ.
