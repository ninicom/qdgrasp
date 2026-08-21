---
document_id: ADR-0001
document_type: decision
title: Bắt buộc truy xuất plan, phiên, số liệu, review và revision
status: accepted
date: 2026-08-21
revises: none
---

# ADR-0001 — Bắt buộc chuỗi bằng chứng

## Bối cảnh

Dự án có nhiều run, simulator, robot profile và protocol không hoàn toàn tương
đương. Một con số đúng trong một pipeline có thể bị dùng sai cho pipeline khác;
việc sửa tài liệu cũ không có lý do cũng làm mất lịch sử quyết định.

## Quyết định

Áp dụng chuỗi hồ sơ bắt buộc:

```text
plan → session → raw artifact → metrics report → third-party review → revision
```

Mọi sửa lịch sử phải nêu nguyên nhân và mức cần thiết `N0`–`N3`. Raw artifact
được nhận dạng bằng SHA-256. Validator cấu trúc chạy ở cuối mỗi phiên.

## Hệ quả

- Báo cáo tốn thêm thời gian nhưng có thể audit và so sánh an toàn.
- Kết quả thiếu artifact/protocol chỉ được coi là exploratory.
- Sửa số liệu `N2/N3` buộc chạy lại review và các gate phụ thuộc.
