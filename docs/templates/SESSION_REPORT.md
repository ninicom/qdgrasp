---
document_id: SESSION-YYYYMMDD-NNN
document_type: session_report
title: Tên ngắn của phiên
status: draft
date: YYYY-MM-DD
session_id: SESSION-YYYYMMDD-NNN
author: tên-người-thực-hiện
revises: none
related_plan: PLAN-V2
---

# SESSION-YYYYMMDD-NNN — Tên phiên

## Mục tiêu phiên

- Mục tiêu cụ thể.
- Tiêu chí dừng/hoàn tất.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Mô tả kết quả, không mô tả ý định | `path/to/output` | verified |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | file/log/commit | `path` | `<sha256>` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `command --flags` | 0 | Mô tả chính xác phần đã kiểm |

## Việc chưa hoàn tất

- Không, hoặc liệt kê rõ work-in-progress/blocker.

## Sửa đổi phiên trước

Không. Nếu có, liên kết revision record và nêu `revision_reason`, `necessity`,
`impact` trong front matter.

## Bàn giao

- Trạng thái hiện tại.
- Bước hợp lệ tiếp theo.
- Điều kiện hoặc dữ liệu phiên sau phải kiểm tra trước.
