---
document_id: ARCHIVE-INDEX
document_type: index
title: Chỉ mục raw archive bất biến
status: active
date: 2026-08-21
revises: none
---

# Raw archive bất biến

Các file được liệt kê ở đây được giữ nguyên byte để bảo toàn lịch sử. Metadata
vòng đời nằm trong sidecar này, không được chèn vào raw file. Một raw archive
không phải nguồn sự thật hiện hành và các assertion bên trong không được xem là
kết quả đã kiểm chứng.

Các nhắc tới RH56E2 trong `PLAN.pre-v2.md` chỉ là lịch sử bất biến. RH56E2 đã bị
loại khỏi toàn bộ scope hoạt động bởi ADR-0007; archive không cấp quyền đưa hand,
URDF, mesh, metadata, fixture, config hoặc checkpoint đó trở lại dự án.

| Artifact | Định danh lịch sử | Trạng thái | Được thay bởi | Revision record | SHA-256 |
|---|---|---|---|---|---|
| `docs/archive/PLAN.pre-v2.md` | `LEGACY-PLAN-PRE-V2` | `superseded` | `PLAN-V2` (`PLAN.md`) | `docs/revisions/REV-20260821-001-plan-v2.md` | `d4d8b1d51723c830bab041575388cd58984e60b77e06013844ca02f03220d36d` |

Kiểm tra từ project root:

```bash
sha256sum -c docs/reviews/evidence/TPR-20260821-002-scope.sha256
```

Nếu hash raw archive thay đổi, không cập nhật hash sidecar để che sai lệch. Phải
đánh dấu artifact bị thay đổi là `invalidated`, điều tra nguyên nhân và tạo
revision record mới.
