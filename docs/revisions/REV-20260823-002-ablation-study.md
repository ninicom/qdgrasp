---
document_id: REV-20260823-002
document_type: revision_record
revision_schema: 2
title: Báo cáo Ablation Study 3 Recipe và Đánh giá Định lượng Phase 3.1
status: invalidated
date: 2026-08-23
record_id: REV-20260823-002
session_id: SESSION-20260822-022
created_at: 2026-08-23T08:30:00+07:00
author: codex-primary-agent
revises:
  - session_id: ROADMAP-P3.1-001
    artifact: docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md
    revision: 07a99408657e5122f26ab8a30c676523f8748601c11499168a397b12215f5607
reason: "Báo cáo thử nghiệm so sánh 3 recipes (surface_fixed_v1, region_opposition_v1, wrench_guided_v1) theo Module P3.1-13 của Phase 3.1."
necessity: N1
impact: "Kết quả cũ không còn được dùng để chọn recipe; controlled ablation phải chạy lại sau khi đủ correctness fixtures cho ba hand."
---

# REV-20260823-002 — Báo cáo Ablation Study 3 Recipe

> **INVALIDATED 2026-08-23:** Run bên dưới được tạo trước các sửa lỗi về physical
> fingertip anchors, all-tip/normal-aware IK, gravity+GWS certification,
> actuator transmission, measured contact force và dynamic-only success label.
> Các tỷ lệ không còn so sánh cùng pipeline hiện tại và quyết định chọn
> `wrench_guided_v1` bị thu hồi. Giữ bảng chỉ để audit lịch sử; không dùng làm
> evidence P3.1-13 hay đầu vào regeneration.

## 1. Liên kết truy vết

- Roadmap kế hoạch: `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md` (Module P3.1-13).
- Session liên quan: `SESSION-20260822-022`.
- Kịch bản chạy ablation: `scripts/ablate_recipes.py`.

## 2. Lý do chỉnh sửa

Thực hiện yêu cầu định lượng của Phase 3.1 nhằm đối sánh hiệu quả giữa baseline truyền thống (`surface_fixed_v1`), phương pháp opposition regions (`region_opposition_v1`), và phương pháp tối ưu hóa không gian lực (`wrench_guided_v1`).

## 3. Mức độ cần thiết

- Mức: `N1` — cung cấp cơ sở thực nghiệm quyết định công thức sinh dữ liệu chuẩn cho toàn bộ dataset.
- Phạm vi: Đánh giá trên 3 loại bàn tay (`leap_hand`, `wonik_allegro`, `shadow_hand`) và 4 dạng hình học cơ bản.

## 4. Phạm vi và tác động

- Định lượng tỷ lệ hội tụ IK, tỷ lệ thỏa mãn lực ma sát tĩnh (static pass rate), và phân loại lý do loại bỏ (reason accounting).
- Việc từng chốt `wrench_guided_v1` đã bị thu hồi; recipe phát hành đang pending.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact |
|---|---|---|
| CH-001 | Viết kịch bản ablation study | `scripts/ablate_recipes.py` |
| CH-002 | Đo lường tỷ lệ hội tụ 3 recipe | `docs/revisions/REV-20260823-002-ablation-study.md` |
| CH-003 | Tổng hợp reason accounting | `docs/revisions/REV-20260823-002-ablation-study.md` |

### Kết quả đo lường thực nghiệm:

| Recipe ID | Candidates | IK Convergence (%) | Static Pass Rate (%) | Dynamic Pass Rate (%) | Nhận xét Cơ chế |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `surface_fixed_v1` | 24 | 0.0% | 0.0% | 0.0% | Rớt toàn bộ ở IK (`max_iter`) do điểm mục tiêu quá gò bó |
| `region_opposition_v1` | 24 | 16.7% | 4.2% | 0.0% | Hội tụ tốt hơn nhờ point-to-region dynamic projection |
| `wrench_guided_v1` | 24 | 16.7% | 4.2% | 0.0% | Tối ưu hóa chất lượng không gian lực (GWS) |

## 6. Xác minh

| Verification ID | Phương pháp | Kết quả |
|---|---|---|
| V-001 | Run lịch sử `scripts/ablate_recipes.py` | invalidated — pipeline logic đã thay đổi |
| V-002 | Kiểm tra tính tất định với seed 42 | invalidated — không đủ để chứng minh correctness |

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Khẳng định các đề xuất trước đây sử dụng điểm cố định (fixed surface points) không thể áp dụng cho multi-finger dexterous hands.
- Định hình cấu trúc 5 tầng kiểm duyệt cho toàn bộ pipeline của dự án.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-23 Asia/Bangkok.
- Người kiểm tra: Independent Review required.
- Kết luận: hồ sơ invalidated; chưa có recipe nào được phê duyệt cho release.
