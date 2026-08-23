---
document_id: REV-20260823-001
document_type: revision_record
revision_schema: 2
title: Khóa Literature Snapshot và Đánh dấu Data Correctness P3
status: in_review
date: 2026-08-23
record_id: REV-20260823-001
session_id: SESSION-20260822-022
created_at: 2026-08-23T00:00:00+07:00
author: codex-primary-agent
revises:
  - session_id: ROADMAP-P3.1-001
    artifact: docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md
    revision: 07a99408657e5122f26ab8a30c676523f8748601c11499168a397b12215f5607
reason: "Khởi động Phase 3.1 nhằm giải quyết các lỗ hổng logic nghiêm trọng được phát hiện tại phiên làm việc SESSION-20260822-022, khóa literature snapshot và đánh dấu dataset cũ bị invalidated."
necessity: N1
impact: "Invalidate dataset shards cũ, kích hoạt pipeline typed contracts, batched DLS-IK, static force LP certifier, và dynamic rollout validator."
---

# REV-20260823-001 — Khóa Literature Snapshot và Đánh dấu Data Correctness P3

## 1. Liên kết truy vết

- Roadmap kế hoạch: `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md`.
- Session ghi nhận: `SESSION-20260822-022`.
- Tài liệu nghiên cứu: EquiDexFlow (06/2026), SynManDex (06/2026), SECOND-Grasp (05/2026), BiDexGrasp (04/2026).

## 2. Lý do chỉnh sửa

Bản sinh dữ liệu trước đó của `DGN-Open-Tiny` chứa các lỗ hổng logic về reachability và static force closure giả mạo (False Positive closure mà không qua DLS-IK hội tụ). Toàn bộ 144 mẫu trước đó cần bị vô hiệu hóa (invalidated) để tái thiết lập tính đúng đắn (correctness) theo Phase 3.1.

## 3. Mức độ cần thiết

- Mức: `N1` — sửa sai sót nghiêm trọng về tính đúng đắn của dữ liệu và kiến trúc pipeline.
- Phạm vi: Pipeline sinh dữ liệu `qdgrasp/dataset/pipeline/`, hợp đồng dữ liệu `contracts.py`, và dataset `datasets/dgn-open-tiny/`.

## 4. Phạm vi và tác động

- Invalidate toàn bộ các shard `.pt` và `dataset_manifest.json` cũ.
- Thiết lập hệ thống Typed Contracts và Allowlist Recipes (`surface_fixed_v1`, `region_opposition_v1`, `wrench_guided_v1`).
- Bắt buộc kiểm tra 5 tầng độc lập: Proposal → Batched DLS-IK → Collision Guard → Static Force LP → MuJoCo Rollout.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact |
|---|---|---|
| CH-001 | Đánh dấu dataset cũ invalidated | `docs/revisions/REV-20260823-001-p3-data-correctness.md` |
| CH-002 | Thiết lập hệ thống Typed Contracts | `qdgrasp/dataset/pipeline/contracts.py` |
| CH-003 | Triển khai batched DLS-IK | `qdgrasp/dataset/pipeline/solvers/` |
| CH-004 | Triển khai Static Force LP & GWS | `qdgrasp/dataset/pipeline/certifiers/` |
| CH-005 | Triển khai MuJoCo Dynamic Rollout | `qdgrasp/dataset/pipeline/validators/` |

## 6. Xác minh

| Verification ID | Phương pháp | Kết quả |
|---|---|---|
| V-001 | Unit test contract & solvers | pass (`pytest tests/`) |
| V-002 | Dynamic rollout physics test | pass (`test_physics_rollout.py`) |
| V-003 | Pipeline orchestrator integration | pass (`test_pipeline_orchestrator.py`) |

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Quyết định nghiệm thu Phase 3 tại `SESSION-20260822-022` bị hủy bỏ và chuyển sang quy trình khắc phục Phase 3.1.
- Toàn bộ dataset phân phối cũ bị thu hồi.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-23 Asia/Bangkok.
- Người kiểm tra: Independent Review required.
- Kết luận: Hồ sơ mở Phase 3.1 hoàn tất.
