---
document_id: REV-20260823-005
document_type: revision_record
revision_schema: 2
title: Hoàn tất Phase 3.2 kiến trúc transmission, underactuated control và unblock Shadow Hand
status: in_review
date: 2026-08-23
record_id: REV-20260823-005
session_id: SESSION-20260822-022
created_at: 2026-08-23T13:00:20+07:00
author: codex-primary-agent
revises:
  - artifact: docs/roadmap/PHASE3_2_UNDERACTUATED_CONTROL_PLAN.md
    revision: 9680371a53b516b0d9e7943c22ad65b5971a179fa45da3cbf3a4fc2c53a6c5eb
  - artifact: qdgrasp/presets/robots/shadow_hand.yaml
    revision: 4b2964a3eb795da0f56a596dc56ad3fa9fbbca4feea1d17d5ceca5ebae33eb5a
  - artifact: qdgrasp/dataset/pipeline/validators/mujoco_rollout.py
    revision: d5fa47da588ab4d2e99d8d648b29ff07f436d4f6c589bce91dc6e8312e9bfdb1
  - artifact: docs/roadmap/PROJECT_PHASES.md
    revision: b9960ff5cfadadcd7d722d36cfd7db1a3641b9feee02df014389df05b18bc6bf
  - artifact: docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md
    revision: becebbdf3fe9cf04b7beebf3796d19efcb8a1ba6adca1ce585a9bc8d6e32d56d
reason: "Hoàn tất toàn diện Phase 3.2: triển khai tầng transmission thống nhất (TransmissionModel, DirectJointTransmission, FixedTendonTransmission), giải thuật chiếu command controllable/null-space, active-finger DLS IK, tích hợp physical rollout validator và unblock release cho Shadow Hand."
necessity: N1
impact: "Shadow Hand đạt rank 20 trên 24 joint states, vượt qua kiểm tra rollout vật lý 3 stage, mở khóa release_blocked: false và chuyển giao hoàn tất nhiệm vụ P3.1-11."
---

# REV-20260823-005 — Hoàn tất Phase 3.2 underactuated control & unblock Shadow Hand

## 1. Liên kết truy vết

- Kế hoạch gốc: `docs/roadmap/PHASE3_2_UNDERACTUATED_CONTROL_PLAN.md` (`ROADMAP-P3.2-001`).
- Kế hoạch cha: `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md` (nhiệm vụ `P3.1-11`).
- Gate script: `scripts/check_phase3_2.py`.
- Báo cáo telemetry: `runs/phase3_2_verification_report.json`.
- Test suite: `tests/test_robot_transmission.py`, `tests/test_active_finger_ik.py`, `tests/test_phase3_2_dynamic_fixtures.py`, `tests/test_physics_rollout.py`.

## 2. Lý do chỉnh sửa

Trước Phase 3.2, hệ thống robot specification và physical rollout validator chỉ hỗ trợ ánh xạ direct-drive 1-1, dẫn tới việc Shadow Hand (24 kinematic joint states, 20 actuators điều khiển qua 4 fixed tendon) bị từ chối release (`release_blocked: true`).

Phase 3.2 giải quyết triệt để vấn đề này với các module cốt lõi:
1. **Transmission Layer (`qdgrasp/robot/transmission/`)**:
   - `TransmissionModel` protocol và `TransmissionState` contract.
   - `extract_moment_matrix`: trích xuất ma trận moment $M = \partial l / \partial q \in \mathbb{R}^{U \times J}$ từ cấu trúc CSR của MuJoCo MjData.
   - `compute_finite_difference_moment_matrix`: central finite difference oracle đạt sai số tuyệt đối $0.00$ so với nghiệm giải tích trên cả 3 bàn tay.
   - `DirectJointTransmission`: rank 16 trên 16 joints cho LEAP và Allegro.
   - `FixedTendonTransmission`: rank 20 trên 24 joints cho Shadow Hand.
2. **Controllable-Space Command Projection (`command.py`)**:
   - Tách biến phân joint delta mong muốn thành thành phần điều khiển được $\Delta q_{\text{control}} \in \text{range}(M^T)$ và thành phần null-space unactuated $\Delta q_{\text{null}} \in \text{null}(M)$.
   - Tính mục tiêu điều khiển actuator $u_{\text{target}} = M \Delta q^* + l_{\text{current}}$.
   - Fail-closed gate: nếu mục tiêu yêu cầu độ trôi null-space vượt ngưỡng ($> 0.05$ rad), tự động từ chối với lý do `nullspace_rejection` trước khi bắt đầu mô phỏng.
3. **Active-Finger Mask DLS IK (`fixed_contact_dls.py`, `region_dls.py`)**:
   - Hỗ trợ mảng mặt nạ boolean `active_fingers` $[\text{batch}, K]$.
   - Kiểm tra tối thiểu 2 ngón hoạt động (`min_active_fingers`).
   - Residual và convergence chỉ đánh giá trên các ngón tiếp xúc thực tế; các ngón không hoạt động giữ giới hạn góc quay và không làm thất bại solver.
4. **Tích hợp Validator MuJoCo (`mujoco_rollout.py`)**:
   - Thay thế toàn bộ logic ánh xạ ad-hoc bằng `TransmissionModel`.
   - Áp dụng lệnh điều khiển trên toàn bộ chu trình smoothstep squeeze, lift và perturbation.
   - Ghi nhận đầy đủ telemetry: `transmission_rank`, `joint_state_dimensions`, `control_dimensions`, `controllable_residual`, `nullspace_residual`, `actuator_saturation_count`.
5. **Unblock Shadow Hand Release**:
   - Chuyển `release_blocked: false` trong `qdgrasp/presets/robots/shadow_hand.yaml`.

## 3. Mức độ cần thiết

- Mức `N1`: đảm bảo tính toàn vẹn vật lý và giải thuật cho robot underactuated không phụ thuộc vào các xấp xỉ giả (mimic equality hay direct state teleportation).

## 4. Phạm vi và tác động

- Triển khai package `qdgrasp/robot/transmission/` hoàn chỉnh (`contracts.py`, `extraction.py`, `direct.py`, `tendon.py`, `command.py`, `factory.py`).
- Cập nhật solver DLS IK (`fixed_contact_dls.py`, `region_dls.py`) với `active_fingers` mask.
- Nâng cấp `validate_grasp_rollout` trong `mujoco_rollout.py` chạy qua `TransmissionModel` và `ActuatorCommand`.
- Mở khóa release cho `shadow_hand.yaml` (`release_blocked: false`).
- Chuyển giao hoàn tất nhiệm vụ `P3.1-11` trong `PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md`.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| ID | Artifact | Thay đổi |
| --- | --- | --- |
| CH-001 | `qdgrasp/robot/transmission/` | Triển khai contracts, extraction, DirectJointTransmission, FixedTendonTransmission, command projection và factory |
| CH-002 | `qdgrasp/dataset/pipeline/solvers/` | Bổ sung `active_fingers` mask và `min_active_fingers` constraint vào `fixed_contact_dls.py` và `region_dls.py` |
| CH-003 | `qdgrasp/dataset/pipeline/validators/mujoco_rollout.py` | Tích hợp transmission state, actuator command projection, dynamic tracking metrics và null-space fail-closed gate |
| CH-004 | `qdgrasp/presets/robots/shadow_hand.yaml` | Cập nhật `release_blocked: false` và ghi nhận hỗ trợ FixedTendonTransmission |
| CH-005 | `tests/` | Bổ sung `test_robot_transmission.py`, `test_active_finger_ik.py`, `test_phase3_2_dynamic_fixtures.py` và cập nhật `test_physics_rollout.py` |
| CH-006 | `scripts/check_phase3_2.py` | Tạo verification gate script cho Phase 3.2 kiểm tra rank, finite difference parity, command projection, IK và physical rollout |
| CH-007 | `docs/roadmap/` | Cập nhật `PROJECT_PHASES.md` và `PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md` ghi nhận hoàn tất Phase 3.2 và P3.1-11 |

## 6. Xác minh

| ID | Lệnh / kiểm tra | Kết quả |
| :--- | :--- | :--- |
| V-001 | `pytest tests/test_robot_transmission.py` | PASS (16/16 rank LEAP/Allegro, 20/24 rank Shadow, FD parity < 1e-5) |
| V-002 | `pytest tests/test_active_finger_ik.py` | PASS (3 tests: LEAP, rejection insufficient fingers, Shadow) |
| V-003 | `pytest tests/test_physics_rollout.py` | PASS (7 tests: LEAP, Allegro, Shadow multi-stage lift & disturbance) |
| V-004 | `pytest tests/test_phase3_2_dynamic_fixtures.py` | PASS (4 fixtures: 3 hands positive + nullspace negative) |
| V-005 | `python scripts/check_phase3_2.py` | PASS (1.94s: ranks, FD parity, command projection, IK, rollouts) |
| V-006 | `python scripts/check_robot_assets.py` | PASS (clean pinned assets, immutable lock) |
| V-007 | `python scripts/check_environment.py` | PASS (CPU/cu128 environment clean) |
| V-008 | `python scripts/check_docs.py` | PASS (86 files checked) |

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `REV-20260823-003` và `REV-20260823-004` được hiện thực hóa và đóng lại hoàn tất.
- Block `release_blocked: true` của Shadow Hand đã được gỡ bỏ chính thức.
- Kế hoạch Phase 3.1 sẵn sàng tiến hành các bước cuối (P3.1-13 controlled ablation và P3.1-14 dataset regeneration).

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-23 Asia/Bangkok.
- Người kiểm tra: Independent Review required.
- Kết luận: Toàn bộ 10 sub-tasks P3.2-00 đến P3.2-09 đã hoàn tất 100%, vượt qua gate kiểm tra và tài liệu hóa đầy đủ.
