---
document_id: REV-20260823-003
document_type: revision_record
revision_schema: 2
title: Sửa mô hình trạng thái Shadow fixed-tendon và chặn release P3
status: in_review
date: 2026-08-23
record_id: REV-20260823-003
session_id: SESSION-20260822-022
created_at: 2026-08-23T11:07:15+07:00
author: codex-primary-agent
revises:
  - artifact: qdgrasp/presets/robots/shadow_hand.yaml
    revision: c4e8bd705eb3ede4bcc0568ea9737fcd82ea768c84058c497097c3aee90e0d05
  - artifact: qdgrasp/robot/spec.py
    revision: 086d3580d73994a55e234d7fec64b0fd25c518fbcbbb1badf144d0eec3666610
  - artifact: qdgrasp/dataset/pipeline/validators/mujoco_rollout.py
    revision: b87b91e58a0b49f4725984e6be87a437ddf3c8d083fcb0b511c6524725d6deb9
  - artifact: scripts/generate_dgn_open_tiny.py
    revision: de47a692911cf963ed4352f287b91f25ccd5540de16ddb16179e51bbae6896f7
  - artifact: scripts/check_phase2.py
    revision: 33c021674e00ad423bc0e5add1bea145120ce431d96cb4a53ce9107c5128c615
  - artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 3b16ffe64b04506b77a8cfc6877fb85fed5b50feafd08ca2f65e76d2efc38f5e
reason: "Fixed tendon của Shadow đã bị diễn giải sai thành mimic equality J1=J2, làm FK 20-DOF không tương thích với MuJoCo 24-state/20-control và có thể sinh nhãn động lực giả."
necessity: N1
impact: "Shadow chuyển sang state 24 joint, bị release_blocked cho tới khi có underactuated solver; LEAP và Allegro giữ nguyên release path."
---

# REV-20260823-003 — Shadow fixed-tendon state correction

## 1. Liên kết truy vết

- P2 gate: `scripts/check_phase2.py`.
- P3.1 plan: `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md`.
- Roadmap tổng: `docs/roadmap/PROJECT_PHASES.md`.
- Session bị ảnh hưởng: `SESSION-20260822-022`.
- Ablation bị vô hiệu hóa: `REV-20260823-002`.

## 2. Lý do chỉnh sửa

MJCF Shadow khai báo bốn fixed tendon, mỗi tendon đo tổng hai joint, ví dụ
`rh_FFJ0 = rh_FFJ2 + rh_FFJ1`. Đây là transmission từ 20 actuator sang 24 joint
states, không phải equality constraint buộc `J1=J2`. Profile cũ bỏ bốn `J1` khỏi
state và khai báo chúng là mimic của `J2`, nên FK có 20 chiều trong khi MuJoCo có
24 qpos độc lập.

Diagnostic tách biệt cho thấy thêm một lỗi frame trong validator: transform
`root→palm` được đo ở q=0 trước khi áp hai wrist joints của Shadow. Sau khi chuyển
initial-joint write lên trước phép đo, palm rotation tracking error không-contact
giảm từ `0.118396 rad` xuống `0.000850 rad`.

## 3. Mức độ cần thiết

- Mức `N1`: nếu không sửa, FK và MuJoCo mô tả hai state space khác nhau nhưng
  generator vẫn có thể phát hành nhãn dynamic như thể chúng tương đương.
- Không có workaround an toàn bằng tolerance, weld stiffness hoặc mimic giả.

## 4. Phạm vi và tác động

- Chạm contract P2 của riêng Shadow, dynamic validator P3, generator release,
  gate P2/P3 và roadmap.
- LEAP và Allegro không đổi số joint hay release eligibility.
- Shadow vẫn dùng được cho parser/FK research fixture, nhưng không được đưa vào
  dataset release trước khi có underactuated control module.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| ID | Artifact | Trước | Sau |
| --- | --- | --- | --- |
| CH-001 | `qdgrasp/presets/robots/shadow_hand.yaml` | 20 independent + 4 mimic; release mở | 24 independent state joints; 20 actuators; `release_blocked=true` |
| CH-002 | `qdgrasp/robot/spec.py` | suy fixed tendon thành mimic ratio | không suy transmission thành kinematic equality |
| CH-003 | `qdgrasp/dataset/pipeline/validators/mujoco_rollout.py` | đo root→palm trước initial wrist; tiếp tục rollout khi 24 targets chỉ có 20 controls | đo frame sau initial joints; fail `underactuated_targets` trước rollout |
| CH-004 | `scripts/generate_dgn_open_tiny.py` | tạo output trước khi phát hiện profile không thể release | validate cả ba profile trước khi tạo thư mục/artifact |
| CH-005 | `scripts/check_phase2.py` | ép Shadow qua published-profile gate | yêu cầu Shadow bị block với đúng reason, vẫn giữ parser/FK gate |
| CH-006 | P3.1 plan / ablation record | chưa nêu blocker; ablation cũ còn claim chọn recipe | blocker được ghi; `REV-20260823-002` invalidated |
| CH-007 | `docs/roadmap/PROJECT_PHASES.md` | latest revision cũ | version 1.5.0 trỏ REV-003; P3 vẫn pending và P4 bị chặn |

### SHA-256 artifact trước/sau

| Artifact | SHA-256 trước (HEAD `f4cf4f7`) | SHA-256 sau |
| --- | --- | --- |
| `qdgrasp/presets/robots/shadow_hand.yaml` | `c4e8bd705eb3ede4bcc0568ea9737fcd82ea768c84058c497097c3aee90e0d05` | `2fca002a0c0702c13e31450ec48900ff94b4bf58911a453f3c26642a9276416a` |
| `qdgrasp/robot/spec.py` | `086d3580d73994a55e234d7fec64b0fd25c518fbcbbb1badf144d0eec3666610` | `32e9fb616487c386eb974868f4dc90791d16fac025865f7505f83b0795ed23fa` |
| `qdgrasp/dataset/pipeline/validators/mujoco_rollout.py` | `b87b91e58a0b49f4725984e6be87a437ddf3c8d083fcb0b511c6524725d6deb9` | `b1b2930fcc8b97894bfc2abeca33b613e259721f15310e15fdbe189d893d0251` |
| `qdgrasp/dataset/pipeline/orchestrator.py` | `82fb8bbff03fca700315573b214f271dca04a9ae257b4c4559e3609b79689548` | `fb133e3e013b9f136eac0850daf538a80d64b246106e93b99595c7075abacc04` |
| `scripts/generate_dgn_open_tiny.py` | `de47a692911cf963ed4352f287b91f25ccd5540de16ddb16179e51bbae6896f7` | `5d5b364668890f0007b5516077d8e6e608853056d95958b0f9cc59d0b4787eed` |
| `scripts/check_phase2.py` | `33c021674e00ad423bc0e5add1bea145120ce431d96cb4a53ce9107c5128c615` | `485c5a27edb5233ad85fe77600cc876f08b18830bda478c0df31656e4a2d00f1` |
| `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md` | `596dc32f39d8fa4831509d9ad0171f38637eef83e36d15b6829d1de19f276836` | `c271b9be51a5f3cb9d6205a2a8b20d6e7d41ba01eb125cd3ae12d0d359e847ea` |
| `docs/roadmap/PROJECT_PHASES.md` | `3b16ffe64b04506b77a8cfc6877fb85fed5b50feafd08ca2f65e76d2efc38f5e` | `a70b59fd98d6336f8537393e73d3c733157f8ad32acb725e4e70c807e01b3ef4` |

## 6. Xác minh

| ID | Lệnh / kiểm tra | Kết quả |
| --- | --- | --- |
| V-001 | `pytest` robot schema/MJCF/FK | pass, Shadow 24 state và không mimic |
| V-002 | Shadow 24-target dynamic request | fail-closed `underactuated_targets`, mapped `24→20` |
| V-003 | Generator với ba release profiles | từ chối Shadow trước khi tạo output |
| V-004 | `scripts/check_phase2.py` | PASS; Shadow 24 joints, 0 mimic |
| V-005 | `scripts/check_phase3.py` | correctness gates 1–4 PASS; dừng đúng ở legacy manifest v1 trước train |

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

Không gỡ `release_blocked` bằng cách khôi phục mimic hoặc bỏ bốn J1. Cần module
underactuated command/control biểu diễn full 24-joint state, lập transmission
matrix 20-control, kiểm controllable subspace và có known-positive Shadow
lift+disturbance fixture. Sau đó phải tạo revision mới, chạy controlled ablation
và regenerate dataset từ clean commit.

Kết quả ablation và quyết định recipe trong `REV-20260823-002` không còn hiệu
lực. Claim Phase 3 complete trong session cũ vẫn bị thu hồi; P3 tiếp tục
`pending`, P4 tiếp tục bị chặn.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-23 Asia/Bangkok.
- Người kiểm tra: Independent Review required.
- Kết luận: implementation correctness đã ghi nhận; revision giữ `in_review` và
  blocker Shadow chưa đóng.
