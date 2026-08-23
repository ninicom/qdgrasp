---
document_id: REV-20260823-009
document_type: revision_record
revision_schema: 2
title: Mở Phase 3.2.1 để sửa full-pipeline correctness và thu hồi false-positive evidence
status: in_review
date: 2026-08-23
record_id: REV-20260823-009
session_id: SESSION-20260822-022
created_at: 2026-08-23T22:30:00+07:00
author: codex-primary-agent
revises:
  - artifact: docs/roadmap/PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md
    revision: new
  - artifact: docs/roadmap/PHASE3_2_UNDERACTUATED_CONTROL_PLAN.md
    revision: ceafd0b4d89b21e245b203d1dd2bb181c076f4e4f66d1b288d19ca882e67c97c
  - artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 6d16117264ad7c7e747b07c3485467c34a504ab49cfe7bde34f6c0cc4c71098e
reason: "Gate P3.2 dùng joint states dựng thủ công và generator thay outcome thật bằng positive fixture; audit solver còn phát hiện normal residual/Jacobian không cùng định nghĩa và active mask không áp vào Hessian, nên claim full-pipeline completion không được chứng minh."
necessity: N2
impact: "P3.2 component evidence được giữ, nhưng full-pipeline completion và Shadow release claim bị thu hồi; P3.2.1 trở thành corrective gate bắt buộc trước ablation, regeneration và P4."
---

# REV-20260823-009 — Phase 3.2.1 full-pipeline correctness

## 1. Liên kết truy vết

- Plan mới: `docs/roadmap/PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md`.
- Plan cha: `docs/roadmap/PHASE3_2_UNDERACTUATED_CONTROL_PLAN.md`.
- P3.1 parent gate: `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md`.
- Finding source: `docs/reports/BUG-REPORT-PIPELINE-2026-08-23.md` và audit
  trực tiếp `run_pipeline_chunk`.
- Claim bị sửa: `REV-20260823-005` và phần P3 trong `PROJECT_PHASES.md`.

## 2. Lý do chỉnh sửa

Gate `scripts/check_phase3_2.py` pass transmission ranks và các rollout dùng
`q_contact/q_open/q_squeeze` dựng thủ công. Nó không gọi proposal, palm fitting,
IK, collision, static certificate và dynamic rollout trong cùng một provenance
chain. Diagnostic box 5 cm trên ba robot x ba recipe x hai candidate cho kết quả
18/18 dừng tại `IK: max_iter`.

Audit code còn xác nhận:

- normal residual dùng configured fingertip axis nhưng Jacobian dùng parent-to-tip;
- active mask không mask Jacobian/Hessian;
- actuator saturation và palm support chưa fail dynamic predicate;
- generator thay `outcomes[0]` bằng positive fixture với kinematics/static
  certificate dựng thủ công.

Do đó completion claim phải được thu hẹp về component evidence và cần một
corrective subphase có causal tests, mutation gates và generated full-flow proof.

## 3. Mức độ cần thiết

- Mức `N2`: sửa sai protocol/evidence và kết luận kỹ thuật, không phải biên tập.
- Nếu không sửa, fabricated positives có thể đi vào dataset/P4 và Shadow có thể
  được release trên gate không kiểm full path.
- P3.1 ablation/regeneration và P4 tiếp tục bị chặn tới closing revision P3.2.1.

## 4. Phạm vi và tác động

- Tạo plan P3.2.1 gồm root-cause matrix, corrected contracts, 15 work packages,
  derivative/collision/transmission/dynamic/full-flow tests và release gate.
- Nâng P3.2 plan 1.0.0 -> 1.1.0 để liên kết corrective subphase.
- Nâng roadmap tổng 1.9.0 -> 1.10.0 và thu hồi claim dynamic parity/full release.
- Revision này chỉ sửa tài liệu; runtime/profile/generator/dataset sẽ được sửa
  trong work package P3.2.1-00 trở đi.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| ID | Artifact | Thay đổi |
| --- | --- | --- |
| CH-001 | `PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md` | Tạo corrective plan active với causal proof, contracts, tests và gate |
| CH-002 | `PHASE3_2_UNDERACTUATED_CONTROL_PLAN.md` | Liên kết P3.2.1 và thu hẹp component/full-flow claim |
| CH-003 | `PROJECT_PHASES.md` | Giữ P3 pending, yêu cầu re-block Shadow và P3.2.1 closure |

### SHA-256 artifact trước/sau

| Artifact | SHA-256 trước | SHA-256 sau |
| --- | --- | --- |
| `PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md` | new | `34c41ac5e795e4153f1073852a2d232acf2104ab245a56189e49e3c5d7c717c4` |
| `PHASE3_2_UNDERACTUATED_CONTROL_PLAN.md` | `ceafd0b4d89b21e245b203d1dd2bb181c076f4e4f66d1b288d19ca882e67c97c` | `36802c835edb435315112a5f5a526251d673a4657727ebbf0f8751fd47da1a41` |
| `PROJECT_PHASES.md` | `6d16117264ad7c7e747b07c3485467c34a504ab49cfe7bde34f6c0cc4c71098e` | `4669880ea7144d1dac52e0aee6f80e6b2cb3610d97d6d1653a854ba43c4f99e6` |

## 6. Xác minh

| ID | Lệnh / kiểm tra | Kết quả |
| --- | --- | --- |
| V-001 | `.venv/bin/python scripts/check_docs.py --root .` | PASS, 91 files checked |
| V-002 | `git diff --check` | PASS |
| V-003 | Manual trace audit | PASS, plan cha, corrective plan, roadmap và revision liên kết hai chiều |

Không chạy runtime/MuJoCo/full pytest trong revision docs-only này. Các finding
runtime đã được tái lập trước khi lập plan; remediation được gate riêng trong
P3.2.1.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `REV-20260823-005` còn hiệu lực cho transmission dimensions/rank và component
  fixtures; claim hoàn tất toàn diện/unblock release không còn đủ.
- `BUG-REPORT-PIPELINE-2026-08-23` vẫn là finding source, nhưng root causes chỉ
  được xác nhận theo matrix RC/H của P3.2.1.
- `DGN-Open-Tiny` hiện tại không được dùng làm positive evidence cho P4.
- Independent full review bắt buộc trước closing revision.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-23 Asia/Bangkok.
- Người kiểm tra: Independent Review required.
- Kết luận: corrective plan đã được lập và liên kết; record giữ `in_review` cho
  tới docs gate và maintainer chấp thuận scope/claim correction.
