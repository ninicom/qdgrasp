---
document_id: REV-20260823-008
document_type: revision_record
revision_schema: 2
title: Mở Phase 3.4 cho contact-rich dynamic grasp generation trên CPU/GPU
status: in_review
date: 2026-08-23
record_id: REV-20260823-008
session_id: SESSION-20260822-022
created_at: 2026-08-23T14:46:11+07:00
author: codex-primary-agent
revises:
  - artifact: docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md
    revision: new
  - artifact: docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md
    revision: 81667834044a02610daffa0fec9b549ab784730be56afa0e782111e81715ac6a
  - artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 7afda53338309051b98c0b0121f475d2877e97e50846084a9158b7c680356eec
reason: "Static scene grasp loại mọi environment contact và đóng băng target trong acquisition, bỏ sót extrinsic-dexterity trajectories có thể reposition object an toàn trước khi terminal grasp hình thành."
necessity: N1
impact: "P3.4 trở thành sibling branch của P3.3, thêm contact-rich trajectory dataset, MuJoCo CPU oracle, MJX-Warp/MuJoCo Warp CUDA search và Kaggle hardware evidence gate."
---

# REV-20260823-008 — Phase 3.4 contact-rich dynamic grasp plan

## 1. Liên kết truy vết

- Plan mới: `docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md`.
- Scene plan cha: `docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md`.
- Transmission/control dependency: `ROADMAP-P3.2-001`, `REV-20260823-005`.
- CUDA policy: `docs/decisions/0006-cuda-hardware-required.md`.
- Kaggle evidence repository: `ninicom/qdgrasp-cuda-kaggle`.
- Roadmap tổng: `docs/roadmap/PROJECT_PHASES.md`.

## 2. Lý do chỉnh sửa

P3.3 sinh grasp trong clutter scene nhưng vẫn dùng direct/static candidate làm
điểm xuất phát: scene được xem gần như cố định trong acquisition và collision
với table/non-target thường là rejection. Cách này không biểu diễn được grasp
cần slide, roll, pivot, push hoặc dùng support contact nhẹ để đưa object vào vị
trí kẹp ổn định.

P3.4 tách một nhánh compute-intensive, trong đó control trajectory được rollout
trực tiếp trong physics. Contact với environment được phân loại và kiểm bằng
force/impulse/work/load/disturbance budget, thay vì binary collision flag. Target
chỉ được di chuyển qua dynamics, không runtime qpos write.

## 3. Mức độ cần thiết

- Mức `N1`: đây là capability mới có schema, search strategy, physics semantics,
  GPU backend và dataset riêng; không phù hợp nhét vào một P3.3 task nhỏ.
- Nếu không tách, static và dynamic labels có thể bị trộn, hoặc safety threshold
  bị hạ để tăng yield mà không còn provenance.

## 4. Phạm vi và tác động

- Tạo plan P3.4 với 18 work packages từ trajectory contracts tới release.
- Pin CPU MuJoCo làm oracle và MJX-Warp/MuJoCo Warp làm CUDA candidate-search
  backend sau compatibility spike.
- Pin CEM là batched search bắt buộc; MPPI là module mở rộng có thể defer rõ.
- Định nghĩa multidimensional contact safety budget và CPU replay policy.
- Định nghĩa `QDGrasp-ContactRich-Tiny` gồm paired static-fail/dynamic-pass cases.
- Mở rộng Kaggle harness Phase 1 để chạy P3.4 trên NVIDIA T4/cu128, không CPU
  fallback và có performance/VRAM/backend parity evidence.
- Không sửa runtime, dependency lock, Kaggle notebook hoặc dataset trong revision
  lập kế hoạch này.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| ID | Artifact | Thay đổi |
| --- | --- | --- |
| CH-001 | `docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md` | Tạo ROADMAP-P3.4-001 active, literature cutoff 2026-08-23, physics/safety contracts, CPU/GPU backend, Kaggle gate và release dataset |
| CH-002 | `docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md` | Nâng 1.0.0 → 1.1.0 và liên kết sibling branch P3.4 |
| CH-003 | `docs/roadmap/PROJECT_PHASES.md` | Nâng 1.8.0 → 1.9.0; thêm contact-rich trajectory scope/artifact/gate vào P3 |

### SHA-256 artifact trước/sau

| Artifact | SHA-256 trước | SHA-256 sau |
| --- | --- | --- |
| `docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md` | new | `4a6132a5be065cdbe25b9ea7c86896558068ec80fb7f80f1e5297a415e7152b7` |
| `docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md` | `81667834044a02610daffa0fec9b549ab784730be56afa0e782111e81715ac6a` | `1e80f955e0dfcbdc5a69b0e3ec3fee27eec9b14c25d6f00471e2fc7140f0c2a9` |
| `docs/roadmap/PROJECT_PHASES.md` | `7afda53338309051b98c0b0121f475d2877e97e50846084a9158b7c680356eec` | `6d16117264ad7c7e747b07c3485467c34a504ab49cfe7bde34f6c0cc4c71098e` |

## 6. Xác minh

| ID | Lệnh / kiểm tra | Kết quả |
| --- | --- | --- |
| V-001 | `scripts/check_docs.py --root .` | thực hiện sau khi tạo record |
| V-002 | `git diff --check` | pass trước record; chạy lại ở gate cuối |
| V-003 | Architecture audit | CPU/GPU, static/dynamic và search/certification responsibilities tách rõ |
| V-004 | Safety audit | allowed collision vẫn chịu hard force/impulse/load/disturbance barriers |
| V-005 | CUDA policy audit | CUDA gate yêu cầu NVIDIA thật, fail-closed và exact external evidence |

Không chạy MuJoCo, GPU, Kaggle hoặc full pytest trong revision docs-only này.
CUDA capability chỉ là kế hoạch cho tới khi external hardware evidence tồn tại.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- P3.3 vẫn là static/direct scene baseline và không bị supersede.
- P3.4 không làm fabricated positive của P3.1 hợp lệ; genuine-label entry gate
  vẫn bắt buộc trước mọi release.
- `REV-20260823-005` tiếp tục là evidence P3.2 transmission parity, chưa phải
  MJX-Warp/MuJoCo Warp parity.
- P3/P4 vẫn `pending`; plan không tạo claim GPU acceleration hoặc yield gain.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-23 Asia/Bangkok.
- Người kiểm tra: Independent Review required.
- Kết luận: P3.4 được lập kế hoạch và liên kết; record giữ `in_review` cho tới
  docs gate pass và maintainer duyệt scope/safety/Kaggle execution contract.
