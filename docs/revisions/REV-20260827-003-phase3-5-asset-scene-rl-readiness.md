---
document_id: REV-20260827-003
document_type: revision_record
revision_schema: 2
title: Mở Phase 3.5 cho asset/scene ingestion và RL simulation readiness
status: in_review
date: 2026-08-27
record_id: REV-20260827-003
session_id: SESSION-20260827-003
created_at: 2026-08-27T11:08:03+07:00
author: codex-primary-agent
revises:
  - artifact: docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md
    revision: new
  - artifact: docs/roadmap/PROJECT_PHASES.md
    revision: aa485210a7d11946682f739fea0bd4db2d99d56a85897576030340e86b137400
reason: "P3.3 có canonical scene và physics settle nhưng chưa có contract ingest raw mesh/CoACD, scene-load-or-generate, vectorized Gym environment hoặc cloud-notebook lifecycle để bàn giao môi trường sẵn sàng cho reinforcement learning."
necessity: N2
impact: "Thêm P3.5 docs-only với raw-mesh/public Python CoACD API viết mới theo full upstream parameters và legacy Stage 0 regression profiles, virtual drop/settle scene, RL environment contract và simulator decision gate; P3 artifact/gate mở rộng nhưng P4 static/offline không bị P3.5 chặn."
---

# REV-20260827-003 — Phase 3.5 asset/scene/RL readiness

## 1. Liên kết truy vết

- Plan mới: `docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md`.
- Scene baseline: `docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md`.
- Dynamic sibling: `docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md`.
- Robot/simulator baseline: `docs/roadmap/PHASE2_EXECUTION_PLAN.md`.
- Roadmap tổng: `docs/roadmap/PROJECT_PHASES.md`.
- Prior source: `ninicom/kaggle_pipeline_modular@d929c175c1bde68f5acd059fbc6f93bec0c03328`,
  thư mục `stage0_coacd` do maintainer chỉ định.
- Yêu cầu nguồn: ingest object/scene, virtual drop region, library CoACD API, RL-ready
  simulator và khả năng chạy tạm trên Kaggle/Colab Pro.

## 2. Lý do chỉnh sửa

P3.3 đã có `SceneSpec`, native scene builder và MuJoCo settle cho dataset scene,
nhưng chưa định nghĩa đường vào khi người dùng chỉ có raw mesh, chưa có public
full-parameter CoACD library function, và chưa có Gymnasium-compatible reset/step
boundary. Prior Stage 0 chỉ là reference/regression fixture, không khóa việc
viết implementation mới. Vì
vậy “scene dataset complete” chưa đồng nghĩa “môi trường sẵn sàng cho RL”.

Việc chọn simulator cũng cần decision gate thay vì pin theo thương hiệu. Stack
hiện tại đã dùng MuJoCo/MJCF cho LEAP, Allegro và Shadow, trong khi môi trường
cloud notebook là ephemeral và loại GPU không được đảm bảo. P3.5 vì thế giữ
MuJoCo CPU làm oracle, thử MJX-Warp trước, chỉ dùng ManiSkill 3 làm fallback sau
import/contact parity, và defer Isaac Lab khỏi critical path v1.

## 3. Mức độ cần thiết

- Mức `N2`: thay đổi scope, artifact và gate của P3; thêm library API/provenance/backend
  decision contract và xác định dependency với P4/P5.
- Nếu không có plan riêng, raw mesh có thể đi thẳng vào dynamic collision không
  kiểm unit/inertia, scene lỗi có thể silently biến thành scene ảo, hoặc notebook
  có thể báo GPU dù simulator đã fallback CPU.
- Không có runtime behavior thay đổi trong record này; implementation chỉ được
  mở sau maintainer/delta review.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | P3.3 scene dataset, chưa có RL env | Plan asset→scene→settle→Gym env | cao nhưng docs-only | thực hiện P3.5 work breakdown |
| API/schema/config | `ObjectAssetManifest`/`SceneSpec` hiện có | dự kiến `AssetIngestRequest`, full-parameter public Python CoACD API, `ObjectAssetManifestV2`, `SceneSnapshot` và RL contracts | cao | viết API mới; giữ Stage 0 legacy profiles/fixtures |
| Dữ liệu/checkpoint | DGN/Scene/ContactRich artifacts | thêm `QDGrasp-RL-Env-Tiny` và notebook checkpoint/evidence | vừa | tạo immutable tiny bundle |
| CPU/GPU | MuJoCo CPU baseline; P3.4 MJX-Warp candidate | CPU oracle + fail-closed 3-hand GPU decision gate | cao | chạy compatibility spike |
| Roadmap | P3.4 là blocker còn lại | P3.4 và P3.5 là sibling pending; P3.5 không chặn P4 static | vừa | maintainer duyệt dependency |

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| ID | Artifact | Thay đổi | SHA-256 sau |
| --- | --- | --- | --- |
| CH-001 | `docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md` | Tạo `ROADMAP-P3.5-001` với research, contracts, backend matrix, work breakdown và gates | `6b86333c9ba3956e55a22b8a7cea657d65c391a2442a747fd7a9ef1f18f0a50e` |
| CH-002 | `docs/roadmap/PROJECT_PHASES.md` | Nâng 1.20.0 → 1.21.0; thêm P3.5 vào P3 scope/artifact/gate và dependency với P4/P5 | `7a7612e432b461682b3547a732a551f9eaf33dff4edab3eaafbaa328cdd86135` |

Research sources trong plan là nguồn chính thức của CoACD, MuJoCo/MJX,
MuJoCo Menagerie/Playground, ManiSkill, NVIDIA Isaac Lab/Gym và Google Colab.

## 6. Xác minh

| ID | Phương pháp | Kết quả | Trạng thái |
| --- | --- | --- | --- |
| V-001 | So khớp scope với P2, P3.3 và P3.4 | P3.5 tái sử dụng `SceneSpec`/MuJoCo oracle, không viết lại settle/data pipeline | pass |
| V-002 | Audit simulator decision | Không pin Isaac theo mặc định; có 3-hand compatibility/fallback gate | pass |
| V-003 | Audit raw-mesh/CoACD | Đã đọc prior Stage 0 và official `run_coacd`; public local function có full typed parameters, legacy profiles và output validation | pass |
| V-004 | Audit cloud lifecycle | Hardware preflight, no CPU fallback, cache/checkpoint/resume và exact hashes có gate | pass |
| V-005 | `git diff --check` | không có whitespace error | pass |
| V-006 | Runtime/tests/GPU/Kaggle/Colab | không chạy trong revision lập kế hoạch docs-only | not_run |

Không có claim rằng MJX-Warp, ManiSkill hoặc Isaac Lab hiện đã chạy được cả ba
hand. Không có claim policy learning hoặc GPU throughput từ revision này.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- P0–P2, P3.1, P3.2/P3.2.1 và P3.3 completion evidence không bị sửa.
- P3.4 vẫn là contact-rich data-generation sibling; P3.5 không thay trajectory
  search và có thể tái sử dụng backend evidence của P3.4 nếu exact profile khớp.
- P4 static/offline có thể bắt đầu sau khi dataset interface khóa; P3.5 là
  prerequisite cho simulation-in-the-loop/RL work ở P5, không phải model MVP.
- P3 tổng vẫn `pending`; plan mới không tự đóng bất kỳ implementation gate nào.
- Cần `delta_review` cho scope/dependency/backend protocol, rồi full independent
  review tại P3.5-18 trước closure.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-27 Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer.
- Kết luận: plan P3.5 và roadmap delta đã được lập; record giữ `in_review` cho
  tới khi maintainer duyệt scope, P4/P5 dependency và simulator decision gate.
