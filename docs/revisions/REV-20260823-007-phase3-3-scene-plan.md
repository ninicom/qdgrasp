---
document_id: REV-20260823-007
document_type: revision_record
revision_schema: 2
title: Mở Phase 3.3 cho scene grasp synthesis và scene dataset
status: in_review
date: 2026-08-23
record_id: REV-20260823-007
session_id: SESSION-20260822-022
created_at: 2026-08-23T14:32:28+07:00
author: codex-primary-agent
revises:
  - artifact: docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md
    revision: new
  - artifact: docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md
    revision: f880b5f40b72b742ff9fb7f17371ab4efe8b10071fdd4cfa3e9361f4b7a00954
  - artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 939ca2b19db6be41d0ae227934cd7209e1dd69293acc8b4d4ab504d976e0b974
  - artifact: docs/revisions/REV-20260823-006-ablation-study-v2.md
    revision: 2f7493907695ba8367427ddc252b3ccc9a0299a539bcd6999d61d5b25df909d5
reason: "P3 hiện chỉ có single-target rollout và chưa có scene schema/import/build/generation contract; đồng thời release generator còn fabricated positive nên không thể mở rộng an toàn sang clutter scene."
necessity: N1
impact: "P3.3 trở thành kế hoạch active cho external scene adapters, native scene builders, whole-scene validation và QDGrasp-Scene-Tiny; P3.1 release gate được mở lại để loại fabricated labels."
---

# REV-20260823-007 — Phase 3.3 scene grasp/data plan

## 1. Liên kết truy vết

- Plan mới: `docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md`.
- Plan data correctness: `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md`.
- Underactuated parity đã hoàn tất: `REV-20260823-005`.
- Roadmap tổng: `docs/roadmap/PROJECT_PHASES.md`.
- Baseline implementation: commit `984df26` trên `feature/phase3-data-layer`.

## 2. Lý do chỉnh sửa

Pipeline hiện có thể compile một hand, một target object và floor để chạy
MuJoCo, nhưng chưa biểu diễn scene nhiều object, support geometry, camera,
visibility, imported source frames hoặc non-target disturbance. Vì vậy nó chưa
thể dùng GraspNet-1Billion/DexGraspNet 2.0/GraspClutter6D như scene source và
chưa thể tạo một scene dataset mới có provenance.

Audit đồng thời phát hiện `scripts/generate_dgn_open_tiny.py` thay một số outcome
bằng `DynamicValidation(passed=True)` thủ công. Cách này vi phạm điều kiện mọi
positive phải đến từ measured squeeze/lift/perturbation. P3.1-12 được mở lại và
P3.3 đặt genuine-positive check làm entry gate.

## 3. Mức độ cần thiết

- Mức `N1`: nếu không tách P3.3, scene import/build/camera/collision/dynamic
  semantics sẽ bị nhét vào object-level generator mà không có contract/gate.
- Fabricated labels là blocker release độc lập; không được hợp thức hóa bằng
  việc scene fixture hoặc unit test pass.

## 4. Phạm vi và tác động

- Tạo plan P3.3 gồm canonical scene schema, adapter allowlist, bốn scene builder
  strategies, scene-aware grasp pipeline, rendering, dataset schema và gate.
- Pin ba adapter v1: GraspNet-1Billion, DexGraspNet 2.0 và GraspClutter6D.
- Pin artifact `QDGrasp-Scene-Tiny` với 12 native scenes và ba clutter tiers.
- Nâng roadmap tổng để P3 có cả object-level và scene-level artifact.
- Mở lại P3.1-12; P3.1-13/14 tiếp tục blocked cho tới khi bỏ fabricated positive.
- Không sửa runtime code hoặc tải external dataset trong revision tài liệu này.
- Chuẩn hóa `REV-20260823-006` theo revision schema và invalidated quyết định
  chọn recipe vì cả ba recipe đều có 0% IK pass.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| ID | Artifact | Thay đổi |
| --- | --- | --- |
| CH-001 | `docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md` | Tạo ROADMAP-P3.3-001 active với literature cutoff 2026-08-23, contracts, 16 work packages, test matrix và resource gates |
| CH-002 | `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md` | Nâng 1.5.0 → 1.6.0; ghi fabricated-positive defect và mở lại P3.1-12 |
| CH-003 | `docs/roadmap/PROJECT_PHASES.md` | Nâng 1.7.0 → 1.8.0; bổ sung P3.3 và `QDGrasp-Scene-Tiny` vào scope/gate P3 |
| CH-004 | `docs/revisions/REV-20260823-006-ablation-study-v2.md` | Sửa status/H2 schema và invalidated kết luận chọn recipe không có positive evidence |

### SHA-256 artifact trước/sau

| Artifact | SHA-256 trước | SHA-256 sau |
| --- | --- | --- |
| `docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md` | new | `81667834044a02610daffa0fec9b549ab784730be56afa0e782111e81715ac6a` |
| `docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md` | `f880b5f40b72b742ff9fb7f17371ab4efe8b10071fdd4cfa3e9361f4b7a00954` | `712b8e7eeb461459fc4564be7dd6a35b3c553075d070f87441074701f9844298` |
| `docs/roadmap/PROJECT_PHASES.md` | `939ca2b19db6be41d0ae227934cd7209e1dd69293acc8b4d4ab504d976e0b974` | `7afda53338309051b98c0b0121f475d2877e97e50846084a9158b7c680356eec` |
| `docs/revisions/REV-20260823-006-ablation-study-v2.md` | `2f7493907695ba8367427ddc252b3ccc9a0299a539bcd6999d61d5b25df909d5` | `a5ba55197382b70b46516d0286b2cf38cee40435ea34f790918d65368b5c89e3` |

## 6. Xác minh

| ID | Lệnh / kiểm tra | Kết quả |
| --- | --- | --- |
| V-001 | `scripts/check_docs.py --root .` | thực hiện sau khi tạo record |
| V-002 | `git diff --check` | pass trước record; chạy lại ở gate cuối |
| V-003 | Manual dependency audit | P3.3 reuse P3.1/P3.2 contracts; không thay model P4 hoặc tải raw dataset |
| V-004 | Literature/source audit | official paper/API/repository links được pin; two-finger labels không được gọi là dexterous ground truth |

Revision chỉ thay đổi tài liệu nên không chạy MuJoCo, external dataset scan hay
full pytest. Đây cũng là lựa chọn an toàn vì workstation từng hard-freeze do I/O.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `REV-20260823-005` vẫn là record hoàn tất transmission/dynamic parity P3.2.
- Completion P3.2 không chứng minh generator phát hành không có fabricated
  labels; claim đó bị tách và chặn tại P3.1-12.
- Ablation report hiện tại không được dùng để bỏ qua entry gate.
- P3 và P4 tiếp tục `pending`; việc tạo plan không đồng nghĩa scene dataset đã
  tồn tại hoặc external license đã được phê duyệt.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-23 Asia/Bangkok.
- Người kiểm tra: Independent Review required.
- Kết luận: plan và dependency đã được ghi; revision giữ `in_review` cho tới khi
  docs gate pass và maintainer duyệt việc mở rộng P3 sang scene-level artifact.
