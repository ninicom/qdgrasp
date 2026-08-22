---
document_id: ADR-0003
document_type: decision
title: Kiến trúc graph-flow cross-embodiment và biên tham khảo paper
status: accepted
date: 2026-08-22
revises: none
supersedes: none
---

# ADR-0003 — Graph-flow trực tiếp với FK/keypoint/contact phụ

## Bối cảnh

DexGrasp cần hỗ trợ nhiều hand khác topology/DoF, chạy trực tiếp ra state thực
thi và có đường nghiên cứu đủ rõ để viết paper. Thiết kế cũng phải clean-room:
paper được dùng như nguồn học thuật cho ý tưởng, công thức và benchmark có trích
dẫn; code, config, data, checkpoint và robot asset chỉ được nhập khi exact
version có license/provenance riêng.

Mười paper do maintainer đề xuất đều tồn tại. Hai correction phải dùng trong mọi
tài liệu/citation sau này:

- Paper tháng 10/2025 hiện có tên **MachaGrasp: Morphology-Aware
  Cross-Embodiment Dexterous Hand Articulation Generation for Grasping**; title
  maintainer đưa là title v1 cũ.
- **DexGraspNet 3.0** là dataset 170M grasp/174K object được giới thiệu trong
  DexVLG, không phải một paper riêng và chưa phải corpus permissive đã duyệt.

## Nguồn nghiên cứu đã xác minh

| Nguồn primary | Ngày/venue đúng | Ý tưởng được phép đưa vào đặc tả | Trạng thái implementation/data |
| --- | --- | --- | --- |
| [GraspGraphNet](https://arxiv.org/abs/2607.11031) | arXiv, 13/07/2026 | URDF-derived variable graph, differentiable FK, world edges, direct flow palm+joint | Chưa có implementation/data có LICENSE; paper-only |
| [KPGrasp](https://arxiv.org/abs/2606.09314) | arXiv, 08/06/2026 | All-Euclidean keypoint flow và scaling study | 21 keypoint cố định cho Shadow; chưa có code/data license; paper-only |
| [EquiDexFlow](https://arxiv.org/abs/2606.12728) | arXiv v1 10/06/2026, v2 12/06 | SE(3) equivariance; joint contact/normal/force; surface/friction projection | Code repo ghi MIT nhưng exact pin/data phải audit; EGAD track có NC, không nhập corpus |
| [EFF-Grasp](https://arxiv.org/abs/2603.16151) | arXiv, 17/03/2026 | Deterministic flow ODE và physics-energy guidance ở inference | Không có implementation/data license được xác minh; paper-only |
| [UniMorphGrasp](https://arxiv.org/abs/2602.00915) | arXiv, 31/01/2026 | Morphology graph và zero-shot protocol | Code “coming soon”; canonical Shadow mapping không được chọn làm core |
| [CADGrasp](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aff7a9ba2c654ad96e24f994c3f11bc-Abstract-Conference.html) | NeurIPS 2025; arXiv đăng 21/01/2026 | Sparse IBS/contact-collision representation cho clutter | Repo không có LICENSE file và phụ thuộc DGN2; paper-only |
| [DextER](https://arxiv.org/abs/2601.16046) | CVPR 2026; arXiv v1 22/01/2026 | Contact token làm bước reasoning trước grasp token | Code/data không có license tường minh; semantic extension sau v1 |
| [MachaGrasp](https://arxiv.org/abs/2510.06068) | v1 07/10/2025; ICRA 2026 accepted | Eigengrasp/morphology embedding và few-shot hand mới | Code repo ghi MIT; weights/assets/dependencies audit riêng |
| [DexVLG](https://arxiv.org/abs/2507.02747) | ICCV 2025 Spotlight; 03/07/2025 | RGB-D/language conditioning và flow pose head | Repo/data/weights chưa có permissive release được xác minh |
| [DexGrasp Anything](https://arxiv.org/abs/2503.08257) | CVPR 2025 Highlight; 11/03/2025 | Physics-aware diffusion, loss/sampling baseline | Code repo ghi MIT; dataset tổng hợp nhiều nguồn nên license tag không đủ cho corpus sạch |

Không copy biểu đồ, prose, pseudocode, config hoặc code từ paper/project chỉ vì
paper có Creative Commons. Quyền paper không tự động cấp quyền cho repository,
dataset, checkpoint, third-party baseline hoặc mesh.

## Quyết định kiến trúc

### 1. Canonical hand representation

`RobotSpec` nhập URDF hoặc MJCF và tạo `HandGraph` biến độ dài:

- node link: parent-frame transform, inertial/geometry summary, surface anchors,
  optional permissive mesh embedding và semantic tag đã xác nhận;
- edge joint: type, axis, origin, limits, mimic/coupling và actuator mapping;
- một palm token và một scalar token cho mỗi actuated named joint;
- graph/profile hash được tính từ canonical fields, source tree và transforms.

Không pad về canonical Shadow 24-DoF như semantic truth. Padding/mask chỉ là chi
tiết batching; joint name + graph node mới là identity thực.

### 2. Core generator

Object point tokens và HandGraph đi qua graph transformer có world edges động.
Conditional flow matching sinh trực tiếp:

```text
palm translation [3] + palm rotation R9D [9]
+ one normalized q-value per actuated named joint
```

Rotation được project về SO(3); q-value được map vào finite joint limits. Đây là
đường inference mặc định vì output thực thi ngay, không phụ thuộc IK/retarget.
Flow và solver được viết mới từ toán học, không sao chép implementation của
GraspGraphNet/DGN2.

### 3. Dual-space consistency

Differentiable FK biến palm/qpos thành tập anchor 3D theo link/topology. Một
auxiliary head dự đoán keypoint/contact tương ứng trong object frame và bị ràng
buộc consistency với FK. Cách này lấy lợi ích spatial reasoning của KPGrasp mà
không khóa vào 21 điểm Shadow và không đưa IK vào critical path.

Ba baseline bắt buộc để paper không nhập nhằng đóng góp:

1. direct executable flow, không auxiliary keypoint;
2. keypoint flow + IK;
3. direct flow + topology-keypoint/FK consistency.

### 4. Physics grounding

Auxiliary state gồm contact point, surface normal, contact force và confidence.
Contact/force projection áp surface và Coulomb-cone constraints bằng phép toán
viết mới/test được. Quality head học từ MuJoCo/MJX labels. Energy guidance cho
penetration, surface distance và self-collision là plugin inference có budget,
được ablate riêng; không che giấu grasp kém bằng optimizer không giới hạn.

### 5. Phạm vi theo giai đoạn

- P0: HandGraph, object encoder, direct executable flow, FK và single/multi-hand
  checkpoints cho LEAP/Allegro/Shadow.
- P1: topology keypoints, contact/normal/force, SE(3) tests và energy guidance.
- P2: CADGrasp-style clutter refiner và MachaGrasp-style few-shot articulation.
- Sau v1: DextER/DexVLG-style language/semantic condition. VLM không nằm trong
  core install hoặc điều kiện release đầu tiên.

## Asset và dữ liệu nghiên cứu

Corpus compatibility cục bộ được khóa ở `robot_assets.lock.yaml`. Bản đầu dùng:

- LEAP official URDF/STL (MIT) + Menagerie MJCF (MIT);
- Wonik official Allegro URDF/STL/ROS metadata (BSD-2-Clause) + Menagerie MJCF
  (BSD-2-Clause);
- Shadow E3M5 Menagerie MJCF/meshes (Apache-2.0); URDF ngoài distribution chỉ là
  parser fixture cho tới khi independent provenance review kết thúc.

Barrett không nằm trong core/default release: source công khai tìm được không có
full rightsholder license rõ bao phủ URDF + mesh. Kết quả tương thích cục bộ
không biến asset đó thành redistributable.

Train data/official weights chỉ dùng DGN-Open tự sinh và asset manifest đã duyệt.
Không train, fine-tune, distill hoặc lấy labels từ DGN2, DexGrasp Anything corpus,
DexGraspNet 3.0 hay dataset có NC/unknown provenance.

## Protocol để có thể viết paper

Trước full training phải khóa protocol và hypotheses:

- H1: graph conditioning tăng success trên held-out morphology so với joint
  vector/padding không graph;
- H2: dual FK-keypoint consistency giảm penetration/collision mà không tăng
  latency như keypoint+IK;
- H3: surface/friction projection tăng force feasibility và giữ SE(3)
  equivariance;
- H4: energy guidance cải thiện physics success trong budget sampling định trước.

Split bắt buộc tách object family và có ít nhất một hand held-out hoặc structural
perturbation (finger removal/link-length/joint-limit variation). Báo cáo phải có
success, collision, penetration, force residual, diversity, latency, memory,
sample count, uncertainty, seeds, compute và exact code/data/model/robot hashes.
Mọi baseline được chạy lại cùng assets/protocol; không chép số từ paper vào bảng
so sánh như thể trực tiếp comparable.

## Hệ quả và giới hạn

- Direct graph flow là core rõ ràng hơn một pipeline canonical-Shadow hoặc IK
  bắt buộc, nhưng batching graph biến độ dài và cross-hand data generation khó
  hơn đáng kể.
- Dual-space/contact heads tạo một trục nghiên cứu có thể ablate, nhưng không tự
  động là novelty; cần literature/patent review trước khi viết claim.
- Model có thể public Apache-2.0 trong repository mới; từng dependency, asset,
  dataset và weight vẫn giữ license/attribution riêng.
- Quyết định này không cấp phép lại cây AGPL hiện tại và không phải tư vấn pháp lý.
