---
document_id: ROADMAP-P4-001
document_type: plan
title: Kế hoạch thi công Phase 4 — QDGrasp-Flow Model MVP
version: 1.1.0
status: active
date: 2026-08-31
revises: ROADMAP-P4-001@1.0.0
related_plan: PLAN-V2
latest_revision_record: docs/revisions/REV-20260831-003-phase4-architecture.md
revision_reason: Sửa cột output của §4 cho khớp code đã viết; ba head nằm chung một forward pass nên chung file.
necessity: N3
impact: Không đổi phạm vi, contract hay cổng; chỉ đổi tên file mà mỗi package giao.
depends_on:
  - ROADMAP-P2-001
  - ROADMAP-P3.1-001
active_hands:
  - leap_hand
  - wonik_allegro
paused_hands:
  - shadow_hand
release_class: not_a_release
---

# Kế hoạch Phase 4 — QDGrasp-Flow Model MVP

`PLAN.md` §M3 và `PROJECT_PHASES.md` mô tả P4 nhưng chưa có execution plan; mọi
phase trước đều có một. Tài liệu này là cái đó: khóa phạm vi, contract, work
breakdown, test matrix và điều kiện đóng trước khi viết model.

P4 xây **kiến trúc và chứng minh nó học được**, không phải một checkpoint tốt.
Training đầy đủ, multi-hand, held-out protocol và benchmark thuộc P5.

## 0. Trạng thái đầu vào

| Thứ P4 cần | Đã có ở đâu | Trạng thái |
| --- | --- | --- |
| `GraspBatch` | `qdgrasp/dataset/batch.py` | complete (P3.1) |
| `DGN-Open-Tiny` | `datasets/dgn-open-tiny/` | complete, 200 sample / 18 object |
| `HandGraph` biến độ dài | `qdgrasp/robot/graph.py` | complete (P2) |
| Differentiable FK | `RobotSpec.forward_kinematics` | complete (P2) |
| Config schema/registry | `qdgrasp/config/` | complete (P1) |

Hai active hand cho HandGraph: LEAP `18 node / 17 edge`, Allegro `22 node /
21 edge`, cùng `node_dim=17`, `edge_dim=9`, 16 actuated joint, 4 fingertip.
Model **không** được giả định số node, số edge hay số joint là hằng số.

## 1. Quyết định thiết kế

### 1.1 Vì sao rectified flow chứ không phải diffusion nhiều bước

`PLAN.md` khóa straight-path velocity matching với solver Euler 5 bước. Lý do
thực dụng: mỗi bước sampling là một forward pass, và P4 phải chạy được trong
notebook budget. Diffusion 50–1000 bước không đổi được chất lượng đủ để bù chi
phí đó ở quy mô tiny.

### 1.2 Vì sao sinh trực tiếp palm + named joint

Retarget hoặc IK trên đường inference mặc định làm model phụ thuộc một solver
ngoài, và solver đó là nơi lỗi trốn. `PLAN.md` yêu cầu flow sinh thẳng
`palm R9D + T + masked named joints`; IK chỉ là baseline/adapter.

### 1.3 Vì sao rotation là 9D chứ không phải quaternion

Quaternion có double cover và không liên tục như một hàm của SO(3); 6D/9D
continuous representation thì có. Model sinh 9D rồi chiếu về SO(3) bằng phép
toán Torch đã test, không dùng SVD của thư viện ngoài trên đường gradient nếu
nó không có backward ổn định.

### 1.4 Cấm tuyệt đối `N×N`

`PLAN.md` §6 yêu cầu profile chứng minh không tạo tensor `N×N`, và tăng token 2×
không làm memory tăng gần 4×. Điều này loại attention toàn cục trên raw point.
Encoder dùng serialization + local window; cross-attention giữa point token và
hand node là `[N_tok, L]` với `L ≤ 24`, không phải `N×N`.

## 2. Phạm vi

### 2.1 Thuộc P4

- Point tokenizer thuần Torch: packed integer key, kiểm overflow, sort/unique/
  scatter, không hash collision, không custom C++/CUDA.
- Serialized point encoder + neck giữ raw-point resolution qua `point_to_token`.
- HandGraph encoder biến độ dài từ `node_features`/`edge_index`/`edge_features`.
- World-edge cross-attention giữa point token và hand node.
- Rectified-flow head sinh `palm_translation [3] + palm_rot9d [9] + joints [J]`
  có mask theo `actuated_joint_names`.
- Differentiable FK nối output với fingertip keypoint; FK consistency loss.
- Quality head học từ `success`/`quality` của batch.
- Model scale `n` (bắt buộc), `s`/`m` chỉ là config, không bắt buộc train.
- Config schema `qdgrasp/model/v1` và registry builder.
- Tiny overfit trên CPU chứng minh kiến trúc học được.

### 2.2 Không thuộc P4

- Không train đủ lâu để công bố chất lượng; không model card, không benchmark.
- Không multi-hand generalization claim; không held-out embodiment.
- Không contact-rich input — nó vẫn `release_blocked` theo P3.4.3.
- Không energy guidance, CADGrasp refiner hay semantic layer.
- Không exporter mới ngoài những gì P1 đã có.

## 3. Contract

### 3.1 Đầu vào

```text
GraspBatch
  points              [B, N, 3]   object point cloud, mét, object frame
  palm_pos            [B, 3]      ground truth
  palm_rot            [B, 3, 3]   ground truth
  joint_angles        [B, J]      ground truth, thứ tự theo profile
  fingertip_positions [B, K, 3]   ground truth
  success             [B]
  quality             [B]
HandGraph (per robot profile, không đổi trong batch cùng robot)
```

### 3.2 Đầu ra

```text
GraspPrediction
  palm_translation [B, 3]
  palm_rotation    [B, 3, 3]   đã chiếu về SO(3)
  joint_angles     [B, J]      đã kẹp về named limits
  joint_mask       [J]         1 ở joint được actuate
  fingertip        [B, K, 3]   từ FK, không phải head riêng
  quality_logit    [B]
```

Mọi output phải hữu hạn. `joint_angles` nằm trong limit. `palm_rotation` phải
thỏa `RᵀR = I` và `det R = 1` trong dung sai `1e-4`.

### 3.3 Loss

```text
flow_velocity + palm_translation + palm_rotation + joint + fk_consistency + quality
```

Total bằng tổng term được log, như contract của P3.5 §6.4. Không term nào được
bù cho một output không hữu hạn: state không hữu hạn là lỗi, không phải phạt.

## 4. Work breakdown

| ID | Công việc | Output | Phụ thuộc |
| --- | --- | --- | --- |
| P4-00 | Khóa scope/contract | tài liệu này | — |
| P4-01 | Point tokenizer thuần Torch | `qdgrasp/models/tokenizer.py` | 00 |
| P4-02 | Serialized point encoder + neck | `qdgrasp/models/encoder.py` | 01 |
| P4-03 | HandGraph encoder biến độ dài | `qdgrasp/models/hand_graph.py` | 00 |
| P4-04 | World-edge cross-attention | `qdgrasp/models/flow.py` (`CrossAttentionBlock`) | 02/03 |
| P4-05 | Rectified-flow palm+joint head | `qdgrasp/models/flow.py` | 04 |
| P4-06 | FK consistency + keypoint | `qdgrasp/models/losses.py` + `RobotSpec.fingertip_positions` | 05/P2 |
| P4-07 | Quality head | `qdgrasp/models/flow.py` (`GraspFlowModel.quality`) | 04 |
| P4-08 | Config schema + registry `qdgrasp-flow-n` | `qdgrasp/models/config.py`, preset YAML | 01–07 |
| P4-09 | Loss assembly + gradient coverage | `qdgrasp/models/losses.py` | 08 |
| P4-10 | Tiny overfit trên CPU | `scripts/overfit_qdgrasp_flow.py` + evidence | 09 |
| P4-11 | CUDA gate harness (`a`) và evidence đo được (`b`) | `scripts/phase4_cuda_gate.py`, `notebooks/phase4_cuda_gate.ipynb`, `evidence/phase4/cuda-*.json` | 10 |
| P4-12 | Independent review | `scripts/phase4_review_packet.py`, `docs/roadmap/PHASE4_REVIEWER_GUIDE.md`, verdict | 11 |

Critical path `00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12`.

`P4-04`, `P4-06` và `P4-07` chia chung một forward pass và một khối
conditioning, nên chúng nằm chung file với `P4-05` thay vì mỗi thứ một module:
tách ra thì mỗi bên phải truyền qua lại conditioning đã tính rồi. Cột output ở
trên ghi vị trí thật; phiên bản `1.0.0` của tài liệu này đoán trước ba tên file
chưa tồn tại.

## 5. Test matrix bắt buộc

### Tokenizer
- Packed key không overflow ở grid size lớn nhất được phép; vượt thì fail.
- Hai point khác cell không bao giờ cùng key; cùng cell thì luôn cùng key.
- Sort/unique/scatter deterministic với cùng input và cùng device.
- Token count tăng đơn điệu theo resolution.

### Encoder/neck
- `point_to_token` rồi `token_to_point` giữ đúng số point.
- Tăng token 2× không làm peak memory tăng gần 4× (`profile` đo được).
- Không tensor nào có shape `[N, N]` trong forward.

### HandGraph
- LEAP `18` node và Allegro `22` node cùng chạy qua một encoder, không padding
  cố định, không giả định số joint.
- Hoán vị thứ tự node không đổi output ở palm node (permutation equivariance của
  message passing).

### Flow head
- Output hữu hạn với input ngẫu nhiên.
- `palm_rotation` thỏa `RᵀR = I`, `det = 1`.
- `joint_angles` trong limit của profile.
- Số bước Euler đổi thì output đổi, nhưng vẫn hữu hạn và vẫn trong limit.

### FK/consistency
- FK của output khớp `fingertip_positions` do `RobotSpec` tính, cùng dung sai.
- Gradient chảy từ fingertip loss về palm và joint head.

### Gradient/lifecycle
- **Mọi** parameter có `requires_grad` nhận gradient hữu hạn sau một backward.
- Build/forward/backward cho scale `n`; `s`/`m` chỉ cần build/forward.
- Cùng seed cho cùng output trên cùng device.

### Overfit
- Trên 8 sample, loss giảm đơn điệu về dưới ngưỡng đã pin, và palm/joint error
  giảm cùng. Đây là bằng chứng kiến trúc học được, **không** phải chất lượng.

### CPU/CUDA
- CPU chạy correctness. CUDA chạy forward/backward/overfit thật; không fallback.
- FP32 CPU/CUDA parity `atol/rtol ≤ 1e-4` theo `PLAN.md` §6.

## 6. Gate đóng P4

```bash
python scripts/check_phase4.py --profile micro
python -m pytest tests/model_flow -q
python scripts/overfit_qdgrasp_flow.py --device cpu --steps 300
git diff --check
```

GPU gate chạy ở notebook riêng:

```bash
python scripts/phase4_cuda_gate.py --device cuda:0 --evidence phase4_cuda_evidence.json
```

## 7. Điều kiện hoàn tất

1. Tokenizer, encoder, HandGraph encoder, conditioning, flow head, FK head và
   quality head có test và không tạo `N×N`.
2. Mọi trainable parameter nhận gradient; output hữu hạn, rotation hợp lệ, joint
   trong limit.
3. Hai active hand cùng chạy qua một model không cần đổi code.
4. Tiny overfit hội tụ dưới ngưỡng đã pin trên CPU **và** trên CUDA.
5. CPU/CUDA FP32 parity trong dung sai.
6. Independent reviewer ký trên packet bất biến, zero S0/S1.

P4 kết thúc ở **kiến trúc chứng minh học được**, không phải "model tốt". Không
số nào ở P4 được trích dẫn như kết quả grasping.
