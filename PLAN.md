---
document_id: PLAN-V2
document_type: plan
title: DexGrasp — thư viện dexterous grasp theo kiến trúc Ultralytics
version: 2.3.0
status: active
date: 2026-08-21
approved_date: 2026-08-21
revises: LEGACY-PLAN-PRE-V2
supersedes: docs/archive/PLAN.pre-v2.md
revision_record: docs/revisions/REV-20260821-001-plan-v2.md
latest_revision_record: docs/revisions/REV-20260822-005-agpl-public-release-basis.md
revision_reason: Kế hoạch cũ không còn đáp ứng yêu cầu nguồn chuẩn, CPU/GPU, kiến trúc YAML mới và kiểm soát giấy phép.
necessity: N3
impact: Thay đổi nguồn phụ thuộc, mô hình lõi, chiến lược đóng gói, điều kiện phát hành và toàn bộ lộ trình triển khai; chưa có code triển khai bị thay đổi.
---

# DexGrasp — kế hoạch triển khai chính thức

Tài liệu này là nguồn sự thật cho phạm vi và thứ tự triển khai. Bản kế hoạch cũ
được giữ nguyên tại `docs/archive/PLAN.pre-v2.md`. Mọi thay đổi kế hoạch phải tuân
theo `docs/governance/DOCUMENTATION_POLICY.md` và phải có revision record nếu sửa
lại quyết định hoặc kết quả của phiên trước.

## 1. Mục tiêu và quyết định nền

- Xây package độc lập `dexgrasp`, public façade `DexGrasp`, CLI `dexgrasp`, với
  lifecycle tương đương core Ultralytics: `train`, `val`, `predict`, `export`,
  `benchmark`, `info`, `load`, `save`, `to`, callbacks, `task_map` và `Results`.
- Fork và pin [Ultralytics v8.4.125 / `329682a`](https://github.com/ultralytics/ultralytics/commit/329682a29d27203582ba30e519340f95abccc6a6);
  không phụ thuộc package `ultralytics` lúc runtime.
- Clone sạch [DexGraspNet2 / `26ecd761`](https://github.com/PKU-EPIC/DexGraspNet2/commit/26ecd76121e3c8218ad53db9840cf34f6b81b076)
  làm oracle cho dữ liệu, mô hình và benchmark. Clone tùy biến hiện có tại
  `/media/quyen/Data/DexGraspNet2` không phải nguồn chuẩn và không được tự ý sửa.
- Clone thêm [GraspGenX / `b942909`](https://github.com/NVlabs/GraspGenX/tree/b9429097728cb1c430dd78b92edf17ba318aad03)
  và [rectified-flow / `14b4925`](https://github.com/lqiang67/rectified-flow/tree/14b4925ad90abdadaca1f7b5caba5555b84e810a)
  làm nguồn code chuẩn cho backbone thuần PyTorch và flow solver.
- Các clone nằm trong `.references/`, bị loại khỏi Git, wheel và source
  distribution. `references.lock.yaml` lưu URL, SHA, license hash và provenance
  của từng phần được port.
- `scripts/check_references.py` khóa content của manifest; khi `.references/`
  có mặt, feature/release gate xác minh origin, exact HEAD, clean worktree và
  artifact/license-evidence hashes của mọi checkout bắt buộc.
- `dexgrasp` sẽ được public toàn bộ source code cho cộng đồng dưới `AGPL-3.0`
  (không phải nội bộ, không phải sản phẩm thương mại độc quyền). Đây là điều
  kiện copyleft mà Ultralytics AGPL-3.0 tự cho phép: phân phối lại bản dẫn xuất
  hợp pháp mà KHÔNG cần văn bản cấp phép thương mại riêng, miễn toàn bộ
  `dexgrasp` (không chỉ phần vay mượn) giữ AGPL-3.0 và source luôn sẵn có, kể cả
  khi chạy như dịch vụ mạng (§13 Affero). Không được đóng gói lại bất kỳ phần
  nào dưới license permissive/độc quyền/dual-license trừ khi có văn bản cấp
  phép thương mại thật từ Ultralytics.
- DexGraspNet2 (`CC-BY-NC-4.0`) KHÔNG được giải quyết bởi quyết định AGPL-3.0
  trên — non-commercial là trục độc lập với lựa chọn license phát hành. Việc
  port code/dữ liệu DGN2 vào `dexgrasp` vẫn là release gate riêng, chờ đánh giá
  pháp lý cụ thể trước khi public.
- Core model bắt buộc chạy CPU FP32 và NVIDIA CUDA FP32/AMP. MuJoCo là evaluator
  CPU chính; Isaac Gym chỉ là bộ đối chiếu GPU tùy chọn.
- CPU phải hoàn tất đúng và không OOM; latency được đo và công bố nhưng chưa là
  release gate.

### Những kết luận của kế hoạch cũ bị thay thế

- MinkowskiEngine có CPU build; loại nó vì gánh nặng đóng gói, bảo trì và hiệu
  năng đa nền tảng, không phải vì tuyệt đối CUDA-only.
- Không ép 40k điểm xuống 4096 rồi mất độ phân giải graspness; luôn giữ
  `point_to_token` để unpool feature về raw points.
- Không dùng một Z/Hilbert sort duy nhất, không thay FPS bằng top-k thuần, không
  dùng Grasp DiT sinh chung wrist và joints, và không giữ sampler likelihood 200
  bước.
- Checkpoint DGN2 cũ chỉ chạy trong upstream oracle; model mới không load trực
  tiếp checkpoint MinkUNet.

## 2. Kiến trúc thư viện và public interfaces

```text
dexgrasp/
├── cfg/                 # defaults, models, datasets, robots, sim
├── data/                # canonical schema, loaders, depth → point cloud
├── engine/              # Model, Trainer, Validator, Predictor, Exporter, Results
├── models/grasp/        # task_map và bốn mode của task grasp
├── nn/
│   ├── modules/         # serialized point blocks, neck, flow/joint/quality heads
│   ├── registry.py      # registry tường minh
│   ├── parser.py        # YAML parser
│   └── tasks.py         # GraspModel
├── robot/               # URDF parser, RobotSpec, FK, profile validation
├── sim/                 # MuJoCo chính, Isaac adapter tùy chọn
└── utils/               # device, callbacks, metrics, checkpoint, profiling
```

Python API:

```python
from dexgrasp import DexGrasp

model = DexGrasp("dexgrasp-flow-n.yaml", robot="my_hand.yaml")
model.train(data="dexgraspnet2.yaml", device=0, max_steps=50000)
metrics = model.val(sim="mujoco")
results = model.predict(points_or_depth, intrinsics=K, device="cpu")
model.export(format="onnx")
```

CLI:

```bash
dexgrasp task=grasp mode=train model=dexgrasp-flow-n.yaml \
  data=dexgraspnet2.yaml robot=my_hand.yaml device=0
```

`GraspResults` phải có `.cpu()`, `.cuda()`, `.to()`, `.numpy()`, `.plot()`,
`.save()` và `.summary()`. Trường chuẩn gồm translation `[K,3]`, rotation
`[K,3,3]`, joints `[K,J]`, score `[K]`, seed point, frame và robot/profile hash.

Checkpoint `.pt` chứa YAML snapshot, `state_dict`/EMA, optimizer, scheduler,
scaler, RNG, global step, dataset manifest, robot profile và source provenance;
không pickle nguyên module. Resume phải tái lập chính xác optimizer, scheduler và
RNG.

### Train configuration surface

- `docs/configuration/TRAIN_ARGUMENTS.yaml` là compatibility manifest máy đọc
  được cho đúng commit Ultralytics đã pin. Nó phải kiểm kê 100% canonical key,
  custom-only key, legacy alias/removed key và tham số điều khiển riêng của
  `Model.train`; không chỉ chép nhóm “Train settings”.
- Mỗi key có default upstream, group, type, train role, disposition
  `retain|adapt|defer|reject`, key DexGrasp đích và device policy. Key YOLO-only
  phải lỗi rõ khi người dùng truyền, không bị bỏ hoặc no-op im lặng.
- Runtime lưu requested/effective config. CPU bắt buộc effective `amp=False`;
  fractional AutoBatch chỉ hợp lệ trên CUDA đơn; mọi device fallback, worker
  clamp, compile fallback và OOM batch reduction phải được log.
- DexGrasp extension chỉ được triển khai sau khi khóa default/range và thêm vào
  checker. Unknown key, dead key hoặc extension còn `required_design` là lỗi.
- Mỗi lần nâng upstream, full checker phải chứng minh zero missing/extra/default/
  type mismatch với source clone và hash mới trước khi merge integration branch.

### YAML

Giữ grammar Ultralytics `- [from, repeats, module, args]`. Parser flatten
`backbone + optional neck + head` với chỉ số layer toàn cục. Registry tường minh
lưu quy tắc input/output channels, repeat và structured outputs; không dùng
`globals()`, `eval()` hoặc module path tùy ý. Khóa lạ hoặc dead key là lỗi.

```yaml
scales:
  n: [0.50, 0.50, 128]
  s: [0.75, 0.75, 192]
  m: [1.00, 1.00, 256]
  l: [1.25, 1.25, 384]
  x: [1.50, 1.50, 512]
```

Tất cả `n/s/m/l/x` phải build được; chỉ phát hành weights `n` và `m` ở v1.
Scale `m` phải đạt 14–16M tham số để đối chiếu DGN2 15.10M; `n` hướng tới
3–5M.

### Robot contract

Mọi URDF hợp lệ được hỗ trợ theo nghĩa checkpoint theo robot:

- Fixed/mimic joints được xử lý tự động; actuated revolute/prismatic joints phải
  có giới hạn hữu hạn.
- Continuous joint hoặc thiếu limit phải có override tường minh.
- Profile khai báo base/wrist/palm/contact/fingertip links, joint order, frame
  transform, actuator/squeeze settings và mesh/package resolution.
- URDF chỉ tạo profile skeleton; semantic links không được suy đoán im lặng.
- Checkpoint bị từ chối nếu profile hash hoặc joint schema không khớp.
- Người dùng cung cấp nhãn wrist/joints cho robot mới; v1 không sinh dữ liệu hay
  retarget nhãn LEAP.

## 3. Họ mô hình `DexGrasp-Flow`

```text
raw XYZ 40k / depth + intrinsics
  → voxel tokenizer thuần Torch
  → 4k–8k active tokens + point_to_token
  → serialized point encoder
  → trace-unpool/FPN neck
  → feature trở lại 40k raw points
  → objectness + graspness
  → local-max + score candidates + radius suppression
  → local geometry tokens
  → 5-step wrist rectified flow (R9D + T)
  → deterministic joint head bounded theo URDF
  → learned quality head
  → top-K grasps
```

- Tokenizer dùng packed `int64` key có kiểm tra overflow, `sort/unique/scatter`,
  không hash collision và không custom C++/CUDA.
- Encoder `m`: channels `[32,64,128,192,256]`, depths `[1,1,2,2,1]`, bốn
  serialization orders có shifted windows, LayerNorm/GELU và PyTorch SDPA.
- Neck là encoder-decoder/FPN dùng pooling parent/inverse và skip connections.
  `RawPointRefine` kết hợp token feature với offset của raw point trong voxel để
  chấm graspness trên đủ 40k điểm.
- Seed selector giữ local maximum theo voxel rồi radius suppression bằng phép
  toán Torch trên tập candidate nhỏ. Không tạo tensor `N×N`.
- `WristFlowHead` dùng straight-path rectified flow, velocity matching và Euler
  solver 5 bước. State là rotation 9D và translation 3D; kết quả chiếu về SO(3)
  bằng SVD.
- `JointHead(local_feature, wrist)` sinh vector theo đúng joint order và map
  sigmoid vào `[lower, upper]`; fixed/mimic joints được dựng lại sau head.
- `QualityHead` học từ positives, hard/free-space negatives và on-policy failures
  được gắn nhãn bằng MuJoCo. Score cuối kết hợp objectness, graspness và quality.
- Input learned của v1 là XYZ. Predictor nhận point cloud hoặc depth+intrinsics;
  RGB được giữ như metadata nhưng chưa fusion vào model.
- Base install không được import MinkowskiEngine, spconv, pointops,
  torch-scatter, torch-cluster, PyTorch3D, Isaac Gym hoặc Open3D.

## 4. Lộ trình

### M0 — Nguồn, license và baseline (1–2 tuần)

- Chốt `dexgrasp` public dưới `AGPL-3.0`, clone/pin bốn upstream, lập
  provenance và baseline manifest.
- Đánh giá pháp lý riêng cho DexGraspNet2 (`CC-BY-NC-4.0`) trước khi port bất
  kỳ code/dữ liệu nào vào `dexgrasp`.
- Đo lại checkpoint DGN2: params, schemas, outputs, scene/view hashes và
  benchmark canonical.
- Khóa corpus 30 scene × 256 view, depth source, camera frame, robot profile và
  stochastic seeds.

### M1 — Fork framework (4–6 tuần)

- Tạo product branch từ Ultralytics pinned commit, đổi namespace thành
  `dexgrasp`.
- Giữ engine/cfg/callback/checkpoint/device spine; loại YOLO tasks, trackers,
  HUB, solutions và integrations ngoài scope.
- Hoàn thiện façade, CLI, `task_map`, `GraspResults`, safe checkpoint và import
  không đổi cwd.

### M2 — Data và robot layer (4–6 tuần)

- Canonical dataset schema cho points, objectness, graspness, wrist pose và qpos
  theo tên joint.
- Converter DGN2 chỉ đọc dữ liệu gốc; lỗi file/shape/NaN phải fail rõ, không
  recursive retry hoặc silent skip.
- URDF/Profile validator, FK, joint-limit mapping, mesh resolver và fixtures.
- Depth-to-point-cloud preprocessing thuần NumPy/Torch.

### M3 — YAML model và kiến trúc (10–14 tuần)

- Registry/parser, tokenizer, serialized encoder, neck, raw-point refinement và
  seed selector.
- Flow, joint và quality heads; build/profile đủ năm scale.
- Overfit tiny dataset, gradient coverage và ablation 4096/8192/16384 tokens,
  3/5/7 flow steps; mặc định 8192 token và 5 bước.

### M4 — Engine, export và device parity (6–8 tuần)

- Trainer/Validator/Predictor với CPU, CUDA, AMP, EMA, exact resume và Kaggle
  session recovery.
- TorchScript và ONNX bundle gồm graph, model YAML, robot profile và preprocessing
  metadata.
- Tokenization/serialization và SVD có thể nằm ngoài ONNX graph trong
  `AutoBackend`; public results phải giống native.
- ONNX Runtime CPU bắt buộc; CUDA runner được kiểm trên Kaggle.

### M5 — MuJoCo evaluator (8–12 tuần)

- Pin MuJoCo minor version, dùng parser URDF chính thức và named joints.
- Giữ trajectory DGN2: pregrasp/cover/grasp/squeeze/lift, 60 Hz, waypoint steps
  `[40,20,20,60]`, nâng 0.2 m và success khi vật thể tăng trên 3 cm.
- Thêm sáu virtual base joints bằng tên ổn định; giữ collision primitives gốc.
- Chạy batch bằng `mujoco.rollout`; cùng seed/config phải cho cùng boolean vector.
- Replay cùng corpus bằng Isaac Gym và MuJoCo, báo confusion matrix/kappa; sau
  hiệu chuẩn đóng băng `mujoco.yaml`.

### M6 — Full training và release (10–18 tuần)

- Train `n`, sau đó `m`, trên Kaggle GPU với checkpoint resumable; không
  warm-start checkpoint MinkUNet.
- Chọn checkpoint bằng validation pose/joint/quality metrics và MuJoCo success,
  không dùng total loss đơn lẻ.
- Phát hành weights chỉ sau khi hoàn tất CPU/CUDA/export/sim gates và benchmark
  30 scene.
- Mỗi lần nâng Ultralytics phải đi qua integration branch và API contract tests.

## 5. Test và nghiệm thu

- Train-argument registry khớp toàn bộ 115 canonical key, 2 extra config kwargs,
  9 legacy name và 1 API control của source đã pin; cả registry-only và full
  source check đều pass với zero missing/extra.
- Config parser từ chối unknown/dead/rejected key; mọi key `retain/adapt` có test
  default, override, serialization, resume mutability và effective CPU/CUDA.
- Mọi YAML n–x build, forward/backward và mọi trainable parameter nhận gradient.
- CPU và CUDA chạy train-smoke, full val, predict, resume và Results conversions;
  không có `.cuda()` hard-code.
- FP32 CPU/CUDA parity `atol/rtol ≤ 1e-4`; AMP `≤ 5e-3`. Preprocess và seed
  indices deterministic.
- Profile chứng minh không tạo `N×N`; tăng token 2× không làm memory tăng gần 4×.
- Model `m` nằm trong 14–16M params; báo riêng backbone/neck/flow/joint/quality.
- i5-8365U/14 GiB phải hoàn tất predict một scene 40k/top-64 và train-smoke mà
  không OOM; latency chỉ được báo cáo.
- TorchScript và ONNX round-trip giữ số grasp, ordering, rotation, translation,
  joints và score trong tolerance.
- Joint outputs luôn hữu hạn và trong giới hạn; rotation hợp lệ; profile mismatch
  phải fail trước inference.
- MuJoCo chạy lặp lại cho cùng kết quả; fixtures kiểm geometry, frames, joint
  order và actuator mapping.
- Quality gate cuối: corpus MuJoCo 30 scene với ba seed, paired bootstrap một
  phía 95%; cận dưới của `success_new_m − success_DGN2` không thấp hơn `−0.02`.
- Mọi số liệu phát hành phải có metrics report, raw artifact/hash và third-party
  review theo bộ tài liệu trong `docs/`.

## 6. Ngoài phạm vi v1

- MPS/XPU/NPU, TensorRT/OpenVINO/CoreML và toàn bộ exporter của YOLO.
- Data generation, LEAP retargeting và một checkpoint universal cho mọi robot.
- HUB, tune, solutions, tracking và integrations không liên quan grasp.
- Legacy DGN2 checkpoint loading trong package mới.
