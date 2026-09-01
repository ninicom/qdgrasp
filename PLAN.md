---
document_id: PLAN-V2
document_type: plan
title: QDGrasp — thư viện dexterous grasp cộng đồng dưới AGPL-3.0
version: 4.6.0
status: active
date: 2026-09-01
approved_date: 2026-08-22
revises: PLAN-V2@4.5.0
supersedes: docs/archive/PLAN.pre-v2.md
revision_record: docs/revisions/REV-20260821-001-plan-v2.md
latest_revision_record: docs/revisions/REV-20260901-001-cross-component-remediation-plan.md
revision_reason: Audit liên mô-đun phát hiện đường thực thi pickle, protocol không đi vào trainer, supervision/Flow semantics mâu thuẫn và artifact lifecycle không giữ đúng danh tính; phải thêm corrective track bắt buộc trước khi tiếp tục P4/P5 hoặc phát hành.
necessity: N3
impact: DGN-Open-Tiny hiện tại và Flow checkpoint/evidence cũ không được dùng làm release evidence; P4/P5 phải qua corrective gates ở §9. MVP round 3 giữ experimental_non_release và cần revalidation sau khi vá fingerprint/checkpoint path.
---

# QDGrasp — kế hoạch triển khai chính thức

Tài liệu này là nguồn sự thật cho phạm vi và thứ tự triển khai. Bản kế hoạch cũ
được giữ nguyên tại `docs/archive/PLAN.pre-v2.md`. Mọi thay đổi kế hoạch phải tuân
theo `docs/governance/DOCUMENTATION_POLICY.md` và phải có revision record nếu sửa
lại quyết định hoặc kết quả của phiên trước.

Quyết định kỹ thuật và giấy phép chi tiết nằm tại
`docs/decisions/0007-agpl-community-library.md` (thay ADR-0002) và clarification DGN2 tại
`docs/decisions/0004-dgn2-paper-reference-boundary.md`; environment/reference
intake nằm tại `docs/decisions/0005-environment-and-reference-intake.md`; CUDA
hardware gate nằm tại `docs/decisions/0006-cuda-hardware-required.md`. Đây là
chính sách kỹ thuật về nguồn và phát hành, không thay thế tư vấn pháp lý chuyên
nghiệp.

Phạm vi robot hiện hành được sửa bởi
`docs/decisions/0008-temporary-shadow-hand-pause.md`: active corpus chỉ gồm LEAP
và Wonik Allegro. Shadow Hand được giữ như paused/experimental compatibility
profile, không tham gia workload/gate/release mới cho tới ADR mở lại.

Corrective execution plan hiện hành cho contact-rich data là
`docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md`. Plan này buộc sửa
đủ safety-budget coverage, trajectory time/frame, exact GPU-to-CPU replay,
CUDA parity/sanitizer, dataset count/split và default-scope enforcement trước
khi mở contact-rich input cho P4. Verdict của nó chỉ là 2/2 active hand; không
được dùng để ghi P3.4 three-hand `pass`.

Thi công theo `docs/roadmap/PHASE3_4_3_EXECUTION_BREAKDOWN.md`. Phần CPU đã
xong với code, test và evidence; gate CUDA chạy trên Kaggle T4 và review độc
lập vẫn là hai điều kiện còn lại, nên `QDGrasp-ContactRich-Active-Tiny` giữ
`release_blocked=true` và loader public từ chối nó theo mặc định.

Đường thực thi ưu tiên tạm thời `docs/roadmap/GRASP_POLICY_MVP_TEMPORARY_PLAN.md`
đã đóng ngày 2026-08-31 (`REV-20260831-001`). Nó tạo ra
`QDGrasp-Leap-Grasp-MVP`, một checkpoint `experimental_non_release` đạt cả ba
tier trong phạm vi LEAP + một cuboid trên bàn + quan sát state. Nó **không** gỡ
blocker nào ở trên, và số của nó không phải bằng chứng release hay bằng chứng
GPU physics. Kết quả âm cần ghi kèm: trong phạm vi đó, controller prior một mình
đã đạt cùng mức, nên MVP không chứng minh giá trị của phần học.

Audit liên mô-đun ngày 2026-09-01 phát hiện các unit contract riêng lẻ có thể
pass trong khi chuỗi dataset → protocol → model → runner → artifact sai nghĩa
hoặc không an toàn. Corrective track bắt buộc tại **§9** có quyền chặn P4/P5 và
release cho tới khi từng gate ở đó có bằng chứng mới. Nó không nới CUDA,
independent-review, license hay release gate đã có.

## 1. Mục tiêu và quyết định nền

- Tên public chính thức là **QDGrasp**; distribution/import/CLI/repository lần
  lượt là `qdgrasp`, `qdgrasp`, `qdgrasp` và `ninicom/qdgrasp`. Tên DexGrasp chỉ
  còn trong hồ sơ lịch sử bất biến hoặc khi nhắc đúng tên dự án/bài báo bên ngoài.
- Xây package độc lập `qdgrasp`, public façade `QDGrasp` và CLI `qdgrasp` có
  trải nghiệm một lệnh, dễ train/val/predict/export và dễ mở rộng cho cộng đồng.
  “Giống YOLO” là mục tiêu về trải nghiệm, tài liệu, model zoo và hệ sinh thái;
  không phải tương thích source hay sao chép implementation của Ultralytics.
- License của code chính thức là `AGPL-3.0-only`, cùng hướng copyleft/network-use
  như hệ sinh thái YOLO hiện đại. Dự án giữ nguyên copyright, license và notice;
  public source không có nghĩa là không còn nghĩa vụ bản quyền.
- Cây `qdgrasp/` và lịch sử hiện tại được tiếp tục như sản phẩm AGPL hợp lệ.
  Ultralytics-derived files còn lại phải giữ header/notice; module mới dùng public
  API QDGrasp và được giảm dần phụ thuộc legacy theo từng phase.
- Repository hiện tại là repository public chính thức. Không tạo clean-history
  Apache repository và không tuyên bố implementation hiện tại là clean-room.
- Mọi đường dẫn tới artifact/config/data/robot trong project là relative path tính
  từ project root hoặc working directory. Cấm commit đường dẫn máy phát triển;
  OS pseudo-files dùng để nhận diện platform không được coi là project artifact.
- Engine mới dùng PyTorch và
  [Lightning Fabric](https://github.com/Lightning-AI/pytorch-lightning) làm nền
  train/device/distributed; façade, config, checkpoint, data và results contract
  được viết mới theo đặc tả QDGrasp.
- Nguồn grasp permissive ưu tiên là
  [GraspGenX `b942909`](https://github.com/NVlabs/GraspGenX/tree/b9429097728cb1c430dd78b92edf17ba318aad03),
  có code Apache-2.0. Chỉ port file đã audit cùng notice tương ứng; không tự động
  mang theo checkpoint, dataset, gripper assets hay dependency chưa kiểm tra.
  GraspGenX sinh wrist pose 6/9 chiều, nên không thay thế joint/qpos head của
  QDGrasp.
- [Pointcept](https://github.com/Pointcept/Pointcept) (MIT) và
  [rectified-flow](https://github.com/lqiang67/rectified-flow) (MIT) là nguồn
  permissive cho point backbone/flow. Ưu tiên triển khai từ paper và toán học;
  nếu dùng code thì pin commit, giữ attribution và ghi file-level provenance.
- DexGraspNet2 (`CC-BY-NC-4.0`) chỉ là **paper/reference-only** để hiểu bài toán
  và các nguyên lý grasping. DGN2 không phải runtime dependency, oracle hay
  nguồn dữ liệu của QDGrasp. Không port code, config, URDF/STL, dataset,
  checkpoint, labels, model weights, distillation target hoặc biểu đạt
  implementation của DGN2 vào sản phẩm. Mọi model, schema, generator, simulator
  protocol và checkpoint QDGrasp được phát triển/train mới.
- Dữ liệu train chính thức được tạo mới bằng generator của QDGrasp trên MuJoCo/
  MJX, từ procedural/CC0 assets và robot assets có license tường minh.
  Dataset, code, weights và assets là bốn miền license độc lập, mỗi miền có
  manifest riêng.
- Model cross-embodiment lấy graph động học biến độ dài từ URDF/MJCF làm điều
  kiện và sinh trực tiếp palm pose + named joint state. Differentiable FK nối
  state thực thi với keypoint/contact/force phụ; không khóa latent hoặc joint
  order vào Shadow/LEAP và không cần retarget làm đường mặc định.
- Corpus tương thích lịch sử đầu tiên gồm LEAP, Allegro và Shadow Hand; exact
  source/commit/tree/license nằm trong `robot_assets.lock.yaml`. Active corpus
  sau ADR-0008 chỉ gồm LEAP và Allegro; Shadow tạm dừng nhưng không bị xóa khỏi
  manifest/provenance. Barrett chỉ là
  fixture nghiên cứu bị chặn phát hành cho tới khi có license đầy đủ bao phủ cả
  URDF lẫn mesh. RH56E2 bị loại khỏi scope, manifest, fixture, model và checkpoint.
- Core model bắt buộc hỗ trợ CPU FP32 cho correctness/CI và NVIDIA CUDA
  FP32/AMP cho train/evaluation/benchmark. Mọi workload hoặc kết quả mang nhãn
  CUDA phải chạy trên GPU NVIDIA thật; không dùng CPU fallback, emulation hoặc
  dry-run làm bằng chứng CUDA pass. MuJoCo là evaluator chuẩn; backend tùy chọn
  không được làm base install kéo license không nằm trong allowlist.
- Environment chuẩn là Linux x86_64, Python `>=3.11` (reference lock dùng 3.11),
  PyTorch 2.11.0, Lightning Fabric 2.6.5 và MuJoCo 3.12.0. CPU dùng wheel
  `+cpu`, NVIDIA dùng `+cu128`; lock và fingerprint nằm trong `environments/`.
  Không cài vào system Python của máy phát triển. Phase model/training không được
  đóng nếu CUDA hardware smoke, train-step, resume và parity chưa chạy.
- Packaging theo pattern YOLO: `pyproject.toml` dùng compatible lower bounds cho
  cài đặt thông thường; exact CPU/cu128 versions chỉ nằm trong research locks.
- Baseline tham chiếu gồm GraspDiffuser, DexDiffuser, CEDex và GraspGenX đã đủ
  để bắt đầu. Chỉ thêm repository mới khi có issue kỹ thuật tái hiện được và
  phải qua `docs/governance/REFERENCE_INTAKE.md`; reference không tự trở thành
  dependency, dataset, asset hay checkpoint hợp lệ.

### Những kết luận của kế hoạch 2.x–3.x bị thay thế

- Không fork hoặc nâng phiên bản Ultralytics; không giữ engine/cfg/callback/
  checkpoint/device spine từ upstream AGPL.
- Không duy trì compatibility manifest 115 train keys, exact CLI key-value,
  `task_map`, `Results` implementation hay grammar YAML của Ultralytics.
- Không theo đuổi Apache-2.0 clean-room hoặc repository/history mới. QDGrasp
  được phát hành trực tiếp dưới AGPL-3.0-only và giữ đầy đủ attribution.
- Không dùng hoặc chạy DGN2 converter/corpus/checkpoint trong pipeline phát hành;
  kết quả DGN2 chỉ có thể được trích dẫn như kết quả đã công bố trong paper,
  không phải baseline artifact bắt buộc hay nguồn nhãn.
- Không dùng Isaac Gym làm oracle bắt buộc; protocol đánh giá mới được định nghĩa
  độc lập trên MuJoCo/MJX và bộ dữ liệu mở của QDGrasp.
- Vẫn giữ các quyết định kỹ thuật độc lập đã hợp lý: pure Torch tokenizer,
  `point_to_token`, raw-point graspness, named-joint contract, CPU/CUDA parity và
  không tạo tensor `N×N`.

## 2. Biên giấy phép và nguồn thay thế

| Nhu cầu | Nguồn cũ bị loại khỏi sản phẩm | Nguồn/triển khai đích | Điều kiện phát hành |
| --- | --- | --- | --- |
| Façade, CLI, lifecycle | Ultralytics AGPL-3.0 | Public API QDGrasp trên Python/PyTorch; UX một lệnh | AGPL-3.0-only, giữ notice của file dẫn xuất |
| Train/device/distributed | Ultralytics engine | Lightning Fabric | Pin commit/version, giữ LICENSE/NOTICE |
| Grasp backbone/conditioning | DGN2 CC-BY-NC và code chưa rõ provenance | File đã audit từ GraspGenX Apache-2.0 hoặc implementation mới từ paper | Không dùng checkpoint/dataset GraspGenX nếu chưa audit riêng |
| Point encoder | MinkowskiEngine/DGN2 implementation | Pure Torch; Pointcept MIT chỉ là nguồn permissive có provenance | Không custom CUDA trong base install |
| Wrist flow | DGN2 diffusion/ODE code | `rectified-flow` MIT hoặc solver viết từ công thức | Pin source, unit test công thức/parity |
| Joint/qpos head | DGN2 LEAP-specific head | Joint-token head viết mới từ `RobotSpec` và named limits | Không dùng DGN2 code, URDF, STL hay weights |
| Config | Ultralytics defaults/grammar | Schema QDGrasp viết mới, YAML khai báo theo tên | Không compatibility-copy keys/defaults upstream |
| Simulator | Isaac Gym/DGN2 adapter | MuJoCo/MJX Apache-2.0 | Protocol, scenes và assets có manifest riêng |
| Robot assets | LEAP assets trong DGN2 | MuJoCo Menagerie; LEAP/Wonik official; nguồn cộng đồng theo từng file MIT/BSD/Apache | Exact source/tree/license trong `robot_assets.lock.yaml`; fixture mơ hồ bị chặn |
| Object assets | DGN2/GraspNet/Acronym corpus | Procedural + CC0 mặc định; GSO/YCB CC-BY chỉ là pack tùy chọn | Official CC0 weights không trộn pack CC-BY/NC |
| Export/runtime | Ultralytics exporters/backends | PyTorch, ONNX và ONNX Runtime adapters viết mới | Chỉ PyTorch/TorchScript/ONNX ở v1 |
| Community pattern | Ultralytics implementation | UX/package/docs/model-zoo kiểu YOLO | Không copy branding hoặc trademark; reuse code phải đúng AGPL/provenance |

Allowlist mặc định cho **code dependency/source**: `AGPL-3.0-only`,
`AGPL-3.0-or-later`, `GPL-3.0`, `Apache-2.0`, `MIT`, `BSD-2-Clause`,
`BSD-3-Clause`, `ISC`, `Zlib`, khi hướng tương thích và notice đã được kiểm.
Code kết hợp được phát hành dưới AGPL-3.0-only; license permissive của thành phần
vẫn phải được giữ. Allowlist mặc định cho **data/assets** là `CC0-1.0`;
`CC-BY-4.0` chỉ ở pack tách riêng có attribution. `CC-BY-NC`, `CC-BY-ND`,
custom/no-license và unknown bị chặn khỏi core/release cho tới khi maintainer phê
duyệt ngoại lệ bằng revision record N3.

Mỗi dependency phải được kiểm ở exact pin và cả dependency bắc cầu. License trên
trang chủ repository không tự động bao phủ model weights, dataset, sample assets,
submodule hoặc file mang notice khác.

## 3. Kiến trúc thư viện và public interfaces

```text
src/qdgrasp/
├── api/                  # façade QDGrasp và public protocols
├── cli/                  # subcommands train/val/predict/export/benchmark
├── config/               # schema/versioning; không tương thích-copy Ultralytics
├── data/                 # GraspBatch, loaders, depth → point cloud, manifests
├── engine/               # Fabric runner, callbacks, checkpoint, metrics
├── models/               # encoder, neck, seed, wrist, joint, quality heads
├── robot/                # RobotSpec, URDF/MJCF, FK, graph, profile validation
├── sim/                  # MuJoCo/MJX generation và evaluation adapters
├── export/               # PyTorch/TorchScript/ONNX bundles
└── provenance/           # SPDX, source/data/weight/asset manifests
```

Python API:

```python
from qdgrasp import QDGrasp

model = QDGrasp("qdgrasp-flow-n.yaml", robot="my_hand.yaml")
model.train(data="dgn-open-v1.yaml", device="cuda:0", max_steps=50_000)
metrics = model.val(data="dgn-open-v1.yaml", sim="mujoco")
results = model.predict(points_or_depth, intrinsics=K, device="cpu")
model.export(format="onnx")
```

CLI dùng subcommand chuẩn, không sao chép cú pháp Ultralytics:

```bash
qdgrasp train --model qdgrasp-flow-n.yaml --data dgn-open-v1.yaml \
  --robot my_hand.yaml --device cuda:0
```

`GraspResults` là dataclass/protocol viết mới với `.cpu()`, `.to()`, `.numpy()`,
`.plot()`, `.save()` và `.summary()`. Trường chuẩn gồm translation `[K,3]`,
rotation `[K,3,3]`, joints theo tên, score `[K]`, seed point, frame, model hash và
robot/profile hash.

Public model bundle không pickle module. Bundle gồm tensors an toàn, config
snapshot, preprocessing schema, robot profile, source/data manifest và chữ ký
hash. Resume state tách khỏi public weights và phải tái lập optimizer, scheduler,
scaler, RNG và global step.

### Configuration contract

- Schema được thiết kế từ yêu cầu grasp, không lấy key/default/docstring từ
  Ultralytics. Mỗi key có type, range, default, device policy và version.
- Runtime lưu requested/effective config. CPU ép `amp=False`; mọi fallback,
  worker clamp, compile fallback và OOM batch reduction được log tường minh.
- Unknown/dead key là lỗi. Migration schema phải có version và test; không giữ
  alias chỉ để mô phỏng CLI upstream.
- Parser dùng registry allowlist; không `eval()`, `globals()` hoặc import module
  tùy ý từ YAML.

Ví dụ model schema độc lập:

```yaml
schema: qdgrasp/model/v1
name: qdgrasp-flow-n
encoder:
  type: serialized_point
  channels: [32, 64, 128, 192]
  depths: [1, 1, 2, 2]
neck:
  type: trace_unpool_fpn
heads:
  wrist: {type: rectified_flow, steps: 5}
  joints: {type: named_joint_tokens}
  quality: {type: grasp_quality}
```

### Robot contract và corpus tương thích

- Fixed/mimic joints được xử lý tự động; actuated revolute/prismatic joints phải
  có giới hạn hữu hạn. Continuous/missing limit cần override tường minh.
- Profile khai báo base/wrist/palm/contact/fingertip links, joint order, frame
  transform, actuator/squeeze settings và mesh/package resolution.
- URDF hoặc MJCF chỉ tạo profile skeleton/kinematic graph; semantic links không
  được suy đoán im lặng. MJCF là đường chuẩn cho MuJoCo, không phải lý do bỏ test
  URDF.
- Checkpoint bị từ chối nếu profile hash hoặc joint schema không khớp.
- Mỗi URDF/mesh/profile public phải có provenance và license manifest. User có
  thể dùng asset riêng cục bộ nhưng asset đó không tự động được phát hành lại.
- Matrix lịch sử Phase 2 phủ LEAP official URDF + Menagerie MJCF, Wonik Allegro
  official URDF + Menagerie MJCF, Shadow E3M5 Menagerie MJCF và một URDF ngoài
  distribution để kiểm parser. Workload/gate mới chỉ bắt buộc LEAP + Allegro;
  Shadow fixture giữ paused theo ADR-0008. Raw asset không được sửa để làm test pass; mọi
  normalization phải là transform tái lập, có hash nguồn/đầu ra và cờ `modified`.

## 4. Model và pipeline dữ liệu mở

```text
raw XYZ / depth + intrinsics          URDF/MJCF + permissive meshes
  → point tokenizer/encoder            → variable-size HandGraph
  → object surface tokens              → link/joint/limit/mesh tokens
                 └── world-edge graph transformer ──┘
  → conditional flow: palm R9D + T + masked named joints
  → differentiable FK → topology-aligned 3D keypoints
  → contact/normal/force auxiliary heads + feasibility projection
  → optional energy guidance → physics quality → top-K grasps
```

- Tokenizer dùng packed integer key có kiểm tra overflow, sort/unique/scatter,
  không hash collision và không custom C++/CUDA.
- Encoder/neck giữ raw-point resolution qua `point_to_token`; serialization order,
  shifted windows và pooling được triển khai/ghi provenance độc lập.
- Flow chính sinh trực tiếp executable palm pose và joint state có mask theo graph,
  lấy cảm hứng từ GraspGraphNet nhưng được viết từ paper/toán học. Straight-path
  velocity matching dùng solver Euler 5 bước; rotation chiếu về SO(3) bằng phép
  toán Torch đã test.
- Keypoint là tập anchor theo link/topology biến độ dài, không phải bộ 21 điểm
  Shadow cố định của KPGrasp. FK consistency buộc keypoint phụ khớp palm/qpos;
  IK chỉ là baseline/adapter và không nằm trên đường inference mặc định.
- Contact, normal và force là auxiliary state; contact được chiếu về surface và
  force về Coulomb cone theo đặc tả toán học kiểu EquiDexFlow. Energy penetration,
  surface-pulling và self-collision kiểu EFF-Grasp là plugin inference có thể tắt,
  không thay thế simulator labels.
- v1 hiện hành phải có checkpoint LEAP, Allegro và một checkpoint multi-hand
  LEAP+Allegro. Shadow checkpoint bị defer theo ADR-0008. Zero-shot hand chỉ
  được tuyên bố khi robot held-out và
  morphology perturbation đều qua protocol định trước.
- Quality head học từ positive, hard/free-space negative và on-policy failures do
  MuJoCo/MJX gắn nhãn.
- Base install không import MinkowskiEngine, spconv, pointops, torch-scatter,
  torch-cluster, PyTorch3D, Isaac Gym, Open3D hoặc dependency license chưa duyệt.

Nguồn nghiên cứu và correction thư mục nằm ở
`docs/decisions/0003-cross-embodiment-flow-architecture.md`. Paper/PDF chỉ cho
phép tham khảo ý tưởng, công thức và benchmark có trích dẫn; code, data,
checkpoint và asset đi kèm không được nhập nếu chưa có exact license evidence.
CADGrasp chỉ là clutter refiner sau v1; MachaGrasp là baseline few-shot
articulation; DextER/DexVLG là semantic layer tương lai.

Pipeline `DGN-Open` nằm trong scope bắt buộc:

1. Sinh procedural objects bằng primitives/CSG/superquadrics; cho phép bổ sung
   CC0 assets có manifest.
2. Nạp robot permissive, sample wrist/contact candidates và giải IK/qpos theo
   named limits.
3. Loại collision, replay squeeze/lift bằng MuJoCo/MJX, lưu success/quality.
4. Render depth/point cloud với camera/frame metadata và deterministic seed.
5. Xuất `GraspBatch` shards, split theo object/shape family và immutable manifest.
6. Phát hành dataset do dự án sở hữu dưới `CC0-1.0`; official weights train chỉ
   trên manifest đã duyệt; official code/checkpoint recipe phát hành theo chính
   sách AGPL của dự án, còn dataset giữ license riêng trong manifest.

GSO/YCB `CC-BY-4.0` có thể tăng độ đa dạng nhưng phải là dataset/weight track riêng
có attribution. Objaverse chỉ được dùng khi lọc từng object theo license; license
của database không thay thế license của từng mesh.

## 5. Lộ trình

### M0 — Nền AGPL, package và license gates (Phase 0)

- Giữ AGPL-3.0 `LICENSE`, thêm `NOTICE`, `THIRD_PARTY.yml`, package metadata,
  DCO 1.1 và contribution policy inbound=outbound.
- Dùng repository hiện tại làm public product repository; giữ lịch sử và notice
  Ultralytics-derived thay vì tạo lịch sử sạch giả tạo.
- Build wheel/sdist, cài wheel ngoài source tree, import `qdgrasp`, chạy CLI và
  yêu cầu mọi notebook dùng package public ở exact commit.
- Pin/audit PyTorch, Lightning Fabric, GraspGenX file candidates, Pointcept,
  rectified-flow, MuJoCo/MJX và mọi dependency bắc cầu.
- CI chạy REUSE/ScanCode hoặc ORT, dependency license allowlist, wheel/sdist
  content audit, secret scan và forbidden-provenance/hash scan.
- Golden behavior tests được viết từ requirements/public papers, không import
  hay chạy cây Ultralytics/DGN2 code trong test suite phát hành.

### M1 — Core framework (4–6 tuần)

- Viết façade, CLI, config schema, Fabric runner, callbacks, result types và
  checkpoint bundles từ đầu.
- Chỉ hỗ trợ PyTorch/TorchScript/ONNX; loại telemetry, account, HUB và SaaS hook.
- Hoàn thiện CPU/CUDA device policy, AMP, EMA, deterministic resume và import
  không đổi cwd.

### M2 — Data generator và robot layer (6–10 tuần)

- Canonical `GraspBatch` và immutable manifests.
- URDF/Profile validator, FK, joint-limit mapping, mesh resolver và fixtures.
- MJCF importer và compatibility matrix active LEAP/Allegro; giữ Shadow parser/
  transmission fixture ở trạng thái paused, không dùng làm gate mới. Test raw
  XML, mesh resolution, joint names/limits, MuJoCo load/forward và transform normalization.
- Procedural/CC0 object generator, candidate sampler, collision filter và
  MuJoCo/MJX label pipeline.
- Phát hành `DGN-Open-Tiny` để overfit/CI trước khi tạo corpus lớn.

### M3 — Model QDGrasp-Flow (10–14 tuần)

- Tokenizer, serialized encoder, neck, raw-point refinement và seed selector.
- HandGraph/world-edge transformer, executable palm+joint flow, differentiable
  FK, topology-keypoint/contact/force heads và quality head; scale n/s/m.
- Overfit tiny dataset, gradient coverage và ablation: no-graph, direct-only,
  keypoint+IK, dual FK-consistency, equivariance, force projection, energy guidance.

### M4 — Engine, export và device parity (6–8 tuần)

- Full train/val/predict, AMP, exact resume và session recovery chạy trên CUDA;
  CPU chỉ giữ correctness/inference/export smoke và parity reference.
- Public weight bundle an toàn; TorchScript/ONNX bundle có schema/robot/preprocess
  metadata.
- ONNX Runtime CPU bắt buộc; CUDA runner được kiểm trong môi trường pin.

### M5 — Scale dữ liệu và evaluator (8–12 tuần)

- Khóa protocol grasp/squeeze/lift độc lập, named joints và deterministic seeds.
- Tạo `DGN-Open-v1`, công bố generator, manifests, splits và attribution.
- Calibrate MuJoCo/MJX parity; DGN2 không chạy trong release. Nếu cần báo cáo
  aggregate research comparison thì chỉ trích dẫn paper/public numbers và phải
  ghi `comparability: limited`, không dùng DGN2 làm nguồn nhãn hay baseline gate.
- Khóa trước paper protocol: train/held-out hand, held-out object family, seed,
  compute, success/collision/penetration/diversity/latency metrics và ablations;
  lưu raw runs, environment, dataset/model/robot hashes trước khi viết kết luận.

### M6 — Full training và community release (10–18 tuần)

- Train `n`, sau đó `m`, chỉ từ dataset manifest đã duyệt.
- Chọn checkpoint bằng pose/joint/quality và physics success, không bằng total
  loss đơn lẻ.
- Phát hành package, source, official weights, dataset cards, SBOM, model cards,
  quickstarts và benchmark artifacts sau khi mọi license/device/export/sim gate
  pass.
- Thiết lập model zoo, issue/PR templates, good-first-issue, RFC và release cadence
  để cộng đồng có đường đóng góp rõ như các hệ sinh thái ML phổ biến.

## 6. Test và nghiệm thu

### Release/license gates bắt buộc

- Public repository giữ toàn bộ history hợp lệ và root license AGPL-3.0-only.
- Source/wheel/sdist/container giữ đúng AGPL và mọi notice MIT/BSD/Apache; không
  chứa NC, ND, custom/no-license hoặc unknown ngoài tài liệu attribution được duyệt.
- Không có import, hash trùng hoặc similarity finding chưa xử lý từ Ultralytics/
  DGN2 implementation/artifacts. Similarity scan chỉ là bằng chứng hỗ trợ, không
  thay provenance review hay cấm tham khảo ý tưởng học thuật.
- Mỗi source, dependency, dataset, weight, URDF và mesh có URL bất biến, version/
  hash, SPDX identifier, license evidence và attribution.
- Mỗi robot release phải vượt raw XML parse, zero missing mesh, finite named
  limits, FK/frame fixture và MuJoCo load/forward; transform sửa inertia/package
  URI phải tái lập và không ghi đè asset gốc.
- `LICENSE`, `NOTICE`, `THIRD_PARTY`, SBOM và package metadata nhất quán; mọi
  Apache/MIT/BSD notice được giữ đúng.
- Mỗi official checkpoint trỏ tới immutable training-data manifest và không có
  DGN2/NC sample, distillation target hoặc pretrained initialization.
- Independent reviewer xác nhận N3 pivot AGPL và full license audit trước public beta.

### Functional/model gates

- Config parser từ chối unknown/dead key; mọi key có test default, override,
  serialization, resume mutability và effective CPU/CUDA behavior.
- Mọi model scale build, forward/backward và mọi trainable parameter nhận gradient.
- CUDA hardware chạy train-smoke, full val, predict, resume và results
  conversions; CPU chạy correctness/inference/export smoke. Không có `.cuda()`
  hard-code và không fallback CPU khi CUDA được yêu cầu.
- FP32 CPU/CUDA parity `atol/rtol ≤ 1e-4`; AMP `≤ 5e-3`. Preprocess, generator và
  seed indices deterministic.
- Profile chứng minh không tạo `N×N`; tăng token 2× không làm memory tăng gần 4×.
- TorchScript/ONNX round-trip giữ số grasp, ordering, rotation, translation,
  named joints và score trong tolerance.
- Joint outputs hữu hạn/trong giới hạn; rotation hợp lệ; profile mismatch fail
  trước inference.
- MuJoCo/MJX fixtures kiểm geometry, frames, joint order, actuator mapping và
  repeatability.
- Mọi số liệu phát hành có metrics report, raw artifact/hash và third-party review
  theo bộ tài liệu trong `docs/`.

## 7. Tiêu chí “cộng đồng như YOLO”

- `pip install qdgrasp`, API một đối tượng và CLI một subcommand cho mỗi lifecycle.
- Model/data/robot schemas ổn định, có migration và examples nhỏ chạy được trên CPU.
- Model zoo n/s/m, weight/model/data cards, benchmark có protocol tái lập.
- Plugin protocols cho dataset, robot, evaluator và exporter mà không yêu cầu fork
  core package.
- Tài liệu quickstart, cookbook, Colab/Kaggle, contribution guide, DCO, RFC,
  semantic versioning, changelog và deprecation policy.
- Không telemetry mặc định, không tài khoản bắt buộc và không khóa tính năng core
  sau dịch vụ độc quyền.

## 8. Ngoài phạm vi v1

- Tuyên bố “không còn bất kỳ nghĩa vụ bản quyền nào”; AGPL và license thành phần
  đều có nghĩa vụ phải tuân thủ.
- Relicense cây/lịch sử Ultralytics hiện tại sang license không tương thích.
- Port code/assets/config/labels/checkpoint của DGN2 hoặc phát hành weights
  train/fine-tune/distill từ DGN2.
- Exact Ultralytics API/config/CLI compatibility hoặc dùng thương hiệu YOLO.
- Universal checkpoint cho mọi robot, MPS/XPU/NPU và exporter ngoài
  PyTorch/TorchScript/ONNX.

## 9. Corrective remediation bắt buộc sau audit liên mô-đun 2026-09-01

Mục này là corrective track có mức cần thiết `N3`. Nó được thêm vì audit đã
tái hiện được lỗi an toàn và lỗi đúng-sai chỉ xuất hiện khi ghép nhiều subsystem;
việc toàn bộ unit test hiện tại pass không phủ định các kết quả đó. Không được
bắt đầu full training, chọn checkpoint, chạy held-out evaluation hoặc phát hành
Flow/DGN trước khi các gate tương ứng trong mục này đóng.

### 9.1 Baseline và trạng thái đóng băng

| Kiểm tra | Kết quả tại audit | Cách đọc bắt buộc |
|---|---:|---|
| Full test suite | `1306 passed, 1 skipped, 93 subtests`, 23 warning | Regression hiện hữu pass; interaction contract bên dưới chưa được test |
| Canonical DGN audit | `FAIL` | Ba source hash drift: `contact_force.py`, `mujoco_rollout.py`, `generate_dgn_open_tiny.py` |
| Phase 5 input gate | LEAP `1/32`, Allegro `2/42` positive trong logical train view | Corpus chưa đủ tín hiệu để train; P5 phải dừng |
| Protocol leakage probe | `12` sample `comp_*` ngoài protocol trên mỗi active hand | Public façade đang train physical split, không train protocol view |
| Ruff | `514` finding, `329` auto-fixable | Chưa phải release gate dùng được; không auto-fix lẫn với semantic changes |
| Mypy | `1480` finding toàn package; `112` trong core khi bỏ missing imports | Legacy/optional và core chưa có biên type-check rõ |

Trạng thái bắt buộc ngay sau audit:

- `datasets/dgn-open-tiny` là **non-release input** cho tới khi regenerate từ
  clean commit và canonical audit pass. Không đổi hash manifest để hợp thức hóa
  byte cũ mà không tái sinh dữ liệu.
- Mọi Flow checkpoint/evidence sinh dưới joint parameterization, quality head
  hoặc target-mask cũ là **superseded for release**. Chúng có thể giữ làm
  artifact lịch sử nhưng không được so trực tiếp với run sau sửa.
- Phase 4 overfit phải chạy lại sau khi sửa latent joint và quality objective.
- Phase 5 giữ blocked cho tới khi data, protocol, Runner và CUDA gate đều pass.
- MVP round 3 hiện khớp fingerprint của world hiện tại, nhưng vẫn giữ
  `experimental_non_release`; evaluator/checkpoint guard phải được vá và
  revalidate trước khi dùng lại. Nếu PPO distribution đổi, checkpoint PPO cũ
  trở thành artifact lịch sử chứ không được migrate như cùng một policy.

### 9.2 Registry các chuỗi lỗi phải khắc phục

| ID | Mức | Chuỗi lỗi đã tái hiện | Kết quả đích |
|---|---|---|---|
| `COR-00` | S0 | Manifest cho phép absolute/escape path; shard và MVP checkpoint dùng `torch.load(weights_only=False)`; Phase 5 có `verify=False`; ContactRich dùng string-prefix containment | Một safe artifact I/O path, `weights_only=True`, schema/path bị ràng buộc và malicious reducer không chạy |
| `COR-01` | S1 | Canonical audit, Phase 5 adapter và public loader dùng ba sample/manifest contract khác nhau; façade không gọi audit/gate | Một `DatasetArtifact.open_verified()` là entry point duy nhất cho audit, gate, façade và Runner |
| `COR-02` | S1 | Physical splitter stratify bên trong shape nhưng tuyên bố family-heldout; protocol lọc hậu kỳ, suy family từ prefix và không đi vào trainer | `ProtocolDatasetView` materialize chính xác `(split, robot, object_id)` và fail trên dư/thiếu/leakage |
| `COR-03` | S1 | Guard chống mixed robot nằm trong helper phụ; Runner dùng `default_collate`; LEAP-bound model nhận sample Allegro vì cùng 16 joint | Batch giữ robot/profile/joint-order identity và model assert hoặc group đúng graph |
| `COR-04` | S1 | Generator dùng zero/identity cho missing kinematics; adapter bỏ validity flags; loss hồi quy mọi sample | Explicit target-validity masks; placeholder không tạo geometric gradient |
| `COR-05` | S1 | `encode_target` ghi joint vật lý còn `decode` áp joint-limit `tanh`; flow loss và joint/FK loss kéo về hai nghiệm | Inverse parameterization có round-trip `<1e-5 rad` trên LEAP/Allegro |
| `COR-06` | S1 | Quality head chỉ nhận observation; K grasp khác nhau có cùng score; test ranking chấp nhận toàn tie | Candidate-aware quality head và ranking test có positive/negative cùng observation |
| `COR-07` | S1/S2 | Validation tiêu RNG train, đổi mode, metric phụ thuộc draw/batch size; EMA được cập nhật nhưng không dùng | RNG streams tách biệt, validation deterministic/sample-weighted, EMA semantics tường minh |
| `COR-08` | S1 | Resume thiếu model/robot/data/protocol identity, nhận LEAP state cho Allegro và âm thầm giữ LR/EMA cũ; actual AMP scaler không được lưu | `resume/v2` exact, validate trước mutation, effective config và scaler được ghi; transfer dùng API khác |
| `COR-09` | S1 | Bundle không gate model/preprocess, `from_bundle` hard-code robot/v1, exact robot gate lại cấm held-out inference | Versioned parser, exact semantic bundle checks và explicit cross-embodiment compatibility binding |
| `COR-10` | S1 | Flow export trace dataclass, stochastic draw và Python token topology; dummy-only tests che lỗi | Tensor-only deterministic export adapter với explicit noise và dynamic-shape parity |
| `COR-11` | S1/S2 | MVP worker bỏ qua checkpoint fingerprint rồi stamp fingerprint hiện tại; demo provenance thiếu content lineage; PPO contract nói bounded nhưng env mới clip | Guard trước episode/report, complete lineage và bounded-action semantics nhất quán |
| `COR-12` | S2 | Zero point padding không có input mask; config có dead/no-op key; packaged legacy còn `exec`, `eval`, unsafe load; lint/type gates không dùng được | Mask/resampling đúng, config key có effect hoặc bị từ chối, legacy bị quarantine/hardening, core static gates sạch |

Các invariant không được giải quyết bằng cảnh báo hoặc lọc im lặng. Sai robot,
split, protocol, schema, fingerprint hoặc artifact identity phải fail trước khi
model/optimizer/env bị mutate hoặc episode/train step đầu tiên chạy.

### 9.3 G0 — Đóng đường chạy sai và khóa regression

**Phạm vi:** `COR-00`…`COR-12`, chưa sửa semantics.

1. Thêm hard-stop ở public train khi canonical audit hoặc positive gate fail.
2. Thêm characterization tests cho mọi repro của audit: malicious shard,
   mixed-hand batch, protocol leakage, placeholder gradient, joint round-trip,
   quality ties, validation RNG, cross-robot resume, semantic bundle mismatch,
   Flow export và foreign MVP fingerprint.
3. Bump dự kiến schema/version; loader cũ phải báo incompatibility rõ thay vì
   load checkpoint như cùng semantics.
4. Không tạo release evidence mới trong giai đoạn này.

**Gate G0:** public command không thể bắt đầu P5 với corpus hiện tại; tất cả
characterization tests fail vì đúng lỗi đã biết trước khi implementation được
sửa, rồi chuyển thành regression pass theo từng giai đoạn sau.

### 9.4 G1 — Safe artifact I/O và một dataset contract duy nhất

**Work package:**

- Tạo helper containment bắt buộc bằng
  `resolved.is_relative_to(root.resolve())`; từ chối absolute path, `..`,
  symlink escape, sibling-prefix và non-regular file.
- Dùng `Literal` cho schema và constrained relative path cho mọi manifest entry.
- Chuyển active dataset/MVP checkpoint load sang `weights_only=True` hoặc format
  tensor/JSON không pickle; xóa production `verify=False`.
- Phân biệt integrity và trust: SHA-256 xác nhận byte không đổi nhưng không xác
  nhận người phát hành. Artifact ngoài repository cần trust policy/signature.
- Viết artifact atomically và không để bundle/manifest nửa chừng trông hợp lệ.
- Hợp nhất sample schema gồm identity, target tensors, validity flags, shape,
  dtype, finiteness và semantic joint order.
- `DatasetArtifact.open_verified()` phải kiểm schema, release flags,
  source/object/robot hashes, path và sample contract trước khi trả dataset.

**Gate G1:**

- Absolute, traversal, sibling-prefix và symlink escape đều bị từ chối.
- Malicious `__reduce__` không thực thi và marker không xuất hiện.
- Sáu DGN shard và checkpoint MVP hiện tại đọc được bằng safe loader.
- Audit, gate, façade và Runner dùng cùng entry point; fixture thiếu tensor train
  không thể pass canonical audit.

### 9.5 G2 — Protocol view, target validity và tái sinh dữ liệu

**Work package:**

- Thay splitter hiện tại bằng group-aware split hoặc coi base physical split là
  non-generalization; không được mô tả stratification là family holdout.
- Family/shape lấy từ object manifest đã hash, không từ prefix object ID.
- Materialize `ProtocolDatasetView` trước Runner; lưu và kiểm
  `dataset_manifest_hash`, `protocol_hash`, `dataset_view_hash`.
- Tách protocol within-hand khỏi held-out embodiment. Với LEAP→Allegro,
  `count(train, wonik_allegro)` phải bằng `0`.
- Khóa ma trận object-per-hand. Positive-control hand-specific phải bị loại khỏi
  cross-hand comparison hoặc có exemption được khai báo và hash.
- Thêm `kinematics_valid`, `pose_target_valid`, `joint_target_valid`,
  `fk_target_valid`; không suy validity từ giá trị zero.
- Flow/pose/joint/FK chỉ dùng sample có target hợp lệ; quality dùng positive và
  negative theo objective đã khóa.
- Sửa generator/split trước, sau đó regenerate DGN từ clean commit hiện tại.

**Gate G2:**

- Canonical audit pass và không còn source drift.
- Physical/logical split không rò family theo claim đã chọn.
- Không có `comp_*` trong logical train view.
- Không sample nào bị lọc im lặng; actual count bằng protocol expected count.
- Proposal-fail không tạo gradient lên flow/pose/joint/FK; thêm placeholder
  negative không đổi pose loss.
- Mỗi declared train hand đạt positive floor. Floor `25` chỉ là input gate,
  không được trình bày như bảo đảm statistical power.

### 9.6 G3 — Robot-aware batch và sửa semantics QDGrasp-Flow

**Work package:**

- Runner dùng một canonical collator giữ `robot_name`, profile hash và ordered
  joint names. Model bound robot assert identity; mixed batch hoặc bị từ chối,
  hoặc được group thành sub-batch dùng đúng HandGraph.
- `encode_target(..., robot)` dùng inverse joint transform:
  `atanh(clamp((q-centre)/half, -1+eps, 1-eps))`.
- Ghi `joint_parameterization` trong model/bundle schema; checkpoint cũ không
  được load như cùng architecture semantics.
- Quality head nhận observation conditioning cùng candidate latent/decoded pose.
  Nhãn target phải gắn với encoded target tương ứng, không với một random draw.
- Bổ sung hard/free-space/on-policy negative cùng observation cho ranking.
- Point cloud ngắn dùng explicit point mask xuyên tokenizer hoặc deterministic
  resampling; zero padding không được tạo voxel thật tại origin.

**Gate G3:**

- Runner-path mixed LEAP/Allegro test raise hoặc group đúng graph.
- `decode(encode(q)) < 1e-5 rad` cho cả hai active hand, kể cả gần limit; gradient
  hữu hạn.
- Giữ observation, đổi candidate làm logit đổi; hoán vị candidate làm score
  hoán vị; overfit được hai candidate cùng object có nhãn trái ngược.
- Prediction bất biến với lượng padding hợp lệ.
- Chạy lại Phase 4 overfit và gradient coverage; evidence mới mang schema/hash
  mới và không so trực tiếp với evidence cũ.

### 9.7 G4 — Runner deterministic, metric đúng và resume/v2

**Work package:**

- Tách generator cho batch order, train sampling, flow noise/time, validation và
  inference. Validation noise cố định theo seed + protocol/view + sample ID,
  độc lập batch grouping.
- Snapshot/restore outer RNG như lớp bảo vệ; trả model về mode trước validation.
- Metric contract trả numerator/count hoặc per-sample tensor; aggregate theo
  sample count và all-reduce cả hai khi distributed.
- Quyết định một EMA contract: khi bật, validation/checkpoint selection/public
  bundle dùng EMA; nếu không hỗ trợ thì bỏ option. Bundle ghi `weights_source`,
  decay và update count.
- `resume/v2` lưu model config/hash, exact training robot, dataset/protocol/view
  digest, optimizer/schedule/effective run config, actual AMP scaler, RNG
  streams, EMA và code/package version.
- Validate identity trước mọi state mutation. `resume` là exact continuation;
  weight transfer dùng `init_from`/API khác.
- Allowlist override như session budget/max steps; ghi old/requested/effective
  value. LR, EMA, batch size, precision hoặc schedule khác phải bị từ chối hoặc
  qua migration tường minh.

**Gate G4:**

- Hai lần validation bit-exact, không đổi outer RNG hoặc model mode.
- Đổi `val_interval` không đổi loss curve/final model.
- Continuous và split+resume Flow run bit-exact trên CPU; CUDA FP32/AMP gate lưu
  và khôi phục scaler thật.
- Metric bất biến theo batch size, kể cả `N % batch_size != 0`.
- Cross-robot, same-size-different-data, semantic-model hoặc protocol mismatch
  fail trước `load_state_dict`; valid resume vẫn bit-exact.
- Khi EMA bật, bundle tensors và validation weights khớp EMA shadow.

### 9.8 G5 — Bundle, cross-embodiment binding và export

**Work package:**

- Parse embedded config bằng versioned schema registry; `robot/v2` round-trip
  không đi qua class `robot/v1` cố định.
- Bundle loader so exact model config/hash và canonical preprocess trước load
  tensor. Runtime result không được gắn live model hash cho source weights có
  semantics khác.
- Tách `training_robot_hash` khỏi `runtime_robot_hash`. Inference transfer chỉ
  qua explicit compatibility binding kiểm graph feature schema, joint capacity/
  mapping, frame, FK/contact capability và protocol permission. Resume vẫn exact.
- Export adapter nhận `points` và explicit initial noise/state; trả tuple tensor
  ổn định gồm translation, rotation, named joints và probability score.
- Tensor hóa tokenizer/export path cho dynamic point/token topology; format chưa
  hỗ trợ phải bị chặn trong capability matrix thay vì rơi vào tracer runtime.

**Gate G5:**

- Flow + robot/v2 save → `from_bundle` → predict round-trip.
- Thay `flow_steps`, voxel size, extent, grasp count hoặc preprocess làm load
  fail trước state mutation.
- LEAP weights chỉ bind Allegro qua explicit compatible path và result ghi cả
  training/runtime robot; incompatible profile fail.
- TorchScript/ONNX eager parity với cùng `points + noise`, nhiều batch/point
  count và cả hai active hand; metadata khớp output thật.

### 9.9 G6 — MVP fingerprint, checkpoint và action contract

**Work package:**

- Compute expected environment fingerprint một lần; truyền vào serial/spawn
  worker, locked evaluation và dev promotion. Validate trước khi tạo env, episode,
  ledger hoặc report.
- Report ghi stored fingerprint, effective fingerprint và match verdict; mismatch
  abort mà không sinh report/ledger có vẻ hợp lệ.
- Checkpoint schema typed, `weights_only=True`, giới hạn shape/size và có parent
  lineage.
- Demonstration manifest hash arrays, ledger, normalizer, BC/PPO config và parent
  checkpoint; cùng summary nhưng khác content phải có digest khác.
- Chọn một action semantics:
  - ưu tiên tanh-squashed Gaussian với Jacobian-correct log-prob; hoặc
  - giữ raw latent Normal nhưng đổi tên/contract, lưu latent+executed action và
    báo clip/saturation rate. Không được gọi distribution là bounded trong khi
    env mới thực hiện clipping.

**Gate G6:**

- Foreign checkpoint fail với `workers=1` và nhiều worker, cả evaluation lẫn
  dev promotion; không có ledger/report sau fail.
- Mutation từng fingerprint/demo field bị phát hiện.
- Unchanged PPO parameters cho ratio `1`; action thực thi hữu hạn/đúng bound và
  log-prob hữu hạn gần boundary.
- MVP round 3 được revalidate bằng guard mới; nếu policy distribution đổi thì
  train/evaluate checkpoint mới thay vì sửa nhãn checkpoint cũ.

### 9.10 G7 — Package boundary, static gates và tái tạo evidence

**Work package:**

- Tách/quarantine legacy namespace khỏi wheel chính, hoặc loại mọi đường
  `exec`, unrestricted `eval`, unsafe checkpoint load và `allow_pickle=True`
  trên input không tin cậy. Optional legacy dependency không được biến thành
  attack surface của base package.
- Mọi config key phải có effect được test hoặc bị từ chối. Các field
  `num_workers`, `pin_memory`, `drop_last`, `seed`, `robot_profiles` không được
  im lặng no-op.
- Cấu hình Ruff/Mypy theo active core và optional/legacy boundary. Sửa semantic
  finding thủ công trước; mechanical import/style cleanup là PR riêng.
- Chạy full suite, docs checker, security tests, CPU/CUDA/AMP gates, real Flow
  export, multi-seed training và held-out evaluation.
- Chỉ sau khi toàn bộ corrective gate pass mới regenerate dataset card, model
  card, run manifests, benchmark/evidence và review packet.

**Gate G7 và điều kiện mở lại release:**

1. `COR-00`…`COR-12` có regression test và owner/verdict rõ.
2. Canonical DGN audit, protocol view audit và positive gate pass.
3. Full suite, active-core Ruff/Mypy, docs checker và package content audit pass.
4. CPU correctness, CUDA FP32/AMP, resume, EMA, export và simulator gates pass.
5. Independent reviewer xác nhận security boundary, protocol isolation,
   artifact identity và evidence disposition.
6. Artifact mới ghi đủ model, training/runtime robot, data, protocol/view,
   preprocess, environment, code và effective run hashes.

### 9.11 Thứ tự PR và dependency

| PR | Nội dung | Phụ thuộc | Không được gộp cùng |
|---|---|---|---|
| `R1` | Characterization tests + hard-stop release/P5 | không | semantic fix |
| `R2` | Safe artifact I/O + unified manifest/sample schema | `R1` | data regeneration |
| `R3` | ProtocolDatasetView + split/hand/object matrix + target masks | `R2` | Flow architecture rewrite |
| `R4` | Robot-aware collator + joint parameterization + candidate quality | `R3` | lint auto-fix |
| `R5` | RNG/metrics/EMA + resume/v2 | `R4` | cross-embodiment transfer |
| `R6` | Bundle parser/compatibility binding + Flow export | `R5` | legacy cleanup |
| `R7` | MVP fingerprint/demo/action remediation | `R2` | unrelated Flow changes |
| `R8` | Regenerate DGN, rerun P4/P5/MVP evidence | `R3`–`R7` | code changes chưa review |
| `R9` | Legacy quarantine + static cleanup + final release packet | `R8` | semantic migrations |

Mỗi PR phải ghi rõ schema migration, artifact bị supersede, negative tests và
rollback. Không dùng một mega-PR vừa đổi semantics, regenerate data, auto-fix
lint và thay evidence vì review sẽ không còn tách được nguyên nhân khỏi kết quả.

### 9.12 Interaction test matrix bắt buộc

Ngoài unit test hiện hữu, CI phải chạy các chuỗi đầy đủ sau:

1. `manifest → verified artifact → protocol view → Runner batch → loss` với
   negative cases cho path, schema, split, robot, object và validity mask.
2. `Flow train → deterministic val → partial resume → bundle` so với continuous
   run trên CPU và CUDA/AMP.
3. `LEAP train bundle → explicit Allegro inference binding → held-out report`,
   đồng thời chứng minh không Allegro sample nào vào train view.
4. `real Flow → TorchScript/ONNX → runtime` trên nhiều point count, cùng explicit
   noise và output schema.
5. `MVP checkpoint → fingerprint guard → spawn workers → ledger/report`, gồm
   checkpoint ngoại lai và demonstration bit mutation.
6. `malicious dataset/checkpoint → public CLI/API`, chứng minh không code được
   thực thi và không partial artifact được công nhận.

Một unit helper pass không được dùng thay interaction test của đúng public path;
đây là nguyên nhân guard mixed-robot, protocol và fingerprint đã tồn tại nhưng
vẫn bị bypass trong audit này.
