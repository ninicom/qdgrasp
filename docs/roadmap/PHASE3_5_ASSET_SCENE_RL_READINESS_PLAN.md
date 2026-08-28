---
document_id: ROADMAP-P3.5-001
document_type: plan
title: Kế hoạch Phase 3.5 — Asset/Scene Ingestion và RL Simulation Readiness
version: 1.2.0
status: active
date: 2026-08-29
revises: ROADMAP-P3.5-001@1.1.0
related_plan: ROADMAP-P3.3-001
depends_on:
  - ROADMAP-P2-001
  - ROADMAP-P3.3-001
optional_dependencies:
  - ROADMAP-P3.4-001
latest_revision_record: docs/revisions/REV-20260829-001-temporary-grasp-policy-mvp.md
execution_priority: deferred_by_ROADMAP-MVP-001
active_hands:
  - leap_hand
  - wonik_allegro
paused_hands:
  - shadow_hand
literature_cutoff: 2026-08-27
gpu_evidence_targets:
  - ninicom/qdgrasp-cuda-kaggle
  - colab-pro-ephemeral
---

# Kế hoạch Phase 3.5 — Asset/Scene Ingestion và RL Simulation Readiness

> **Trạng thái thực thi tạm thời:** contract P3.5 vẫn `active` và còn nguyên giá
> trị dài hạn, nhưng implementation mới được hạ ưu tiên bởi
> [`ROADMAP-MVP-001`](GRASP_POLICY_MVP_TEMPORARY_PLAN.md). MVP chỉ dựng LEAP,
> one-object table environment và state-based policy; nó không làm P3.5 pass.
> Sau MVP, raw mesh/CoACD, scene load/generate/drop, two-hand vector environment,
> backend parity và independent review tiếp tục từ tài liệu này.

P3.5 biến object/scene artifact của P2–P3.3 thành một môi trường mô phỏng có thể
reset/step theo contract học tăng cường. Đầu vào có thể là scene hoàn chỉnh,
object manifest đã chuẩn hóa hoặc chỉ một raw mesh. Nếu không có scene, hệ thống
tạo một scene ảo có support và spawn region hữu hạn, thả vật bằng physics, chờ
settle và lưu lại trạng thái khởi tạo có thể tái lập.

Theo `ADR-0008`, compatibility/release gate hiện dùng 2/2 active hand: LEAP và
Wonik Allegro. Shadow preset/evidence vẫn được giữ nhưng không được load trong
default environment, backend spike hoặc RL readiness dataset cho tới quyết định
mở lại.

P3.5 **không viết lại P3.3**. `SceneSpec`, native scene builder, MuJoCo settle và
asset provenance của P3.3 là baseline/oracle. Phần mới của P3.5 là:

- ingest raw mesh thành visual/collision/physics asset có manifest;
- public Python API CoACD chạy local trong thư viện, được thiết kế mới trên
  upstream CoACD đầy đủ; `kaggle_pipeline_modular/stage0_coacd` là reference và
  regression fixture khi người dùng chỉ có raw mesh;
- lựa chọn “load scene nếu có, tạo virtual drop scene nếu không có”;
- vectorized RL environment contract, deterministic reset và domain randomization;
- backend GPU có gate tương thích thật trên Kaggle/Colab, không giả GPU;
- checkpoint/evidence chịu được runtime notebook bị thu hồi.

P3.5 chuẩn bị simulation-in-the-loop cho P5. Nó không đổi topology QDGrasp-Flow
của P4 và không phải điều kiện chặn việc bắt đầu P4 static/offline sau khi
interface dataset đã khóa. Theo governance hiện tại, P3 tổng chỉ được ghi
`complete` khi mọi subphase được tuyên bố bắt buộc đã đóng hoặc có revision đổi
dependency rõ ràng.

## 1. Câu hỏi thiết kế và kết luận nghiên cứu

### 1.1 Có nên chọn Isaac Sim/Isaac Lab ngay không?

Chưa. Isaac Lab mạnh cho RL song song, sensor/RTX và scene USD phức tạp, nhưng
full Isaac workflow hiện yêu cầu một stack nặng hơn notebook tạm thời. Tài liệu
Isaac Lab khuyến nghị Ubuntu/Linux workstation, tối thiểu khoảng 32 GB RAM và
16 GB VRAM; container cần Docker và NVIDIA Container Toolkit. Asset còn cần
cache để tránh tải lại. Đây là lựa chọn tốt cho workstation/cloud GPU persistent,
không phải baseline đáng tin cậy cho một Kaggle/Colab runtime bất kỳ.

Nguồn:

- [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/)
- [Isaac Lab Docker deployment](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html)
- [Isaac Lab RL performance](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)

Không dùng Isaac Gym cho code mới: NVIDIA đã đánh dấu đây là legacy và hướng
người dùng sang Isaac Lab.

Nguồn: [NVIDIA Isaac Gym archive](https://developer.nvidia.com/isaac-gym/download).

### 1.2 Backend phù hợp với QDGrasp hiện tại

Quyết định tạm thời là một kiến trúc nhiều tầng, nhưng chỉ một backend GPU được
chọn làm release target sau compatibility spike:

| Vai trò | Backend | Quyết định |
| --- | --- | --- |
| Oracle/replay | MuJoCo CPU hiện có | Bắt buộc; giữ cùng MJCF, contact semantics và fixture P2/P3 |
| GPU candidate A | MJX-Warp/MuJoCo Warp | Thử đầu tiên; cùng hệ MuJoCo, cài bằng wheel, hợp với notebook và batched RL |
| GPU candidate B | ManiSkill 3/SAPIEN GPU | Fallback nếu A fail; Gymnasium/PyTorch-native, pip-install, mạnh cho manipulation và heterogeneous scene |
| Backend mở rộng | Isaac Lab | Không chặn P3.5; chỉ mở khi có môi trường persistent qua preflight và có parity artifact |
| Watchlist | Genesis | Không vào gate v1 trước khi có parity/contact/provenance evidence cho hai active hand |

MJX-Warp là candidate ưu tiên vì tài liệu MuJoCo mô tả đây là implementation GPU
đầy đủ tính năng nhất, hỗ trợ batched model/data và tối ưu contact/constraint.
Catalog MuJoCo Menagerie hiện đánh dấu LEAP, Wonik Allegro và Shadow Hand là chưa
được chứng nhận `MJX`. Gate hiện chỉ spike LEAP/Allegro; Shadow finding được giữ
cho resumption nhưng không chặn backend decision khi ADR-0008 còn hiệu lực.

Nguồn:

- [MuJoCo XLA/MJX](https://mujoco.readthedocs.io/en/latest/mjx.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)

ManiSkill 3 là fallback hợp lý vì hỗ trợ CPU/GPU simulation, Gymnasium API,
custom task, URDF/MJCF và GPU-parallel manipulation. Nhưng MJCF importer của nó
không bảo toàn mọi motor/solver/collision attribute, nên cũng không được chọn
trước khi LEAP/Allegro import-parity pass. Shadow import-parity được defer rõ
theo ADR-0008, không được ghi pass.

Nguồn:

- [ManiSkill quickstart](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)
- [ManiSkill installation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [ManiSkill custom robots](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html)

### 1.3 Kaggle và Colab Pro

Kaggle là target evidence chính vì dự án đã có CUDA harness riêng. Colab Pro là
portability target và đường chạy bổ sung, không phải nơi lưu artifact duy nhất.
Không pin tên GPU theo subscription: Colab công bố loại GPU, runtime lifetime và
quota có thể thay đổi. Mỗi notebook phải preflight phần cứng thật và fail-closed
nếu backend thực tế là CPU.

Nguồn: [Google Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits).

## 2. Phạm vi và ngoài phạm vi

### 2.1 Thuộc P3.5

- Nạp object từ `ObjectAssetManifest` hoặc raw `.obj`, `.ply`, `.stl`, `.glb`.
- Tách visual mesh và collision representation; mọi scale cuối dùng mét.
- Triangulate/normalize có provenance; không tự sửa asset gốc tại chỗ.
- Public Python API CoACD local-first, typed result và content-addressed cache.
- Nạp canonical `SceneSpec`/MJCF scene; adapter USD chỉ là optional extension.
- Khi thiếu scene, tạo table/tray/bin virtual scene và bounded spawn region.
- Thả một hoặc nhiều object bằng gravity; certifier quyết định settled/invalid.
- Gymnasium-compatible single/vector environment cho state-based RL trước.
- Observation/action/reward/termination/info contracts độc lập simulator.
- Deterministic reset, topology bucketing, domain randomization và snapshot.
- MuJoCo CPU oracle cùng một GPU backend được chọn sau compatibility spike.
- Kaggle GPU evidence và Colab portability smoke có checkpoint/resume.

### 2.2 Không thuộc P3.5

- Không train hoặc công bố policy SOTA; P3.5 chỉ chứng minh environment ready.
- Không đổi P4 thành trajectory policy hoặc thay đổi model architecture.
- Không đồng nhất kết quả physics giữa engine bằng bit-exact dài hạn.
- Không có HTTP/service/external CoACD API và không auto-upload mesh ra ngoài.
- Không đoán unit, mass hoặc license để làm asset “pass”.
- Không yêu cầu photorealistic RTX/ray tracing cho gate state-based v1.
- Không dùng reward tuning trên test scene để tạo learning claim.
- Không đưa Isaac Lab, ManiSkill hay MJX vào dependency mặc định trước decision
  record và optional environment lock riêng.

## 3. Contract ingest object

### 3.1 `AssetIngestRequest`

```text
AssetIngestRequest
  source:
    manifest_ref? | local_mesh_path? | mesh_bytes?
  input_sha256
  source_format
  units: m | cm | mm | explicit_scale
  scale_to_meters
  up_axis, forward_axis, source_frame
  physics:
    mass? | density?
    center_of_mass?
    inertia?
    friction?, restitution?
  collision_policy:
    existing | convex_if_possible | coacd
  license_record
  redistributable
```

Đúng một source phải có. `scale_to_meters`, hash và license là bắt buộc trước
release. Nếu không có mass nhưng có density đã khóa, mass/inertia được tính từ
collision volume và ghi rõ là derived; nếu cả hai thiếu thì ingest chỉ có thể ở
trạng thái `geometry_ready`, chưa được spawn thành dynamic object.

### 3.2 Pipeline chuẩn hóa

1. Resolve source trong sandbox/root cho phép; không nhận path traversal.
2. Hash raw byte trước mọi transform.
3. Load scene/mesh, flatten transform có kiểm soát và triangulate.
4. Loại NaN/Inf, degenerate face, duplicate vertex theo config đã hash.
5. Kiểm bounds, triangle count, connected components và unit scale.
6. Giữ visual mesh riêng; collision mesh không thay visual artifact.
7. Nếu collision primitive/convex có sẵn và pass thì tái sử dụng.
8. Nếu raw mesh không dùng được cho dynamic collision, gọi public Python API
   `decompose_collision_mesh()` ngay trong process/library hoặc worker local.
9. API trả typed result/error; không gọi network và không tự ghi vào dataset.
10. Validate convex parts, mass/inertia, simulator compile và drop micro-fixture.
11. Gọi writer riêng để ghi immutable `ObjectAssetManifestV2` với mọi
   input/output/config/tool/source hash.

Không “repair cho tới khi pass” bằng heuristic ẩn. Mỗi transform tạo derivative
record và asset nguồn giữ nguyên byte.

## 4. Public Python API CoACD trong thư viện

CoACD hỗ trợ Python wheel, nhận triangle mesh và trả danh sách convex hull.
Phiên bản hiện tại có real-metric mode, phù hợp threshold theo mét. License MIT
phải được ghi trong third-party manifest.

Nguồn upstream: [CoACD official repository](https://github.com/SarahWeiii/CoACD)
và [official Python signature](https://github.com/SarahWeiii/CoACD/blob/main/python/package/__init__.py),
đã audit tại commit `c5f4b8368b47c4f10023b1b09a1cbea98e4c1249`
(version `1.0.13`, 2026-08-19). Execution lock vẫn phải pin exact wheel/source
được compatibility test, không tự động trôi theo `main`.

### 4.1 Prior implementation được maintainer chỉ định làm reference

Source tham chiếu nghiệp vụ là private repository
[`ninicom/kaggle_pipeline_modular`](https://github.com/ninicom/kaggle_pipeline_modular),
pin tại commit `d929c175c1bde68f5acd059fbc6f93bec0c03328`, thư mục
`stage0_coacd/source/Grasping/dexonomy/scripts/object_setup/`.

Đây là Stage 0 cơ bản và không khóa implementation P3.5. Các bài học/contract
cần giữ, không đồng nghĩa copy nguyên code:

- `remesh_watertight()` — gọi ManifoldPlus để vá mesh thành watertight;
- `simplify()` — giảm lưới về target face budget;
- `run_coacd_standard()` — normalize bbox diagonal, CoACD seed cố định và restore
  metric scale;
- `piece_inertial()` — tính mass/COM/inertia cho convex piece theo density;
- `prepare()` — cho thấy orchestration một object, nhưng side effect phải tách
  khỏi core API mới;
- `convert_raw_object.py` — contract xuất convex pieces/MJCF/URDF/info, chỉ dùng
  làm migration reference, không giữ layout Dexonomy làm public contract.

`coacd_explorer.py` có route Flask `/api/decompose`, nhưng đó là visualization
tool cục bộ để so threshold. P3.5 **không** biến route này thành production API,
không mở port và không gửi mesh ra service khác.

Có một khác biệt provenance phải khóa bằng named profile:

- Stage 0 artifact thực tế gọi `--threshold 0.1`, seed `0`, simplify `5000`;
- `prepare_graspnet_objects.py` đặt default/commented standard là threshold
  `0.4`, cùng normalize bbox diagonal `2 m`, seed `0`.

Không được ghi một số là “default cũ” mơ hồ. Evidence tái lập Stage 0 phải dùng
profile `legacy_kaggle_stage0_0_1_v1`; profile
`legacy_dexgraspnet_standard_0_4_v1` là lựa chọn khác và không được dùng để tái
tạo Stage 0 artifact.

Mặc định P3.5 viết implementation mới dựa trên official CoACD Python contract;
prior Stage 0 chỉ dùng để định nghĩa named legacy profiles và regression cases.
Nếu implementation thực tế copy bất kỳ đoạn code nào từ private repository,
revision phải ghi exact source commit, authorship/permission và license
provenance. ManifoldPlus/CoACD giữ notice/license riêng; không copy third-party
tree vào package mà thiếu source/license record.

### 4.2 Public façade

API người dùng dự kiến:

```python
from qdgrasp.objects import (
    CoACDConfig,
    CoACDResult,
    decompose_collision_mesh,
    build_collision_asset,
)

result: CoACDResult = decompose_collision_mesh(
    mesh,
    config=CoACDConfig.profile("legacy_kaggle_stage0_0_1_v1"),
)

asset = build_collision_asset(
    mesh,
    mass_kg=0.2,  # giá trị được người dùng cung cấp, không phải default
    config=CoACDConfig.profile("legacy_kaggle_stage0_0_1_v1"),
)
```

Contract:

```text
decompose_collision_mesh(mesh_or_path, *, config, cache_dir?) -> CoACDResult
build_collision_asset(mesh_or_path, *, mass_kg?|density_kg_m3?, config,
                      cache_dir?) -> CollisionAsset

CoACDConfig
  profile: upstream_default_v1 | qdgrasp_metric_v1
           | legacy_kaggle_stage0_0_1_v1
           | legacy_dexgraspnet_standard_0_4_v1 | custom
  mesh_preprocess: MeshPreprocessConfig
  algorithm: CoACDAlgorithmConfig
  execution: CoACDExecutionConfig

MeshPreprocessConfig
  unit_scale_to_meters
  triangulate
  repair_mode: none | manifoldplus
  manifold_depth
  simplify_faces?
  normalize_diagonal_m?

CoACDAlgorithmConfig  # expose toàn bộ official run_coacd parameters
  threshold
  max_convex_hull
  preprocess_mode: auto | on | off
  preprocess_resolution
  resolution
  mcts_nodes
  mcts_iterations
  mcts_max_depth
  pca
  merge
  decimate
  max_ch_vertex
  extrude
  extrude_margin
  apx_mode: ch | box
  seed
  real_metric

CoACDExecutionConfig  # wrapper-level, không truyền nhầm vào upstream
  log_level
  timeout_s?
  max_input_vertices
  max_input_faces
  max_output_parts
  max_workers

CoACDResult
  prepared_mesh
  convex_parts[]
  input/prepared/part hashes
  source_to_metric_transform
  tool/build/config/profile hashes
  piece_count, volume, elapsed_s, warnings
```

`decompose_collision_mesh()` không tự ghi file, không tự chọn mass và không gọi
network. `build_collision_asset()` chỉ tạo typed in-memory asset; writer/manifest
layer quyết định nơi ghi. Batch helper có thể dùng local `ProcessPoolExecutor`
như Stage 0, nhưng public single-object API vẫn chạy được CPU với một lời gọi.

Không dùng `**kwargs` xuyên thẳng tới CoACD. Mọi official parameter phải có
typed field, range/enum validation và xuất hiện trong config hash. Parameter mới
ở upstream phải làm schema/version thay đổi hoặc bị báo unsupported; không được
silently ignore. `real_metric=true` định nghĩa threshold theo mét và không được
kết hợp với legacy diagonal normalization nếu chưa có rule rõ ràng.

`mass_kg` và `density_kg_m3` loại trừ nhau. Không dùng default `0.2 kg` của batch
script cũ cho object tùy ý: thiếu cả hai thì chỉ trả geometry result, chưa tạo
dynamic physics asset.

Config thực tế phải bám API của exact CoACD release; field không được hỗ trợ bị
từ chối, không silently ignore. Cache key:

```text
sha256(input_bytes || source_to_metric_transform || preprocessing_config
       || coacd_config || tool_build_digest || platform_tag)
```

Typed errors tối thiểu: `MeshValidationError`, `MeshRepairUnavailable`,
`CoACDExecutionError`, `TooManyConvexPartsError`, `CollisionValidationError`.
Lỗi không được trả empty/single-hull giả để pipeline tiếp tục.

## 5. Scene selection và virtual drop scene

### 5.1 Quy tắc chọn scene

```text
resolve_scene(scene_ref?, object_refs[], virtual_scene_config?)
  scene_ref hợp lệ   -> load + validate + compile
  scene_ref bị hỏng  -> fail; không tự thay bằng scene ảo
  scene_ref không có -> generate virtual drop scene
```

Phân biệt “không cung cấp scene” với “scene được cung cấp nhưng lỗi” ngăn việc
silently đổi bài toán. Scene load v1 ưu tiên canonical `SceneSpec` của P3.3 và
MJCF. USD/USDA/USDZ chỉ mở bằng adapter riêng khi Isaac backend được duyệt.

### 5.2 `VirtualDropSceneSpec`

```text
VirtualDropSceneSpec
  environment: table | tray | bin
  support_geometry, support_pose, support_material
  spawn_region: aabb | obb
  drop_height_range_m
  object_count_range
  initial_clearance_m
  gravity, timestep, solver_profile
  boundary_policy: reject | contain
  settle_thresholds:
    linear_velocity_mps
    angular_velocity_radps
    kinetic_energy_j
    pose_delta_m_rad
    consecutive_steps
    timeout_steps
  seed_streams:
    asset, layout, orientation, drop, material, observation
```

Scene ảo v1 có support hữu hạn và vùng spawn được render/debug rõ. Object được
đặt ở trạng thái ban đầu không overlap, sau đó chỉ chuyển động do gravity/contact;
không ghi qpos để ép settle. Table là baseline; tray/bin thêm sau khi table gate
pass để tránh che lỗi bằng thành container.

### 5.3 Settle certifier

Một scene chỉ `settled=true` khi mọi dynamic object đồng thời thỏa velocity,
kinetic-energy và pose-delta thresholds trong số bước liên tiếp đã pin. Các lớp
fail gồm:

- `initial_overlap`;
- `non_finite_state`;
- `solver_warning` hoặc contact-buffer overflow;
- `escaped_spawn_region`/`fell_off_support`;
- `excessive_penetration` hoặc tunneling;
- `settle_timeout`;
- `backend_divergence` khi CPU replay đổi validity class.

Output là một `SceneSnapshot` immutable: exact object poses/velocities, contact
summary, seed/config/backend/tool hashes và settle trace rút gọn. Snapshot được
cache ngoài training loop; reset có thể chọn replay snapshot hoặc re-drop theo
curriculum, và luôn ghi mode vào `info`.

## 6. RL environment contract

### 6.1 Gymnasium boundary

P3.5 tạo environment registry tối thiểu:

- `QDGrasp-ObjectSettle-v0`: debug ingest/drop/settle, không dùng để claim grasp;
- `QDGrasp-DexAcquire-v0`: hand tiếp cận, đóng ngón, giữ và lift target;
- `QDGrasp-DexAcquireScene-v0`: scene clutter có target/non-target accounting.

Mọi environment có:

```text
reset(seed, options) -> observation, info
step(action) -> observation, reward, terminated, truncated, info
```

Single và vector mode cùng semantics. `terminated` chỉ biểu diễn success/failure
của task; hết horizon/compute budget là `truncated`.

### 6.2 Observation

State-based v1:

- named joint position/velocity/effort cùng joint mask;
- palm pose/twist và active HandGraph/profile ID;
- target pose/twist, relative palm-target transform;
- support/non-target relative state;
- contact summary theo link/object/contact class;
- previous action, episode phase và time remaining;
- asset/scene/snapshot IDs trong `info`, không nhét string vào policy tensor.

Optional observation `pointcloud`/`rgbd` chỉ mở sau state-based gate; renderer và
physics timing báo riêng.

### 6.3 Action

Action canonical không phụ thuộc engine:

```text
ActionSpec
  palm_command?: delta_pose_6d | twist_6d | fixed
  joint_command: named_position_target | named_delta_target
  active_joint_mask
  control_dt
```

Controller adapter áp limits, coupling/tendon policy và action latency. Không
cho policy ghi trực tiếp object pose hoặc simulator internal state.

### 6.4 Reward và outcome

Reward chỉ là tổng của các term được log riêng:

```text
reach + contact_progress + enclosure + lift + retention
- penetration - unsafe_impulse - non_target_disturbance
- action_rate - joint_limit - drop
```

Hard safety/validity barrier không được bù bằng reward dương. `info` phải chứa
raw metric dùng để tính reward và terminal reason. P3.5 gate chỉ yêu cầu scripted
fixture/random-policy stability và một tiny learnability smoke; không claim
general policy success.

### 6.5 Reset, randomization và topology

- Tách seed stream cho asset, scene, drop, physics, observation noise và policy.
- Randomize mass/friction/scale trong range preregistered và giữ manifest gốc.
- Scene/robot có topology khác nhau được bucket theo `scene_signature`; không
  recompile mỗi environment step.
- Snapshot/reset không được mang contact/solver warm state từ episode trước.
- Evaluation split khóa object family, scene template và seed; training không
  đọc test reward statistics để đổi threshold.

## 7. Simulator parity và backend decision gate

### 7.1 CPU oracle

MuJoCo CPU tiếp tục là semantic oracle vì đã qua P2/P3.3 trong lịch sử. Active
gate hiện replay LEAP và Allegro. Oracle
kiểm model compile, named joint/transmission, passive force, actuator response,
free-object drop, contact class, settle và scripted grasp/lift.

### 7.2 MJX-Warp spike — candidate A

Spike phải kiểm riêng LEAP và Allegro:

1. `put_model`/compile không unsupported feature;
2. joint/actuator/tendon counts và named mapping khớp CPU;
3. reset state nằm trong limits;
4. no-contact horizon parity;
5. single-contact/drop outcome parity;
6. squeeze/lift scripted outcome class parity;
7. contact sensor không overflow;
8. batched step thật trên CUDA và nhanh hơn CPU sau warmup;
9. deterministic seed/accounting ở cùng backend;
10. memory/compile time nằm trong budget notebook.

Candidate A chỉ thắng nếu **2/2 active hand** pass. Shadow mang trạng thái
`paused_by_ADR-0008`, không được ghi như một pass; vẫn cấm fallback CPU im lặng.

### 7.3 ManiSkill spike — candidate B

Chỉ chạy nếu A fail hoặc thiếu feature cần thiết. Spike dùng derived/import
artifact có provenance và kiểm:

- link/joint/actuator/coupling mapping;
- visual/collision transform và self-collision exclusions;
- tendon/passive behavior hoặc explicit approximation record;
- contact/drop/lift outcome class với MuJoCo CPU;
- GPU vectorization/headless install trên notebook.

Nếu một hand cần approximation làm đổi task semantics, ManiSkill không thể làm
release backend v1; khi đó P3.5 giữ MuJoCo CPU và báo GPU blocker.

### 7.4 Isaac Lab optional spike

Isaac Lab không nằm trong gate đóng P3.5 v1. Một revision sau có thể thêm USD
export/import, PhysX material/contact parity, tiled rendering và large-scale RL
khi preflight đạt yêu cầu hệ thống. Không cài full Isaac Sim vào Kaggle/Colab
chỉ để có tên backend trong report.

## 8. Kaggle/Colab deployment contract

### 8.1 Preflight bắt buộc

Notebook ghi vào evidence trước khi cài/chạy:

- platform/runtime ID không chứa secret;
- GPU name, compute capability, driver, CUDA và VRAM;
- CPU/RAM/disk còn trống;
- Python/PyTorch/JAX/Warp/MuJoCo hoặc ManiSkill/SAPIEN versions;
- khả năng import, device allocation và một GPU kernel thật;
- renderer/Vulkan capability tách khỏi state-physics capability;
- repo commit, environment lock và asset bundle hash.

`requested_backend=gpu` mà tensor/simulator state nằm CPU phải thoát nonzero.

### 8.2 Cài đặt tạm thời

- Dùng optional lock riêng (`phase3_5-mjx-warp` hoặc `phase3_5-maniskill`), không
  sửa base environment cho tới decision record.
- Cài exact commit/wheel versions, không `latest`/nightly trong evidence run.
- Cache wheel và immutable asset bundle ở Kaggle Dataset/Google Drive hoặc object
  store; cache không phải source of truth nếu thiếu checksum.
- Không phụ thuộc Docker/root cho notebook baseline.
- Headless state-based run trước; render vài QA frame ở batch nhỏ sau.

### 8.3 Chịu được runtime bị thu hồi

- Checkpoint policy/optimizer/RNG/environment counters atomically theo interval.
- Evidence JSONL flush theo episode/batch; index cuối có hash toàn bundle.
- Resume xác minh exact code/config/asset/backend hashes trước khi tiếp tục.
- Không lưu hàng nghìn file nhỏ trên Drive; shard bundle và upload theo chunk.
- Timeout handler xuất checkpoint/evidence trước khi notebook dừng nếu còn thời gian.

Kaggle cần một fail-closed GPU smoke cho release. Colab Pro cần một portability
smoke hoặc record `not_run` kèm lý do; Colab không được dùng làm bằng chứng về
một loại GPU cố định.

## 9. Work breakdown

| ID | Module | Output chính | Phụ thuộc |
| --- | --- | --- | --- |
| P3.5-00 | Khóa scope/backend decision protocol | contracts + revision | P2/P3.3 |
| P3.5-01 | Asset ingest/schema inventory | compatibility matrix | 00 |
| P3.5-02 | Raw mesh normalization/validation | `assets/ingest.py` | 01 |
| P3.5-03 | Viết mới full-parameter CoACD core + legacy profiles | `qdgrasp/objects/coacd.py` | 02 |
| P3.5-04 | Public Python façade + typed results/errors/cache | `qdgrasp.objects` exports | 03 |
| P3.5-05 | `ObjectAssetManifestV2` writer/loader | asset modules | 02/04 |
| P3.5-06 | Scene resolver/load adapters | `scenes/resolver.py` | 05/P3.3 |
| P3.5-07 | Virtual drop scene builder | `scenes/virtual_drop.py` | 06 |
| P3.5-08 | Settle certifier/snapshot cache | `scenes/settle.py` | 07 |
| P3.5-09 | Simulator-neutral RL protocol | `rl/contracts.py` | 08/P2 |
| P3.5-10 | Gymnasium environments | `rl/envs/` | 09 |
| P3.5-11 | Observation/action/reward adapters | `rl/tasks/` | 10 |
| P3.5-12 | Reset/randomization/topology buckets | `rl/randomization.py` | 10/11 |
| P3.5-13 | MJX-Warp 3-hand compatibility spike | report + raw evidence | 09/P3.4 optional |
| P3.5-14 | ManiSkill fallback compatibility spike | report + derived assets | 13 fail/defer |
| P3.5-15 | Backend decision record + optional lock | ADR/revision | 13 or 14 |
| P3.5-16 | Kaggle/Colab harness + checkpoint/resume | notebook/scripts/evidence | 15 |
| P3.5-17 | Tiny RL readiness artifact/audit | `QDGrasp-RL-Env-Tiny` | 05/08/12/16 |
| P3.5-18 | Independent review và roadmap handoff | review/revision/gates | 17 |

Critical path:

```text
00 → 01 → 02 → (03 → 04) → 05 → 06 → 07 → 08
   → 09 → 10 → 11 → 12 → 13 → [14 nếu cần] → 15 → 16 → 17 → 18
```

P3.5-04 chặn đóng P3.5: user-facing CoACD capability phải gọi được bằng import
trong library, không chỉ tồn tại dưới script/Kaggle stage. Isaac Lab spike không
nằm critical path.

## 10. Test matrix bắt buộc

### Asset/CoACD

- Raw byte/input hash và normalized output deterministic.
- Missing/ambiguous unit fail; scale không bị áp hai lần.
- Non-triangle source được triangulate có record; NaN/degenerate/oversized reject.
- Convex input không bị decomposition thừa.
- CoACD parts convex, finite, đúng bounds/volume tolerance và compile cả backend.
- `legacy_kaggle_stage0_0_1_v1` parity fixture khóa threshold `0.1`, seed `0`, simplify
  `5000`, normalize diagonal `2 m` và output class với prior Stage 0.
- `legacy_dexgraspnet_standard_0_4_v1`, `qdgrasp_metric_v1` và
  `upstream_default_v1` không bị trộn cache/profile.
- Toàn bộ official parameters (`max_convex_hull`, preprocess/resolution/MCTS,
  PCA/merge/decimate, vertex cap, extrude, approximation mode, seed và
  real-metric) round-trip config/hash và đến đúng upstream argument.
- Invalid enum/range và tổ hợp `real_metric`/legacy normalization mâu thuẫn bị reject.
- Public function không network/HTTP, không implicit dataset write và trả typed error.
- Path/bytes input, part/triangle/time/resource limits fail closed.
- Thiếu mass/density vẫn trả geometry result nhưng không tạo dynamic asset.

### Scene/drop/settle

- Có scene hợp lệ thì load, không tạo scene ảo.
- Scene được cung cấp nhưng lỗi phải fail, không fallback.
- Không scene thì cùng seed tạo cùng support/spawn/initial state.
- Object chỉ rơi do physics sau initial placement, không teleport khi settle.
- Falling off, timeout, NaN, tunneling và penetration có reason riêng.
- Snapshot replay giữ outcome class và không giữ stale contact state.
- Multi-object spawn không initial-overlap; target/non-target IDs ổn định.

### RL contract

- Gymnasium API/space/dtype/device/batch shape pass cho 3 hand.
- Named-joint mask/coupling không đổi theo backend.
- `terminated`/`truncated` tách đúng; terminal reason duy nhất.
- Reward total bằng tổng logged terms; hard barrier không được bù.
- Reset seed streams độc lập; evaluation split không leak.
- Random policy không tạo NaN/Inf; scripted acquire/lift fixture có expected class.
- Tiny PPO/learner smoke cho reward tăng trên trivial fixture, không gọi là model
  quality hoặc generalization evidence.

### CPU/GPU/cloud

- MuJoCo CPU oracle pass trước GPU candidate.
- GPU request không thể fallback CPU.
- 2/2 active hand compile/step/contact/drop/lift parity theo tolerance/class;
  Shadow ghi `paused_by_ADR-0008`.
- Compile/warmup báo riêng khỏi steady-state throughput.
- Contact buffer/memory cap overflow là failure.
- Kaggle exact-commit evidence; Colab resume/portability smoke.
- Runtime restart đọc lại checkpoint chỉ khi hash khớp.

## 11. Resource, security và provenance policy

- Local dry-run báo input bytes/triangles, estimated CoACD cost, convex-part cap,
  worlds × steps và peak memory trước execute.
- CoACD function mặc định giới hạn thread; batch worker giới hạn CPU/RAM/time.
- Không render đồng thời peak physics batch trên GPU VRAM thấp.
- Asset/source license đi cùng derivative; prior Stage 0 source commit được ghi
  như reference/regression source. Chỉ thêm port provenance/NOTICE nếu có copy.
- Secret/token chỉ lấy từ notebook secret store, không vào output/log/checkpoint.
- External artifact dùng content hash; URL không đủ làm identity.
- Mọi benchmark ghi cold-start, compile, warmup và steady-state riêng.
- Không chọn simulator theo FPS duy nhất: 3-hand semantics và contact parity đứng
  trước throughput.

## 12. Gate đóng P3.5

Các lệnh dự kiến sau implementation:

```bash
.venv/bin/python scripts/check_docs.py --root .
.venv/bin/python scripts/check_phase3_5.py --backend mujoco-cpu --profile micro
.venv/bin/python -m pytest tests/assets_ingest/ tests/rl_env/ -q
git diff --check
```

GPU gate chạy trong external notebook harness bằng backend đã được chọn:

```bash
python scripts/phase3_5_gpu_rl_readiness.py \
  --backend <mjx-warp|maniskill-gpu> \
  --device cuda:0 \
  --profile notebook-micro \
  --evidence phase3_5_gpu_evidence.json
```

Không đóng P3.5 chỉ bằng việc import simulator, render được một ảnh hoặc train
được một hand. Gate yêu cầu asset → scene/drop → settle → reset/step → scripted
outcome xuyên suốt cho LEAP và Allegro; manifest ghi Shadow paused.

## 13. Điều kiện hoàn tất

1. Raw mesh hợp lệ được chuyển thành visual/collision/physics asset immutable;
   unit, mass/inertia, license, config/tool và hash đầy đủ.
2. Public Python CoACD API mới expose/validate/hash toàn bộ official parameters,
   có legacy Stage 0 profile parity, typed result/error, local cache, không
   network và gọi được trực tiếp từ installed wheel.
3. Scene resolver phân biệt load/fail/generate đúng, và virtual object drop/settle
   không teleport, deterministic theo seed.
4. `SceneSnapshot` replay và MuJoCo CPU oracle xác nhận validity class.
5. Gymnasium single/vector environment, observation/action/reward/termination,
   reset/randomization contracts pass cho hai active hand.
6. Một backend GPU duy nhất được chọn bằng decision record sau 2-hand parity;
   không có silent fallback hoặc approximation không khai báo.
7. Kaggle GPU evidence exact-commit pass; Colab Pro portability/resume được chạy
   hoặc ghi rõ `not_run` mà không biến thành release claim.
8. `QDGrasp-RL-Env-Tiny` có object-only, loaded-scene và generated-scene cases,
   positive scripted/negative/random fixtures và raw evidence hashes.
9. Independent reviewer kiểm asset transforms/units, CoACD/security, settle
   semantics, Gym API, reward accounting, backend parity và cloud evidence.

P3.5 kết thúc ở trạng thái **RL environment ready**, không phải “RL policy đã
học tốt”. Policy training, benchmark và simulator-in-the-loop evaluation thuộc
P5 hoặc một revision model/training riêng.
