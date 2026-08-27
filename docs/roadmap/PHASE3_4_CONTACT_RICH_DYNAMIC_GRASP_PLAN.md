---
document_id: ROADMAP-P3.4-001
document_type: plan
title: Kế hoạch Phase 3.4 — Contact-Rich Dynamic Grasp Synthesis
version: 1.3.0
status: active
date: 2026-08-27
revises: ROADMAP-P3.4-001@1.2.0
related_plan: ROADMAP-P3.3-001
depends_on:
  - ROADMAP-P3.1-001
  - ROADMAP-P3.2-001
  - ROADMAP-P3.3-001
latest_revision_record: docs/revisions/REV-20260827-009-temporary-shadow-hand-pause.md
execution_state: paused_by_ADR-0008
literature_cutoff: 2026-08-27
gpu_evidence_target: ninicom/qdgrasp-cuda-kaggle
---

# Kế hoạch Phase 3.4 — Contact-Rich Dynamic Grasp Synthesis

P3.4 bổ sung một nhánh sinh dữ liệu khác P3.3: grasp không còn là một pose tĩnh
được đặt vào scene rồi kiểm tra, mà là **một trajectory được sinh trực tiếp trong
physics scene**. Bàn tay được phép đẩy, trượt, lăn, pivot hoặc kẹp target trong
khi target đang dịch chuyển; contact nhẹ với bàn/support hoặc object lân cận
được chấp nhận nếu không vượt safety budget.

Mục tiêu là tìm được các grasp thực tế mà static collision filter sẽ loại bỏ,
đặc biệt với object nằm sát bàn, sát thành hộp hoặc bị che bởi clutter. Chi phí
tìm kiếm cao hơn nhiều vì mỗi candidate là một rollout nhiều bước. CPU vẫn là
correctness backend và có thể sinh dữ liệu với batch nhỏ; NVIDIA GPU là backend
throughput bắt buộc cho scale generation và phải được xác minh trên Kaggle theo
quy trình fail-closed của Phase 1.

## 1. Vấn đề và giả thuyết cần kiểm chứng

### 1.1 Giới hạn của static grasp

Static pipeline thường giả định:

1. scene/object không đổi trong lúc bàn tay approach;
2. mọi contact với bàn hoặc non-target object là collision phải reject;
3. target chỉ bắt đầu chuyển động sau khi grasp đã đóng;
4. chất lượng được đánh giá ở một endpoint pose.

Các giả định này bỏ sót extrinsic dexterity. Trong thực tế, bàn có thể làm điểm
tựa để pivot một vật mỏng, ngón có thể slide vật ra khỏi góc, hoặc contact nhẹ
với object bên cạnh có thể tạo khoảng trống trước khi enclosure hoàn tất.

### 1.2 Giả thuyết P3.4

Tại cùng object/scene split và cùng safety budget:

- dynamic contact-rich search sẽ tạo thêm valid grasp so với static P3.3;
- phần tăng phải đến từ trajectory vật lý thật, không hạ penetration/force gate;
- scene disturbance và hand load vẫn dưới threshold đã khóa trước khi chạy;
- terminal grasp phải tự giữ target khi support assistance kết thúc.

Đây là hypothesis cần ablation. Plan không mặc định kết quả sẽ tốt hơn; nếu yield
không tăng hoặc chỉ tăng nhờ unsafe contact, record phải báo thất bại thay vì đổi
threshold sau khi xem kết quả.

## 2. Quan hệ với P3.3

P3.3 và P3.4 là hai nhánh sibling dùng chung `SceneSpec`:

```text
                         ┌─ P3.3 static scene candidate → static/dynamic check
SceneSpec + observation ─┤
                         └─ P3.4 trajectory search → contact-rich acquisition
                                                    → terminal grasp check
```

- P3.3 giữ baseline direct grasp nhanh, dễ audit và ít compute.
- P3.4 dùng khi direct approach bị blocked hoặc khi strategy được chọn rõ.
- P3.4 không thay đổi label cũ thành positive; nó tạo trajectory sample mới.
- Hai nhánh hợp nhất ở dataset manifest qua `generation_mode`, không nhập logic
  P3.4 vào P3.3 orchestrator.
- P4 hiện tại không tự động chuyển thành policy học trajectory. Consuming
  `DynamicGraspTrajectory` cần revision model/training riêng.

## 3. Literature snapshot 2024–2026

### 3.1 DexGraspVLA — non-prehensile step trước grasp

Nguồn: [DexGraspVLA, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38953).

DexGraspVLA mở rộng dexterous grasping sang non-prehensile behavior: hệ thống có
thể thay đổi trạng thái vật/scene trước khi grasp trực tiếp, thay vì yêu cầu một
pose cuối luôn reachable từ observation ban đầu. Paper dùng hierarchical
language planner và diffusion action controller.

**Quyết định:** P3.4 kế thừa decomposition “reposition rồi acquire”, nhưng sinh
trajectory bằng model-based physics search, chưa train VLA hoặc diffusion policy.
Action sequence và contact events được lưu để policy phase sau có thể học lại.

### 3.2 ClutterDexGrasp — dynamic và safety curriculum trong clutter

Nguồn: [ClutterDexGrasp, CoRL 2025](https://arxiv.org/abs/2506.14317) và
[project page](https://clutterdexgrasp.github.io/).

ClutterDexGrasp cho thấy closed-loop interaction và safety curriculum quan trọng
khi target bị che hoặc không có direct path. Scene density và unsafe interaction
được tăng dần thay vì dùng một difficulty duy nhất.

**Quyết định:** P3.4 có clutter/safety curriculum cho generator: bắt đầu với
target-table contact, sau đó một non-target, rồi dense clutter. Curriculum chỉ
điều khiển sampling budget; tiêu chí release cuối không yếu hơn ở tier dễ.

### 3.3 Dexterous Functional Pre-Grasp Manipulation

Nguồn: [Dexterous Functional Pre-Grasp Manipulation, 2024](https://arxiv.org/abs/2403.12421).

Paper định nghĩa đúng khoảng trống P3.4: object thường phải được reposition hoặc
reorient trước khi có functional grasp. Hệ thống dùng extrinsic dexterity và
feedback để phối hợp hand pose, object pose và contact qua thời gian.

**Quyết định:** target pose không bị đóng băng. Dataset lưu `T_world_object[t]`,
contact mode và terminal grasp; target motion chỉ được tạo bởi physics/contact,
không ghi trực tiếp object qpos trong rollout.

### 3.4 Contact-Grounded Policy — contact là trajectory, không phải bit

Nguồn: [Contact-Grounded Policy, 20/06/2026](https://openreview.net/forum?id=xTf2W00n95).

CGP mô hình hóa contact đa điểm biến đổi liên tục, dự đoán đồng thời robot state
và tactile/contact response rồi dùng compliance controller để hiện thực contact.
Điểm liên quan là một contact sequence không thể mô tả bằng collision yes/no ở
endpoint.

**Quyết định:** P3.4 lưu time-varying contact graph, force/impulse/slip và command
trajectory. Không giả tactile image; simulator contact signal được ghi dưới tên
`sim_contact`, không gọi là hardware tactile.

### 3.5 Grasp-to-Act — terminal grasp phải chịu dynamic load

Nguồn: [Grasp to Act, IEEE RA-L 2026](https://doi.org/10.1109/LRA.2026.3677744).

Paper đánh giá grasp dưới contact-rich dynamic task thay vì chỉ lift tĩnh, và
đo in-hand slip trong hammering/sawing/cutting/stirring/scooping.

**Quyết định:** sau acquisition, P3.4 luôn có support-release, lift và bounded
disturbance. Một trajectory đưa vật vào tay nhưng terminal grasp trượt ngay khi
rời bàn vẫn là negative.

### 3.6 MuJoCo Warp/MJX-Warp — backend GPU cho batched contact rollout

Nguồn: [MuJoCo Warp repository](https://github.com/google-deepmind/mujoco_warp)
và [MJX documentation](https://github.com/google-deepmind/mujoco/blob/main/doc/mjx.rst).

MuJoCo Warp là backend MuJoCo tối ưu cho NVIDIA GPU, hỗ trợ parallel worlds và
GPU batch rendering. MJX docs nêu MJX-Warp là lựa chọn tối ưu NVIDIA và có
transmission/contact support rộng hơn MJX-JAX; Warp hiện chưa cung cấp
differentiability hoàn chỉnh.

**Quyết định:** backend CUDA v1 dùng MJX-Warp/MuJoCo Warp với batched sampling
không gradient. CEM/MPPI/shooting phù hợp hơn giả định differentiable contact.
MuJoCo CPU là oracle replay. Exact package/version chỉ được pin sau compatibility
spike với robot tendon, mocap/weld, contact force extraction và cu128 lock.

## 4. Physical semantics: contact được phép nhưng có điều kiện

### 4.1 Contact classes

Mỗi contact event thuộc một class:

- `target_intentional`: finger/palm được phép chạm target theo strategy;
- `support_assisted`: target hoặc hand dùng table/bin/shelf làm điểm tựa;
- `neighbor_incidental`: contact nhẹ với non-target object;
- `self_contact_allowed`: pair nằm trong robot allowlist;
- `forbidden`: link/geom pair không được phép;
- `damaging`: bất kỳ pair nào vượt safety budget.

“Allowed” không đồng nghĩa miễn penalty. Search objective vẫn ưu tiên lực nhỏ,
thời gian ngắn và ít làm thay đổi scene.

### 4.2 Safety budget đa đại lượng

Không dùng một ngưỡng force duy nhất. `ContactSafetyBudget` tối thiểu gồm:

```text
peak_normal_force_N
peak_tangential_force_N
normal_impulse_Ns
tangential_impulse_Ns
contact_duration_s
contact_work_J
max_penetration_m
max_wrist_force_N / max_wrist_torque_Nm
max_joint_or_tendon_load
max_non_target_translation_m / rotation_rad
max_non_target_velocity_mps
```

Budget được pin theo robot profile, link class, material class và environment.
Threshold mặc định là simulation/research constraint bảo thủ, **không phải chứng
nhận bàn tay thật không hỏng**. Hardware safety claim cần manufacturer limit,
calibration, safety factor và revision riêng.

### 4.3 Điều kiện terminal success

Trajectory chỉ positive nếu:

1. target được di chuyển bằng contact dynamics, không teleport;
2. không contact nào `forbidden` hoặc `damaging`;
3. non-target disturbance nằm trong budget;
4. hand chuyển từ environmental assistance sang enclosure hợp lệ;
5. target rời support và được nâng đủ cao;
6. grasp sống sót qua disturbance với measured contacts và finite state;
7. CPU oracle replay xác nhận cùng outcome class.

## 5. Typed trajectory contracts

```text
DynamicGraspRequest
  scene_state_ref, observation_ref, target_object_id
  robot_profile, strategy_id, safety_budget_id
  horizon, control_dt, seed, backend_request

DynamicGraspTrajectory
  time: [T]
  palm_pose: [T, 7]
  joint_state: [T, J]
  actuator_command: [T, U]
  object_pose: [T, O, 7]
  object_velocity: [T, O, 6]
  contact_graph: ragged ContactEvent stream
  stage: [T]  # approach/reposition/enclose/lift/perturb
  terminal_grasp: GraspBatch-compatible record

ContactEvent
  time_index, geom/body/link pair, contact_class
  point, frame, normal/tangential force, impulse, slip
  penetration, relative_velocity, work, budget_margin

DynamicSearchOutcome
  trajectory_ref, passed, failure_stage, failure_reason
  objective_terms, peak/cumulative safety metrics
  gpu_search_evidence?, cpu_replay_evidence
```

Trajectory storage dùng keyframes + fixed-rate state samples + sparse contact
events. Raw every-step arrays chỉ là optional debug artifact có retention policy;
không làm release dataset phình theo simulator timestep.

## 6. Backend contract

Module `qdgrasp/sim/batched/` cung cấp `BatchedContactBackend`:

```text
compile(scene_signature, robot_profile, batch_capacity)
reset(request_batch)
step(control_batch, steps)
observe() -> state/contact tensors
rollout(control_sequences) -> trajectory summaries
export_finalists(indices) -> replayable CPU requests
```

Hai implementation v1:

- `MuJoCoCpuBackend`: oracle, batch nhỏ/tuần tự, FP64 state nơi MuJoCo hỗ trợ.
- `MjWarpCudaBackend`: NVIDIA GPU, nhiều worlds cùng compiled topology.

Scene được bucket theo `scene_signature` gồm robot, số/type geoms, joints,
supports và solver settings. Không recompile mỗi candidate. Object pose/mass/
friction nằm trong data/batched parameters khi backend cho phép; thay topology
phải sang bucket khác.

Yêu cầu fail-closed:

- `backend=cuda` mà không có NVIDIA/Warp phải fail, không fallback CPU;
- unsupported MuJoCo feature phải báo trước search;
- NaN/Inf, contact overflow hoặc truncated contact buffer reject cả world;
- compile/warmup time báo riêng, không trộn vào steady-state throughput;
- finalist GPU phải replay bằng CPU trước release.

## 7. Các strategy module

### 7.1 `static_seeded_contact_rollout`

Lấy pose P3.3 bị blocked hoặc gần-feasible làm seed, sau đó cho phép target và
scene phản ứng vật lý trong approach/squeeze. Đây là baseline rẻ nhất để đo lợi
ích của việc bỏ giả định object frozen.

### 7.2 `primitive_sequence`

Search trên chuỗi primitive có parameter:

- `push`, `slide`, `roll`, `pivot_on_support`;
- `hook`, `cage`, `squeeze`;
- `support_release`, `lift`, `perturb`.

Primitive chỉ là control prior, không đặt object pose đích trực tiếp. Transition
giữa primitive dựa trên observed contact/object state.

### 7.3 `batched_cem`

Cross-Entropy Method sample control spline/primitive parameters trên GPU, rollout
batch, giữ elite và cập nhật distribution trong số iteration hữu hạn. Objective
tách riêng terminal grasp, progress, safety margin, scene disturbance, energy và
trajectory length để audit.

### 7.4 `batched_mppi`

MPPI dùng receding horizon cho scene mà target motion sau contact khó dự đoán.
Mỗi lần chỉ execute prefix của trajectory tốt nhất rồi replan từ physics state.
Module này đắt hơn CEM và chỉ mở sau khi CEM/CPU oracle parity pass.

### 7.5 `local_contact_refine`

Refine control knots quanh một successful trajectory để tăng safety margin và
giảm impulse/energy. Không thay label threshold và không dùng test split để tune.

Gradient-based contact optimizer/MJX-JAX, RL policy và learned value model nằm
ngoài v1; muốn thêm phải là strategy module + revision riêng.

## 8. Search objective và accounting

Objective có dạng khai báo, không hard-code vào strategy:

```text
score = terminal_grasp_quality
      + target_accessibility_progress
      + enclosure_progress
      + safety_margin
      - forbidden_contact_barrier
      - hand_load_cost
      - non_target_disturbance_cost
      - slip_and_penetration_cost
      - control_energy_and_time_cost
```

`forbidden_contact_barrier` và `damaging` là hard rejection, không được bù bằng
terminal score cao. Reason accounting ghi denominator ở từng stage:

- sampled;
- numerically stable;
- safe-contact feasible;
- terminal enclosure;
- support released;
- lift passed;
- perturbation passed;
- CPU replay confirmed.

## 9. CPU/GPU parity và release policy

Contact-rich trajectories có thể phân kỳ nhanh giữa backend, nên không yêu cầu
mọi state dài hạn bit-exact. Parity chia ba tầng:

1. **No-contact short horizon:** qpos/qvel/actuator deviation `<= 1e-4`.
2. **Pinned single-contact:** contact normal, impulse và object delta trong
   tolerance đã preregister; không đổi contact class.
3. **Full trajectory:** cùng success/failure class và không vi phạm safety budget.

GPU dùng để search/rank. CPU replay dùng cùng initial state, command sequence,
friction/mass và timestep. Nếu CPU đổi outcome hoặc vượt budget, sample mang
reason `backend_divergence` và không được release positive.

CPU generation vẫn hỗ trợ đầy đủ semantics với `batch_size=1` hoặc nhỏ; nó không
được gọi là CUDA evidence và không dùng để đóng performance gate.

## 10. Kaggle CUDA gate theo Phase 1

P3.4 mở rộng repository notebook công khai hiện có
`ninicom/qdgrasp-cuda-kaggle`, không tạo một GPU evidence repository khác.

Script library dự kiến:

```text
scripts/phase3_4_cuda_contact_search.py
```

Script phải:

1. fail nếu `torch.cuda.is_available()` false, Warp không thấy `cuda:0` hoặc
   backend thực tế là CPU;
2. ghi GPU model, driver, CUDA, PyTorch, MuJoCo, MJX/MuJoCo Warp versions;
3. cài/chạy exact public QDGrasp commit và exact optional GPU lock;
4. chạy lại P1 CUDA smoke và P2 CUDA FK parity trước P3.4;
5. compile một native micro scene cho LEAP, Allegro và Shadow;
6. chạy pinned CPU/GPU parity fixtures;
7. chạy bounded batched CEM smoke và một CPU-confirmed finalist/hand;
8. đo compile time, warmup, trajectories/s, peak VRAM và failure counts;
9. ghi `qdgrasp/evidence/phase3.4-cuda/v1` JSON cùng raw log hash;
10. thoát nonzero nếu có fallback, NaN, overflow, OOM hoặc thiếu CPU replay.

Kaggle notebook dùng GPU NVIDIA thật (ban đầu T4/cu128 như Phase 1), không chứa
credential và không tải external scene dataset. Micro scene native được tạo từ
exact seed trong notebook. VRAM budget mặc định `<= 14 GiB`; batch cap được pin
trước run, không loop tăng batch cho tới OOM.

Performance acceptance ban đầu:

- GPU steady-state throughput lớn hơn CPU cùng Kaggle kernel ít nhất `2×` ở
  batch đã pin; compile/warmup báo riêng;
- ít nhất 64 worlds chạy đồng thời trên T4 mà không OOM;
- ba hand có một GPU-searched, CPU-confirmed trajectory fixture;
- previous P1/P2 CUDA gates vẫn pass.

Nếu MJX-Warp/MuJoCo Warp không tương thích tendon/weld/contact observation, P3.4
giữ blocked và lập backend decision record; không thay bằng mock CUDA hoặc bỏ
Shadow khỏi gate.

## 11. Dataset `QDGrasp-ContactRich-Tiny`

Dataset nhỏ đầu tiên gồm:

- cùng canonical scene contract của `QDGrasp-Scene-Tiny`;
- ba hand, ba environment classes và ít nhất hai clutter tiers;
- static-seeded, primitive và CEM generation modes;
- positive/negative trajectory với contact/safety evidence;
- paired cases: cùng scene/target, static P3.3 fail nhưng P3.4 dynamic pass;
- CPU reference và GPU search provenance tách riêng;
- split theo object family, scene template và trajectory seed.

Mỗi hand cần ít nhất:

- một table-assisted pivot/slide positive;
- một sparse-clutter reposition positive;
- negative do damaging hand/support contact;
- negative do non-target disturbance;
- negative do GPU/CPU backend divergence fixture.

Dataset manifest ghi candidate budget, rollout horizon, objective weights,
safety-budget hash, backend/version, CPU replay rate, yield và compute cost. Không
chỉ lưu positive; failure trajectories cần thiết để học critic/safety model sau.

## 12. Work breakdown

| ID | Module | Output chính | Phụ thuộc |
| --- | --- | --- | --- |
| P3.4-00 | Khóa hypothesis, safety semantics và genuine-label entry gate | contracts/revision/failing tests | P3.1/P3.3 |
| P3.4-01 | Typed trajectory/contact contracts | `dataset/dynamic_contracts.py` | 00 |
| P3.4-02 | Batched backend protocol + scene signature bucketing | `sim/batched/contracts.py` | 01/P3.3 |
| P3.4-03 | MuJoCo CPU oracle backend | `sim/batched/mujoco_cpu.py` | 02/P3.2 |
| P3.4-04 | MJX-Warp compatibility spike và decision record | report + optional lock | 02 |
| P3.4-05 | MJWarp CUDA backend | `sim/batched/mjwarp_cuda.py` | 04 |
| P3.4-06 | Contact observer + multidimensional safety budget | `dynamic/safety.py` | 01/03 |
| P3.4-07 | Primitive-sequence controller | `dynamic/primitives.py` | 03/06 |
| P3.4-08 | Static-seeded contact rollout | `dynamic/static_seeded.py` | 07/P3.3 |
| P3.4-09 | Batched CEM search | `dynamic/cem.py` | 05/07 |
| P3.4-10 | Batched MPPI strategy | `dynamic/mppi.py` | 09 |
| P3.4-11 | Local contact trajectory refinement | `dynamic/refine.py` | 09 |
| P3.4-12 | CPU replay + terminal grasp certifier | `dynamic/certify.py` | 03/06/09 |
| P3.4-13 | Trajectory writer/loader và sparse event storage | dataset modules | 01/12 |
| P3.4-14 | Static-vs-dynamic controlled ablation | report + raw evidence | 08/09/12 |
| P3.4-15 | Kaggle CUDA harness và evidence | script/notebook/evidence JSON | 05/09/12 |
| P3.4-16 | Generate/audit/render `QDGrasp-ContactRich-Tiny` | release artifact | 13/14/15 |
| P3.4-17 | Independent review và roadmap handoff | revision/gates | 16 |

Thứ tự:
`00 → 01 → 02 → (03 || 04 → 05) → 06 → 07 → 08 → 09 → (10 || 11) →
12 → 13 → (14 || 15) → 16 → 17`.

P3.4-10 MPPI không chặn v1 nếu CEM đạt acceptance; nếu defer phải ghi rõ module
status và dataset không được claim MPPI coverage.

## 13. Test matrix bắt buộc

### Contract/safety unit tests

- Ragged contact events round-trip deterministic, đúng units/frame.
- Force, impulse, work và duration tích phân đúng trên synthetic trajectory.
- Allowed contact vượt bất kỳ hard budget nào trở thành `damaging`.
- Peak force thấp nhưng cumulative impulse/work cao vẫn reject.
- Non-target displacement/rotation accounting theo đúng object ID.
- Không thể ghi positive thiếu support-release/lift/CPU replay evidence.

### Physics behavior

- Target chỉ chuyển động do contacts/external physics, không runtime qpos write.
- Table pivot fixture pass với safe force và fail khi cùng path vượt force/impulse.
- Static blocked fixture trở thành dynamic positive mà không penetration spike.
- Neighbor incidental contact dưới budget được phép; trên budget reject.
- Wrong-object lift, support-only hold và hand/table crushing đều negative.
- LEAP/Allegro/Shadow dùng cùng strategy/backend interfaces.

### Search determinism/accounting

- Cùng seed/backend/config tạo cùng sampled controls và reason accounting.
- CEM elite update/bounds unit tests; no empty/NaN elite silent recovery.
- Batch/single CPU parity cho micro horizon.
- Search budget hết trả `budget_exhausted`, không tự tăng iteration/worlds.
- Objective term logging cộng đúng total score nhưng hard barrier không thể bị bù.

### GPU/Kaggle

- CUDA request fail trên CPU-only host.
- Device tensors/state/contact buffers thật sự nằm trên CUDA/Warp device.
- Pinned no-contact/single-contact/full-trajectory parity.
- Memory cap, batch cap, timeout và no-fallback tests.
- Evidence JSON exact commit/config/profile hash và CPU replay outcome.

## 14. Resource safety

### Local CPU

- Mặc định `batch_size=1`, một scene/hand, horizon và iteration cap hữu hạn.
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- Không chạy full CEM/MPPI ablation trên workstation từng hard-freeze.
- Trajectory debug dump tắt mặc định; chỉ ghi keyframes/contact events.
- Mọi CLI có dry-run in estimated worlds × steps × bytes trước khi execute.

### GPU

- Batch được cấp phát một lần theo preflight estimate, không tăng dần tới OOM.
- Contact/event buffers có cap và overflow là failure có telemetry.
- Checkpoint generation theo scene/seed atomically; resume không duplicate sample.
- Không chạy renderer đồng thời với peak search batch nếu vượt memory budget.
- Kaggle job có wall-time checkpoint và xuất evidence trước khi kết thúc session.

## 15. Gate đóng P3.4

```bash
.venv/bin/python scripts/check_docs.py --root .
.venv/bin/python scripts/check_phase3_4.py --backend cpu --profile micro
.venv/bin/python -m pytest tests/dynamic_grasp/ -q
git diff --check
```

CUDA gate chạy ở external Kaggle harness:

```bash
python scripts/phase3_4_cuda_contact_search.py \
  --device cuda:0 --profile kaggle-t4-micro \
  --evidence phase3_4_cuda_evidence.json
```

Không đóng P3.4 chỉ bằng CPU. CPU pass chứng minh correctness; Kaggle GPU pass
chứng minh CUDA backend/throughput. Ngược lại, CUDA search pass mà CPU replay
không xác nhận finalist cũng không tạo release positive.

## 16. Điều kiện hoàn tất

1. Dynamic trajectory được sinh trong scene physics và target có thể di chuyển
   trong acquisition mà không teleport.
2. Contact với support/non-target được phân loại và kiểm safety budget đa đại
   lượng, không còn binary collision-only semantics.
3. Có paired evidence static-fail/dynamic-pass cho cả ba hand.
4. GPU search trên Kaggle T4/cu128 pass fail-closed gate và nhanh hơn CPU theo
   criterion đã pin.
5. Mọi positive GPU đều được CPU oracle replay xác nhận.
6. `QDGrasp-ContactRich-Tiny` regenerate deterministic, có trajectory/contact/
   compute/provenance hashes và rendered stage evidence.
7. Không thay threshold/objective sau khi xem test split; controlled ablation và
   yield/compute trade-off được ghi bằng revision.
8. Independent reviewer kiểm force/impulse math, backend parity, scene damage
   semantics và ba hand fixtures.

P3.4 làm tăng coverage grasp bằng compute có kiểm soát; nó không tuyên bố rằng
mọi collision đều an toàn và không thay thế hardware safety validation.

## 17. Corrective phase P3.4.1

Ba nhánh P3.4 đã có measured/structural blocker. Kernel v9 tại 1024 worlds đạt
speed `4.444x` so với gate `>=2.0x`, nhưng full CUDA gate vẫn fail vì 29/1024
world non-finite và telemetry VRAM hiện chỉ đo PyTorch allocator. Shadow 40/40
hard-reject do damaging self-contact có cả ở no-closure; chưa có independent
reviewer. Không điều kiện nào được đổi thành pass bằng cách diễn giải lại
evidence.

Kế hoạch sửa chi tiết nằm tại
[`ROADMAP-P3.4.1-001`](PHASE3_4_1_FIX_PLAN.md). P3.4.1 giữ nguyên ngưỡng §10,
ưu tiên overflow/non-finite triage và đo VRAM đúng allocator; device-resident
MJWarp + CUDA Graph chỉ kích hoạt nếu speed sau stability fix tụt dưới `2x`.
Plan tạo versioned corrective delta cho Shadow thay vì sửa lịch sử P3.2/P3.3 và
đóng gói exact immutable review packet để một reviewer khác phát hành verdict.
P3.4 vẫn `active/pending` cho tới khi GPU gate, Shadow positive và independent
review cùng pass.

## 18. Correctness-recovery successor P3.4.2

Session report `SESSION-20260827-002` đã loại các phương án Shadow A/B/C trong
miền thử và ghi GPU divergence trên toàn identical-world cohort. Sau khi đối
chiếu code/evidence, hai kết luận cần thêm gate trước khi sửa runtime: GPU chưa
có fail-closed sanitizer verdict, còn failure `actuator_tracking` của Shadow
đang gộp joint tracking với actuator/tendon-coordinate tracking. Review packet
P3.4.1 cũng khóa revision cũ, worktree bẩn và chứa disclosure đã bị rút lại.

[`ROADMAP-P3.4.2-001`](PHASE3_4_2_CORRECTNESS_RECOVERY_PLAN.md) supersede đường
execution của P3.4.1 nhưng giữ P3.4.1 làm diagnostic history. P3.4.2 yêu cầu sửa
sanitizer harness thành fail-closed, tách Shadow predicate trước khi chọn
projected-target/safe-set/mixed-mode controller, rồi dựng lại immutable packet
cho reviewer độc lập. Gate §10/§16, safety budget và yêu cầu ba hand không đổi.

## 19. Tạm dừng bởi ADR-0008

Maintainer đã tạm dừng Shadow Hand khỏi active corpus vì chi phí cấu hình
underactuated/contact-control. Exact P3.4 contract này yêu cầu ba hand nên không
thể đóng trong thời gian pause; trạng thái execution là `paused_by_ADR-0008`,
không phải `pass` hay `complete`.

Không rewrite §10/§11/§16 hoặc evidence lịch sử thành hai hand. Nếu dự án cần
phát hành contact-rich dataset LEAP+Allegro trong thời gian pause, phải lập một
successor scope/plan riêng với tên dataset/verdict mới; không được gắn verdict đó
vào P3.4 ba-hand. Chi tiết policy và điều kiện mở lại nằm tại
[`ADR-0008`](../decisions/0008-temporary-shadow-hand-pause.md).
