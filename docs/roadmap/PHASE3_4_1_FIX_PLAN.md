---
document_id: ROADMAP-P3.4.1-001
document_type: plan
title: Kế hoạch Phase 3.4.1 — Sửa throughput, Shadow recipe và independent review
version: 1.1.0
status: superseded
date: 2026-08-27
revises: ROADMAP-P3.4.1-001@1.0.0
parent_plan: ROADMAP-P3.4-001
successor_plan: ROADMAP-P3.4.2-001
depends_on:
  - ROADMAP-P3.2.1-001
  - ROADMAP-P3.3-001
  - ROADMAP-P3.4-001
latest_revision_record: docs/revisions/REV-20260827-007-phase3-4-2-correctness-recovery-plan.md
literature_cutoff: 2026-08-27
evidence_root: evidence/phase3_4_1
---

# Kế hoạch Phase 3.4.1 — Corrective fix cho P3.4

P3.4.1 không phải thủ tục đóng phase. Đây là corrective phase để xử lý ba nhánh
blocker đã có bằng chứng: GPU gate, Shadow recipe/collision và independent
review. Kernel v9 đã đổi trạng thái nhánh GPU: speed tại 1024 worlds đạt
`4.444x`, nhưng full gate vẫn fail vì 29/1024 world non-finite và VRAM vẫn được
đo bằng allocator không đúng. Cho tới khi cả ba nhánh pass, P3.4 vẫn `pending`.

## 1. Baseline bất biến và điều không được làm

Bốn phát hiện sau là baseline, không bị ghi đè bởi lần chạy mới:

| Blocker | Bằng chứng hiện tại | Kết luận được phép |
| --- | --- | --- |
| `B-GPU-64` | Tesla T4, 64 worlds, LEAP scene 91 geom: speedup lịch sử `0.764x` (v9 rerun: `0.694x`) | Giữ như evidence cho batch floor; không còn là gating point hiện tại |
| `B-GPU-1024` | v9 tại operating point đã pin: GPU `12781.4`, CPU `2876.3` steps/s, `4.444x`; 29/1024 world non-finite; `peak_vram_gib=0.0` lấy từ PyTorch | Speed subcriterion đạt; full GPU gate vẫn fail vì stability, còn VRAM verdict chưa hợp lệ |
| `B-SHADOW` | no-closure vẫn `damaging_contact`; `rh_lfproximal`/`rh_lfmetacarpal` đạt 323 N; CEM 40/40 hard-reject | Không còn là lỗi search trên closure scale |
| `B-REVIEW` | artifact và evidence do cùng tác giả tạo; chưa có reviewer độc lập | Tác giả không được tự ký verdict |

Nguồn local chuẩn:

- `evidence/phase3_4/p15-throughput/`;
- `evidence/phase3_4/p16-contactrich-tiny/`;
- `docs/governance/THIRD_PARTY_REVIEW.md`;
- `docs/revisions/REV-20260827-005-phase3-4-implementation-status.md`.

P3.4.1 cấm:

1. đổi ngưỡng `2.0x`, đổi safety budget hoặc bỏ Shadow sau khi đã thấy kết quả;
2. chọn batch/GPU/scene tốt nhất sau nhiều lần chạy rồi chỉ công bố lần pass;
3. sửa/xóa evidence `0.764x`, `4.444x`, danh sách 29 world lỗi hoặc manifest
   release-blocked cũ;
4. làm CPU baseline chậm đi, giảm output bắt buộc hoặc bỏ đồng bộ cuối chỉ để
   tăng speedup;
5. coi speed subcriterion pass là full GPU gate pass, gọi 29 NaN là OOM, hoặc
   dùng tolerance rejection để bỏ qua chúng;
6. coi throughput pass là VRAM pass, hoặc ngược lại;
7. cho tác giả ký review packet do chính mình chuẩn bị.

## 2. Research snapshot 2026 và hệ quả kiến trúc

### 2.1 MJWarp device-resident + CUDA Graph là kiến trúc có điều kiện

Tài liệu MuJoCo Warp hiện tại nói một `step` gồm nhiều kernel launch và khuyến
nghị CUDA Graph khi lặp lại; đồng thời yêu cầu siết `nconmax`/`naconmax`/`njmax`
theo workload vì memory và compute tăng theo các capacity này. Benchmark upstream
dùng hàng nghìn world, capacity khai báo rõ, event trace và overflow fail-closed.

Hệ quả cho QDGrasp: nếu speed sau khi sửa stability tụt dưới `2x`, kiến trúc sửa
performance chính là **device-resident rollout executor**, không phải tiếp tục
tăng batch. Hiện v9 đã đạt `4.444x`, nên không tối ưu launch trước khi tìm được
nguyên nhân 29 world non-finite. Capacity vẫn phải lấy từ calibration split rồi
khóa trước validation split.

Nguồn chính thức:

- [MuJoCo Warp — performance tuning](https://mujoco.readthedocs.io/en/latest/mjwarp/);
- [MuJoCo Warp benchmark suite](https://github.com/google-deepmind/mujoco_warp/blob/main/benchmarks/README.md);
- [MuJoCo Warp repository và event trace](https://github.com/google-deepmind/mujoco_warp).

### 2.2 Backend hiện tại có host-device synchronization trong mỗi tick

`MjWarpCudaBackend.rollout()` gọi `step(..., steps=1)` cho từng tick. Mỗi lần
`step()` gọi `warp.synchronize()` rồi `observe()` kéo `qpos`, `qvel`, `xpos`,
`xquat`, `cvel` và `ncon` về NumPy. Với horizon 100, vùng steady-state hiện có
99 ranh giới sync/readback. Đây là technical debt và có thể làm GPU host-bound,
nhưng v9 cho thấy speed đã vượt gate ở batch 1024. Vì vậy code inspection này
không phải lý do ưu tiên refactor trước stability; event trace quyết định có cần
sửa hay không.

Nsight Systems 2026 phân biệt GPU starvation/low utilization và khuyến nghị
truy CPU blocking, synchronization và DtoH copy thay vì suy đoán từ throughput.
P3.4.1 dùng NVTX ranges `reset`, `ctrl_upload`, `physics_graph`,
`safety_reduce`, `terminal_readback` và lưu raw `.nsys-rep`/summary.

Nguồn: [NVIDIA Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).

### 2.3 Số VRAM `~0 GiB` hiện không đo Warp

Harness dùng `torch.cuda.max_memory_allocated()`. API này báo peak memory của
tensor PyTorch; simulator lại cấp phát qua Warp. Vì vậy `~0 GiB` không chứng minh
Warp dùng 0 GiB và cũng không chứng minh GPU nhàn. Các số throughput `0.764x`
và `4.444x` vẫn là raw timing hợp lệ cho lần chạy tương ứng; diễn giải
VRAM/occupancy và `vram_within_budget=true` từ bộ đếm PyTorch thì không hợp lệ.

P3.4.1 thay bằng ba số tách biệt:

1. device free-memory delta trước/sau allocation qua `warp.Device.free_memory`;
2. process/device memory qua NVML hoặc `nvidia-smi` sampling;
3. breakdown `Model`, `Data`, `Other memory` từ `mjwarp-testspeed --memory`.

Nguồn:

- [PyTorch `max_memory_allocated`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html);
- [Warp `Device.free_memory`](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.Device.html);
- [NVIDIA NVML device memory](https://docs.nvidia.com/deploy/nvml-api/structnvmlMemory__t.html).

### 2.4 Overflow và non-finite phải được phân loại trước performance refactor

MJWarp dùng fixed-size buffers; tài liệu chính thức nói overflow dẫn tới
undefined behavior và ghi bitmask theo world tại `Data.overflow`. Backend hiện
chỉ nhìn `qpos/qvel` non-finite, chưa đưa `Data.overflow` vào summary. Vì vậy
việc đầu tiên là kiểm xem 29 world hỏng có overflow trước NaN hay không.

Nếu không overflow, Warp cung cấp `verify_cuda`, `verify_fp` và debug mode để
định vị kernel đầu tiên; NVIDIA Compute Sanitizer tách `memcheck`, `initcheck`,
`racecheck` và `synccheck`. Các mode này làm chậm và không dùng với CUDA Graph,
chỉ chạy trên minimized reproducer.

Nguồn:

- [MJWarp overflow detection](https://mujoco.readthedocs.io/en/latest/mjwarp/#overflow-detection);
- [Warp debugging](https://nvidia.github.io/warp/latest/user_guide/debugging.html);
- [NVIDIA Compute Sanitizer 2026](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

### 2.5 Shadow là lỗi recipe/proxy cần phân loại trước khi sửa

Upstream Menagerie hiện mô hình hóa `rh_lfmetacarpal` bằng box và
`rh_lfproximal` bằng capsule, với `rh_lfknuckle` nằm giữa trong cây body. MuJoCo
tự bỏ collision cho cặp parent-child trực tiếp, không tự bỏ cho cặp
grandparent-grandchild. Upstream chỉ khai báo exclude cho wrist/forearm và một
cặp thumb, không exclude cặp little-finger đang lỗi.

Đây là bằng chứng rằng contact có thể được engine tạo đúng theo model, chưa phải
bằng chứng cặp đó nên bị exclude. Local recipe lại đặt MF/RF/LF gần trạng thái
đóng dù grasp chỉ dùng thumb/index. Suy luận cần kiểm chứng là initial recipe đã
đưa proxy little-finger vào overlap hoặc near-overlap trước khi closure search.

Nguồn:

- [Menagerie Shadow `right_hand.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/main/shadow_hand/right_hand.xml);
- [MuJoCo — surprising collisions và collision filtering](https://mujoco.readthedocs.io/en/latest/overview.html#surprising-collisions);
- [Menagerie Shadow derivation notes](https://github.com/google-deepmind/mujoco_menagerie/blob/main/shadow_hand/README.md).

### 2.6 Kiến trúc 2026 dự phòng, không thay backend giữa gate

Isaac Lab 3.0 beta có multi-backend factory, kit-less `newton_mjwarp`, CUDA
Graph và asset Shadow/Allegro; `mjlab` cung cấp Isaac-Lab-style manager API trực
tiếp trên MJWarp. Đây là hướng tốt cho P3.5/RL và là spike dự phòng nếu native
executor không bảo trì được. Tuy nhiên Isaac Lab/Newton vẫn beta và việc chuyển
framework sẽ đổi integration surface, nên không dùng làm cách nhanh để xóa
baseline `0.764x`.

Nguồn:

- [Isaac Lab 3.0 multi-backend release](https://github.com/isaac-sim/IsaacLab/releases);
- [Isaac Lab Newton/MJWarp solver](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.html);
- [`mjlab` — Isaac Lab API trên MJWarp](https://github.com/mujocolab/mjlab).

## 3. Nhánh A — Speed đã pass, fix stability và telemetry của GPU gate

### 3.1 Trạng thái v9 và decision tree mới

Operating point được khóa là `leap_hand_scene@1024`, Tesla T4, horizon 100,
exact CPU oracle/scene/commands và ngưỡng `>=2.0x`. v9 đo `4.444x`, nên không còn
lý do tối ưu throughput trước. Kết quả 64 worlds vẫn được báo cạnh nó nhưng
không thay thế 1024-world result.

```text
v9 @ 1024: speed 4.444x PASS, 29/1024 non-finite FAIL, VRAM unknown
├─ Data.overflow != 0 trước NaN → capacity/overflow fix
├─ no overflow, bad indices ổn định → initialization/world-index bug
├─ no overflow, bad indices đổi giữa run → race/uninitialized/atomic instability
└─ sau stability fix
   ├─ median speed >= 2x + VRAM <=14 GiB → GPU gate resolved
   └─ median speed < 2x → mới kích hoạt graph/performance refactor
```

Final bundle chạy ba fresh-process repetition đã khai báo trước, công bố từng
run và dùng median cho speed gate; không bỏ run thấp như outlier nếu chưa chứng
minh lỗi đo. Mọi run phải compile/warmup tách riêng, có final synchronization,
zero non-finite/overflow/OOM và cùng output semantics với CPU.

### 3.2 GPU-0 — Bảo toàn v9 và sửa diễn giải evidence

- giữ raw `4.444x`, `0.694x`, rejected indices và kernel log;
- ghi speed subcriterion `pass`, full CUDA gate `fail_stability`;
- đánh dấu `peak_vram_gib=0.0`/`vram_within_budget=true` là telemetry invalid,
  không sửa raw JSON để che provenance;
- tách `nonfinite_worlds`, `overflow_worlds` và `oom` thành ba field khác nhau;
  29 non-finite không được gọi là 29 OOM.

### 3.3 GPU-1 — Stability triage theo thứ tự nguyên nhân

Kernel repin hiện tại đã thêm `qpos_shape`, contiguous-tail và distinct-row
diagnostics. P3.4.1 tái sử dụng chúng như bước localization, không coi shape
match hoặc non-contiguous indices là root-cause verdict.

1. Đọc `Data.overflow` bitmask ở rollout boundary và ở first-bad-tick debug
   run; decode `OverflowType`, không chỉ nhìn `qpos/qvel`.
2. Chạy lại exact identical-world cohort ba fresh process; so sánh tập rejected
   index. Same indices gợi ý index/capacity/init; changing indices gợi ý race hoặc
   uninitialized memory, nhưng cả hai vẫn cần tool evidence.
3. Permute cùng initial state/command qua world indices và chạy duplicate-world
   equivalence tại tick 0, 1, first divergence và terminal.
4. Dùng binary search horizon để tìm kernel/stage đầu sinh non-finite; lưu tên
   array, world, tick và finite predecessor, không dump full 1024-world state.
5. Trên minimized reproducer, bật lần lượt `verify_cuda`, `verify_fp`, Warp debug;
   sau đó Compute Sanitizer `memcheck → initcheck → racecheck → synccheck`.
6. Chạy upstream `mjwarp-testspeed` trên exact exported model/capacity. Nếu
   upstream cũng hỏng, tạo minimal upstream reproducer; nếu chỉ QDGrasp hỏng,
   sửa wrapper/reset/control assignment trước.

Debug mode được phép sync từng launch và không dùng làm performance evidence.
Sau CUDA error phải chạy fresh process vì context có thể đã ở error state.

### 3.4 GPU-2 — Fix theo classification, không tune mù

| Finding | Fix cho phép | Bằng chứng bắt buộc |
| --- | --- | --- |
| `Data.overflow` trước NaN | đo rồi pin `nconmax/naconmax/njmax/nccdmax`; overflow fail-closed | zero overflow 3 run + memory budget |
| bad world cố định theo index | reset/initialization/indexing audit; mọi data field initialized | permutation/duplicate equivalence |
| bad world thay đổi giữa run | sửa race/uninitialized/atomic path; deterministic mode chỉ dùng chẩn đoán | sanitizer sạch + run-to-run stability |
| upstream MJWarp 1.16.0 bug | pin patch hoặc version mới qua compatibility spike | tendon/weld/contact/force + CPU parity |
| solver numerical instability thật | scale/state/solver root cause; không hạ safety | first-bad-tick evidence + CPU comparison |

Không được đơn giản reject 2.8% world rồi tiếp tục search: identical inputs phải
tạo valid finite cohort trước khi GPU finalist đáng tin.

### 3.5 GPU-3 — Đo VRAM và profile đúng allocator

Final evidence gồm:

- device free-memory delta qua Warp trước/sau `Model`/`Data` allocation;
- NVML process/device sampling và total/free/used;
- `mjwarp-testspeed --memory --measure_alloc --measure_solver`;
- overflow=`error`, event trace và Nsight/NVTX nếu Kaggle cho phép;
- exact lock, GPU UUID/model/driver/CUDA/Warp/MJWarp/MuJoCo.

VRAM pass chỉ khi device-level peak `<=14 GiB`; peak PyTorch được phép báo như
metric phụ nhưng không tham gia verdict.

### 3.6 GPU-4 — Performance refactor chỉ kích hoạt nếu cần sau stability

Nếu stability fix làm median speed xuống dưới `2x`, triển khai device-resident
horizon executor: upload control một lần, preallocate trước capture, CUDA Graph
cho `mjw.step` + GPU safety reduction, không `.numpy()`/full-state DtoH mỗi tick,
và một bounded terminal readback có final sync trong elapsed time. Debug mode
snapshot từng tick phải có parity với release mode trên horizon nhỏ.

Sau đó mới cân nhắc bỏ visual-only geom bằng `discardvisual` (dynamics giữ nguyên
nhưng raycast semantics có thể đổi), solver options, compact solver/sleep hoặc
MJWarp upgrade. Mỗi thay đổi phải pass state/contact/force/stability parity.

### 3.7 Backend dự phòng

| Phương án | Khi dùng | Vai trò gate |
| --- | --- | --- |
| Native MJWarp 1.16 fix | mặc định | ít thay đổi evidence nhất |
| MJWarp newer pinned version | upstream 1.16 bug đã cô lập | compatibility spike bắt buộc |
| `mjlab` trên MJWarp | native integration không bảo trì được | candidate ADR, không tự thay baseline |
| Isaac Lab 3.0 `newton_mjwarp` | cần đồng bộ P3.5/RL architecture | deferred vì beta |
| Isaac Sim/PhysX | cần RTX/sensor fidelity | ngoài P3.4.1, gate/oracle mới |

Không backend nào được chọn chỉ vì có speedup đẹp hơn.

## 4. Nhánh B — Shadow corrective delta

### 4.1 Quyết định phạm vi khuyến nghị

Không viết lại lịch sử P3.2/P3.3 và không đổi chúng thành “chưa từng pass”. Tạo
một **corrective delta** có recipe hash/version mới. Evidence và review cũ vẫn
bất biến; mọi kết luận phụ thuộc Shadow recipe cũ được đánh dấu superseded cho
recipe đó sau khi delta pass.

P3.2/P3.3 chỉ giữ trạng thái `complete` sau delta nếu toàn bộ impacted gates
được replay và reviewer kết luận correction không phá acceptance cũ. Trong lúc
chưa replay xong, shared Shadow recipe có trạng thái `corrective_review_pending`
và P3.4 tiếp tục blocked. Đây là mở lại phạm vi kỹ thuật bị ảnh hưởng, không mở
lại hoặc sửa nội dung lịch sử.

### 4.2 SHADOW-0 — Localization trước khi sửa

Tạo contact-pair audit theo stage cho `q_open`, approach sweep, `q_contact`,
squeeze, lift và perturbation:

- signed geom distance/penetration, contact normal/force/impulse/duration;
- body/geom pair, joint vector và active-finger mask;
- actuator error/force để phân biệt interpenetration ban đầu và controller ép;
- run gravity/control-off và no-closure để tách recipe khỏi control search;
- visual overlay collision proxy trên visual mesh tại first-contact frame.

Kết quả phải phân cặp lỗi vào đúng một lớp:

1. `invalid_posture`: inactive finger recipe tự đi vào collision;
2. `invalid_proxy`: collision geom sai rõ so với visual/CAD envelope;
3. `missing_structural_exclusion`: cặp link không thể va chạm vật lý nhưng model
   cho phép;
4. `legitimate_self_contact`: posture thật sự tự va chạm và phải reject.

### 4.3 SHADOW-1 — Ba phương án sửa theo thứ tự

**Phương án A — recipe-only, khuyến nghị đầu tiên.** Giữ active set thumb/index;
đưa MF/RF/LF về canonical open/clearance posture thay vì 1.2/1.2/1.4 rad. Tạo
approach/squeeze trajectory có swept clearance cho các ngón inactive. Không đổi
mesh, contact material, safety budget hoặc force threshold.

**Phương án B — compile-time `<exclude>` có chứng minh.** Chỉ dùng nếu body-pair
audit chứng minh `rh_lfmetacarpal`/`rh_lfproximal` là structural adjacency/proxy
artifact không nên sinh contact ở toàn joint-limit domain cần hỗ trợ. Exclude
được khai báo trong MJCF/MjSpec trước compile, không mutate
`geom_contype/conaffinity` sau compile. Phải có negative control chứng minh
object contact và legitimate inter-finger contact vẫn hoạt động.

**Phương án C — re-author collision proxy.** Chỉ dùng nếu overlay/CAD audit xác
nhận primitive box/capsule sai envelope. Ưu tiên primitive/capsule mới; convex
decomposition/CoACD chỉ là fallback cho link mesh phức tạp, với fixed transform,
mass/inertia, tolerance và part cap. Không tune decomposition cho tới khi run
pass; geometry fidelity được chấm trên calibration fixtures độc lập.

Không dùng giảm contact stiffness, giảm actuator gain hoặc tăng force budget làm
fix đầu tiên. Các thay đổi đó có thể che overlap nhưng không sửa recipe/model.

### 4.4 SHADOW-2 — Acceptance và regression blast radius

Shadow delta chỉ pass khi:

1. `q_open` và no-closure không có damaging self-contact;
2. approach/squeeze sweep không penetration quá safety budget;
3. có ít nhất một measured Shadow positive qua support-release, lift,
   perturbation và CPU certification;
4. negative damaging-contact fixture vẫn reject;
5. same 40-candidate diagnostic không còn 40/40 chết vì pre-closure collision;
   search không bắt buộc 40/40 pass;
6. LEAP/Allegro và các Shadow P2/P3.2/P3.3 impacted fixtures không regression;
7. recipe/asset/collision-policy hash mới được ghi; manifest cũ không bị sửa;
8. dataset mới ghi rõ `supersedes_recipe_hash`, không tuyên bố evidence cũ sai.

## 5. Nhánh C — Independent review không thể tự động “fix” bởi tác giả

### 5.1 Packet tác giả được phép chuẩn bị

Sau khi A và B tạo release candidate bất biến, tác giả chuẩn bị packet:

- exact clean commit SHA và dependency locks;
- manifest SHA-256 của source, configs, evidence và generated dataset;
- baseline fail `0.764x`, toàn bộ rerun kể cả fail, final gating result;
- raw performance traces, memory/overflow/parity evidence;
- Shadow before/after pair audit, recipe diff, blast-radius replay;
- frozen §10/§16 checklist và lệnh tái lập;
- disclosure về mọi thay đổi threshold/objective: giá trị hợp lệ là `none`;
- mẫu verdict `PASS`/`FAIL`/`BLOCKED`, không điền chữ ký thay reviewer.

### 5.2 Reviewer hợp lệ

Ưu tiên reviewer external/human. Nếu dùng reviewer nội bộ độc lập theo tiền lệ
`TPR-20260825-001`, phải chạy trong context/task riêng, không tham gia viết hoặc
sửa artifact, không nhận kết luận mong muốn, và chỉ có quyền đọc exact commit +
packet. Báo cáo phải tự gọi đúng loại `external` hoặc `internal_independent`;
không gọi internal review là third-party.

Reviewer kiểm tối thiểu:

1. gate `2.0x` không đổi, `4.444x` không bị gọi là full pass khi có 29 NaN,
   và VRAM được đo ngoài PyTorch allocator;
2. GPU/CPU workload, observation semantics, final sync, overflow và reject
   denominator;
3. Shadow fix không hạ safety, không vô hiệu hóa legitimate collision;
4. P3.2/P3.3 impacted claims đã replay đúng blast radius;
5. exact commit/evidence hash và clean-tree state;
6. mọi severity S0–S3 đã resolve hoặc verdict không phải pass.

Artifact đổi sau verdict làm verdict mất hiệu lực; cần delta review hoặc full
review theo governance. Tác giả hiện tại không có hành động kỹ thuật nào có thể
tự gỡ `B-REVIEW`; cần người dùng chỉ định reviewer hoặc tạo review task độc lập.

## 6. Work breakdown P3.4.1

| ID | Output | Phụ thuộc | Gate |
| --- | --- | --- | --- |
| P3.4.1-00 | freeze baseline, exact RC/evidence schema | P3.4 evidence | không sửa evidence cũ |
| P3.4.1-01 | overflow/first-bad-tick/repeat stability triage | 00 + v9 | 29-world failure classified |
| P3.4.1-02 | versioned GPU stability fix | 01 | zero non-finite/overflow |
| P3.4.1-03 | allocator-correct VRAM + upstream/wrapper profiling | 02 | device peak measured |
| P3.4.1-04 | conditional CUDA Graph/performance refactor | 02/03, chỉ khi `<2x` | semantics unchanged |
| P3.4.1-05 | final T4 GPU bundle | 03 hoặc 04 | median `>=2.0x`, VRAM `<=14 GiB`, zero invalid worlds |
| P3.4.1-06 | Shadow pair/stage localization | 00 | classification evidence |
| P3.4.1-07 | versioned Shadow recipe/collision delta | 06 | no-closure safe |
| P3.4.1-08 | impacted P2/P3.2/P3.3 replay + ContactRich v2 | 07 | Shadow positive; regressions pass |
| P3.4.1-09 | immutable independent-review packet | 05/08 | exact hashes, all failures included |
| P3.4.1-10 | reviewer verdict và roadmap handoff | 09 | independent pass |

`01` và `06` chạy độc lập. `02–03` là bắt buộc vì v9 đã fail stability/telemetry;
`04` chỉ chạy nếu speed sau stability fix tụt dưới `2x`. `09–10` không bắt đầu
final verdict trước khi `05` và `08` đều pass.

## 7. Evidence layout và trạng thái

```text
evidence/phase3_4_1/
├── baseline-manifest.json
├── perf-v9-1024/
│   ├── raw-runs/
│   ├── event-trace/
│   ├── memory/
│   └── parity/
├── shadow-corrective-delta/
│   ├── pair-audit/
│   ├── geometry-overlay/
│   ├── recipe-diff/
│   └── regression/
└── review-packet/
    ├── MANIFEST.sha256
    ├── frozen-checklist.md
    └── reproduction.md
```

Trạng thái hợp lệ của P3.4.1 là `active`, `blocked` hoặc `complete`. `complete`
chỉ khi P3.4.1-05, P3.4.1-08 và P3.4.1-10 cùng pass. Hiện v9 nhanh `4.444x`
nhưng có 29 world non-finite và VRAM chưa đo đúng; thêm Shadow chưa positive và
chưa có reviewer, nên trạng thái không phải `complete`.

## 8. Gate hoàn tất P3.4.1 và tác động tới P4

1. T4/1024-world representative gate đạt `>=2.0x` bằng measurement hợp lệ,
   device-level peak VRAM `<=14 GiB`, zero non-finite/OOM/overflow/fallback qua
   ba fresh-process repetition.
2. GPU/CPU parity, output semantics, finalist CPU replay và reject accounting
   pass trên exact release candidate.
3. Shadow có positive thật; no-closure không damaging; safety threshold không
   đổi; impacted P2/P3.2/P3.3 regression pass.
4. Independent reviewer phát hành verdict pass trên exact immutable commit.
5. Roadmap chỉ đổi P3.4 thành `complete` sau một revision riêng ghi toàn bộ hash.

P4 static/offline có thể tiếp tục trên interface không phụ thuộc ContactRich theo
roadmap hiện hành. Mọi claim hoặc training input phụ thuộc
`QDGrasp-ContactRich-Tiny` phải chờ execution successor P3.4.2 complete; không
dùng dataset `release_blocked=true` như release data.

## 9. Successor P3.4.2

`SESSION-20260827-002` đã đưa P3.4.1 tới ranh giới của ba blocker nhưng phát hiện
hai decision point mà §3/§4 chưa mô hình hóa đủ: sanitizer evidence chưa
fail-closed và `actuator_tracking` của Shadow đang gộp actuator-coordinate với
joint-space error. P3.4.1 vì vậy được giữ làm diagnostic history với trạng thái
`superseded`, không được ghi `complete`.

Execution successor nằm tại
[`ROADMAP-P3.4.2-001`](PHASE3_4_2_CORRECTNESS_RECOVERY_PLAN.md). P3.4.2 giữ toàn
bộ baseline/gate của plan này, thêm corrected package ledger, fail-closed GPU
triage, Shadow projected-target/safe-set decision tree và rebuild review packet
trên exact clean release candidate. Evidence P3.4.1 không bị sửa hoặc xóa.
