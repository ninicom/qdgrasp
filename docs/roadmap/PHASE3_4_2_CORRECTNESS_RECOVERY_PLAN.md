---
document_id: ROADMAP-P3.4.2-001
document_type: plan
title: Kế hoạch Phase 3.4.2 — GPU correctness recovery, Shadow control semantics và release review
version: 1.2.0
status: superseded
date: 2026-08-27
revises: ROADMAP-P3.4.2-001@1.1.0
parent_plan: ROADMAP-P3.4-001
successor_to: ROADMAP-P3.4.1-001
depends_on:
  - ROADMAP-P3.4-001
  - ROADMAP-P3.4.1-001
source_report: docs/sessions/SESSION-20260827-P3.4-AND-P3.4.1.md
latest_revision_record: docs/revisions/REV-20260827-009-temporary-shadow-hand-pause.md
superseded_by: ADR-0008
execution_state: paused_by_ADR-0008
literature_cutoff: 2026-08-27
evidence_root: evidence/phase3_4_2
---

# Phase 3.4.2 — Correctness recovery và release review

> **Trạng thái hiện tại:** execution của plan này tạm dừng theo `ADR-0008` vì
> Shadow không còn thuộc active corpus. Nội dung bên dưới được giữ như exact
> three-hand closure/resumption contract, không phải backlog mặc định và không
> được dùng để tuyên bố P3.4 đã pass.

P3.4.2 tiếp nhận ranh giới kỹ thuật được ghi tại
`SESSION-20260827-002`. Đây không phải một phase tối ưu hiệu năng mới và không
phải cách đổi tên blocker để đóng P3.4. P3.4.2 chỉ hoàn tất khi:

1. GPU search hữu hạn, ổn định và có semantics tương đương CPU oracle;
2. Shadow có dynamic positive thật mà không bỏ qua self-contact hoặc tracking;
3. review packet được dựng lại trên exact clean commit và người kiểm độc lập ký
   verdict `PASS`.

P3.4.1 được giữ như diagnostic history. Các số đo `0.764x`, `4.444x`,
`4.537x`, 29/34 non-finite world, Shadow 323 N và mọi lần thử thất bại đều là
evidence bất biến; P3.4.2 không sửa hoặc xóa chúng.

## 1. Đính chính baseline trước khi triển khai

### 1.1 Báo cáo chưa phải release ledger

P3.4.2-00 phải phát hành một status ledger phân biệt ba trạng thái
`implemented`, `gate_passed` và `deferred`. Không tiếp tục dùng một con số tổng
duy nhất vì ba nguồn hiện không thống nhất:

- tiêu đề báo cáo ghi `14/18`;
- bảng trong cùng báo cáo liệt kê output cho 16 package và chỉ nêu `10`, `17`
  là chưa hoàn tất;
- project roadmap trước báo cáo ghi `15/18` tại một revision cũ hơn.

Không được tự sửa tiêu đề báo cáo đã hoàn tất. Ledger successor phải dẫn từng
package `00–17` tới commit/evidence và giải thích package có implementation nhưng
chưa pass gate.

### 1.2 GPU mới được localization, chưa có root-cause verdict

Hai run T4/1024-world đạt `4.444x` và `4.537x`, nên speed subcriterion `>=2x`
đã pass tại operating point hiện tại. Full GPU gate vẫn fail vì 29 rồi 34 world
non-finite và 990/990 finite survivor có state khác nhau từ cùng state/control.

Các số đó chứng minh GPU output hiện không đủ tái lập. Chúng chưa chứng minh
duy nhất một trong hai nguyên nhân `race` hoặc `uninitialized memory`. Solver
nhạy với thứ tự atomic, wrapper control/reset, fixed-buffer overflow và lỗi
upstream MJWarp vẫn là hypothesis cho tới khi có tool/first-bad-tick evidence.
P3.4.2 dùng trạng thái `unclassified_gpu_divergence`, không ghi `root-caused`.

### 1.3 Shadow chưa đủ bằng chứng để đổi controller ngay

Recipe cũ đưa MF/RF/LF vào self-contact 323 N. Sweep đã loại các phương án A/B/C
đúng trong miền được thử: uniform flexion, một số abduction, collision exclude
và proxy re-authoring. Tuy nhiên failure `actuator_tracking` hiện là phép AND của:

- actuator/tendon-coordinate tracking;
- toàn bộ joint-space tracking.

Bằng chứng hiện chưa ghi actuator/joint nào gây max error, ở stage nào, và lỗi
nằm ở requested joint target hay projected reachable target. Vì vậy câu
"position controller không giữ được ngón inactive" vẫn là hypothesis. Gate đầu
tiên của nhánh Shadow là tách predicate; mixed-mode controller chỉ được kích hoạt
nếu actuator-space tracking thực sự fail.

### 1.4 Review packet hiện tại bị vô hiệu

Packet P3.4.1 không phải final RC packet vì:

- khóa commit `12897bb...`, cũ hơn source report và sanitizer harness;
- tự ghi `worktree_clean=false`;
- manifest hiện có 60 entry trong khi báo cáo ghi 56;
- disclosure còn nói sweep Shadow parameterize tendon sai, claim đã được rút lại;
- chưa chứa sanitizer verdict, GPU stability fix hoặc Shadow successor evidence.

Packet này chỉ được giữ làm evidence của lần đóng gói thử. Không giao reviewer
để xin verdict trên packet đó.

### 1.5 Điều bất biến

P3.4.2 cấm:

1. hạ gate speed `2x`, safety budget, tracking tolerance hoặc contact threshold
   sau khi xem kết quả;
2. bỏ Shadow, bỏ world lỗi hoặc chỉ công bố run tốt;
3. gọi `distinct_finite_rows == 1` là tiêu chí duy nhất cho numerical parity;
4. bỏ joint/actuator tracking của inactive fingers mà không thay bằng một
   safe-set predicate đo được;
5. chuyển backend sang Isaac/PhysX chỉ để tránh sửa evidence MJWarp;
6. mời reviewer trước khi có exact clean RC hoặc để tác giả tự ký verdict.

## 2. Quyết định kiến trúc

### 2.1 Giữ native MuJoCo Warp làm đường đóng P3.4

Native MJWarp có ít blast radius nhất và speed đã vượt gate. CUDA Graph,
device-resident horizon executor hoặc framework migration không được ưu tiên
trước correctness. CUDA Graph chỉ kích hoạt nếu stability fix làm median speed
tụt dưới `2x`.

Nếu minimal upstream reproducer cũng fail, được phép pin một MJWarp patch/newer
version qua compatibility matrix riêng. Matrix phải replay tendon, weld,
contact/force, CPU parity và ba hand trước khi đổi lock. Isaac Lab/Newton,
`mjlab` hoặc PhysX vẫn là P3.5/RL spike; chúng không tự thừa kế gate P3.4.

### 2.2 Shadow dùng controllable-space semantics

Shadow có 24 joint nhưng 20 actuator và fixed tendon. Mọi target phải lưu ba
biểu diễn riêng:

1. `requested_joint_target` — ý định ở joint space;
2. `projected_reachable_target` — target nằm trong controllable subspace;
3. `actuator_coordinate_target` — lệnh thật gửi vào model.

Validator không được so actual state với một target bất khả thi rồi gọi đó là
controller failure. Ngược lại, projected target không được dùng để che
nullspace/task residual: residual và saturation gate hiện tại vẫn fail-closed.

### 2.3 Inactive finger không được miễn kiểm

Nếu exact joint tracking không phải semantics phù hợp cho inactive tendon group,
thay nó bằng một **safe-set predicate**, không bỏ predicate. Safe set gồm:

- không damaging self-contact hoặc penetration quá budget;
- nằm trong joint/velocity/actuator limits;
- không actuator saturation và actuator coordinate bám target;
- không cản swept path của active fingers/target;
- giữ ổn định qua squeeze, lift và perturbation.

## 3. Nhánh A — GPU correctness và final T4 gate

### 3.1 GPU-A0 — Sửa sanitizer harness thành fail-closed

Sanitizer cell hiện mới là launcher, chưa có archived verdict trong repo.
Trước khi chạy lại phải sửa protocol evidence:

1. dùng cùng seed, cùng initial `qpos/qvel/act`, cùng control tensor và lưu hash;
2. chạy `memcheck → initcheck → racecheck → synccheck` trên fresh process;
3. dùng nonzero `--error-exitcode`, không dùng `--error-exitcode 0`;
4. không catch exception rồi trả exit 0; infrastructure failure là failure;
5. lưu toàn bộ stdout/stderr, command, return code và tool summary, không chỉ
   4.000 ký tự cuối hoặc đếm dòng chứa từ `error`/`hazard`;
6. ghi `Data.overflow`, non-finite và OOM thành ba field tách biệt;
7. sanitizer/debug run không được dùng làm performance evidence.

Một clean sanitizer run chỉ loại các lớp lỗi mà tool quan sát được. Nó không tự
chứng minh deterministic contact/atomic ordering hoặc numerical parity.

### 3.2 GPU-A1 — Localization tới first bad stage

Trên exact minimized scene:

- kiểm state equality tại tick 0, 1, first divergence và terminal;
- binary-search horizon, lưu array/world/component đầu tiên vượt tolerance;
- permute duplicate world qua index để tách world-index defect;
- đọc/decode `Data.overflow` tại rollout boundary và first-bad-tick;
- bật lần lượt Warp `verify_cuda`, `verify_fp` và debug mode;
- chạy exact exported model bằng upstream `mjwarp-testspeed` hoặc minimal
  upstream driver để tách wrapper với MJWarp.

Decision tree:

```text
overflow trước divergence
└─ pin/calibrate fixed capacities, overflow fail-closed

sanitizer hoặc first-bad-tick chỉ ra QDGrasp wrapper
└─ sửa reset/control/index/readback trong backend

minimal upstream MJWarp cũng fail
└─ pin patch/newer version sau compatibility matrix; tạo upstream reproducer

sanitizer sạch nhưng duplicate worlds lệch
└─ điều tra atomic/contact ordering và solver sensitivity; không ghi "no bug"
```

### 3.3 GPU-A2 — Correctness gate

`distinct_finite_rows` tiếp tục là diagnostic, không phải release gate duy nhất.
GPU correctness pass khi cả ba fresh-process repetition đồng thời đạt:

1. zero non-finite, overflow, OOM và CPU fallback;
2. qpos/qvel/contact/force nằm trong tolerance đã pin trước validation;
3. hard-reject label, failure reason, finalist set và ranking không đổi ngoài
   tie policy đã khai báo;
4. mọi finalist được CPU oracle replay và cùng verdict;
5. không có sanitizer finding chưa resolve ở severity release-blocking.

Tolerance phải lấy từ CPU/GPU parity contract hoặc calibration split, không tune
trên final validation. Exact row equality có thể dùng cho duplicate-world unit
fixture nếu backend hứa deterministic execution, nhưng không thay outcome parity.

### 3.4 GPU-A3 — Performance và VRAM sau correctness

Final T4 bundle giữ operating point 1024 worlds, horizon/workload/output semantics
như CPU oracle và gồm ba fresh process:

- fail ngay nếu thiếu CUDA/Warp `cuda:0` hoặc backend/fallback thực tế là CPU;
- chạy lại P1 CUDA smoke và P2 CUDA FK parity trên cùng environment;
- compile native micro scene cho LEAP, Allegro và Shadow, kể cả Shadow tendon;
- chạy đủ parity no-contact, pinned single-contact và full-trajectory;
- chạy bounded batched CEM và có ít nhất một GPU-searched, CPU-confirmed
  trajectory fixture cho từng hand;
- median speed `>=2.0x` với final synchronization trong elapsed time;
- device/process peak VRAM `<=14 GiB` qua Warp free-memory + NVML hoặc
  `nvidia-smi`; PyTorch allocator chỉ là metric phụ;
- exact GPU/driver/CUDA/Warp/MJWarp/MuJoCo lock;
- raw timing, compile/warmup, trajectories/s, failure denominator và JSON/log
  hash từng run, không loại outlier sau khi xem số;
- process exit nonzero nếu có fallback, non-finite, overflow, OOM hoặc thiếu
  CPU replay.

Chỉ nếu correctness pass nhưng median speed `<2x` mới kích hoạt device-resident
rollout/CUDA Graph work package.

## 4. Nhánh B — Shadow predicate, safe set và conditional controller

### 4.1 SHADOW-B0 — Tách failure predicate

Replay posture `0.00`, `0.05` và recipe cũ với instrumentation sau cho từng
stage `approach/squeeze/lift/perturbation`:

- max actuator-coordinate error kèm actuator name, target và actual;
- max joint error kèm joint name, requested/projected target và actual;
- `controllable_residual`, `nullspace_residual`, saturation mask;
- contact pair/force/penetration và inactive-finger safe-set margin;
- task lift, support release, duty cycle và active-finger contact.

Không dùng failure label tổng hợp `actuator_tracking` để quyết định kiến trúc.

### 4.2 SHADOW-B1 — Sửa semantics nếu validator so sai target

Nếu actuator coordinate bám target nhưng joint-space check fail vì so actual
với `requested_joint_target` ngoài controllable subspace:

1. validator so tracking với `projected_reachable_target`;
2. vẫn lưu requested target và fail nếu nullspace/task residual vượt gate cũ;
3. active fingertip/task displacement, safety, saturation và support-release
   gates giữ nguyên;
4. thêm mutation test: target cố ý bất khả thi phải reject, không được project
   im lặng thành một grasp khác.

Đây là validator correctness fix, không phải nới tracking tolerance.

### 4.3 SHADOW-B2 — Tìm reachable safe set trong actuator space

Nếu actuator tracking cũng fail, tìm feasibility trong actuator/tendon space
thay vì tiếp tục sweep một uniform joint flexion:

- biến chính: MF/RF/LF tendon actuator coordinates;
- biến phụ có chứng minh hình học: abduction/metacarpal coordinates;
- trajectory nhiều stage từ open → approach → squeeze, không chỉ static target;
- objective lexicographic: feasibility/safety trước, task quality sau;
- calibration split khóa search bounds, controller parameters và budget trước
  validation split.

Candidate chỉ hợp lệ nếu actuator trackable, không saturation, toàn swept path
an toàn và thumb/index task vẫn đạt. Không thu nhỏ proxy hoặc exclude contact thật.

### 4.4 SHADOW-B3 — Mixed-mode controller có điều kiện

Chỉ kích hoạt nếu B2 chứng minh current position protocol không có reachable safe
set trong search domain đã pin. Controller version mới có hai role:

- active fingers: position/task-space controller hiện hành;
- inactive tendon groups: tendon-space impedance hoặc safe-set controller với
  bounded torque/velocity và collision margin.

Per-role gain được phép là tham số của kiến trúc mới nhưng phải khóa trên
calibration fixtures trước validation. Không giảm global contact stiffness,
safety budget hoặc threshold. Inactive finger pass bằng safe-set invariance và
actuator bounds, không phải bị loại khỏi denominator.

Nếu B3 vẫn không có feasible Shadow positive, **P3.4 tiếp tục blocked**. Maintainer
có thể mở một roadmap/compatibility-contract revision mới, nhưng revision đó
không thỏa gate P3.4 hiện tại và không được dùng để ghi P3.4 `complete`. Không
agent hoặc ADR nào được tự bỏ Shadow khỏi gate của exact plan này.

### 4.5 SHADOW-B4 — Acceptance và regression

Shadow successor chỉ pass khi:

1. q_open/no-closure/swept path không damaging self-contact;
2. zero controller saturation hoặc tracking failure chưa giải thích;
3. support-release, lift, perturbation và CPU certification pass;
4. negative damaging-contact và impossible-target fixtures vẫn reject;
5. LEAP/Allegro không regression;
6. impacted P2/P3.2/P3.2.1/P3.3 Shadow gates replay;
7. recipe/controller/protocol hashes mới được ghi và evidence cũ bất biến;
8. Shadow có cả table-assisted pivot/slide và sparse-clutter reposition positive;
9. `QDGrasp-ContactRich-Tiny-v2` đáp ứng đầy đủ §8.2, không chỉ thêm một Shadow
   sample, và `release_blocked=false` chỉ sau toàn bộ gate.

## 5. Nhánh C — Rebuild review packet và independent verdict

### 5.1 REVIEW-C0 — Invalidate packet cũ, dựng packet mới

P3.4.2 packet chỉ được dựng sau GPU-A3 và SHADOW-B4. Packet bắt buộc có:

- exact clean commit SHA và dependency locks;
- manifest SHA-256 được đếm từ file thật;
- corrected P3.4 package ledger;
- mọi GPU run, sanitizer log, overflow/VRAM/parity evidence;
- Shadow requested/projected/actuator tracking evidence và controller decision;
- before/after hashes, regression replay và ContactRich v2 manifest;
- frozen checklist, reproduction commands và verdict để trống;
- disclosure nêu rõ mọi threshold/objective/controller change.

Packet generator phải fail nếu worktree bẩn, manifest không khớp, source report
claim đã rút lại còn tồn tại hoặc evidence dependency mới hơn packet commit.

### 5.2 REVIEW-C1 — Reviewer độc lập

Reviewer hợp lệ là external/human hoặc `internal_independent` trong task/context
riêng, không tham gia viết artifact và không nhận verdict mong muốn. Reviewer
chỉ đọc exact RC commit/packet, có quyền trả `PASS`, `FAIL` hoặc `BLOCKED`.

Phạm vi review tối thiểu kế thừa nguyên văn ý nghĩa §16:

- force/impulse/work/duration math và rolling-window budget;
- CPU/GPU parity, search/replay provenance và reject denominator;
- scene damage semantics cho support/non-target/self-contact;
- LEAP/Allegro/Shadow fixtures, gồm paired static-fail/dynamic-pass;
- deterministic regeneration, rendered stage evidence và controlled ablation;
- exact commit/manifest/clean-tree state và mọi S0–S3 finding.

Mọi thay đổi source/config/runtime/evidence/dataset/packet sau verdict làm verdict
mất hiệu lực. Commit hậu review chỉ được thêm independent verdict và governance
closure record; automated diff guard phải chứng minh không đổi reviewed artifact.
Tác giả P3.4/P3.4.1/P3.4.2 không được tự ký.

## 6. Work breakdown

| ID | Output | Phụ thuộc | Gate hoàn tất |
| --- | --- | --- | --- |
| `P3.4.2-00` | corrected 00–17 status ledger + executable closure checker + packet invalidation | source report/P3.4.1 | implemented/pass/deferred tách rõ; parent matrix machine-checked |
| `P3.4.2-01` | fail-closed sanitizer/overflow harness | 00 | full logs, nonzero failure, exact inputs |
| `P3.4.2-02` | GPU first-bad-tick + wrapper/upstream classification | 01 | root cause có tool evidence |
| `P3.4.2-03` | versioned GPU correctness fix | 02 | zero invalid + outcome parity |
| `P3.4.2-04` | full parent-§10 T4 bundle | 03 | P1/P2 + 3 hand + 3 parity tier + per-hand finalist + median `>=2x`, peak `<=14 GiB` |
| `P3.4.2-05` | Shadow tracking predicate decomposition | 00 | joint/actuator/stage attribution |
| `P3.4.2-06` | validator projected-target fix hoặc actuator safe-set search | 05 | classification-dependent gate pass |
| `P3.4.2-07` | conditional mixed-mode controller | 06, chỉ khi cần | inactive safe-set + active task pass |
| `P3.4.2-08` | impacted replay + full ContactRich v2 + ablation | 06 hoặc 07 | parent §11/§16 dataset gates, deterministic regeneration, render và trade-off pass |
| `P3.4.2-09` | immutable RC review packet | 04 + 08 | clean exact hashes, parent coverage matrix complete, no stale claim |
| `P3.4.2-10` | full-scope independent verdict | 09 | reviewer `PASS`, zero S0–S3 open |
| `P3.4.2-11` | executable final gate + P3.4 closure revision và P4/P3.5 handoff | 10 | §15 commands pass trên exact reviewed RC; roadmap/dataset status truthful |

`01–04` và `05–08` độc lập sau `00`. Review không chạy song song với artifact
mutation; `09` bắt đầu sau khi hai technical branch bất biến.

## 7. Evidence layout

```text
evidence/phase3_4_2/
├── baseline-ledger/
├── gpu-correctness/
│   ├── sanitizer/
│   ├── overflow-first-bad-tick/
│   ├── upstream-isolation/
│   ├── parity/
│   └── final-t4/
├── shadow-control/
│   ├── tracking-decomposition/
│   ├── reachable-safe-set/
│   ├── controller-delta/
│   └── regression/
├── contactrich-v2/
└── review-packet/
```

Mỗi generated artifact ghi schema version, exact source/config/dependency hash,
command, environment và failure status. Raw evidence không được sửa để cập nhật
diễn giải; đính chính nằm trong revision record hoặc successor artifact.

## 8. Parent gate coverage matrix

### 8.1 Quy tắc kế thừa

P3.4.2 là corrective successor, không thay thế acceptance contract của parent.
Toàn bộ P3.4 §9–§16 vẫn normative. Nếu matrix dưới đây thiếu evidence hoặc có
verdict khác `PASS`, executable closure checker phải exit nonzero dù GPU/Shadow
blocker riêng lẻ đã được sửa.

`P3.4.2-00` triển khai `scripts/check_phase3_4_2.py` như read-only aggregator.
Checker đọc exact evidence/manifest/review verdict; không tự sinh evidence,
không gọi network và không chuyển `unknown` thành `pass`.

### 8.2 Coverage P3.4 §10 — CUDA/Kaggle

| Parent requirement | Owner | Evidence bắt buộc | Pass condition |
| --- | --- | --- | --- |
| CUDA/Warp thật, không fallback | `01/04` | device/backend identity + negative CPU-only probe | request fail trên CPU; T4 run dùng `cuda:0` |
| Exact environment/commit | `04` | commit, lock, GPU/driver/CUDA/PyTorch/MuJoCo/Warp/MJWarp | mọi field present và hash khớp packet |
| Previous CUDA gates | `04` | P1 smoke + P2 FK parity logs | cả hai exit 0 trong cùng Kaggle environment |
| Ba native hand | `04` | LEAP/Allegro/Shadow compile records | 3/3 compile; Shadow tendon/weld giữ nguyên |
| Ba tầng parity §9 | `03/04` | no-contact, pinned single-contact, full-trajectory fixtures | `<=1e-4` no-contact; preregistered contact tolerance; cùng outcome/safety class |
| Bounded GPU search | `04` | CEM budget/horizon/seed + reason ledger | bounded run, denominator đầy đủ |
| Finalist từng hand | `04` | GPU trajectory + exact CPU replay | ít nhất 1 GPU-searched/CPU-confirmed fixture cho mỗi hand |
| Performance | `04` | 3 fresh raw timings, compile/warmup tách riêng | median steady-state `>=2.0x`; ít nhất 64 worlds, operating point 1024 |
| Memory/capacity | `01/04` | Warp/NVML peak + batch/contact buffer caps | peak `<=14 GiB`, zero overflow/OOM |
| Fail-closed output | `01/04` | process return code + v1 evidence JSON/raw-log hash | nonzero khi fallback/non-finite/overflow/OOM/missing replay |

Thay MJWarp version sau upstream isolation làm toàn bộ bảng trên stale; phải chạy
lại compatibility matrix và final bundle, không được reuse speed của version cũ.

### 8.3 Coverage P3.4 §11 — `QDGrasp-ContactRich-Tiny-v2`

`P3.4.2-08` không pass bằng cách chỉ thêm một Shadow sample. Release artifact
phải đồng thời có:

1. canonical P3.3 scene contract, đủ ba hand, ba environment class và ít nhất
   hai clutter tier;
2. static-seeded, primitive và CEM generation modes; MPPI được phép defer nhưng
   manifest phải ghi `mppi_coverage=false`;
3. paired static-fail/dynamic-pass cho **LEAP, Allegro và Shadow**;
4. với mỗi hand: ít nhất một table-assisted pivot/slide positive và một
   sparse-clutter reposition positive;
5. negative do damaging hand/support contact, non-target disturbance và
   CPU/GPU backend divergence;
6. positive/negative trajectory đều có contact/safety evidence, không chỉ lưu
   positive;
7. CPU reference và GPU search provenance tách riêng; split theo object family,
   scene template và trajectory seed;
8. manifest ghi candidate budget, horizon, objective weights, safety-budget
   hash, backend/version, CPU replay rate, yield và compute cost;
9. hai clean regeneration từ exact RC tạo byte-identical dataset/manifest và
   deterministic trajectory/event ordering;
10. rendered evidence cho approach/squeeze/support-release/lift/perturbation và
    visual QA hash khớp manifest.

Mỗi positive phải xuất phát từ physics contact; target teleport/runtime qpos
write là hard failure. Mọi GPU positive phải CPU replay cùng initial state,
command, mass/friction và timestep.

### 8.4 Coverage P3.4 §16 — Điều kiện hoàn tất

| §16 | Điều kiện parent | P3.4.2 owner/evidence | Closure verdict |
| --- | --- | --- | --- |
| 1 | physics trajectory, target không teleport | `08`: source guard + paired trajectory + rendered stages | required pass |
| 2 | support/non-target contact + multidimensional safety | `08`: contact ledger, negative fixtures, force/impulse/work/duration tests | required pass |
| 3 | paired static-fail/dynamic-pass cả ba hand | `08`: six minimum positives và paired IDs | required pass |
| 4 | fail-closed T4 CUDA nhanh hơn CPU | `01–04`: toàn bộ §8.2 | required pass |
| 5 | mọi GPU positive được CPU replay | `04/08`: one-to-one provenance/replay ledger | required pass |
| 6 | deterministic dataset + hashes + render | `08`: hai regeneration + §8.3 manifest/render | required pass |
| 7 | không post-test tuning + ablation/trade-off | `08/09`: frozen config, controlled ablation, yield/compute revision | required pass |
| 8 | independent full-scope review | `09/10`: exact clean packet + scope §5.2 | reviewer `PASS` |

### 8.5 Controlled ablation và change disclosure

Sau Shadow/GPU fix, chạy lại static-vs-dynamic controlled ablation trên split đã
khóa. Báo cáo phải gồm numerator/denominator, positive yield theo hand/scene,
compute/trajectory, safety rejection distribution và confidence/uncertainty phù
hợp với sample size. Kết quả `no_measured_difference` được phép nếu đúng số đo;
không được đổi hypothesis hoặc lọc run để ép kết quả dương.

Packet disclosure liệt kê before/after cho objective weight, tracking semantics,
controller/gain, safety budget, contact stiffness và dataset selection. Mọi tham
số được khóa trên calibration trước validation; thay đổi sau validation buộc tạo
RC mới và chạy lại impacted gate.

### 8.6 Exact closure command matrix

Trên exact clean RC commit, trước khi đóng packet:

```bash
.venv/bin/python scripts/check_docs.py --root .
.venv/bin/python scripts/check_phase3_4.py --backend cpu --profile micro
.venv/bin/python scripts/check_phase3_4_2.py \
  --root . \
  --evidence-root evidence/phase3_4_2 \
  --require-parent-gates
.venv/bin/python -m pytest tests/dynamic_grasp/ -q
git diff --check
```

Trên Kaggle T4/cu128 exact public commit/lock:

```bash
python scripts/phase3_4_cuda_contact_search.py \
  --device cuda:0 \
  --profile kaggle-t4-micro \
  --evidence phase3_4_cuda_evidence.json
```

Sau independent verdict, closure run đọc packet ở chế độ read-only:

```bash
.venv/bin/python scripts/check_phase3_4_2.py \
  --root . \
  --evidence-root evidence/phase3_4_2 \
  --require-parent-gates \
  --require-independent-pass \
  --reviewed-commit <EXACT_RC_SHA>
```

Checker cuối phải xác minh diff từ `<EXACT_RC_SHA>` chỉ chứa independent verdict
và allowlisted governance closure record. Bất kỳ thay đổi runtime/config/evidence/
dataset/packet nào buộc invalidate verdict và quay lại `P3.4.2-09`.

## 9. Gate đóng P3.4.2 và P3.4

P3.4.2 chỉ `complete` khi cùng lúc:

1. package ledger đã được reconciled và không còn packet cũ bị gọi là ready;
2. toàn bộ §8.2 CUDA/Kaggle matrix pass ba fresh process;
3. Shadow và ba-hand paired/dataset requirements §8.3 pass;
4. toàn bộ tám parent condition trong §8.4 có evidence `PASS`;
5. controlled ablation/disclosure §8.5 hoàn tất, không post-test tuning;
6. exact command matrix §8.6 pass trên reviewed RC;
7. independent reviewer ký `PASS`, zero S0–S3 open;
8. ContactRich v2 `release_blocked=false` và manifest/hash khớp packet;
9. revision riêng cập nhật P3.4 thành `complete` mà không viết lại evidence cũ.

Thiếu bất kỳ mục nào giữ cả P3.4.2 và P3.4 chưa đóng. Scope revision bỏ
Shadow, backend fallback, partial dataset hoặc reviewer `BLOCKED` không được tính
là closure của exact P3.4 contract.

## 10. Tác động tới P4 và P3.5 sau ADR-0008

- P4 static/offline tiếp tục với active hand LEAP/Allegro trên interface/dataset
  không phụ thuộc ContactRich release-blocked.
- P3.5 asset loader, scene synthesis, local CoACD API và environment contract
  tiếp tục theo gate 2/2 active hand của ADR-0008; không chờ P3.4.2.
- Không dùng ContactRich v1 làm release training input hoặc claim RL readiness.
- Mọi Shadow/three-hand RL training hoặc ContactRich claim vẫn phải chờ plan này
  được mở lại, P3.4.2-08 và independent verdict P3.4.2-10.

## 11. Non-goals

P3.4.2 không:

- triển khai MPPI đã deferred ở P3.4-10;
- thay safety budget hoặc hợp thức hóa self-contact 323 N;
- chứng minh sim-to-real/hardware safety;
- chuyển toàn bộ RL stack sang Isaac Sim;
- xóa P3.4.1, session report hoặc packet cũ;
- tự phát hành independent verdict.

## 12. Resumption policy

Không thực thi các package Shadow/three-hand closure của P3.4.2 trong normal
workload khi ADR-0008 còn hiệu lực. Plan chỉ trở lại `active` bằng revision mới
thỏa điều kiện mở lại trong ADR-0008. GPU diagnostics độc lập có thể được tái sử
dụng trong successor hai-hand plan, nhưng evidence/verdict mới không tự đóng
P3.4.2 hoặc P3.4 ba-hand.
