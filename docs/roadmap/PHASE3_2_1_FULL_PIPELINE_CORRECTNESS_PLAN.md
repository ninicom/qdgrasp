---
document_id: ROADMAP-P3.2.1-001
document_type: plan
title: Kế hoạch Phase 3.2.1 — Full-Pipeline Correctness & Causal Remediation
version: 1.2.0
status: complete
date: 2026-08-24
revises: ROADMAP-P3.2.1-001@1.1.0
related_plan: ROADMAP-P3.2-001
latest_revision_record: docs/revisions/REV-20260824-001-phase3-2-1-closure.md
---

# Kế hoạch Phase 3.2.1 — Full-Pipeline Correctness & Causal Remediation

P3.2.1 là corrective subphase của P3.2. P3.2 đã chứng minh được một phần riêng
biệt: parse transmission, rank, finite-difference parity tại fixture đã pin,
joint-space projection và rollout của các grasp dựng thủ công. Bằng chứng đó
không chứng minh pipeline có thể tự sinh grasp theo luồng:

```text
proposal
  -> palm hypothesis
  -> IK
  -> collision admission
  -> static certificate
  -> actuator command plan
  -> MuJoCo squeeze/lift/perturbation
  -> measured dynamic label
```

Chẩn đoán trực tiếp trên box 5 cm, ba robot, ba recipe và hai candidate/recipe
cho kết quả 18/18 candidate dừng tại `IK: max_iter`; không candidate nào đi tới
dynamic rollout. Đồng thời gate P3.2 và generator đang sử dụng grasp/joint state
dựng thủ công. Vì vậy P3.2.1 không tuning threshold để tăng yield; subphase này
trước hết sửa invariant toán học, loại false-positive evidence, rồi chứng minh
quan hệ nhân-quả bằng characterization, mutation và end-to-end tests.

## 1. Trạng thái và claim correction

### 1.1 Claim còn hiệu lực

- LEAP và Allegro có transmission rank 16/16 trên compiled model đã pin.
- Shadow có 24 joint states, 20 actuator controls và transmission rank 20.
- Fixed tendon không phải mimic/equality giữa hai joint.
- Analytic moment matrix khớp finite difference tại state đã kiểm.
- MuJoCo validator có thể chạy squeeze/lift/perturbation với các fixture dựng
  thủ công đã pin.

### 1.2 Claim bị thu hồi cho tới khi P3.2.1 đóng

- Không gọi các fixture hardcode là `end-to-end pipeline`.
- Không dùng fixture hardcode để gỡ `release_blocked` của Shadow.
- Không xem dataset positive được thay vào `outcomes[0]` là output generator.
- Không kết luận active-finger IK đúng chỉ từ target bằng trạng thái ban đầu.
- Không kết luận dynamic parity khi saturation/tracking/palm-support chưa nằm
  trong success predicate.

### 1.3 Biện pháp fail-closed tức thời

- Shadow trở lại `release_blocked: true` cho tới gate P3.2.1.
- Xóa positive substitution trong `scripts/generate_dgn_open_tiny.py`.
- Dataset sinh từ code có substitution được đánh dấu `invalidated`; không sửa
  shard cũ tại chỗ và không dùng cho P4.
- Tách status `component_fixture_pass` và `pipeline_generated_pass` trong report.

## 2. Root causes đã xác nhận và giả thuyết phải kiểm

| ID | Loại | Finding | Cơ chế thất bại | Oracle bắt buộc |
| --- | --- | --- | --- | --- |
| RC-01 | xác nhận từ code | Residual normal dùng configured contact axis nhưng autodiff Jacobian dùng vector parent-to-tip | Solver chấm và tối ưu hai hàm khác nhau | autodiff so với central finite difference của cùng residual |
| RC-02 | xác nhận từ code | Active mask chỉ áp vào error, không áp vào các hàng Jacobian khi lập Hessian | Inactive fingers vẫn đổi curvature và nghiệm | thay target inactive không được đổi active solution |
| RC-03 | xác nhận từ code | `_fit_palm_pose` có `direction_weight=0` và Kabsch không có approach/floor/collision constraint | Wrist/palm orientation mơ hồ hoặc xuyên vật | transform recovery + forbidden-geom collision oracle |
| RC-04 | xác nhận từ code | `region_opposition` sample opposing faces với replacement | Nhiều active fingers có thể nhận cùng face/region | contact identity, spacing và surface membership tests |
| RC-05 | xác nhận từ code | Validator chỉ chặn null-space rejection, không chặn actuator saturation | Clipped command vẫn có thể được label PASS | saturation phải return trước `mj_step` |
| RC-06 | xác nhận từ code | Tracking errors chỉ được ghi telemetry | Actuator/palm/tip không bám target vẫn có thể PASS | tracking mutation phải đảo PASS thành FAIL |
| RC-07 | xác nhận từ code | Palm contact được đo nhưng không nằm trong success predicate | Palm support có thể giả grasp bằng fingertip | palm-only fixture phải fail |
| RC-08 | xác nhận từ code | Generator thay outcome thật bằng kinematics/static/dynamic fixture dựng thủ công | Positive shard không có provenance liên tục | provenance chain và no-substitution regression |
| H-01 | cần kiểm | Kabsch pose là nguyên nhân chính của 18/18 IK failure | Pose seed nằm ngoài basin hội tụ | controlled pose intervention giữ proposal/solver cố định |
| H-02 | cần kiểm | XML-native gains gây chatter/instability trong core rollout | Closed-loop poles/contact solver không đủ damping | no-contact step response + bounded contact sweep |
| H-03 | cần kiểm | Preliminary wrench ranking tương quan yếu với post-IK certificate | Sampler và certifier đánh đại lượng khác convention/scale | rank correlation trên cùng achieved contacts |

Một finding chỉ được chuyển từ `H-*` sang `RC-*` khi intervention thay đúng một
biến, failure signature biến mất và control case không đổi ngoài tolerance.

## 3. Contract kiến trúc sửa đổi

### 3.1 Contact proposal

`ContactProposal` phải bổ sung hoặc mang evidence tương đương:

```text
candidate_id
active_fingers: [K] bool
target_points hoặc contact_regions
face_ids và inward_normals cùng một surface sample
opposition_axis
approach_cone
object_frame
proposal_score_components
```

Không ép tất cả fingertips active. Một proposal tối thiểu có hai finger groups,
không duplicate active region và mọi active target nằm trên mesh trong tolerance
đã pin. `wrench_guided` chỉ ranking; certificate cuối luôn chạy độc lập trên
achieved contacts sau IK.

### 3.2 Palm hypothesis

Không tăng tùy ý `direction_weight` của Kabsch. Initializer production phải:

1. chọn opposition pair/group chính;
2. dựng object grasp frame từ opposition axis và một approach direction trong
   cone hợp lệ;
3. dựng canonical hand grasp frame từ FK của seed;
4. enumerate hữu hạn roll/seed hypotheses;
5. loại hypothesis vi phạm floor, palm, wrist hoặc forbidden-link clearance;
6. refine cục bộ palm twist `delta xi in se(3)` cùng joint delta với trust region.

Kabsch chỉ được giữ làm hypothesis phụ có cùng admission checks. Score hypothesis
phải tách contact residual, normal residual, forbidden penetration, clearance và
joint margin; không gộp thành một số mà mất evidence.

### 3.3 Kinematic solver

Residual, convergence, Jacobian và final evidence phải gọi cùng một primitive
`contact_state(spec, palm, q)` dùng configured contact offsets/axes.

Với weight/mask `W`, normal equation chuẩn là:

```text
Jw = W J
ew = W e
H  = Jw^T Jw + lambda^2 I + regularization
g  = Jw^T ew + regularization_gradient
```

Inactive fingers không tham gia task Hessian. Collision/joint-limit guards của
chúng được xử lý bằng constraint/regularizer riêng và phải có telemetry riêng.
Failure reason tối thiểu: `converged`, `stagnation`, `joint_limit`, `singular`,
`line_search_failed`, `insufficient_active_fingers`, `max_iter`.

### 3.4 Collision admission

Trimesh point/segment probing là prefilter. Oracle cuối dùng cùng compiled
MuJoCo robot/object geoms với dynamic rollout:

- chỉ cho phép contact giữa declared active fingertip geoms và target object;
- cấm palm/wrist/non-contact-link penetration;
- lưu geom pair, contact distance và maximum penetration;
- kiểm floor clearance tại pregrasp/contact pose;
- không dùng AABB làm collision oracle cho object cong/compound.

### 3.5 Controllable task-space command

Raw joint null-space norm không đủ để quyết định một contact task có điều khiển
được hay không. Tại state hiện tại:

```text
M = dl/dq
P = M^+ M
J_a = active-contact task Jacobian
dx = desired active-contact displacement
```

Command solver tìm `dq_c` trong `range(M^T)`:

```text
min ||J_a dq_c - dx||^2
    + alpha ||dq_c||^2
    + beta  ||M dq_c||^2
subject to joint margin, step limit và control range
```

Sau đó `du = M dq_c`. Admission dùng active-task residual, saturation, control
range và state validity. Global null-space residual vẫn được lưu nhưng không tự
động reject một task có nghiệm controllable tương đương.

Output mới hoặc contract tương đương:

```text
GraspCommandPlan
  q_pregrasp
  q_contact
  q_preload
  active_fingers
  control_start
  control_target
  task_residual
  nullspace_residual
  saturated
  rejection_reason
```

### 3.6 Dynamic controller và labeler

`RolloutProtocol` pin timestep, gains source, stage durations, lift trajectory,
disturbance normalization, tracking tolerance và stability limits. Gain lấy từ
robot profile/compiled MJCF; không hardcode `Kd=-0.15` và gọi là critical damping.
Nếu cần hiệu chỉnh, dùng bounded step-response sweep và một selection rule khóa
trước dựa trên overshoot, settling time, acceleration và MuJoCo warnings.

Dynamic success bắt buộc:

```text
stable
and actuator_tracking_pass
and active_contact_sustained
and not palm_support
and not floor_support_after_lift
and penetration_pass
and lift_pass
and disturbance_survival_pass
```

Active contact được đo trên time window bằng normal impulse, contact duty cycle
và sustained load; không quyết định bằng force của một frame cuối hay scenario
name. No-contact rollout dùng để đo numerical noise floor trước khi pin threshold.

## 4. Work breakdown và thứ tự bắt buộc

| ID | Hạng mục | Output | Gate cục bộ | Trạng thái 2026-08-24 |
| --- | --- | --- | --- | --- |
| P3.2.1-00 | Thu hồi false-positive evidence và re-block release | profile, generator, manifest status, revision | no-substitution regression | complete |
| P3.2.1-01 | Tạo characterization harness và frozen failure corpus | JSON evidence của 3 hand x 3 recipe | tái lập 18-candidate baseline | complete |
| P3.2.1-02 | Hợp nhất contact-state primitive | shared FK contact primitive | residual parity tests | complete |
| P3.2.1-03 | Sửa normal Jacobian và weighted active mask | fixed/region DLS | FD Jacobian + metamorphic mask | complete |
| P3.2.1-04 | Harden solver reason/telemetry/line search | solver contracts | reachable/unreachable oracle | complete |
| P3.2.1-05 | Sửa proposal active set, opposition và identity | proposal modules | surface/spacing/determinism tests | complete |
| P3.2.1-06 | Thay palm initialization bằng grasp-frame hypotheses | initializer + local refinement | pose/collision equivariance tests | complete |
| P3.2.1-07 | Tích hợp exact MuJoCo collision admission | collision validator | allowed/forbidden geom tests | complete |
| P3.2.1-08 | Implement task-space transmission command plan | command layer | controllability/saturation tests | complete |
| P3.2.1-09 | Tách controller protocol và dynamic predicate | rollout/observer contracts | tracking/palm/floor/stability mutations | complete |
| P3.2.1-10 | Generated-reachable full-flow fixtures | tests + raw telemetry | positive không nhận oracle q/palm/contact | complete, 3/3 hand positive |
| P3.2.1-11 | Canonical object full-flow matrix | box/cylinder/superquadric/compound evidence | mỗi hand có generated positive | complete, 12/12 cell measured; 0 canonical positive |
| P3.2.1-12 | Negative mutation gate | mutation suite | từng mutation fail đúng stage | complete, including zero-damping pre-step rejection |
| P3.2.1-13 | Clean deterministic regeneration | new staging dataset | two-run hash/stage parity | complete, byte-identical manifests |
| P3.2.1-14 | Independent review và release decision | review + closing revision | reviewer chấp thuận claim scope | complete; TPR-20260824-001 pass |

P10–P13 evidence nằm tại
`evidence/phase3_2_1/p10-p13-release-gate/`. Hai generated manifests có
canonical JSON payload SHA-256 giống nhau
`5a34f9d8e7f6568c7dc28e1b5f70c0421b1910ef9fe038d75e33efec55b3c563`;
canonical manifest payload có SHA-256
`20450c16e2bc74ca6079a7407b9d2573c05784335dcdac3c88f0a6531c3a8eaa`.
Kết quả `0/12` canonical positive được giữ như measured limitation, không thay
bằng P10 fixture pass. Review độc lập `TPR-20260824-001` đã chấp thuận claim
scope sau khi finding active-count fail-closed được sửa và review lại sạch.
Closing revision `REV-20260824-001` chuyển Shadow sang `release_blocked: false`;
việc này không phải claim canonical yield hoặc hoàn tất toàn bộ P3.

Thứ tự bắt buộc:

```text
00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07
   -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14
```

Không tuning gains, threshold, candidate budget hoặc disturbance trước khi frozen
failure corpus và corresponding oracle test tồn tại. Mọi thay đổi budget sau khi
xem kết quả phải có revision và chạy lại toàn matrix.

## 5. Test strategy

### 5.1 Derivative and solver oracles

- So sánh autodiff với central finite difference cho fingertip position và
  configured contact axis tại ít nhất 10 interior joint states/robot.
- Kiểm active target giữ nguyên trong khi inactive target đổi lớn: active
  solution/residual/iteration không đổi ngoài tolerance.
- Inverse-generated target từ `q_truth`, solver bắt đầu ở `q_seed != q_truth` và
  phải khôi phục contact task trong tolerance.
- Unreachable, joint-limit, singular và insufficient-finger cases trả đúng reason.
- Batch/single và finger-order permutation parity.

### 5.2 Palm/proposal/collision oracles

- Exact rigid-transform recovery trên synthetic correspondence.
- Contact-order permutation không làm palm flip.
- Object rotation quanh gravity axis cho nghiệm equivariant.
- Mọi active point/normal/face ID nhất quán sau region materialization.
- Duplicate region, palm penetration, wrist penetration, floor collision và
  forbidden-link contact đều fail trước dynamic rollout.

### 5.3 Transmission and command tests

- Analytic/FD moment parity tại nhiều states, không chỉ q=0.
- Named joint/actuator permutation parity.
- Active task reachable qua controllable subspace pass dù raw desired q chứa
  thành phần null-space không ảnh hưởng task.
- Active task ngoài controllable task space fail trước rollout.
- Saturation, duplicate/unknown mapping và broken tendon coefficient fail closed.
- Không có `mj_step` sau pre-rollout command rejection.

### 5.4 Dynamic predicate mutations

- no-contact, palm-only, floor-supported, excessive penetration;
- actuator tracking failure và mocap/palm tracking failure;
- zero/unstable damping protocol và injected MuJoCo warning;
- perturbation làm rơi object;
- scenario/category ghi `pass` nhưng measured outcome fail;
- observer trả contact noise dưới calibrated floor.

Mỗi mutation phải làm gate fail đúng `failure_stage`. Mutation không làm verdict
đổi nghĩa là gate thiếu coverage và P3.2.1 chưa được đóng.

### 5.5 Full-flow fixtures

Hai tầng fixture được giữ tách biệt:

1. `generated-reachable`: oracle chỉ dùng để dựng object có nghiệm tồn tại;
   pipeline không được nhận `q_truth`, palm pose hoặc oracle contacts.
2. `canonical-independent`: box, cylinder, superquadric và compound object độc
   lập với hand morphology.

Mọi positive phải có provenance liên tục:

```text
proposal_id -> palm_hypothesis_id -> IK evidence -> achieved surface contacts
-> static certificate -> command plan -> actuator trajectory -> measured label
```

Fixture hardcode q vẫn được giữ trong component tests nhưng không được tính vào
`pipeline_generated_pass`, dataset yield hoặc release gate.

## 6. Evidence schema và causal proof

Mỗi experiment ghi:

- hypothesis/root-cause ID;
- exact code/config/robot/object/protocol hashes;
- seed và candidate budget;
- control/intervention pair chỉ khác một biến;
- stage accounting và per-stage failure reason;
- raw residual, collision, transmission, tracking và load telemetry;
- expected failure signature và observed result;
- verdict `supported`, `rejected` hoặc `inconclusive`.

Không gọi một hypothesis là root cause chỉ vì patch làm yield tăng. Root cause
được xác nhận khi patch xóa đúng failure signature, negative control vẫn fail và
unrelated metrics không drift ngoài tolerance.

## 7. Resource and execution safety

Mọi diagnostic/full-flow command chạy bounded:

```bash
timeout <seconds> env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  PYTHONHASHSEED=0 \
  .venv/bin/python <command>
```

- Một hand/fixture tại một thời điểm trên CPU oracle.
- Candidate/iteration/rollout budgets là hữu hạn và ghi trong evidence.
- NaN/Inf, MuJoCo warning hoặc timeout kết thúc run với failure reason; không
  tự retry bằng gain/timestep/budget khác.
- Dataset mới sinh vào clean staging path; không ghi đè release hiện tại.

## 8. Gate đóng P3.2.1

Gate script mới: `scripts/check_phase3_2_1.py`. Gate phải kiểm:

1. không còn positive substitution/fabricated certificate trong generator;
2. derivative, active-mask và solver oracle pass cho ba hand;
3. proposal/palm/collision invariants pass;
4. transmission task-space command, saturation và broken mapping tests pass;
5. dynamic negative mutations fail đúng stage;
6. mỗi hand có ít nhất một positive từ full pipeline trong budget đã pin;
7. two-run deterministic stage accounting và artifact hashes;
8. fixture/component pass được báo riêng, không thay full-flow evidence.

Verification cuối:

```bash
.venv/bin/python scripts/check_environment.py --profile cpu
.venv/bin/python scripts/check_docs.py --root .
.venv/bin/python scripts/check_phase2.py
.venv/bin/python scripts/check_phase3_2.py
.venv/bin/python scripts/check_phase3_2_1.py
.venv/bin/python -m pytest tests/ -q
git diff --check
git status --short
```

## 9. Điều kiện hoàn tất và release decision

P3.2.1 chỉ complete khi:

- Không còn positive fabricated hoặc provenance gap.
- Jacobian/residual/mask dùng cùng task definition và qua derivative oracle.
- Palm initialization có approach/collision/floor constraints rõ.
- Shadow command được xét trong active task space qua 24-state/20-control thật.
- Saturation, tracking failure và palm support không thể nhận PASS.
- LEAP, Allegro và Shadow đều có ít nhất một generated full-flow positive.
- Negative mutations bị bắt đúng stage.
- Clean deterministic regeneration pass hai lần.
- Independent reviewer xác nhận math, fixtures, mutation gate và release scope.

Sau đó mới được:

- chuyển Shadow `release_blocked: false` bằng closing revision;
- cập nhật P3.1-11;
- chạy controlled ablation P3.1-13;
- regenerate dataset chính thức P3.1-14;
- dùng dataset cho P4.
