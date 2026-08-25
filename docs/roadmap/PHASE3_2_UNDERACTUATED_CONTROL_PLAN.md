---
document_id: ROADMAP-P3.2-001
document_type: plan
title: Kế hoạch Phase 3.2 — Underactuated Hand Control & Dynamic Parity
version: 1.3.0
status: complete
date: 2026-08-25
revises: ROADMAP-P3.2-001@1.2.0
related_plan: ROADMAP-P3.1-001
latest_revision_record: docs/revisions/REV-20260825-002-phase3-2-to-phase3-3-handoff.md
---

# Kế hoạch Phase 3.2 — Underactuated Hand Control & Dynamic Parity

P3.2 là phase con sửa khoảng trống kiến trúc được phát hiện trong P3.1: pipeline
đã điều khiển đúng LEAP và Allegro 16-state/16-control, nhưng chưa có abstraction
cho Shadow 24-state/20-control. Mục tiêu của P3.2 là đưa cả ba hand qua cùng một
transmission contract và dynamic protocol mà không giả fixed tendon thành mimic,
không teleport joint state và không hạ physics gate.

P3.2 không đóng P3 tổng. Khi P3.2 pass, công việc quay lại P3.1-11 để xác nhận
dynamic fixtures ba hand, rồi mới chạy P3.1-13 controlled ablation, P3.1-14
regeneration và P3.1-15 release gates.

## Corrective subphase P3.2.1

Audit sau implementation xác nhận gate P3.2 hiện chứng minh transmission và
rollout component fixtures dựng thủ công, chưa chứng minh full generator path.
Chạy `run_pipeline_chunk` trên box 5 cm với ba robot, ba recipe và hai candidate
mỗi recipe cho kết quả 18/18 candidate dừng tại `IK: max_iter`. Generator còn
thay một outcome thật bằng positive fixture dựng thủ công.

Vì vậy [`ROADMAP-P3.2.1-001`](PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md) được
mở làm corrective gate bắt buộc. Gate này đã hoàn tất ngày 2026-08-24 theo
`TPR-20260824-001` và `REV-20260824-001`: generated-reachable full flow pass cho
cả ba hand, mutation/determinism gates pass và Shadow được gỡ release block.
Fixture P3.2 vẫn chỉ là component evidence; canonical-independent matrix P3.2.1
hiện `0/12`, nên closure không phải claim canonical yield hay dataset release.

## 1. Baseline và blocker

- Baseline implementation: commit `58cafe5` trên `feature/phase3-data-layer`.
- Revision phát hiện: `REV-20260823-003`.
- Tại thời điểm mở corrective plan, Shadow profile đúng là **24 joint states /
  20 actuator controls** và được đặt `release_blocked=true`; closing revision
  P3.2.1 đã chuyển cờ này sang `false`.
- Dynamic validator baseline fail closed bằng `underactuated_targets` khi một full
  24-joint target chỉ map được sang 20 transmissions.
- LEAP/Allegro component fixtures pass ở baseline; Shadow generated full-flow
  positive được xác nhận trong P3.2.1.

### 1.1 Kết luận vật lý không được đảo ngược

Fixed tendon của MuJoCo định nghĩa một chiều dài tổng hợp từ nhiều joint, ví dụ
`l = q_J1 + q_J2`. Actuator tác động lên tọa độ tendon `l`; nó không tạo equality
constraint `q_J1 = q_J2`. Vì thế:

- `q ∈ R^24` là state hợp lệ của Shadow;
- `u ∈ R^20` là control;
- bốn chiều null-space không thể nhận target độc lập từ actuator;
- full-state IK vẫn được phép tìm `q`, nhưng command layer chỉ được hứa điều
  khiển phần nằm trong controllable transmission space.

Nguồn chuẩn triển khai: [MuJoCo transmission model](https://mujoco.readthedocs.io/en/latest/computation/index.html#transmission)
và [fixed tendon XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon-fixed).
Compiled `MjModel`/`MjData`, đặc biệt actuator transmission IDs, tendon
coefficients, `actuator_length` và `actuator_moment`, là oracle runtime; không
đoán transmission từ tên joint/actuator.

## 2. Mục tiêu và ngoài phạm vi

### 2.1 Thuộc P3.2

- Typed transmission contract dùng chung cho direct-joint và fixed-tendon.
- Extract transmission từ compiled MuJoCo model, có rank/null-space evidence.
- Chuyển desired kinematic state/contact motion sang actuator-space command.
- Active-finger mask cho IK command và residual accounting.
- Dynamic metrics tách joint-state residual, actuator-coordinate residual,
  passive drift và active-contact tracking.
- Known-positive/negative dynamic fixtures cho LEAP, Allegro và Shadow chạy qua
  cùng public interface.
- Gỡ `release_blocked` của Shadow chỉ bằng revision riêng sau khi mọi gate pass.

### 2.2 Không thuộc P3.2

- Không thêm neural policy, learned controller, RL hoặc system identification.
- Không đổi model architecture P4, training objective hoặc checkpoint schema.
- Không chạy controlled ablation ba recipe và không regenerate dataset; hai việc
  đó tiếp tục ở P3.1 sau khi P3.2 đóng.
- Không sửa MJCF upstream để thêm equality constraint không tồn tại.
- Không thêm actuator giả cho bốn passive joints.
- Không dùng direct `data.qpos` write sau frame đầu, `body_pos` mutation hoặc
  weld stiffness bất thường để ép fixture pass.

## 3. Contract kiến trúc

### 3.1 Representation

Module mới nằm dưới `qdgrasp/robot/transmission/` và tối thiểu có:

```text
TransmissionState
  joint_names: tuple[str, ...]       # J state dimensions
  actuator_names: tuple[str, ...]    # U control dimensions
  joint_position: [B, J]
  actuator_coordinate: [B, U]        # joint angle hoặc tendon length
  moment_matrix: [B, U, J]           # dl/dq tại state hiện tại
  rank: [B]

ActuatorCommand
  control_target: [B, U]
  projected_joint_delta: [B, J]
  controllable_residual: [B]
  nullspace_residual: [B]
  saturated: [B, U]
  reason: [B]
```

`moment_matrix` dùng thứ tự named joints của `RobotSpec`, không dùng raw qpos
index làm API. Free joint của hand/object không nằm trong `J`.

### 3.2 Strategy modules

- `DirectJointTransmission`: LEAP/Allegro; mỗi actuator-coordinate là một joint
  coordinate. Vẫn extract mapping từ compiled model thay vì giả identity theo
  thứ tự file.
- `FixedTendonTransmission`: Shadow; direct wrist/thumb/finger actuators kết hợp
  bốn tendon rows. Tọa độ actuator tendon lấy từ `target_data.actuator_length`,
  moment row lấy từ `data.actuator_moment` và kiểm lại bằng finite difference.
- Registry chỉ cho phép strategy đã pin theo compiled transmission types; unknown
  transmission fail closed, không import class từ YAML.

### 3.3 Command projection

Với desired delta `Δq*`, command solver tìm actuator-space delta bằng damped
least squares/SVD trên `M = dl/dq`:

```text
Δl = M Δq*
Δq_control = M⁺ Δl
Δq_null = Δq* - Δq_control
```

Không tuyên bố tái tạo full `Δq*` khi `Δq_null != 0`. Candidate bị reject nếu
active-contact motion phụ thuộc vào null-space vượt tolerance. Regularization
gồm joint margin, control range và thay đổi actuator nhỏ; không tối ưu bằng cách
ghi trực tiếp passive qpos trong rollout.

Đối với tendon phi tuyến, `M` được cập nhật tại state hiện tại và command có
step limit; fixed tendon hiện tại tuyến tính nhưng interface không hard-code giả
định đó.

### 3.4 Active-finger IK

Command IK nhận explicit boolean mask `[B, K]`:

- residual/convergence chỉ tính trên active fingertips;
- inactive fingers vẫn chịu joint-limit/collision guard;
- palm/wrist regularization ngăn solver dùng chuyển động toàn bàn tay để che
  một pinch cục bộ;
- static certifier và dynamic validator lưu đúng danh sách active fingers;
- ít nhất hai finger groups active mới đủ điều kiện dynamic success.

Proposal IK của dataset vẫn có thể dùng toàn bộ fingertips. Active mask chỉ là
contract rõ ràng cho grasp/contact set thực tế, không phải cách bỏ qua fingertip
xấu sau khi đã xem kết quả.

## 4. Dynamic protocol và metrics

### 4.1 Protocol bắt buộc

1. Khởi tạo full joint state đúng một lần trước `mj_forward` đầu tiên.
2. Đo root-to-palm transform sau khi initial wrist joints đã được áp.
3. Chuyển initial/target state sang actuator coordinates qua transmission module.
4. Squeeze bằng actuator controls với smoothstep và control-range clipping.
5. Lift bằng mocap-weld root trajectory; không mutate model/body/qpos runtime.
6. Perturb object bằng `xfrc_applied`, sau đó đo contact loads thật.

### 4.2 Metrics bắt buộc

- `transmission_rank`, `joint_state_dimensions`, `control_dimensions`.
- `max_actuator_coordinate_error` và per-actuator error.
- `active_fingertip_tracking_error`; inactive error báo riêng.
- `controllable_residual`, `nullspace_residual`, saturation count.
- passive-joint drift và tendon-length drift.
- palm/root position/rotation tracking.
- active finger count, measured wrench, cone violation, floor support,
  penetration maximum toàn trajectory và lift achieved.

Success không phụ thuộc việc passive joints bám một target không điều khiển được.
Nó phụ thuộc actuator coordinates bám command, active contacts giữ vật, physics
gates pass và mọi state hữu hạn. Tuy nhiên null-space residual vượt threshold
phải reject trước rollout, không được che bằng dynamic success ngẫu nhiên.

## 5. Work breakdown

| ID | Hạng mục | Output chính | Phụ thuộc |
| --- | --- | --- | --- |
| P3.2-00 | Khóa transmission/command contracts và failing fixtures | contracts, test matrix, revision mở | không |
| P3.2-01 | Extract named joint/actuator mapping từ compiled MuJoCo | `robot/transmission/model.py` | 00 |
| P3.2-02 | Implement direct-joint strategy và parity LEAP/Allegro | `direct.py`, tests | 01 |
| P3.2-03 | Implement fixed-tendon strategy, rank và finite-difference oracle | `fixed_tendon.py`, tests | 01 |
| P3.2-04 | Thêm active-finger mask vào fixed/region DLS command path | solver contracts/tests | 00 |
| P3.2-05 | Implement controllable-space projection và rejection reasons | `command.py`, tests | 02/03/04 |
| P3.2-06 | Tích hợp transmission controller vào MuJoCo validator/orchestrator | validator, observer, metrics | 05 |
| P3.2-07 | Known-positive/negative fixtures ba hand | tests + rendered evidence | 06 |
| P3.2-08 | Harden gate, wheel/import/docs và safety budget | scripts/tests/docs | 07 |
| P3.2-09 | Revision gỡ Shadow release block và bàn giao về P3.1 | profile, revision, roadmap | 08 + independent review |

Thứ tự bắt buộc:
`00 → 01 → (02 || 03 || 04) → 05 → 06 → 07 → 08 → 09`.

Không gỡ `release_blocked`, chạy ablation hoặc regenerate dataset ở giữa chuỗi.

## 6. Test matrix bắt buộc

### 6.1 Transmission unit tests

- Synthetic 2-joint/1-tendon model có coefficients đã biết; extracted moment
  row và actuator length phải đúng analytic result.
- `actuator_moment` gần central finite difference của `actuator_length`.
- Named permutation của joints/actuators không đổi kết quả.
- LEAP/Allegro có rank 16 trên 16 state dimensions.
- Shadow có 24 state, 20 controls, rank 20 tại fixtures đã pin.
- Absolute/unknown/duplicate joint mapping fail closed.
- Control clipping và saturation được báo, không im lặng đổi target evidence.

### 6.2 Projection và active-mask tests

- Desired delta trong row-space có null-space residual gần 0.
- Desired delta thuần null-space bị reject.
- Mixed batch giữ reason riêng từng sample.
- Active index+thumb reachable pass dù inactive finger được phép drift trong
  tolerance collision; cùng target với all-finger mask phải fail nếu một inactive
  finger không đạt.
- Không active mask nào có ít hơn hai fingers được dynamic-accept.
- Batch/single parity và deterministic result theo seed/config.

### 6.3 Dynamic fixtures

Mỗi hand phải có ít nhất một known-positive và các known-negative sau:

- no contact;
- palm-only;
- floor-supported object;
- excessive penetration spike;
- actuator target ngoài control range;
- null-space-only Shadow command;
- broken tendon coefficient/mapping.

Known-positive của cả ba hand phải pass cùng stage contract:

- ít nhất hai active fingers;
- lift tối thiểu `0.5 × commanded lift`;
- nonzero perturbation sống sót;
- không floor/palm support;
- penetration `<= 0.002 m`;
- finite state, zero MuJoCo warnings;
- actuator-coordinate tracking trong tolerance đã pin;
- measured contact loads, không dùng predicted force làm nhãn.

Disturbance được phép morphology/mass-normalized, nhưng công thức normalization
phải chung, ghi đơn vị và hash trong protocol; không pin tùy ý một wrench cho
từng fixture sau khi xem pass/fail.

## 7. Safety và giới hạn tài nguyên

Do workstation từng hard-freeze trong ablation, mọi lệnh P3.2 phải:

```bash
timeout <bounded-seconds> env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONHASHSEED=0 <command>
```

- Chạy một hand/fixture tại một thời điểm.
- FK search nếu cần phải có candidate cap hữu hạn và không tự tăng budget.
- Không chạy full ablation trong P3.2.
- Full pytest chỉ chạy sau khi các file test liên quan đã pass riêng.
- Render tối đa các stage/view đã pin; không ghi video/frame sequence dài.
- Nếu timeout, MuJoCo warning, NaN/Inf hoặc I/O stall xuất hiện, dừng và ghi
  evidence; không retry vô hạn với stiffness/timestep khác nhau.

## 8. Gate đóng P3.2

```bash
.venv/bin/python scripts/check_environment.py --profile cpu
.venv/bin/python scripts/check_docs.py --root .
.venv/bin/python scripts/check_phase2.py
.venv/bin/python scripts/check_phase3_2.py
.venv/bin/python -m pytest tests/ -q
git diff --check
git status --short
```

`scripts/check_phase3_2.py` phải kiểm:

1. transmission dimensions/rank cho ba hand;
2. finite-difference transmission parity;
3. controllable/null-space command fixtures;
4. active-mask IK behavior;
5. known-positive/negative dynamic fixtures ba hand;
6. Shadow còn blocked khi gate chưa hoàn tất và chỉ được unblocked bởi revision
   P3.2-09.

## 9. Điều kiện hoàn tất và bàn giao

P3.2 chỉ complete khi:

1. Ba hand dùng cùng transmission interface; không hand-name branch trong
   orchestrator/controller.
2. Shadow known-positive pass bằng 24-state/20-control thật, không mimic/equality
   giả và không runtime qpos write.
3. Negative controllability fixtures fail trước rollout với reason chính xác.
4. Full CPU suite, docs, P2 và P3.2 gates pass trên clean commit.
5. Ảnh squeeze/lift/perturbation của ba hand được render từ chính passing states.
6. Independent reviewer xác nhận transmission math, Shadow fixture và việc gỡ
   `release_blocked`.
7. Revision đóng P3.2 cập nhật P3.1-11 sang complete; P3.1-13/14/15 vẫn pending.

Sau đó mới được:

- chạy controlled ablation ba recipes theo safety budget của P3.1;
- chọn recipe/mixture bằng revision evidence;
- regenerate `DGN-Open-Tiny` hai lần từ clean staging;
- audit hash/class/robot/split coverage và đóng P3 tổng.

### Bàn giao thực tế sang P3.3 — 2026-08-25

P3.2 và corrective P3.2.1 đã đóng tại commit `c7f4061`; transmission/control,
generated-reachable full flow, mutation, determinism và independent review đã
được nghiệm thu trong `REV-20260824-001`. Baseline này được bàn giao cho
`ROADMAP-P3.3-001` để tiếp tục scene contracts và scene-aware validation.

Các work package P3.1 còn lại là release-data backlog độc lập và vẫn giữ gate
riêng; chúng không đổi trạng thái complete của P3.2, đồng thời không cho phép
P3.3 phát hành positive scene dataset trước khi object-level release gate pass.
