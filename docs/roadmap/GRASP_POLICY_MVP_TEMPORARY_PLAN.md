---
document_id: ROADMAP-MVP-001
document_type: plan
title: Kế hoạch tạm thời Grasp Policy MVP — LEAP vertical slice
version: 1.0.0
status: active
date: 2026-08-29
related_plan: ROADMAP-001
temporarily_precedes:
  - ROADMAP-P3.4.3-001
  - ROADMAP-P3.5-001
  - ROADMAP-001#P4
  - ROADMAP-001#P5
release_class: experimental_non_release
active_hand: leap_hand
environment_id: QDGrasp-DexAcquire-MVP-v0
artifact_id: QDGrasp-Leap-Grasp-MVP
latest_revision_record: docs/revisions/REV-20260829-001-temporary-grasp-policy-mvp.md
---

# Kế hoạch tạm thời Grasp Policy MVP — LEAP vertical slice

## 0. Quyết định thực thi

Plan này là **đường thực thi ưu tiên tạm thời** để tạo sớm một checkpoint gắp
vật chạy end-to-end và có tỷ lệ thành công cao trong phạm vi hẹp. Trong thời gian
plan ở trạng thái `active`, công việc mới ưu tiên theo thứ tự:

1. làm môi trường LEAP tối thiểu chạy được;
2. tạo expert trajectory có positive thật;
3. học policy, lưu checkpoint và đánh giá sau khi reload;
4. chỉ sau đó quay lại mở rộng correctness/generalization theo P3.4.3, P3.5,
   P4 và P5.

Quyền ưu tiên này **không supersede và không đóng giả** plan cũ. Mọi blocker,
gate và finding đã đo của P3.4.3/P3.5/P4/P5 vẫn giữ nguyên. Artifact của plan
này luôn mang nhãn `experimental_non_release`; nó không được dùng để tuyên bố
P3, P4 hay P5 complete.

## 1. Mục tiêu MVP

Tạo một policy cho **LEAP Hand** có thể tiếp cận, đóng tay, nâng và giữ một vật
đơn trên mặt bàn trong MuJoCo, với đầu vào state và phạm vi object/pose được khóa
trước. Đầu ra tối thiểu phải gồm:

- environment `QDGrasp-DexAcquire-MVP-v0` có `reset/step` xác định rõ;
- expert dataset sinh từ controller/recipe hiện có và chỉ giữ positive vật lý;
- checkpoint actor-critic reload được từ installed package;
- train/eval config, seed list, hash và raw episode ledger;
- script/notebook có resume cho local, Kaggle và Colab Pro;
- báo cáo success/failure trên tập eval đã khóa trước vòng tune cuối.

MVP ưu tiên **độ tin cậy trong một vertical slice** hơn độ rộng. “Thành công cao”
trong tài liệu này chỉ có nghĩa là đạt gate M5 trong phạm vi §2; không có nghĩa
generalize tới raw mesh, clutter, scene tùy ý, tay khác hoặc vật thật.

## 2. Phạm vi đã khóa

| Thành phần | MVP bắt buộc | Hoãn sau MVP |
| --- | --- | --- |
| Robot | LEAP Hand, một preset/version pin | Wonik Allegro; Shadow vẫn paused theo ADR-0008 |
| Simulator | MuJoCo CPU làm physics/oracle | MuJoCo Warp, ManiSkill, Isaac Lab và GPU physics parity |
| Scene | một table cố định, một target, không clutter | scene resolver, virtual drop, bin/tray/clutter |
| Object | cuboid family chuẩn hóa, 6 size train + 4 size held-out | raw mesh, CoACD, arbitrary object/category |
| Khởi tạo | pose/yaw và friction trong miền pin; object settle trước episode | camera/scene randomization rộng |
| Observation | privileged simulator state | RGB/depth/point cloud/tactile học được |
| Action | residual palm pose + finger synergy quanh controller prior | full 16-DOF policy từ đầu, torque control |
| Learning | behavior cloning warm start + residual PPO | diffusion/flow generator, multi-task/offline RL |
| Claim | experimental narrow-scope checkpoint | release model, multi-hand hoặc sim-to-real claim |

Tập object cụ thể, khoảng size/mass/friction, palm spawn box và randomization
range phải nằm trong một versioned YAML. Không được thay các khoảng này sau khi
đã khóa eval seeds; thay đổi là một MVP version mới.

## 3. Kiến trúc nhanh nhất để ra model

### 3.1 Controller prior

Dùng state machine `approach -> enclose -> lift -> retain` từ recipe LEAP đã có
positive vật lý làm prior. Controller phải đạt ít nhất `90/100` episode trên
canonical development set trước khi tạo demonstration. Nếu chưa đạt, sửa
environment/controller trước; không dùng PPO để che lỗi scene/contact.

### 3.2 Policy và action

Policy MLP nhỏ, mặc định hai hidden layer 256 unit, nhận state đã normalize và
sinh Gaussian residual có bound. Action MVP tối đa tám chiều:

- `delta_xyz` và `delta_rot` của palm target;
- hai finger-synergy residual áp lên named joints theo coupling đã pin.

Residual được cộng vào lệnh của controller prior rồi clamp bởi workspace, joint
limit, velocity/acceleration và contact safety budget. Policy không có action
trực tiếp lên object state, solver state hoặc termination flag.

### 3.3 Observation

Observation v0 chỉ dùng state để loại computer vision khỏi critical path:

- LEAP joint position/velocity và palm pose/twist;
- target pose/twist trong palm frame;
- fingertip position, contact bit và force summary;
- previous action, controller phase và normalized time-to-timeout.

Mọi field, shape, unit và frame có schema/version. Normalization statistics chỉ
fit từ train split và được lưu cùng checkpoint.

### 3.4 Học hai bước

1. **Behavior cloning:** học residual/expert action từ trajectory thành công,
   tạo baseline nhanh và ổn định.
2. **Residual PPO:** fine-tune trong simulator để xử lý pose/friction variation.
   PPO dùng clipped objective, GAE và bounded Gaussian actor; hyperparameter nằm
   trong config hash, không hard-code trong notebook.

Nếu checkpoint BC đã đạt toàn bộ gate M5 thì vẫn chạy một PPO confirmation ngắn;
chỉ promote PPO nếu không làm giảm locked-eval success quá 2 điểm phần trăm. Luôn
giữ checkpoint BC làm rollback.

## 4. Environment và định nghĩa thành công

Một episode chỉ `success=true` khi đồng thời:

1. target được nâng ít nhất `0.05 m` so với settled height;
2. giữ trên ngưỡng đó liên tục ít nhất `0.50 s`;
3. terminal có contact của ít nhất hai finger group với target;
4. target không còn support-assisted bởi table ở cửa sổ retain;
5. không NaN/Inf, không hard penetration/force/impulse violation;
6. không teleport/reset target sau bước đầu và không vượt timeout.

Mọi timeout, safety termination, invalid state và simulator error đều tính là
failure trong mẫu số. Reward dùng để học không thay verdict trên:

`approach progress + target contact + enclosure + lift progress + retain bonus`

trừ `penetration + excess force/impulse + action-rate + drop + timeout`. Reward
component được log riêng từng episode để tránh tối ưu nhầm proxy.

## 5. Dữ liệu và chống leakage

- Sinh demonstration từ nhiều seed/pose trong **train range**, bỏ trajectory
  không đạt chính predicate §4 nhưng vẫn ghi failure statistics của generator.
- Chia theo `object_variant_id + seed`, không chia theo frame; trajectory cùng
  episode không được rơi vào nhiều split.
- `dev` dùng chọn hyperparameter. `locked_eval` được tạo và hash trước vòng tune
  cuối, policy không được quan sát reward/outcome của tập này trong training.
- Held-out object size không xuất hiện trong expert, replay buffer hay
  normalization fit.
- Mỗi checkpoint gắn commit, environment/config/normalizer/dataset hashes.

## 6. Work package và thứ tự thi công

| ID | Công việc | Đầu ra bắt buộc | Điều kiện qua |
| --- | --- | --- | --- |
| MVP-00 | Khóa scope/config/eval protocol | YAML schema, object ranges, seed manifests | config round-trip; eval manifest immutable |
| MVP-01 | Adapter environment tối thiểu | `QDGrasp-DexAcquire-MVP-v0` | deterministic reset; random policy không được false-positive |
| MVP-02 | Controller prior + expert recorder | trajectory train/dev và generator ledger | controller canonical `>=90/100`; zero invalid accepted demo |
| MVP-03 | BC trainer/checkpoint | actor + normalizer + resume state | reload parity; dev success `>=75%` |
| MVP-04 | Residual PPO fine-tune | best/last checkpoint và learning curves | không safety bypass; best không thấp hơn BC quá 2 pp |
| MVP-05 | Locked evaluation | raw per-episode ledger + aggregate/95% CI | đạt đủ ba tier ở §7 |
| MVP-06 | Cloud handoff | Kaggle/Colab runner, cache và resume guide | fresh runtime cài exact commit, resume, eval checkpoint |
| MVP-07 | Đóng MVP | model card experimental + revision record | artifact hash đủ; limitations và backlog link đầy đủ |

Dependency cứng là `00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07`. Có thể dựng
cloud harness song song sau MVP-01, nhưng không được báo success trước MVP-05.

## 7. Gate nghiệm thu model

| Tier | Miền đánh giá | Cỡ mẫu tối thiểu | Gate |
| --- | --- | ---: | ---: |
| A — canonical | pose/size trung tâm, deterministic fixture | 100 episode | raw success `>=95%` |
| B — in-scope randomized | 6 train size, pose/yaw/friction đã pin | 300 locked seeds | raw success `>=85%` và Wilson 95% lower bound `>=80%` |
| C — held-out size | 4 size không dùng trong train, cùng cuboid family | 200 locked seeds | raw success `>=70%` |

Ngoài success rate, cả ba tier phải có `invalid_state=0`, `safety_violation=0` và
`checkpoint_reload_mismatch=0`. Phải báo thêm drop, timeout, approach miss,
contact loss, penetration và simulator error theo failure bucket; không chỉ báo
một số trung bình đẹp.

Gate B là tiêu chí để gọi checkpoint là **high-success MVP**. Nếu chỉ A pass,
artifact được gọi là demo controller/model, chưa được gọi MVP hoàn tất.

## 8. Ngân sách thực thi và stop rule

- Local chỉ dùng cho smoke, deterministic replay và episode inspection.
- Kaggle/Colab GPU dùng train neural network; physics MVP vẫn là MuJoCo CPU,
  không biến notebook GPU thành bằng chứng GPU physics.
- Mỗi cloud run mục tiêu tối đa 6 giờ, checkpoint định kỳ, lưu RNG/optimizer và
  resume được sau runtime thu hồi.
- Tối đa ba vòng tune sau khi có BC baseline. Nếu B vẫn dưới 85%, dừng scale và
  xử lý failure bucket lớn nhất; không mở rộng simulator/object/hand cùng lúc.
- Eval locked chỉ chạy một lần cho candidate cuối mỗi vòng; thay protocol hoặc
  seed bắt buộc bump version và giải thích trong revision.

## 9. Các phần cố ý hoãn

Các phần sau không bị xóa, chỉ trở lại execution queue sau MVP-07:

- P3.4.3: CUDA evidence/review và contact-rich release-grade closure;
- P3.5: raw mesh/CoACD, scene load/generate/drop/settle, Gym vector contract,
  two-active-hand/backend parity và independent review;
- P4: point-cloud encoder, HandGraph conditioning và palm+joint flow/diffusion;
- P5: multi-hand, held-out category, large-scale simulator evaluation/ablation;
- Shadow Hand: tiếp tục paused theo ADR-0008.

Không được lấy việc hoãn này làm căn cứ xóa blocker, đổi failure thành pass hoặc
phát hành dataset/checkpoint đang `release_blocked`.

## 10. Điều kiện rời plan tạm thời

Plan chỉ được ghi `MVP complete` khi MVP-00–07 có artifact thật và §7 pass sau
checkpoint reload. Khi đó roadmap chính được mở lại theo thứ tự dựa trên failure
evidence của MVP:

1. đóng những correctness item trực tiếp ảnh hưởng success/safety;
2. mở rộng object/pose rồi observation;
3. thêm Allegro và simulator/backend thứ hai;
4. hoàn tất raw asset/scene pipeline và release governance.

Nếu MVP không đạt sau ba vòng tune, trạng thái là `blocked_with_evidence`, không
phải complete; báo cáo phải giữ checkpoint tốt nhất, failure ledger và đề xuất
scope/architecture tiếp theo.
