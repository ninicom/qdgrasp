---
document_id: ROADMAP-P3.4.3-002
document_type: plan
title: Phân rã thực thi Phase 3.4.3 thành sub-phase S0–S12
version: 1.0.0
status: active
date: 2026-08-27
parent_plan: ROADMAP-P3.4.3-001
related_decision: ADR-0008
requirements_manifest: docs/roadmap/phase3_4_3_requirements.yaml
active_hands:
  - leap_hand
  - wonik_allegro
paused_hands:
  - shadow_hand
---

# Phase 3.4.3 — phân rã thực thi

`ROADMAP-P3.4.3-001` là contract. Tài liệu này là **thứ tự thi công**: nó chia
contract đó thành mười ba sub-phase có thể làm xong từng cái một, mỗi cái kết
thúc bằng test xanh và một status transition trong
`docs/roadmap/phase3_4_3_requirements.yaml`.

Phân rã không thêm, bớt hay nới bất kỳ requirement nào. Mọi B/G/C ID dưới đây
đến từ contract; ID không xuất hiện ở đây là lỗi mapping, không phải miễn trừ.

## 0. Quy tắc chuyển trạng thái

Một requirement chỉ chuyển khỏi `pending` khi có đủ bốn thứ, và bốn thứ này là
bốn trạng thái riêng — `implemented` không mua được `evidenced`:

1. `implementation_refs` trỏ tới code thật đang chạy;
2. `test_ids` trỏ tới test tồn tại và pass;
3. `evidence_refs` trỏ tới artifact tồn tại trên commit sạch;
4. reviewer disposition cho các gate cần review.

Sub-phase nào không chạy được trong môi trường hiện tại thì requirement của nó
nhận `blocked` kèm `blocker_reason` đo được, **không** nhận `passed` và cũng
không bị bỏ khỏi manifest.

## 1. Bảng sub-phase

| Sub-phase | Nội dung | Gate contract | Blocker đóng | Môi trường |
| --- | --- | --- | --- | --- |
| S0 | Requirements ledger + closure checker + exit-code semantics | G00, C08 | B-09 | CPU |
| S1 | Active-scope registry cho ADR-0008 | G05 | B-10 | CPU |
| S2 | Typed contract v2 fail-closed + schema version | C01 | B-11 | CPU |
| S3 | SafetyBudget v2 + rolling impulse/episode | G01, G02 | B-01, B-02 | CPU |
| S4 | Contact taxonomy, scene damage, terminal semantics | C03 | B-12, B-15 | CPU |
| S5 | Trajectory v2 (timebase/palm/frame) + ReplayCapsule v1 | G03, G06 | B-04, B-06 | CPU |
| S6 | Certification và release invariants | G04 | B-05 | CPU |
| S7 | Objective/CEM/refinement fail-closed | C04 | B-16 | CPU |
| S8 | Backend hydration, scene signature, CPU summary v2 | C02 | B-13, B-14 | CPU |
| S9 | GPU contact summary + exact finalist export | G07 | B-03 | CPU code, GPU evidence |
| S10 | CUDA capability/parity/perf + operations harness | G08, C07 | B-08, B-19 | **Kaggle T4 bắt buộc** |
| S11 | ContactRich-Active-Tiny + coverage/ablation/handoff artifact | G09, C05, C06 | B-07, B-17, B-18, B-20 | CPU + GPU-derived |
| S12 | Immutable packet, independent review, P4 handoff | G10, G11 | — | reviewer độc lập |

Thứ tự phụ thuộc giữ nguyên đồ thị mục 3 của contract. S0–S8 không cần GPU và
phải xong trước; S9 viết code và oracle test trên CPU nhưng verdict GPU nằm ở
S10; S11 chỉ tạo release artifact khi S0–S10 pass.

## 2. Định nghĩa từng sub-phase

### S0 — Ledger và closure checker

Sản phẩm: loader/validator cho manifest requirement, một closure runner mới, và
sửa exit-code của checker cũ.

- `qdgrasp/roadmap/requirements.py`: đọc manifest, kiểm duplicate ID, unknown
  status, mapping hai chiều, và tính verdict đóng.
- `scripts/check_phase3_4_3.py`: `--scope active`, `--profile cpu|release`,
  in một JSON verdict duy nhất, exit `0` chỉ khi exact scope pass; `1` fail;
  `2` paused/not-applicable; `3` incomplete/partial.
- `scripts/check_phase3_4.py`: `PARTIAL` không còn thoát `0` (B-09).

Exit: manifest hợp lệ, `mapped_requirements == total_requirements`, checker
trả `3` khi còn requirement `pending`.

### S1 — Active-scope registry

Sản phẩm: một nguồn sự thật cho active/paused hand thay vì tuple lặp lại.

- `qdgrasp/config/active_scope.py`: `ACTIVE_HANDS`, `PAUSED_HANDS`,
  `resolve_workload_hands()`, `require_release_scope()`.
- Default generation/training/backend/RL config resolve đúng 2/2 active.
- Shadow chỉ vào workload khi `experimental_shadow=True`; artifact khi đó mang
  `non_release=True`.

Exit: không default path nào chọn Shadow; preset và compatibility test của
Shadow vẫn chạy được.

### S2 — Typed contract v2

Sản phẩm: `DynamicGraspRequest`, `DynamicGraspTrajectory`, `ContactEvent`,
`DynamicSearchOutcome` fail-closed đầy đủ, cộng schema version cho từng payload.

Kiểm tối thiểu: non-finite, sai rank/shape, time không tăng, `Δtime` lệch sample
period, quaternion không chuẩn hóa, stage order sai, contact index ngoài rollout,
frame không trực giao, positive thiếu typed CPU certificate.

Exit: invalid corpus fail-closed, valid round-trip byte-stable.

### S3 — SafetyBudget v2

Sản phẩm: mapping bắt buộc field → sensor/aggregation/threshold/reason, sáu
observer còn thiếu (wrist force, wrist torque, joint/tendon load, non-target
translation/rotation/velocity), và tách `peak` / `windowed` /
`per-contact cumulative` / `per-object cumulative`.

Rolling impulse chuyển từ tumbling block sang ring buffer theo timestamp; contact
episode tách khi pair biến mất; nhiều contact point trên cùng geom pair không
nhân đôi thời gian.

Exit: coverage 100% field, zero `unavailable_fields`, mutation từng threshold
tạo đúng reason, xung lực nằm giữa biên hai cửa sổ vẫn bị phát hiện.

### S4 — Taxonomy và terminal semantics

Sản phẩm: pair semantics có kiểu (target-support, robot-support, target-robot,
non-target-*, self-contact allowlist theo profile, unknown/forbidden), verdict
scene-damage, và terminal certifier chỉ xét **target**-support khi đánh giá
support release. Primitive timeout tạo `transition_timeout:<condition>` negative
chứ không đồng nghĩa thành công.

### S5 — Trajectory v2 và ReplayCapsule v1

Sản phẩm: timestamp từ `data.time` hoặc `index * sample_every * dt`, palm pose
từ exact palm body id với quaternion thật, contact event mang simulator-step
index riêng, schema `qdgrasp/dynamic-trajectory/v2`, và một capsule đủ để CPU
replay finalist mà không cần process memory của search.

### S6 — Certification và release

Sản phẩm: `certify_replay` chỉ certified khi cả hai `passed=true`;
`release_decision` v2 nhận outcome refs; positive yêu cầu typed CPU evidence với
backend id, capsule hash, command hash, model hash, timestep, terminal và safety
cert; reason ledger giữ denominator bảo toàn.

### S7 — Objective, CEM, refinement

Sản phẩm: objective schema đủ term và hash vào manifest; NaN/Inf/missing term
hard-reject; forbidden/damaging luôn `-inf`; CEM bounded, deterministic, trả
`no_feasible_elite` / `budget_exhausted` thay vì refit; refine chỉ nhận
CPU-confirmed positive và không đổi threshold.

### S8 — Backend parity

Sản phẩm: `reset(request_batch)` hydrate đúng per-world state; world isolation;
`SceneSignature` hash toàn bộ topology-affecting fields; `num_worlds` là live
worlds còn capacity là field riêng; CPU và GPU cùng phát `RolloutSummary v2`.

### S9 — GPU summary và finalist export

Sản phẩm: MJWarp rollout thu contact identity/force/frame/impulse/penetration/
work/overflow; `hard_reject` bao gồm forbidden/damaging/budget violation; export
finalist trả capsule exact. Code và oracle-parity test chạy CPU; verdict GPU ở S10.

### S10 — CUDA gate trên Kaggle

Máy phát triển không có thiết bị NVIDIA, nhưng S10 **không được bỏ**: gate này
chạy trên Kaggle T4 qua harness notebook, đúng như §14–§15 của contract gốc.
Việc của sub-phase này là làm harness chạy được và tạo evidence thật:

1. mở rộng `scripts/build_phase3_4_notebook.py` thành harness P3.4.3 với exact
   public commit và exact CUDA/Warp lock;
2. capability probe đòi **đồng thời** contact force, frame, distance,
   identity/count và overflow telemetry;
3. ba tầng parity (no-contact, pinned single contact, full active-hand
   finalist) đối chiếu CPU oracle capsule sinh ở S5–S9;
4. performance gate với median nhiều run, GPU/CPU `>=2x`, VRAM `<=14 GiB` đo
   bằng NVML;
5. sanitizer, overflow/truncation, OOM và fallback đều zero;
6. dry-run, resource cap, atomic checkpoint/resume và retention theo C07;
7. ghi environment fingerprint, raw log hash và commit vào packet immutable.

Trong lúc chưa có run Kaggle, requirement S10 giữ `pending` với
`blocker_reason` chỉ rõ nó đang chờ **execution** trên T4, không phải chờ
implementation. Không CPU number nào được ghi dưới schema CUDA, và không
threshold nào được nới để né gate.

### S11 — Dataset artifact

`QDGrasp-ContactRich-Active-Tiny` với 2 hand × 3 environment class × 2 clutter
tier × 3 generation mode, positive/negative fixture bắt buộc, paired
static-fail/dynamic-pass, controlled ablation, split group-disjoint, loader
public v2, dataset card và rendered stage QA.

### S12 — Packet và review

Immutable packet trên commit sạch, reviewer độc lập ký exact completeness
manifest hash, rồi mới mở P4 contact-rich input.

## 3. Điều kiện dừng

Phân rã này không đổi Definition of Done của contract. Khi S10 chưa có run
Kaggle thật hoặc S12 chưa có reviewer ký, trạng thái tổng của P3.4.3 là
`active`, và `release_blocked` giữ `true`.
