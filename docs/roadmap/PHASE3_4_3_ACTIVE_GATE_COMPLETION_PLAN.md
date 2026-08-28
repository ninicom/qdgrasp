---
document_id: ROADMAP-P3.4.3-001
document_type: plan
title: Kế hoạch Phase 3.4.3 — hoàn tất correctness gate cho contact-rich active hands
version: 1.3.0
status: blocked
date: 2026-08-28
revises: ROADMAP-P3.4.3-001@1.2.0
parent_plan: ROADMAP-P3.4-001
related_decision: ADR-0008
related_plans:
  - ROADMAP-P3.4.1-001
  - ROADMAP-P3.4.2-001
active_hands:
  - leap_hand
  - wonik_allegro
paused_hands:
  - shadow_hand
evidence_root: evidence/phase3_4_3
requirements_manifest: docs/roadmap/phase3_4_3_requirements.yaml
latest_revision_record: docs/revisions/REV-20260828-016-phase3-4-3-review-remediation.md
---

# Phase 3.4.3 — Active contact-rich gate completion

## 1. Mục tiêu và ranh giới verdict

Plan này xử lý các lỗ hổng logic còn lại sau static review P3.4, tạo một
successor contract có thể kiểm tra bằng máy cho hai active hand và bàn giao
contact-rich input đúng cho P4.

Plan **không** đổi kết luận lịch sử:

- P3.4 gốc yêu cầu LEAP + Allegro + Shadow và vẫn
  `paused_by_ADR-0008`, không phải `pass`;
- Shadow không chạy trong default generation, training, backend selection,
  release hoặc RL gate của plan này;
- artifact hai-hand không dùng lại verdict hay tên release ba-hand cũ;
- muốn đóng lại contract ba-hand phải có ADR mở Shadow và thực hiện
  `ROADMAP-P3.4.2-001` hoặc successor được duyệt.

Artifact đích của plan này là
`QDGrasp-ContactRich-Active-Tiny`, schema successor và verdict riêng
`P3.4.3-ACTIVE-PASS`. Chỉ verdict đó mới được dùng để gỡ block contact-rich
input của P4. Nó phải luôn mang:

- `active_hands=[leap_hand, wonik_allegro]`;
- `paused_hands=[shadow_hand]`;
- `coverage=2/2_active`;
- `three_hand_coverage=false`;
- liên kết `ADR-0008`.

## 2. Baseline lỗi phải sửa

Static audit tại commit `a3eb016` xác định hai mươi nhóm blocker:

| ID | Blocker | Mức | Hệ quả hiện tại |
| --- | --- | --- | --- |
| B-01 | Safety verdict bỏ sáu giới hạn đã khai báo | P0 | Có thể phát hành positive vượt wrist/joint/scene budget |
| B-02 | Impulse dùng tumbling window và reset trước verdict | P0 | Có thể bỏ lọt impulse vượt ngưỡng qua biên cửa sổ |
| B-03 | MJWarp summary không quan sát contact/safety | P0 | World hữu hạn nhưng unsafe vẫn thành finalist |
| B-04 | GPU finalist không mang exact control sequence | P0 | CPU không replay đúng candidate đã được GPU xếp hạng |
| B-05 | Certifier có thể chứng nhận hai matching failures | P1 | Failure có thể bị fold thành release positive |
| B-06 | Contact-rich trajectory sai timebase/palm pose | P1 | Trajectory không đúng vật lý và không dùng an toàn cho learning |
| B-07 | Manifest trộn pause metadata với sample/split | P1 | Count/hash/split contract không khớp shard |
| B-08 | CUDA capability/perf harness có false-positive/crash | P1 | Gate có thể support sai hoặc lỗi ở nhánh pass |
| B-09 | Checker stale và `PARTIAL` trả exit 0 | P1 | CI có thể hiểu nhầm partial là phase pass |
| B-10 | Runtime default vẫn tự chọn Shadow | P2 | Default workload vi phạm ADR-0008 |
| B-11 | Typed contracts không reject đầy đủ non-finite/shape/time/quaternion sai | P0 | Malformed trajectory/evidence có thể đi tới writer/certifier |
| B-12 | SceneRoles nhập nhằng target-support/hand-support và allowlist self-contact quá rộng | P0 | Terminal support và scene-damage verdict có thể sai |
| B-13 | Backend reset không hydrate request state; scene signature chưa khóa đủ topology | P0 | Các world/candidate không thực sự đại diện request và cache có thể reuse sai model |
| B-14 | CPU oracle summary cũng thiếu contact/objective semantics chung | P0 | Không có oracle độc lập cho GPU contact-rich summary |
| B-15 | Primitive timeout có thể advance dù transition vật lý chưa đạt | P0 | Support-release/lift/perturb có thể bị bỏ qua nhưng sequence vẫn kết thúc |
| B-16 | Objective/CEM không fail-closed với missing/NaN term hoặc all-rejected elite | P0 | Search có thể tiếp tục/refit từ cohort không feasible |
| B-17 | Dataset plan chưa khóa đủ environment/clutter/mode/negative coverage | P1 | Tiny artifact có thể đúng hash nhưng không đạt coverage vật lý gốc |
| B-18 | Thiếu controlled static-vs-dynamic ablation và tuning-leakage guard | P1 | Không chứng minh phần tăng đến từ dynamics thay vì threshold/tuning |
| B-19 | Dry-run/resource cap/atomic checkpoint/resume/dedup/retention chưa thành gate | P1 | Cloud run có thể OOM, hết giờ hoặc tạo duplicate/partial artifact |
| B-20 | Schema migration, public loader/API, dataset card và rendered QA chưa khóa | P1 | Artifact không đủ điều kiện bàn giao cho P4 hoặc phát hành |

Mọi blocker P0/P1 là release-blocking. B-10 là scope-blocking: phải xong trước
khi tạo artifact mới dù preset Shadow vẫn được giữ. Danh sách B-01–B-20 là
closed set cho revision này; blocker mới phát hiện phải nhận ID mới và làm
`COMPLETENESS` fail cho tới khi plan được revision.

## 3. Thứ tự phụ thuộc

```text
G00 status/contract lock
 ├─> G01 complete safety semantics ─> G02 contact observer correctness
 ├─> G03 trajectory integrity
 ├─> G04 certificate/release invariants
 └─> G05 ADR runtime scope enforcement

G01 + G02 + G03 + G04
 └─> G06 CPU oracle capsule
      └─> G07 GPU contact summary + exact finalist export
           └─> G08 CUDA capability/parity/performance gate

G03 + G04 + G05 + G06
 └─> G09 dataset v2 generation/manifest

G08 + G09
 └─> G10 immutable evidence + independent review
      └─> G11 P4 contact-rich handoff
```

Không chạy G08/G09 để tạo release evidence trước khi G01–G07 khóa xong code và
schema. Kết quả thăm dò sớm chỉ ghi `diagnostic`, không được promote thành
release artifact.

## 4. Work package và gate

### G00 — Status ledger và machine-readable verdict

**Việc làm**

1. Tạo một ledger duy nhất cho P3.4 work package với ba trường độc lập:
   `implementation_state`, `gate_state`, `execution_state`.
2. Thay hard-coded status mâu thuẫn trong `scripts/check_phase3_4.py`.
3. Tách ba command:
   - CPU correctness check;
   - active release-candidate check;
   - historical three-hand status report.
4. Quy ước exit code:
   - `0`: exact requested gate pass;
   - `1`: gate fail;
   - `2`: paused/not-applicable;
   - `3`: incomplete/partial;
   - lỗi cấu hình dùng nonzero riêng nhưng không được map thành pass.
5. Output JSON bắt buộc có `verdict`, `release_blocked`,
   `closure_scope`, `active_hands`, `paused_hands` và artifact refs.

**Gate pass**

- Không còn câu “CUDA chưa implement” hoặc “dataset chưa tồn tại” khi ledger
  hiện hành nói ngược lại.
- Không đường `PARTIAL`, `PAUSED`, `BLOCKED` nào thoát 0.
- Checker active không thể tạo claim `three_hand_coverage=true`.

### G01 — SafetyBudget v2: mọi giới hạn đều có observer

**Việc làm**

1. Lập mapping bắt buộc từ từng field của `ContactSafetyBudget` tới:
   sensor/source, đơn vị, aggregation, threshold và failure reason.
2. Bổ sung đo:
   - wrist force và torque;
   - joint actuator/tendon load;
   - non-target translation;
   - non-target rotation;
   - non-target velocity.
3. Tách `peak`, `windowed`, `per-contact cumulative` và
   `per-object cumulative`; không dùng một số thay nhiều semantics.
4. Budget field không có sensor hợp lệ phải fail compile/preflight, không
   silently bỏ qua.
5. Kết quả safety chứa `measured_fields`, `unavailable_fields`,
   `violated_fields` và minimum margin kèm field tạo margin.

**Gate pass**

- Coverage matrix đạt 100% field, không `unavailable_fields`.
- Mutation từng threshold tạo đúng failure reason và không bị terminal score
  mua lại.
- Positive chỉ hợp lệ khi mọi limit có số hữu hạn và margin không âm.

### G02 — Rolling impulse và contact episode correctness

**Việc làm**

1. Thay accumulator reset theo block bằng deque/ring buffer theo timestamp cho
   đúng cửa sổ trượt.
2. Tính verdict trước/sau update theo protocol cố định và tài liệu hóa endpoint.
3. Tách contact episode khi pair biến mất; recontact không thừa hưởng duration/
   work của episode trước nếu schema định nghĩa per-episode.
4. Xử lý nhiều contact point trên cùng geom pair mà không nhân đôi thời gian.
5. Dùng simulator timestep thật; observer không được đếm hai lần một control
   interval.

**Gate pass**

- Xung lực nằm giữa biên hai cửa sổ vẫn bị phát hiện.
- Dịch thời điểm cùng waveform dưới một timestep không đổi verdict.
- Analytic force traces khớp normal/tangential impulse trong tolerance đã pin.
- Không NaN/Inf hoặc negative duration/work.

### G03 — Trajectory v2: timebase, frames và palm pose

**Việc làm**

1. Timestamp lấy từ `data.time` hoặc sample index nhân đúng
   `sample_every * simulator_dt`.
2. Contact event mang simulator-step index và sample association riêng; không
   ép mọi tail event về last state index.
3. Palm pose lấy exact palm body ID, world position và quaternion thực; không
   dùng body index 1 hay identity quaternion.
4. Ghi rõ quaternion order, world/object/robot frames và state sampling rate.
5. Thêm invariant:
   - time tăng nghiêm ngặt;
   - `Δtime` bằng declared sample period;
   - quaternion chuẩn hóa;
   - mọi contact timestamp nằm trong rollout horizon.
6. Nâng schema thành `qdgrasp/dynamic-trajectory/v2`; v1 loader chỉ đọc ở
   legacy mode và không cho release.

**Gate pass**

- Pose Palm trong record khớp simulator body state tại mọi sampled frame.
- Duration từ timestamp khớp rollout duration.
- Serialize/deserialize byte-stable và không mất frame/time metadata.

### G04 — Certification và release invariants

**Việc làm**

1. `certify_replay` chỉ certified khi search và replay đều `passed=true`;
   matching failures được giữ làm negative parity evidence.
2. `release_decision` nhận hoặc xác minh original outcome refs, không chỉ
   nhận hai boolean certificate rời.
3. `DynamicSearchOutcome(passed=True)` yêu cầu typed CPU evidence:
   backend ID, exact input/capsule hash, command hash, model hash, timestep,
   terminal cert và safety cert.
4. Từ chối empty dict, string truthy, stale schema hoặc evidence hash không
   khớp.
5. Reason ledger đếm riêng searched, GPU-survived, replayed, CPU-confirmed và
   released; mẫu không replay không nằm trong denominator released.

**Gate pass**

- Hai matching failures không thể tạo positive.
- GPU-only evidence, mismatched command hash, stale model hash và unsafe replay
  đều fail closed.
- Positive contract chỉ có một đường tạo qua release decision v2.

### G05 — ADR-0008 runtime scope enforcement

**Việc làm**

1. Tạo central active-scope registry/config thay vì lặp tuple hand ở nhiều file.
2. Default data/generation/training/backend/RL config chỉ chọn LEAP + Allegro.
3. Shadow chỉ chạy khi `experimental_shadow=true` cùng explicit diagnostic
   purpose; artifact luôn gắn `non_release=true`.
4. Giữ parser, preset, asset provenance và targeted compatibility tests của
   Shadow.
5. Checker quét default config để từ chối Shadow selection không explicit.

**Gate pass**

- Mọi default workload resolve đúng 2/2 active hand.
- Không thể tạo release manifest/checkpoint có Shadow khi ADR còn hiệu lực.
- Explicit Shadow diagnostic vẫn hoạt động và không bị ghi thành release
  evidence.

### G06 — ReplayCapsule v1 cho CPU oracle

**Việc làm**

1. Định nghĩa typed `ReplayCapsule` chứa:
   - scene/model/profile/asset hashes;
   - initial qpos/qvel/mocap/object state;
   - exact control sequence và dtype;
   - timestep, solver options, friction/mass;
   - seed, strategy parameters và safety budget hash.
2. Hash canonical payload; command tensor có shape/dtype/hash rõ ràng.
3. CPU oracle replay capsule trực tiếp, không regenerate candidate từ seed.
4. Lưu capsule cho finalist và negative parity sample; bulk population không
   bắt buộc lưu full tensor nếu không thành finalist.

**Gate pass**

- Replaying cùng capsule hai lần trên CPU cho cùng outcome và evidence hash.
- Thay một control/state byte làm hash đổi và certificate cũ bị từ chối.
- Finalist export/import không phụ thuộc process memory của search.

### G07 — GPU contact summaries và exact finalist export

**Việc làm**

1. MJWarp rollout phải thu contact count, identity/class, force/frame, impulse,
   penetration, work và overflow/truncation telemetry cần cho SafetyBudget v2.
2. `RolloutSummary.hard_reject` bao gồm forbidden/damaging/budget violation,
   không chỉ NaN/Inf.
3. Export finalist trả `ReplayCapsule` exact hoặc immutable ref tới capsule,
   không chỉ `DynamicGraspRequest`.
4. World có contact buffer overflow, unavailable field hoặc non-finite safety
   metric bị reject toàn world.
5. Nếu pinned MJWarp version còn defect upstream uninitialized read:
   - thử compatibility matrix với phiên bản mới đã pin;
   - nếu chưa có bản hợp lệ, chọn backend fallback đã qua capability/parity
     gate hoặc giữ GPU gate blocked;
   - không vá bằng cách bỏ sanitizer hay lọc world xấu sau rollout.

**Gate pass**

- Unsafe finite world bị hard-reject.
- Mỗi surviving finalist có capsule exact và CPU replay result.
- Identical-world cohort không divergence không giải thích; sanitizer fail
  closed.

### G08 — CUDA capability, parity và performance

**Việc làm**

1. Capability probe yêu cầu **đồng thời** contact force, frame, distance,
   identity/count và overflow telemetry; `pos` một mình không đủ.
2. Sửa key VRAM thống nhất thành một schema; không hard-code OOM bằng 0.
3. Scene benchmark fail phải làm toàn gate fail có lý do, không tạo thiếu-key
   exception.
4. Đo device memory bao phủ Warp allocator bằng NVML/device telemetry; PyTorch
   allocator chỉ là số phụ.
5. Chạy ba tầng parity:
   - no-contact short horizon;
   - pinned single contact;
   - full active-hand finalist cùng safety/outcome class.
6. Performance gate dùng representative active-hand workload, median nhiều run,
   giữ criterion GPU/CPU `>=2x`, VRAM `<=14 GiB`.
7. Zero NaN/Inf, zero overflow/truncation, zero sanitizer invalid read.

**Gate pass**

- Capability matrix 2/2 active hand pass.
- CPU/GPU parity pass cả ba tầng.
- Mọi finalist được CPU xác nhận.
- Median speed và device VRAM đạt threshold; raw per-run evidence được giữ.
- Nhánh success của script thoát 0 và in đúng key; mọi nhánh fail thoát nonzero.

### G09 — QDGrasp-ContactRich-Active-Tiny

**Việc làm**

1. Dùng dataset ID/schema mới; không overwrite ContactRich v1 evidence.
2. Tách:
   - `samples`: trajectory thực có shard;
   - `coverage_status`: active/paused/not-applicable metadata.
3. Counts được tính từ shard records và cross-check với shard header/hash.
4. Tạo train/val split thực, group theo source scene/object/seed để không rò
   candidate lineage.
5. Negative control phải có expected failure predicate:
   - nếu control pass, generation gate fail hoặc record là
     `unexpected_control_outcome`, không gọi nó negative;
   - không tính pause metadata là negative.
6. Mỗi active hand cần tối thiểu một CPU-confirmed dynamic positive và các
   negative class đã pin; sample GPU-derived có capsule/replay evidence.
7. `release_blocked=false` chỉ khi G00–G08 đều pass trên exact clean commit.

**Gate pass**

- `manifest.counts.samples == Σ shard.count`.
- Positive/negative counts khớp record outcome; train và val đều không rỗng.
- Không lineage leakage; deterministic regeneration cho cùng lock/seed.
- Loader mặc định chấp nhận v2 release và từ chối v1/dirty/blocked artifact.

### G10 — Immutable packet và independent review

**Packet bắt buộc**

- exact clean code commit và environment locks;
- ledger G00–G09;
- safety field coverage/mutation report;
- analytic impulse traces;
- trajectory frame/time audit;
- CPU/GPU parity capsules;
- sanitizer, overflow, VRAM và throughput raw logs;
- dataset manifest/shard hashes và regeneration comparison;
- danh sách known limitations, active/paused disclosure.

**Independent review**

Reviewer không được là người viết phần lớn patch hoặc tạo evidence quyết định.
Review tối thiểu kiểm:

1. force/impulse/window math;
2. omitted-budget-field không còn tồn tại;
3. exact GPU→CPU replay lineage;
4. manifest count/split/hash;
5. ADR scope disclosure;
6. checker exit-code mutation;
7. không claim three-hand/P3.4 historical pass.

**Gate pass**

- Packet immutable trên clean commit.
- Reviewer verdict `PASS`, zero unresolved severity S0/S1; S2/S3 phải có
  disposition được maintainer ký.

### G11 — P4 contact-rich handoff

P4 contact-rich input chỉ mở khi:

- G00–G10 đều pass;
- artifact ID là `QDGrasp-ContactRich-Active-Tiny`;
- loader public đọc được train/val;
- dataset card ghi 2/2 active, Shadow paused và no hardware-safety claim;
- interface freeze gồm trajectory v2, ReplayCapsule v1 và manifest v2.

Nếu G08 blocked do upstream GPU defect nhưng G01–G07/G09 CPU-only đều pass,
P4 static/offline vẫn tiếp tục; contact-rich GPU-derived training input vẫn
blocked. Không đổi threshold để né blocker.

## 5. File/module dự kiến tác động

| Nhóm | File/module chính |
| --- | --- |
| Status/gate | `scripts/check_phase3_4.py`, checker successor, ledger JSON |
| Safety | `qdgrasp/dataset/dynamic_contracts.py`, `qdgrasp/dynamic/safety.py` |
| Trajectory | `qdgrasp/dynamic/wrapped_rollout.py`, `qdgrasp/dataset/dynamic_shards.py` |
| Certification | `qdgrasp/dynamic/certify.py`, release outcome factory |
| Backend | `qdgrasp/sim/batched/contracts.py`, CPU/MJWarp backends |
| CUDA | `scripts/phase3_4_cuda_contact_search.py`, sanitizer/notebook harness |
| Dataset | generator successor, manifest/schema/loader |
| Scope | data/robot/default workload registry and ADR checker |

Không sửa raw evidence cũ. Schema v1 chỉ nhận compatibility/security fixes;
release semantics mới đi qua version mới.

## 6. Completeness overlay bắt buộc

G00–G11 sửa trực tiếp các lỗi audit đầu tiên. Các gate C01–C08 dưới đây là lớp
đối chiếu ngược với toàn bộ contract P3.4 gốc. Chúng là **normative gate**, không
phải backlog phụ. Một G gate pass nhưng C gate liên quan chưa pass vẫn không đủ
đóng P3.4.3.

### C01 — Typed contract và schema fail-closed

Phạm vi: P3.4-00, P3.4-01, P3.4-13 và B-11.

1. `DynamicGraspRequest` phải validate non-empty refs, allowed backend,
   finite/positive horizon-dt, seed policy và profile/safety-budget compatibility.
2. `DynamicGraspTrajectory` phải validate:
   - mọi array finite và đúng rank/shape;
   - time tăng nghiêm ngặt, bắt đầu/offset được khai báo và khớp sample period;
   - palm/object quaternion finite, chuẩn hóa, đúng order;
   - stage order hợp lệ và có required terminal stages;
   - joint/actuator dimensions khớp robot profile;
   - contact event timestamp/index nằm trong rollout.
3. `ContactEvent` phải validate point/frame shape, orthonormal frame tolerance,
   force/impulse/work/duration/penetration không âm, finite units và identity refs.
4. `DynamicSearchOutcome` phải từ chối:
   - positive thiếu typed CPU certificate;
   - unknown failure reason/stage;
   - missing hoặc non-finite required objective/safety terms;
   - evidence schema/hash/version không khớp.
5. Tạo versioned schemas cho trajectory, contact event, replay capsule, outcome,
   dataset manifest và gate evidence. Unknown key fail; migration v1→v2 không
   được tự promote v1 artifact thành release-ready.
6. Public writer/loader round-trip deterministic; shard header count và hash
   được kiểm khi đọc, không chỉ khi ghi.

**Pass:** property/mutation corpus chứng minh mọi invalid field fail closed,
valid round-trip byte-stable và public schema docs khớp implementation.

### C02 — Scene, request hydration và backend protocol parity

Phạm vi: P3.4-02, P3.4-03, P3.4-04, P3.4-05 và B-13/B-14.

1. `reset(request_batch)` phải hydrate chính xác per-world initial qpos/qvel,
   mocap, object pose/velocity, mass, friction, target ID và seed-derived state;
   không reset mọi request về cùng model default trừ khi capsule khai đúng như vậy.
2. World isolation gate: command/state của world A không thay đổi world B.
3. `SceneSignature` phải hash toàn bộ topology-affecting fields:
   robot asset/profile/transmission/equality hash, geom type/count/collision parts,
   joint/dof/actuator/tendon counts, supports/non-target topology, contact/event
   capacities, solver/integrator/timestep/cone/options và material/contact params
   không thể batch.
4. Per-world data nào backend không batch được phải tạo bucket riêng hoặc
   preflight reject; không silently share.
5. `num_worlds` nghĩa là live worlds ở cả CPU/GPU; capacity là field riêng.
6. CPU và GPU cùng tạo `RolloutSummary v2` với objective, safety, failure và
   contact semantics giống nhau. CPU là oracle, không được để dict rỗng.
7. CPU backend kiểm NaN/Inf, MuJoCo warnings, contact/constraint overflow,
   truncated event stream và target identity; GPU kiểm cùng failure taxonomy.
8. Unsupported tendon/site/equality/mocap/contact feature fail trước search.
9. Compile/reset/observe/step/rollout/export state machine có invalid-call errors
   nhất quán; reset sau rollout xóa hoàn toàn state/invalid flags/timing run cũ.

**Pass:** micro + active-hand fixtures chứng minh request hydration, bucket
separation, world isolation, live/capacity semantics và CPU/GPU summary schema.

### C03 — Contact taxonomy, scene damage và physical terminal semantics

Phạm vi: §4, P3.4-06, P3.4-07, P3.4-08 và B-12/B-15.

1. Contact classification dựa trên typed pair semantics, không chỉ “pair có
   support”:
   - target-support;
   - robot-support;
   - target-robot;
   - non-target-support/robot/target;
   - robot self-contact exact allowlist;
   - unknown/forbidden.
2. Self-contact allowlist phải đến từ versioned robot profile/link-pair policy;
   cấm tạo cartesian product cho toàn bộ robot geoms.
3. Budget pin theo robot profile + link class + material + environment; missing
   specialization fail preflight.
4. Force/frame sign và wrench transform được kiểm bằng analytic/pinned contact;
   wrist wrench và tendon/joint load dùng đúng frame/actuator semantics.
5. Target motion chỉ do physics/contact/external wrench đã khai báo; cấm runtime
   write object qpos để đạt pose.
6. Primitive transition timeout không được đồng nghĩa condition success.
   Timeout tạo explicit `transition_timeout:<condition>` negative hoặc
   `budget_exhausted`.
7. Positive phải đi qua đúng stage:
   environmental assist → enclosure → target support released → lift threshold
   → bounded perturbation → retained grasp, finite state.
8. Terminal cert chỉ xem target-support khi đánh giá support release; hand-floor
   contact riêng không làm target “still supported”.
9. Wrong-object lift, support-only hold, palm/table crushing, target teleport,
   insufficient enclosure/lift và perturbation slip đều là typed negatives.
10. Static-seeded observer không được cộng dt hai lần cho một control interval;
    simulator/control/sample clocks được tách rõ.

**Pass:** table-pivot, sparse-clutter, neighbor-incidental, wrong-object và
scene-damage fixtures tạo đúng class/reason trên CPU oracle.

### C04 — Strategy, objective, CEM và refinement correctness

Phạm vi: P3.4-07, P3.4-09, P3.4-10, P3.4-11 và B-16.

1. Primitive parameters, order, frame, bounds và maximum duration được hash.
   Capability matrix phải chứng minh từng primitive được claim có command và
   transition semantics thật; enum-only không tính implemented. Tối thiểu
   push/slide/pivot, hook/cage/squeeze, support-release/lift/perturb có fixture.
   Primitive chưa có semantics phải `deferred_not_claimed`.
   Transition dựa trên observed contact/object state; clock-only chỉ hợp lệ khi
   condition được khai báo rõ là duration.
2. Objective schema bắt buộc đủ mọi term đã pin; weights finite, sign/range hợp
   lệ và hash vào manifest. Không default missing term về 0 trong release path.
   Control energy và elapsed time là hai số đo thật, không dùng step count thay
   cho cả hai.
3. NaN/+Inf/-Inf objective term hoặc inconsistent passed/failure state hard
   reject candidate và ghi reason.
4. Forbidden/damaging/barrier candidate luôn score `-inf`; không terminal term
   nào mua lại được.
5. CEM:
   - bounded population/iterations/worlds;
   - CUDA path thực sự rollout batch `[W,T,U]`, không loop từng candidate qua
     CPU callback rồi gọi đó là batched GPU search;
   - candidate-index ↔ world-index ↔ capsule mapping ổn định;
   - deterministic sampled controls cùng seed/config/backend;
   - all-rejected hoặc no-finite elite trả `no_feasible_elite`, không refit;
   - budget hết trả `budget_exhausted`;
   - elite count/mean/std/bounds có invariant;
   - best outcome không được `None` với success verdict.
6. Reason ledger có denominator bảo toàn ở từng stage và reconcile:
   sampled = sum released/failure/unfinished dispositions.
7. Local refine chỉ nhận CPU-confirmed positive, chỉ thay control params, không
   thay threshold/objective/split; regression candidate không được chấp nhận.
8. MPPI vẫn optional: nếu deferred, ledger và dataset card ghi
   `deferred_not_claimed`; không có MPPI coverage claim. Nếu implement sau,
   phải qua cùng capsule/safety/parity gates.

**Pass:** deterministic search fixtures, all-reject/NaN/missing-term mutations,
bounded-budget proof và refinement non-regression đều pass.

### C05 — Dataset coverage và controlled ablation

Phạm vi: §11, P3.4-14, P3.4-16 và B-17/B-18.

`QDGrasp-ContactRich-Active-Tiny` phải có tối thiểu:

1. **2/2 active hand**: LEAP và Wonik Allegro; Shadow chỉ ở
   `coverage_status=paused_by_ADR-0008`.
2. Dùng canonical scene/state/observation contract của
   `QDGrasp-Scene-Tiny` hoặc versioned successor; không dựng đường dataset
   riêng bỏ qua target identity, frame, camera và scene hash.
3. **Ba environment class** được preregister, ví dụ table/tray/bin hoặc exact
   successor equivalents.
4. **Hai clutter tier**: sparse và denser bounded tier.
5. **Ba generation mode**: static-seeded, primitive sequence và bounded CEM.
6. Mỗi active hand có:
   - table-assisted pivot/slide positive;
   - sparse-clutter reposition positive;
   - damaging hand/support negative;
   - non-target disturbance negative;
   - GPU/CPU divergence negative fixture;
   - no-closure/insufficient-lift hoặc perturbation-fail control.
7. Paired evidence cùng scene/target/state:
   P3.3 static fail hoặc blocked → P3.4.3 dynamic CPU-confirmed pass.
8. Controlled static-vs-dynamic ablation khóa trước:
   same scene/state/candidate budget/horizon/objective/safety/seed; chỉ khác
   frozen-vs-reactive physics factor được tuyên bố.
9. Báo yield, failure distribution, CPU replay rate, compute/trajectory,
   trajectories/s và confidence/raw counts; `no_measured_difference` vẫn là
   verdict hợp lệ, không được đổi threshold để tạo uplift.
10. Objective weights, safety thresholds, candidate budgets và split policy pin
   trước khi xem val/test. Mọi thay đổi sau đó tạo revision và invalidate affected
   comparison.
11. Train/val/test không rỗng, group-disjoint theo object family + scene template
    + source state/candidate lineage + trajectory seed.
12. Unexpected negative-control pass làm gate fail hoặc disposition riêng; không
    được tính nó là negative.
13. CPU reference provenance và GPU search provenance là hai typed field/ref
    riêng; không suy backend từ filename hoặc gộp CPU number dưới CUDA schema.
14. Manifest lưu candidate budget, horizon, objective/safety hash, backend/lock,
    capsule/CPU replay refs, yield, compute, active/paused disclosure và license.

**Pass:** coverage checker xác nhận toàn bộ Cartesian requirements đã khai báo,
split leakage = 0, counts reconcile shard và controlled ablation report immutable.

### C06 — Storage, provenance, render và public handoff artifact

Phạm vi: P3.4-13, P3.4-16 và B-20.

1. Storage là fixed-rate keyframes + sparse contacts; giảm simulator timestep
   không làm release size tăng theo raw integrator steps ngoài tolerance/protocol.
2. Debug every-step dump tắt mặc định, có retention policy và không nằm trong
   public shard.
3. `terminal_grasp` round-trip với GraspBatch-compatible named-joint record.
4. Mọi sample/shard/manifest có trajectory, contact, capsule, compute,
   source/config/asset/profile/scene/environment/provenance hashes.
5. Clean regeneration hai lần trên exact lock/seed tạo byte-identical manifest
   và shard; dirty worktree luôn `release_blocked=true`.
6. Render stage evidence tối thiểu approach/reposition/enclose/support-release/
   lift/perturb từ pinned views; render metadata trỏ đúng trajectory hash.
7. Visual QA kiểm target identity, support release, no teleport, penetration,
   wrong-object motion và frame/orientation; renderer không dùng để thay physics
   verdict.
8. Public loader mặc định chỉ nhận v2 clean/unblocked, verify hash/path traversal/
   count/schema và từ chối v1/dirty/invalidated.
9. Dataset card ghi scope 2/2, Shadow pause, simulator-only contact, license,
   limitations, no hardware-safety claim và ablation verdict.
10. Wheel/sdist chứa public schemas/loaders cần thiết nhưng không nhúng raw
    evidence, private path hoặc external restricted asset.

**Pass:** release audit + loader round-trip + rendered QA + packaging manifest
đều trỏ cùng immutable dataset root hash.

### C07 — Resource, cloud harness, checkpoint/resume và prior gates

Phạm vi: §10, §14, P3.4-15 và B-08/B-19.

1. Local CLI mặc định batch 1, bounded horizon/iterations, thread caps và
   `--dry-run` in worlds × steps × estimated host/device/event-buffer bytes.
2. GPU preflight pin batch/capacity trước run; không tăng dần tới OOM.
3. Contact/event/constraint buffer cap nằm trong signature/evidence; overflow
   fail world và gate, không resize âm thầm sau khi thấy kết quả.
4. Generation checkpoint atomic theo scene/state/seed/candidate; resume
   idempotent, không duplicate/drop sample, partial file không được coi completed.
5. Wall-time guard luôn flush ledger/raw evidence trước cloud timeout.
6. Renderer không chạy cùng peak search nếu preflight vượt memory budget.
7. Kaggle notebook:
   - cài exact public QDGrasp commit + exact CUDA/Warp lock;
   - không chứa credential/secret/private path;
   - chạy P1 CUDA smoke và P2 active-hand CUDA FK parity trước P3.4.3;
   - xác minh device tensors/contact buffers thật trên CUDA;
   - ghi environment fingerprint và raw log hashes.
8. Benchmark công bằng:
   same initial capsules/commands/horizon/precision/observation work;
   reset sau warmup; compile/warmup tách steady state; ít nhất 3 run và median;
   không loại run thấp nếu chưa có documented invalidation.
9. Threshold giữ GPU/CPU `>=2x`, at least 64 simultaneous worlds, representative
   operating point preregistered, device VRAM `<=14 GiB`, zero OOM/fallback/
   nonfinite/overflow/truncation/sanitizer error.
10. Kaggle primary và Colab Pro secondary replication nếu môi trường hỗ trợ;
    khác GPU/driver ghi comparability, không gộp số tùy tiện.

**Pass:** fresh cloud run từ clean public commit có thể resume, tạo cùng semantic
artifact, vượt full CUDA gate và không phụ thuộc state bí mật của notebook.

### C08 — Closed-world completeness manifest và closure checker

Manifest `docs/roadmap/phase3_4_3_requirements.yaml` là inventory
machine-readable ban đầu và phải được checker quản lý xuyên suốt implementation.
Nó liệt kê toàn bộ:

- B-01–B-20;
- G00–G11;
- C01–C08;
- original P3.4-00–P3.4-17;
- §4.1–§4.3, §5–§11, §13–§16 của `ROADMAP-P3.4-001`;
- ADR-0008 active/paused disclosure;
- upstream MJWarp defect/compatibility decision.

Mỗi requirement bắt buộc có:

`id, normative_source, owner, implementation_refs, test_ids, evidence_refs,
status, blocker_reason, supersession_policy`.

Checker completeness phải fail nếu:

1. requirement thiếu mapping/test/evidence;
2. unknown/untracked gate xuất hiện;
3. status không thuộc `pending|passed|failed|blocked|paused|deferred_not_claimed`;
4. `passed` nhưng artifact/hash không tồn tại hoặc worktree dirty;
5. deferred item được claim coverage;
6. P3.4.3 pass trong khi bất kỳ B/G/C required item chưa pass;
7. historical P3.4/Shadow bị ghi pass;
8. reviewer/packet commit không khớp candidate.

**Pass:** `mapped_requirements == total_requirements`,
`unmapped=0`, `unknown=0`, mọi required status pass và independent reviewer
ký exact completeness manifest hash.

## 7. Original work-package traceability

| Original ID | Successor gate bắt buộc | Output đóng |
| --- | --- | --- |
| P3.4-00 | G00, C01, C08 | hypothesis/safety/entry ledger khóa |
| P3.4-01 | C01, G03, G04 | typed v2 contracts + invalid corpus |
| P3.4-02 | C02 | backend protocol + complete scene signature |
| P3.4-03 | C02, G06 | CPU oracle contact summary + replay capsule |
| P3.4-04 | C02, G08, C07 | pinned compatibility matrix/decision |
| P3.4-05 | G07, G08, C02 | fail-closed CUDA backend |
| P3.4-06 | G01, G02, C03 | full safety/taxonomy observer |
| P3.4-07 | C03, C04 | primitive transition semantics |
| P3.4-08 | G03, C03, C05 | static-seeded measured baseline |
| P3.4-09 | G07, C04 | bounded deterministic CEM |
| P3.4-10 | C04 | implemented+gated hoặc deferred_not_claimed |
| P3.4-11 | C04 | local refinement non-regression |
| P3.4-12 | G04, G06, C03 | terminal+safety+CPU replay certifier |
| P3.4-13 | G03, G09, C01, C06 | trajectory/shard writer-loader v2 |
| P3.4-14 | C05 | controlled static-vs-dynamic ablation |
| P3.4-15 | G08, C07 | public Kaggle CUDA packet |
| P3.4-16 | G09, C05, C06 | ContactRich-Active-Tiny RC |
| P3.4-17 | G10, G11, C08 | independent verdict + P4 handoff |

Không package nào được đánh dấu successor-pass chỉ vì file/module tồn tại.
`implemented`, `tested`, `evidenced`, `reviewed` là bốn trạng thái riêng.

## 8. Original completion-criteria traceability

| §16 criterion đã scope lại 2/2 active | Gate |
| --- | --- |
| Physics target motion, no teleport | C03, C05 |
| Contact/support/non-target multi-budget | G01, G02, C03 |
| Paired static-fail/dynamic-pass cho 2/2 active hand | C05 |
| Real T4/cu128 fail-closed và >=2x | G08, C07 |
| Mọi GPU positive được CPU replay exact | G06, G07, G08 |
| Deterministic dataset + trajectory/contact/compute/provenance hash + render | G09, C06 |
| Không post-hoc threshold/objective; ablation/yield-compute | C04, C05 |
| Independent force/parity/scene-damage/active-hand review | G10, C08 |

Đây là scope successor được ADR-0008 cho phép; nó không sửa bảng §16 lịch sử
ba-hand.

## 9. Evidence matrix

| Gate | CPU local | Kaggle T4 | Colab Pro GPU | Reviewer |
| --- | --- | --- | --- | --- |
| G00–G06, C01–C04 | bắt buộc | không | không | delta |
| G07, C02–C04 | oracle/capsule | bắt buộc | khuyến nghị cross-check | full |
| G08, C07 | CPU reference | primary | secondary replication | full |
| G09, C05–C06 | regeneration 1 | GPU-derived samples | regeneration 2 | full |
| G10–G11, C08 | packet build | raw GPU logs | replication refs | ký verdict |

Kaggle/Colab artifact phải ghi GPU, driver, CUDA, Python, package lock, commit,
notebook hash và wall-clock timestamp. Không dùng tên notebook mutable làm pin.

## 10. Milestone thực thi

| Milestone | Nội dung | Điều kiện sang bước |
| --- | --- | --- |
| M0 — Completeness lock | C08 requirements inventory draft | mapping coverage 100% |
| M1 — Contract lock | G00, G04, G05, C01 | API/schema invalid-corpus pass |
| M2 — CPU physics correctness | G01–G03, G06, C02–C04 | safety/time/replay/search mutations pass |
| M3 — GPU correctness | G07, C02, C03 | exact finalist + oracle summary parity |
| M4 — CUDA/operations gate | G08, C07 | parity/perf/VRAM/sanitizer/resume pass |
| M5 — Dataset + ablation RC | G09, C05, C06 | coverage/ablation/render/regeneration pass |
| M6 — Completeness freeze | C08 final manifest | zero unmapped/unknown/open required |
| M7 — Release review | G10 | independent PASS, zero S0–S3 open |
| M8 — Handoff | G11 | P4 contact-rich input unblocked |

M1–M2 không cần GPU và nên hoàn tất trước. M3–M4 chạy khi có Kaggle T4/Colab
session; không chờ cloud để sửa CPU contracts.

## 11. Closure command surface

Implementation phải cung cấp một entry gate tương đương:

```text
python scripts/check_phase3_4_3.py --scope active --profile cpu
python scripts/check_contactrich_active.py DATASET_ROOT --require-release
python -m pytest tests/dynamic_grasp/ tests/contactrich_active/ -q
python scripts/check_docs.py --root .
git diff --check
```

CUDA command chạy trên exact public Kaggle/Colab environment và nhận immutable
CPU packet/hash. Tên script có thể đổi trước schema lock, nhưng capability không
được thiếu. Final closure runner phải:

1. gọi mọi CPU/dataset/docs/completeness gate;
2. verify external CUDA evidence signature/hash/commit;
3. verify independent review packet hash;
4. trả 0 duy nhất khi exact active scope pass;
5. in one JSON verdict không cần suy diễn từ log.

## 12. Definition of Done

Plan chỉ `complete` khi đồng thời:

1. hai mươi blocker B-01–B-20 có patch, regression/mutation evidence và ledger
   `gate_state=passed`;
2. G00–G11 và C01–C08 đều pass; completeness manifest có zero unmapped/unknown;
3. original P3.4-00–17 có successor disposition và artifact/evidence mapping;
4. SafetyBudget v2 kiểm 100% declared limits theo profile/link/material/environment;
5. contact taxonomy, support release, scene damage và terminal perturbation đúng;
6. trajectory v2 có đúng timebase/palm/frame/stage/event semantics;
8. objective/CEM/refinement deterministic, bounded và fail-closed;
9. không matching failure, GPU-only hoặc hash-mismatch nào thành release positive;
10. GPU contact summary fail-closed và mọi finalist có exact ReplayCapsule;
11. active-hand CUDA parity/sanitizer/overflow/VRAM/performance/prior-gate pass;
12. resource preflight, atomic checkpoint, resume và dedup pass trên fresh cloud run;
13. ContactRich-Active-Tiny đạt 2 hand × 3 environment × 2 clutter tier × 3 mode
    coverage đã khai báo cùng required positive/negative fixtures;
14. paired static-fail/dynamic-pass và controlled ablation immutable hoàn tất;
15. train/val/test lineage-disjoint, counts/hash/replay đúng và regenerate sạch;
16. trajectory/contact/compute/provenance hashes cùng rendered stage QA pass;
17. public loader/schema/package/dataset-card handoff pass;
18. checker không trả 0 cho partial/paused/blocked/deferred/dirty;
19. runtime default tuân ADR-0008 và không claim Shadow/three-hand;
20. independent reviewer phát hành PASS trên exact clean packet, zero S0–S3 open.

Nếu thiếu bất kỳ mục nào, trạng thái là `active` hoặc `blocked`, không dùng
`complete`. Hoàn tất plan này không tự đổi P3.4 historical three-hand thành
pass.

### 12.1 Trạng thái chốt (2026-08-28)

Trước lượt static review ngày 2026-08-28, ledger ghi ba claim `passed` mà
review logic sau đó phát hiện không
còn bảo toàn: closure checker có thể tin evidence tự khai, active-scope
audit bỏ sót Python release entry point còn chọn Shadow, và §16.3 so sánh
GWS margin đã chuẩn hóa với norm wrench trộn N/Nm. Vì vậy revision
1.3.0 mở lại chín requirement liên quan. WRK-R1 tới WRK-R6 đã thi công xong;
bảy trong chín requirement đó đóng lại theo re-close matrix §13.8, còn `C05`
và `R-DOD-03` giữ `failed` vì cặp paired không xuất hiện dưới tiêu chí
resistance. Ledger hiện tại: 65 passed, 14 failed, 5 blocked, 1 deferred,
0 pending — `manifest b56deb1efa39`.
Verdict vẫn `FAIL` và plan vẫn `blocked`.

| Mục DoD | Trạng thái sau review | Lý do |
| --- | --- | --- |
| 11 — CUDA parity/sanitizer | **failed, đã đo** | contact parity, non-finite world và initcheck chưa đạt |
| 14 — paired static-fail/dynamic-pass | **failed, reviewer mở lại** | metric không cùng chuẩn và mass sweep không controlled |
| 19 — ADR-0008 runtime default | **failed, reviewer mở lại** | generator Python còn có thể release Shadow |
| 20 — independent reviewer PASS | **blocked** | packet stale/tự tham chiếu và verifier chưa gắn verdict với packet |

Kết quả CUDA cũ giữ nguyên: speed đạt 5.47×/14.04× nhưng single-contact
parity lệch 8.39 mm so với tolerance 2 mm, 84/1024 LEAP world non-finite và
initcheck còn uninitialised read. Speed không mua lại correctness. Các bằng
chứng từ run v7–v11 vẫn là negative evidence hợp lệ, nhưng không thể
đóng G07/G08/C07.

Claim §16.3 tại commit `83bb755` bị thu hồi chờ remediation. Giá trị
`quality_margin` hiện được tính trong GWS có unit contact primitive và torque
scale theo kích thước vật; `declared_disturbance` lại trả norm trực tiếp của
vector gồm force và torque. Hai số không thể so sánh để chứng minh khả
năng chống nhiễu. Hơn nữa, LEAP mass sweep override mass cho dynamic arm
nhưng static threshold vẫn suy từ mass gốc. Evidence `8 cases` không được
dùng làm closure evidence.

Review packet hiện không sẵn sàng cho independent verdict: packet ghi commit
`36887b4` trong khi candidate đã ở `83bb755`, hash manifest đã stale, và
`collect_evidence` hash chính packet cũ trước khi ghi đè. G10/C08/R-DOD-08
chỉ được mở review sau khi hoàn tất toàn bộ §13.

## 13. Reviewer remediation addendum

Addendum này không tạo successor scope mới và không thay đổi ADR-0008.
Nó mở lại các ID hiện có theo fail-closed semantics. Không requirement
nào bên dưới được trả về `passed` chỉ vì patch hoặc test tồn tại;
phải có acceptance evidence mới trên exact candidate.

### 13.1 Reopening matrix

| Finding | Requirement mở lại | Trạng thái | Evidence cũ |
| --- | --- | --- | --- |
| RRV-01 — closure evidence/review verifier có false-pass path | B-09, G00 | failed | invalidated cho closure claim |
| RRV-02 — Shadow còn trong default Python release workload | B-10, G05, R-ADR-01 | failed | S1 scope audit không đủ coverage |
| RRV-03 — §16.3 metric không cùng đơn vị/chuẩn | B-18, C05, P3.4-14, R-DOD-03 | failed | static-vs-dynamic v1 invalidated |
| RRV-04 — mass sweep không dùng cùng disturbance | B-18, C05, P3.4-14, R-DOD-03 | failed | claim `8 cases` invalidated |
| RRV-05 — packet stale và tự tham chiếu | G10, C08, R-DOD-08 | blocked | packet hiện tại không reviewable |
| RRV-06 — plan/packet/guide mâu thuẫn | G10, C08, R-DOD-08 | blocked | derived prose phải regenerate |

### 13.2 WRK-R1 — Dimensionally correct static resistance

Thay `quality_margin < norm(perturbation_wrench)` bằng resistance problem có
đơn vị và force bound rõ ràng. Implementation tham chiếu:

```text
maximize alpha
subject to
    G f + w_gravity + alpha * w_disturbance = 0
    f_i in friction_cone_i
    0 <= normal_force_i <= force_limit_i
```

- `alpha` không có đơn vị; static resistance pass khi `alpha >= 1`.
- Mô-men trong `G` và `w_disturbance` phải cùng scale bằng
  `1 / characteristic_length` nếu cần conditioning.
- `force_limit_i` lấy từ profile/material/contact safety budget; không có
  force cap thì resistance factor không được xem là physical certificate.
- Legacy API giữ `quality_margin_threshold=0.0`, nhưng field đó không còn
  được dùng làm §16.3 release evidence.

**Acceptance:** analytic fixtures có known feasible/infeasible `alpha`, scale
invariance khi đổi length unit, monotonicity theo force limit và disturbance,
gravity + perturbation equilibrium residual trong tolerance đã pin.

### 13.3 WRK-R2 — Controlled paired ablation

Tạo `CandidateSnapshot` immutable gồm scene/target/robot state, command,
contact proposal, mass/geometry/friction, safety budget, horizon, seed và applied
wrench. Static và dynamic arm fork từ cùng snapshot; chỉ
`physics_mode=frozen|reactive` được khác.

- Contact lấy sau dynamic rollout chỉ là diagnostic, không phải primary
  paired evidence.
- Tách một hàm `resolve_perturbation_wrench(...)` dùng chung cho validator
  và ablation; không sao chép công thức.
- Mỗi mass-sweep record lưu mass, applied wrench/hash, snapshot hash,
  static `alpha`, dynamic outcome và safety verdict.
- Parameter hash bao phủ mass grid, wall geometry, disturbance policy, recipe,
  safety/objective hash và implementation commit.

**Acceptance:** same-snapshot hash cho hai arm, exact one-factor diff audit pass,
không post-treatment contact leakage, regenerate byte-identical, và paired
static-fail/dynamic-pass xuất hiện trên 2/2 active hand bằng resistance
criterion WRK-R1; nếu không có thì R-DOD-03 giữ `failed`.

### 13.4 WRK-R3 — Closure evidence trust chain

Closure CLI tách ba input: CUDA evidence bundle, immutable review packet và
signed reviewer verdict.

- CUDA verifier chỉ nhận exact schema, bắt buộc parity/sanitizer/non-finite/
  speed/VRAM/raw-log hash, và tự tính verdict từ metric.
- Packet verifier tính lại canonical hash, khóa candidate commit, code/lock/
  config/notebook/raw evidence hashes và không tin self-declared verdict.
- Reviewer verdict phải ký packet hash, candidate commit, reviewer identity và
  open-finding counts. `reviewer != author` không được xác minh chỉ bằng
  hai chuỗi tự khai.
- Release profile từ chối `--skip-tests`; missing suites là failure.
- Requirement ledger kiểm test path và evidence schema/verdict/hash, không chỉ
  sự tồn tại của file.
- Independent PASS yêu cầu zero open S0–S3. Finding đã disposition phải
  chuyển sang `closed`, không để `open`.

**Acceptance:** mutation corpus chứng minh forged PASS JSON, wrong packet hash,
wrong commit, unsigned reviewer, open S2/S3, missing raw log, nonexistent test và
release `--skip-tests` đều trả nonzero.

### 13.5 WRK-R4 — ADR-0008 enforcement toàn repository

- Release mới dùng `DEFAULT_ROBOT_PROFILES` và `require_release_scope()`.
- Artifact ba-hand lịch sử phải có ID trong
  `HISTORICAL_THREE_HAND_ARTIFACTS`, explicit `--historical-reproduction` và
  `non_release=true`; config allowlist không tự làm Python generator an toàn.
- Audit quét YAML/YML và Python entry point bằng AST; YAML parse error là
  finding thay vì bỏ qua.
- Mọi manifest gọi `require_release_scope()` trước khi có thể ghi
  `release_blocked=false`.

**Acceptance:** không default CLI/generator/training/backend/RL path nào chọn
Shadow; experimental/historical path bắt buộc explicit purpose và không thể
promote output thành release.

### 13.6 WRK-R5 — Review packet không vòng lặp

Packet là artifact sinh từ clean candidate commit, không là input của chính
nó. `collect_evidence` loại output target; packet được dựng trong thư mục
tạm hoặc CI artifact store, hash canonical payload không chứa
`packet_sha256`, sau đó reviewer ký digest. Packet được phép commit trong
attestation commit sau, nhưng `candidate_commit` luôn là code commit đã review.

**Acceptance:** rebuild hai lần từ cùng candidate cho cùng canonical digest
ngoài timestamp được tách khỏi signed payload; không self-reference; packet,
verdict và closure runner cùng xác nhận một candidate/hash.

### 13.7 WRK-R6 — Single source of truth cho trạng thái

Requirements manifest là nguồn trạng thái duy nhất. Counts, blocker table,
reviewer checklist và packet completeness block được sinh từ manifest.
Plan version và manifest `plan_version` phải khớp. Revision cũ không bị viết
lại; revision mới ghi `supersedes` và nêu claim nào bị thu hồi.

**Acceptance:** không còn câu mâu thuẫn giữa plan, ledger, review guide,
dataset card và packet; generated counts khớp audit result; docs gate cấm status
snapshot tự gõ nếu không mang manifest hash.

### 13.8 Thứ tự thực thi và re-close

```text
WRK-R3 closure fail-closed -----> WRK-R5 immutable packet -----> WRK-R6 docs
WRK-R1 resistance -----> WRK-R2 controlled ablation ----------/
WRK-R4 Shadow scope ------------------------------------------/

GPU upstream fix hoặc fallback backend -----> G07/G08/C07
toàn bộ nhánh trên pass ----------------> independent review ----> G10/G11
```

Thứ tự bắt buộc:

1. WRK-R3 và WRK-R4 trước để không sinh thêm evidence có thể false-pass.
2. WRK-R1 trước WRK-R2; không regenerate ablation bằng metric cũ.
3. Hoàn tất code freeze rồi mới dựng WRK-R5 packet.
4. WRK-R6 chỉ regenerate derived prose sau khi ledger và evidence đã khóa.
5. Independent reviewer là bước cuối; tác giả không tự phát hành verdict.

Re-close matrix:

| Gate | Điều kiện trả về `passed` |
| --- | --- |
| B-09, G00 | WRK-R3 mutation evidence pass trên release profile |
| B-10, G05, R-ADR-01 | WRK-R4 full-repository audit zero finding |
| B-18, C05, P3.4-14, R-DOD-03 | WRK-R1 + WRK-R2 có valid evidence trên 2/2 active hand |
| G07, G08, C07 | GPU parity/sanitizer/non-finite/VRAM/performance đều pass |
| G10, C08, R-DOD-08 | WRK-R5/R6 xong và signed independent verdict zero open S0–S3 |
| G11, P3.4-17 | Mọi G00–G10/C01–C08 required gate pass; P4 handoff explicit |

Trong thời gian remediation, `release_blocked=true`; không packet, dataset card,
P4 handoff hay summary nào được claim `P3.4.3-ACTIVE-PASS`.
