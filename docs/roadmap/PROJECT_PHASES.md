---
document_id: ROADMAP-001
document_type: roadmap
title: Roadmap tổng thể QDGrasp theo tám phase
version: 1.26.0
status: active
date: 2026-08-27
revises: ROADMAP-001@1.25.0
related_plan: PLAN-V2
latest_revision_record: docs/revisions/REV-20260827-012-phase3-4-3-completeness-expansion.md
---

# Roadmap tổng thể QDGrasp

Tài liệu này là bản nhìn cấp cao để tổ chức thực thi. `PLAN.md` và các ADR vẫn
là nguồn chuẩn cho chi tiết kỹ thuật, license và tiêu chí nghiệm thu.

**Current robot scope (`ADR-0008`):** active corpus/gate mới chỉ gồm LEAP và
Wonik Allegro. Shadow Hand tạm dừng vì khó cấu hình underactuated/contact-control;
preset/evidence lịch sử được giữ nhưng không dùng trong default workload,
release hoặc RL pipeline. Các phase ba-hand đã đóng vẫn là historical truth;
P3.4 ba-hand chưa đóng thì giữ paused, không đổi thành pass.

## Nhịp thực hiện

- Toàn dự án chia thành **8 phase**, mỗi phase dự kiến **4–6 tuần** với một nhóm
  core nhỏ. Tổng thời gian tuần tự khoảng **32–48 tuần**; làm song song chỉ khi
  interface đầu vào/đầu ra của phase trước đã được khóa.
- Mỗi phase phân bổ gần giống nhau: 60% implementation, 25% test/evidence và
  15% tài liệu/provenance/review.
- Phase kết thúc theo gate, không kết thúc chỉ vì hết thời gian. Work chưa qua
  gate được chuyển rõ sang backlog, không ghi “hoàn tất”.
- Mỗi phase phải tạo ít nhất một artifact chạy được, một bộ test/evidence và một
  tài liệu bàn giao.

## Tổng quan các phase

| Phase | Mục tiêu chính | Artifact cuối phase | Gate chuyển phase | Trạng thái |
| --- | --- | --- | --- | --- |
| P0 — Foundation | Khóa scope, AGPL boundary, library package, environment, references và public repositories | Plan/ADR, wheel/sdist, environment locks, manifests và Kaggle harness riêng | Clean wheel import/CLI; CPU pass; CUDA hardware smoke pass; không có secret/RH56E2 trong active artifacts | complete |
| P1 — Framework | Dựng package, CLI, YAML schema, runner và checkpoint contract | Skeleton có `train/val/predict/export` trên dummy model | API/config round-trip, CPU smoke và CUDA dummy train-step pass | complete |
| P2 — Robot layer | Chuẩn hóa URDF/MJCF, HandGraph, FK, limits, frames và simulator adapter | LEAP/Allegro/Shadow cùng chạy qua một `RobotSpec` | Parse/mesh/FK/MuJoCo fixtures pass cho ba hand | complete |
| P3 — Data layer | Xây procedural objects/scenes, static/contact-rich grasp data và môi trường asset→scene→RL có thể tái lập | `DGN-Open-Tiny`, `QDGrasp-Scene-Tiny`, `QDGrasp-ContactRich-Tiny` và `QDGrasp-RL-Env-Tiny` có manifest/evidence | Object/scene/trajectory regenerate; raw-mesh/CoACD, virtual-drop/settle, Gym API, frame/physics/CPU-GPU audit và tiny loaders pass | pending |
| P4 — Model MVP | Xây object encoder, HandGraph conditioning và palm+joint flow | QDGrasp-Flow `n` overfit được tiny dataset | CUDA forward/backward, gradient coverage, finite joints/rotations và tiny overfit pass | pending |
| P5 — Training & evaluation | Hoàn thiện multi-hand training, quality/contact heads, evaluator và ablation | Checkpoint multi-hand đầu tiên cùng benchmark report | CUDA train/eval/resume, held-out protocol, physics success và ablations tái lập | pending |
| P6 — Scale & delivery | Scale data/model, tối ưu memory/latency, resume, TorchScript và ONNX | Model `n/s/m`, export bundles và reproducible train recipes | CPU/CUDA/AMP parity, export round-trip, memory/latency gates pass | pending |
| P7 — Release & paper | Đóng package, model/data cards, paper evidence và community workflow | Public release candidate, model zoo, dataset card và paper artifact bundle | License/security/release review; raw metrics/hash; independent review pass | pending |

## Phạm vi từng phase

### P0 — Foundation

- Hoàn tất boundary DGN2 paper-only và license/provenance cho code, data,
  checkpoint, robot/object assets.
- Phát hành code dưới AGPL-3.0-only, giữ notice của source dẫn xuất và source
  permissive; không còn kế hoạch Apache clean-room.
- Build/cài package `qdgrasp` từ wheel ngoài source tree. Notebook CUDA nằm ở
  repository riêng và phải cài exact public commit của library.
- Giữ bốn architecture baseline đã pin; dùng quy trình reference-on-demand khi
  có ca khó.
- Chốt Python/PyTorch/MuJoCo, CPU/cu128 locks, CI gates và nơi lưu raw evidence;
  cu128 phải được smoke-test trên GPU NVIDIA thật trước khi đóng P0.
- Chốt repository/history dùng cho implementation và release.
- Loại RH56E2 khỏi active scope; archive lịch sử chỉ được chú thích bằng sidecar.

### P1 — Framework

- Public façade và CLI tối thiểu, không mở DSL layer tùy ý.
- YAML chỉ chọn preset/stage/module có schema; unknown/dead key bị từ chối.
- Runner dựa trên Lightning Fabric, deterministic seed, logging, callbacks,
  checkpoint/resume và result bundle.
- Dummy model/data cho phép kiểm toàn lifecycle trước khi model thật xuất hiện.

### P2 — Robot layer

- `RobotSpec`, named joints, joint limits, mimic/coupling policy và HandGraph.
- URDF/MJCF mesh resolver, frame transforms, FK và batch kinematics.
- MuJoCo adapter cho grasp/squeeze/lift fixtures.
- Compatibility matrix bắt buộc cho LEAP, Allegro và Shadow Hand.

### P3 — Data layer

- Procedural/CC0 object generation và asset manifest.
- Candidate palm/contact sampling, IK hỗ trợ, collision filtering và physics
  validation.
- `GraspBatch` schema, shards, splits, seed, camera/frame metadata và hashes.
- P3 vẫn `pending`, nhưng Phase 3.2 và corrective Phase 3.2.1
  ([`ROADMAP-P3.2.1-001`](PHASE3_2_1_FULL_PIPELINE_CORRECTNESS_PLAN.md)) đã
  complete. Generated-reachable full flow pass cho LEAP, Allegro và Shadow;
  solver math, palm initialization, exact collision, task-space command,
  dynamic predicate và deterministic evidence đã qua independent review.
  Shadow được gỡ release block bởi `REV-20260824-001`. Controlled ablation
  P3.1-13 đã chọn `region_opposition_v1` trong validated positive-control scope
  (`REV-20260825-001`), nhưng canonical-independent yield vẫn `0/12` và không
  chứng minh generalization. P3.1 đã đóng bằng `REV-20260827-001`: regeneration
  cho thấy mười hai object procedural sinh `0` positive ở cả sáu shard, nên
  release nhận sáu positive-control object gắn theo hand. Blocker `wonik_allegro`
  được gỡ sau khi lưới kinematics định vị nút thắt là IK convergence chứ không
  phải palm floor clearance: mở opposition task sang 45 mm đo được 2 dynamic
  positive, trong khi bốn variant giữ 40 mm đo được 0. `DGN-Open-Tiny` hiện có
  200 sample, 18 object, 7 measured positive trên **6/6 shard**,
  `invalidated=false`, `release_blocked=false`, hai clean regeneration
  byte-identical và `scripts/check_phase3.py` PASS. Canonical procedural yield
  vẫn `0` và không có claim generalization. Giả thuyết nâng IK budget đã bị đo
  và bác bỏ (`REV-20260827-002`): solver dừng cách tolerance 28–179 lần về
  position và 60–150° về normal, tức không nằm trong basin chứ không phải hết
  budget, nên canonical `0/12` là bài toán **proposal feasibility** chứ không
  phải solver tuning. Mọi thí nghiệm P3.1 đã chạy, gồm cả hướng chết, được ghi
  tại `evidence/phase3_1/EXPERIMENTS.md`. P3 tổng vẫn `pending` vì implementation
  P3.4 contact-rich chưa bắt đầu và P3.5 RL-readiness mới được lập kế hoạch.
- Phase 3.3 ([`ROADMAP-P3.3-001`](PHASE3_3_SCENE_GRASP_DATA_PLAN.md)) mở rộng
  pipeline sang clutter scene: adapter GraspNet-1Billion, DexGraspNet 2.0 và
  GraspClutter6D; native scene builders; whole-scene collision; multi-object
  rollout; camera observations và `QDGrasp-Scene-Tiny`. P3.2/P3.2.1 đã bàn giao
  baseline và P3.3 hiện ở release candidate. Adapter official-layout micro,
  native builders, renderer/observation, target crop, whole-scene clearance,
  measured multi-object rollout, shards và loader đều đã qua gate.
  `QDGrasp-Scene-Tiny` tại `e9c87b8` có 12 scene/24 observation/33 scene-state,
  17 grasp, ba measured positive cho LEAP/Allegro/Shadow, ba negative class và
  108 artifact hash. Micro regeneration byte parity, release audit, resource
  policy và rendered QA đã pass. Independent review `TPR-20260825-001` tái lập
  scoped gates, mutation probes và frame/collision/license/visual evidence trên
  exact candidate `870ef91`; verdict `pass`, không có S0–S3. P3.3 vì vậy
  `complete` (`REV-20260825-007`). External real-source roots chưa cấu hình nên
  closure không tạo source-scale compatibility claim. P3 tổng thể vẫn `pending`
  vì P3.4 contact-rich trajectory và P3.5 RL simulation readiness còn độc lập
  chưa đóng.
- Phase 3.4 ([`ROADMAP-P3.4-001`](PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md))
  đạt **15/18 work package** tại `16d74c5` (`REV-20260827-005`), tất cả pass CPU
  gate `scripts/check_phase3_4.py` với 117 test. Backend decision đã giải quyết
  trên Tesla T4: MuJoCo Warp 1.16.0 compile được cả ba tay kể cả 4 tendon của
  Shadow, verdict `supported`. **Phase chưa đóng.** `QDGrasp-ContactRich-Tiny`
  đã sinh với 9 sample và **4 measured dynamic positive** (LEAP lift 4.9 cm,
  Allegro 4.1 cm, có contact `support_assisted` thật), nhưng
  `release_blocked=true` vì `shadow_hand` chưa có positive: recipe của nó ép
  `rh_lfproximal` vào `rh_lfmetacarpal` ở 323 N bền vững, thứ mà P3.2.1 không
  kiểm. Chi tiết tại `evidence/phase3_4/p16-contactrich-tiny/`; CEM 40 candidate
  trên closure scale của Shadow cho 40/40 hard-reject, nên đây không phải bài
  toán search control mà là recipe/collision model, thuộc P3.2/P3.3. GPU speed
  §10 ban đầu fail `0.764x` ở 64 worlds; v9 tại operating point đã pin 1024
  worlds đạt **`4.444x`**, vượt yêu cầu `2x`. Tuy nhiên full CUDA gate vẫn fail:
  29/1024 identical worlds non-finite, còn `peak_vram_gib=0.0` được đo qua
  PyTorch nên không bao phủ Warp allocator
  (`evidence/phase3_4/p15-throughput/`). P3.4-05/15 vì vậy còn stability,
  allocator-correct VRAM và parity; P3.4-17 independent review không thể do tác
  giả tự phát hành; P3.4-10 MPPI deferred theo đúng kế hoạch. Ablation
  static-vs-dynamic chạy được nhưng verdict `no_measured_difference` và **không**
  xác nhận giả thuyết P3.4. Phạm vi gốc: target được phép
  dịch chuyển do physics, support/non-target contact được chấp nhận dưới safety
  budget, GPU batched generation dùng MJX-Warp/MuJoCo Warp và CUDA evidence chạy
  trên Kaggle như Phase 1. CPU vẫn là correctness/oracle replay backend.
- Phase 3.4.1
  ([`ROADMAP-P3.4.1-001`](PHASE3_4_1_FIX_PLAN.md)) là diagnostic/corrective plan
  đã được `ROADMAP-P3.4.2-001` supersede, không phải giấy phép tự đóng P3.4. Plan
  giữ nguyên gate `>=2x`; v9
  đã đạt `4.444x` tại 1024 worlds nên speed subcriterion pass, nhưng ưu tiên hiện
  tại là phân loại 29 non-finite world qua `Data.overflow`, repeat/index
  invariance, Warp debug và Compute Sanitizer. Bộ đo VRAM hiện tại dùng
  `torch.cuda.max_memory_allocated()` nên không bao phủ allocation của Warp;
  con số `~0 GiB` không được dùng làm bằng chứng GPU idle hoặc VRAM pass, và sẽ
  được thay bằng Warp/NVML/Nsight measurement. CUDA Graph/device-resident
  refactor chỉ chạy nếu speed sau stability fix tụt dưới `2x`. Shadow dùng versioned
  corrective delta: ưu tiên mở các ngón inactive và swept-collision audit; chỉ
  exclude/re-author proxy khi có geometric evidence, không tăng safety budget.
  Evidence P3.2/P3.3 cũ giữ bất biến, impacted gates phải replay. Tác giả chỉ
  chuẩn bị immutable packet; external hoặc internal-independent reviewer khác
  mới được ký verdict. P4 contact-rich input tiếp tục bị chặn, còn P4
  static/offline chỉ dùng interface/dataset không phụ thuộc release-blocked data.
- Phase 3.4.2
  ([`ROADMAP-P3.4.2-001`](PHASE3_4_2_CORRECTNESS_RECOVERY_PLAN.md)) là exact
  three-hand closure contract hiện `superseded/paused_by_ADR-0008`. Trước khi bị
  pause, plan yêu cầu reconcile ledger P3.4
  vì headline/bảng/roadmap hiện ghi ba package count khác nhau và invalidate
  review packet cũ vì packet khóa commit cũ, worktree bẩn, manifest count lệch
  và còn disclosure Shadow đã rút lại. GPU branch giữ native MJWarp, sửa
  sanitizer harness thành fail-closed, localization first-bad-tick/overflow và
  chỉ đóng khi zero invalid + outcome parity + median T4 `>=2x` + device VRAM
  `<=14 GiB`. Shadow branch tách actuator/tendon tracking khỏi joint tracking;
  chỉ sửa projected-target semantics hoặc kích hoạt actuator-space safe set /
  mixed-mode controller theo classification, không miễn kiểm inactive fingers.
  Vì Shadow tạm dừng, các package closure này không còn là default backlog và
  P3.4 ba-hand giữ trạng thái chưa đạt. P4 static/offline và P3.5 active-two-hand
  asset/scene work có thể tiếp tục; ContactRich v1 vẫn release-blocked.
- Phase 3.4.3
  ([`ROADMAP-P3.4.3-001`](PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md)) là
  corrective execution plan hiện hành cho contact-rich active hands. Plan sửa
  các gate còn hở trong safety-budget coverage, rolling impulse, trajectory
  time/frame, certification, exact GPU-to-CPU replay, CUDA capability/parity,
  dataset counts/splits và runtime enforcement của ADR-0008. Artifact đích là
  `QDGrasp-ContactRich-Active-Tiny` với claim 2/2 active hand; nó không đổi
  P3.4 ba-hand thành pass. P4 contact-rich input chỉ mở sau immutable packet và
  independent verdict `P3.4.3-ACTIVE-PASS`. Revision 1.1 bổ sung closed-world
  completeness contract: mọi P3.4-00–17, §4–§16, blocker B-01–B-20 và gate
  G00–G11/C01–C08 phải có mapping, test, evidence và zero-unmapped verdict.
- Phase 3.5 ([`ROADMAP-P3.5-001`](PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md))
  thêm pipeline nạp raw mesh/object/scene và tạo simulation-ready asset. Raw
  mesh được chuẩn hóa theo mét, tách visual/collision, rồi public Python CoACD
  API trong library tạo convex parts local-first trước immutable manifest. API
  được viết mới với full typed CoACD parameters; source
  `kaggle_pipeline_modular/stage0_coacd` chỉ làm legacy profile/regression
  reference, không phải implementation bắt buộc. Đây không phải HTTP service và
  không gửi mesh ra ngoài. Nếu không có scene,
  hệ thống tạo table/tray/bin ảo, bounded spawn region, thả vật bằng physics và
  chỉ nhận snapshot sau deterministic settle certification. P3.5 cung cấp
  Gymnasium single/vector contract, reset/randomization/checkpoint và GPU
  notebook evidence để sẵn sàng cho RL ở P5. MuJoCo CPU là oracle; MJX-Warp là
  GPU candidate đầu tiên, ManiSkill 3 là fallback qua active-hand compatibility gate;
  Isaac Lab chỉ là optional backend cho persistent RTX/cloud, không chặn v1.
  Theo ADR-0008, P3.5 gate hiện dùng LEAP+Allegro 2/2; Shadow được ghi
  `paused_by_ADR-0008`, không được suy thành three-hand coverage.
  P3.5 không chặn việc bắt đầu P4 static/offline sau khi dataset interface đã
  khóa, nhưng vẫn phải đóng hoặc được scope lại trước khi ghi P3 tổng `complete`.
- `DGN-Open-Tiny` đủ nhỏ cho CI/overfit và tái tạo từ đầu.

### P4 — Model MVP

- Point-cloud object encoder và variable-length HandGraph conditioning.
- Sinh trực tiếp palm pose + named joint state bằng flow/diffusion backend.
- Differentiable FK nối pose với keypoint/contact/force auxiliary heads.
- Baseline tối thiểu: direct-only, contact-first và evaluator-guided refinement.
- Overfit, backward và memory smoke của model phải chạy bằng CUDA; CPU chỉ là
  correctness reference.

### P5 — Training & evaluation

- Full training loop, sampling, quality head, hard negatives và simulator replay.
- Multi-hand train, held-out object family và held-out embodiment protocols.
- Ablation khóa trước khi chạy: graph/no-graph, direct/contact-first, FK
  consistency, energy guidance và evaluator refinement.
- Metrics gồm success, collision, penetration, diversity, coverage và latency.

### P6 — Scale & delivery

- Mở rộng `DGN-Open-v1`, model sizes `n/s/m` và distributed training.
- Profile memory để ngăn cấu trúc `N×N`; tối ưu batching và caching có evidence.
- Exact resume, safetensors weight bundle, TorchScript/ONNX và ONNX Runtime CPU.
- CPU/CUDA FP32 và AMP parity trên cùng dataset/model/robot hashes.

### P7 — Release & paper

- Package, quickstart, examples CPU, configuration/robot/data documentation.
- Model/data cards, SBOM, third-party notices, benchmark artifacts và release
  report.
- Paper dùng raw runs, environment lock, seeds, confidence intervals và ablation
  đã khóa; DGN2 chỉ là literature context.
- Issue/PR/RFC templates, good-first-issue, model zoo và release cadence.

## Công việc xuyên suốt

Các luồng sau chạy trong mọi phase và không được dồn về cuối:

| Luồng | Yêu cầu mỗi phase |
| --- | --- |
| License/provenance | Cập nhật exact pin, SPDX, NOTICE và artifact boundary |
| Test | Unit + integration phù hợp phase; regression cũ tiếp tục pass |
| Reproducibility | Lưu config, seed, environment/data/model/robot hashes |
| Documentation | Cập nhật contract, decision và session evidence cùng code |
| Performance | Train/AMP/distributed/benchmark chạy trên CUDA thật; CPU chỉ là correctness/smoke baseline |
| Research | Hypothesis/ablation trước experiment; không chọn kết quả sau khi xem số liệu |

## Xử lý ca khó

Khi một phase gặp blocker kỹ thuật, không mở rộng reference tùy ý. Quy trình là:

1. Tạo minimal reproduction và ghi expected/actual behavior.
2. Kiểm bốn baseline hiện có và paper primary liên quan.
3. Nếu vẫn thiếu, thực hiện `docs/governance/REFERENCE_INTAKE.md` rồi pin source
   mới dưới `.references/`.
4. Trích contract/test/toán học cần thiết; mặc định viết implementation mới.
5. Ghi cả kết luận đúng lẫn negative finding vào session evidence.

Reference mới không tự làm phase “quay lại từ đầu”; chỉ khi nó thay schema,
license, model topology hoặc protocol mới cần ADR/revision và đánh giá lại gate.

## Quy tắc ưu tiên

Trong một phase, thứ tự ưu tiên là correctness → reproducibility → license →
performance → tiện ích. Không scale data/model trước khi tiny overfit và physics
fixtures pass. Không viết paper claim trước khi protocol, environment và raw
artifact format được khóa.
