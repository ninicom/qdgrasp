---
document_id: ROADMAP-P3.3-001
document_type: plan
title: Kế hoạch Phase 3.3 — Scene Grasp Synthesis & Scene Dataset
version: 1.4.0
status: active
date: 2026-08-25
revises: ROADMAP-P3.3-001@1.3.0
related_plan: ROADMAP-P3.1-001
depends_on:
  - ROADMAP-P3.1-001
  - ROADMAP-P3.2-001
literature_cutoff: 2026-08-23
latest_revision_record: docs/revisions/REV-20260825-004-phase3-3-adapters.md
---

# Kế hoạch Phase 3.3 — Scene Grasp Synthesis & Scene Dataset

P3.3 mở rộng P3 từ grasp một object độc lập sang **grasp trong scene có clutter**.
Hệ thống phải vừa đọc được scene từ dataset bên ngoài, vừa tự dựng scene mới,
sinh candidate cho một target object, kiểm collision với toàn scene, chạy rollout
vật lý có các object lân cận và phát hành một scene dataset tái lập.

Artifact nhỏ đầu tiên là `QDGrasp-Scene-Tiny`. Nó không sao chép toàn bộ
GraspNet-1Billion hay dataset bên thứ ba vào repository; adapter đọc nguồn do
người dùng cung cấp, normalize về contract nội bộ và ghi provenance/hash.

Nhánh contact-rich trajectory search được tách thành
[`ROADMAP-P3.4-001`](PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md). P3.3 giữ
static/direct scene baseline; P3.4 cho phép target và scene dịch chuyển trong
acquisition dưới safety budget, rồi hợp nhất qua dataset manifest.

## 1. Baseline, dependency và entry gate

- Baseline bàn giao: commit `c7f4061` trên `feature/phase3-data-layer`.
- P3.2/P3.2.1 đã complete transmission/control parity và generated-reachable
  full-flow correctness cho LEAP, Allegro và Shadow (`REV-20260824-001`).
- P3.1-12 đã xóa fabricated positive; P3.1-13 đã chọn
  `region_opposition_v1`. Regeneration và release closure P3.1-14/15 còn mở.
- `build_rollout_scene_model()` hiện đã ghép hand, một target object và floor
  trong MuJoCo, nhưng chưa có scene schema, non-target objects, camera rig,
  imported-scene adapter hay scene-level dataset writer.
- `scripts/generate_dgn_open_tiny.py` không còn nhánh thay outcome bằng positive
  thủ công; regression `tests/test_no_positive_substitution.py` fail closed nếu
  positive thiếu passing rollout evidence.

### Entry gate bắt buộc

Trước khi P3.3 sinh bất kỳ positive label nào:

1. Xóa toàn bộ synthetic-positive injection khỏi generator.
2. Mọi `dynamic_valid=true` phải có trajectory evidence đủ stage từ validator
   thật: squeeze, lift và perturbation.
3. Dataset writer phải fail closed nếu evidence thiếu, mock, stage không đủ hoặc
   source được đánh dấu `test_fixture_only`.
4. P3.1 phải regenerate object-level dataset sạch trước khi scene-level release
   dùng pipeline đó.

Schema/import adapter và unit test của P3.3 có thể phát triển song song với phần
còn lại của P3.1. Dynamic scene generation chỉ mở sau entry gate.

## 2. Literature snapshot và quyết định triển khai

### 2.1 GraspNet-1Billion — adapter tương thích nền

Nguồn: [paper CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf),
[dataset format](https://graspnet.net/datasets.html) và
[graspnetAPI](https://graspnetapi.readthedocs.io/en/latest/grasp_format.html).

Dataset có 190 cluttered scenes, 97.280 RGB-D frames từ Kinect Azure và RealSense
D435, 88 object models, object poses, masks, camera calibration và hơn một tỷ
parallel-jaw grasp labels. Điểm mạnh cho P3.3 là scene replay có real sensor
observations và object-level pose; điểm yếu là grasp annotation dành cho kẹp hai
ngón, không phải LEAP/Allegro/Shadow.

**Quyết định:** adapter `graspnet1b` nhập scene layout, camera, depth/RGB, mask,
object identity/pose và collision labels. Parallel-jaw labels chỉ là optional
proposal/evaluation channel có provenance riêng; không chuyển trực tiếp thành
dexterous positive. Dexterous label phải được QDGrasp sinh và kiểm lại.

### 2.2 DexGraspNet 2.0 — nguồn tham chiếu scene dexterous chính

Nguồn: [paper CoRL/PMLR 2025](https://proceedings.mlr.press/v270/zhang25j.html)
và [repository/dataset chính thức](https://github.com/PKU-EPIC/DexGraspNet2).

Dataset công bố 1.319 objects, 8.270 synthetic clutter scenes và khoảng 426,6
triệu grasp labels cho LEAP. Paper cho thấy local geometry quanh target hữu ích
hơn ép một model hiểu đồng đều toàn scene; scene variation là thành phần dữ liệu
quan trọng, không chỉ tăng số grasp trên cùng object. Một phần scene benchmark
kế thừa layout GraspNet-1Billion, phần còn lại mở rộng bằng synthetic objects.

**Quyết định:** adapter `dexgraspnet2` là reference chính cho scene-level
dexterous format và benchmark parity. Import label LEAP phải giữ nguyên source
hand/version và chỉ mang trạng thái `external_label`; muốn đưa vào
`dynamic_valid` của QDGrasp phải replay bằng robot asset/protocol hiện tại.
Không suy label Allegro/Shadow từ LEAP bằng đổi tên joint.

### 2.3 GraspClutter6D — scene thật mật độ và occlusion cao

Nguồn: [paper/dataset 2025](https://arxiv.org/abs/2504.06866) và
[dataset page](https://sites.google.com/view/graspclutter6d/dataset).

Dataset nhắm tới 1.000 real scenes, 200 objects trong 75 environment
configurations, bốn RGB-D cameras, 52.000 frames, 736.000 object poses và 9,3 tỷ
feasible parallel-jaw annotations. Giá trị của nó với P3.3 là clutter thật,
occlusion cao và nhiều camera/layout hơn GraspNet-1Billion.

**Quyết định:** adapter `graspclutter6d` dùng để kiểm frame/calibration,
multi-camera observation, instance visibility và scene collision. Nhãn kẹp hai
ngón tuân cùng boundary như GraspNet: dùng làm external proposal/metric, không
phải dexterous ground truth.

### 2.4 CADGrasp — contact và collision phải nhìn toàn scene

Nguồn: [CADGrasp, 2026](https://arxiv.org/abs/2601.15039) và
[official repository](https://github.com/matthewmzy/CADGrasp).

CADGrasp dùng Interaction Bisector Surface để biểu diễn đồng thời khoảng cách
contact với target và khoảng hở collision với phần còn lại của scene. Bài học
quan trọng là target-local contact tốt vẫn có thể tạo grasp bất khả thi nếu
palm, mu bàn tay, wrist hoặc đường approach xuyên object lân cận.

**Quyết định:** P3.3 không bắt buộc triển khai IBS network, nhưng tạo module
`scene_clearance` riêng với target distance, obstacle distance, swept-volume và
per-link clearance. Contract cho phép thêm IBS/SDF backend sau này mà không đổi
dataset schema.

### 2.5 ClutterDexGrasp — difficulty curriculum và closed-loop safety

Nguồn: [ClutterDexGrasp, CoRL 2025](https://arxiv.org/abs/2506.14317) và
[official repository](https://github.com/QiyangYan/ClutterDexGrasp).

Phương pháp tổ chức scene theo clutter density, tách target-oriented planning và
closed-loop interaction, đồng thời dùng safety curriculum. Điều này chỉ ra rằng
“scene pass” không thể chỉ là hand chạm đúng target; hand không được hất/văng
non-target objects hoặc tận dụng chúng như giá đỡ giả.

**Quyết định:** scene generator có difficulty profile `single`, `sparse`,
`dense`; validator ghi non-target displacement, collision impulse, target
isolation và disturbance survival. P3.3 sinh pose/trajectory labels, chưa thêm
teacher policy, imitation learning hoặc diffusion policy.

### 2.6 GraspGen — BYOD và adapter boundary hiện đại

Nguồn: [GraspGen/GraspDataGen, ICRA 2026](https://github.com/NVlabs/GraspGen).

GraspGen công bố pipeline modular cho nhiều gripper, partial/full point cloud,
single object/clutter và workflow bring-your-own-dataset; dữ liệu phát hành có
hơn 57 triệu grasps trên 8.515 Objaverse XL objects cho hai gripper và suction.
Nó không phải dexterous scene ground truth cho ba hand của QDGrasp, nhưng là mẫu
tốt cho adapter, gripper-specific metadata và scene point-cloud collision filter.

**Quyết định:** contract P3.3 phải cho phép adapter/plugin mới và external object
catalog mà không sửa orchestrator. Không kéo Isaac Lab/Omniverse vào dependency
mặc định; backend đó chỉ có thể xuất hiện qua revision riêng.

### 2.7 Contact representation trong real-world clutter — hướng theo dõi 2026

Nguồn: [Nature Machine Intelligence, 12/08/2026](https://www.nature.com/articles/s42256-026-01292-y).

Paper mới nhất tại literature cutoff tách contact opportunity dùng chung khỏi
kinematics riêng của từng hand và kiểm tra trên clutter với nhiều embodiment.
Đây phù hợp trực tiếp với tuple contact/normal hiện có và HandGraph của dự án.

**Quyết định:** P3.3 lưu contact opportunity ở scene/object frame độc lập với
robot solution. Vì chưa xác minh được một public scene dataset/API ổn định từ
paper này, nó là design reference/watchlist, không phải adapter bắt buộc v1.

## 3. Phạm vi và ngoài phạm vi

### 3.1 Thuộc P3.3

- Canonical `SceneSpec`, `SceneObservation`, `SceneGraspOutcome` và manifest.
- Adapter cho GraspNet-1Billion, DexGraspNet 2.0 và GraspClutter6D.
- Native scene builder từ procedural objects hoặc external object catalog.
- Tabletop, bin và shelf/container support qua environment module.
- Physics settle, scene replay và deterministic scene composition.
- Target selection, partial point-cloud crop, scene-aware proposal/IK/collision.
- Multi-object MuJoCo rollout và sequential remove/re-settle generation.
- RGB/depth/instance/normal rendering từ camera rig đã pin.
- Phát hành `QDGrasp-Scene-Tiny` và schema mở rộng cho `QDGrasp-Scene-v1`.

### 3.2 Không thuộc P3.3

- Không train scene diffusion/VLA/teacher-student policy.
- Không sao chép hoặc redistribute raw third-party dataset khi license chưa cho.
- Không tự động tải hàng trăm GB/TB trong test hoặc gate.
- Không coi parallel-jaw label là dexterous-hand positive.
- Không synthesize object pose từ RGB-D khi source không có pose; đó là module
  perception/reconstruction riêng trong phase sau.
- Không thêm arm motion planning đầy đủ. P3.3 kiểm wrist/forearm proxy và swept
  approach; full arm reachability cần revision riêng.

## 4. Contract kiến trúc

### 4.1 Canonical scene schema

```text
SceneSpec
  scene_id, source_dataset, source_version, source_split
  environment: table | bin | shelf | custom
  objects[]: object_id, asset_ref, T_world_object, scale, mass, friction
  supports[]: body/geom specs and transforms
  cameras[]: camera_id, intrinsics, distortion, T_world_camera
  gravity, timestep, solver_profile, settle_seed
  source_record_hash, license_record, redistributable

SceneObservation
  scene_id, camera_id, frame_id, timestamp
  rgb_ref?, depth_ref?, point_cloud_ref
  instance_mask_ref?, normal_ref?, visibility_by_object
  T_world_camera and calibration hash

SceneGraspOutcome
  scene_id, target_object_id, robot_profile, candidate_id
  contact_opportunity in object/world frame
  palm/joint/actuator command and active fingers
  approach_path and swept-clearance metrics
  static certificate and dynamic trajectory evidence
  target motion, non-target motion/impulse, scene-state hashes
  label stage, reason, recipe/protocol/source hashes
```

Mọi transform dùng metre, radian, right-handed coordinates và ma trận
`T_destination_source`. Adapter phải lưu transform gốc→canonical và có test
round-trip; không đoán axis convention từ filename.

### 4.2 Adapter interface

Mỗi adapter nằm trong `qdgrasp/scenes/adapters/` và implement:

```text
probe(root) -> SourceDatasetInfo
index(root, split, limits) -> SceneIndex
load_scene(scene_key) -> SceneSpec
load_observation(scene_key, camera_key, frame_key) -> SceneObservation
load_external_grasps?(...) -> ExternalGraspSet
audit(scene_key) -> SourceEvidence
```

Registry là allowlist: `native`, `graspnet1b`, `dexgraspnet2`,
`graspclutter6d`. Root chỉ nhận qua CLI/config/environment, không hard-code
đường dẫn cá nhân. Adapter optional không làm clean-wheel import phụ thuộc SDK
bên thứ ba; thiếu dependency phải báo hướng cài rõ ràng.

### 4.3 Ba lớp dữ liệu

1. **External source view:** read-only, lazy, giữ nguyên file nguồn.
2. **Normalized cache:** canonical index/transform/selected geometry; có thể xóa
   và dựng lại, không được gọi là release dataset.
3. **QDGrasp release:** chỉ chứa asset được phép phân phối, scene native hoặc
   derived records hợp lệ; mỗi record liên kết hash tới nguồn.

## 5. Các module tạo scene

### 5.1 `replay_imported`

Khôi phục object poses, support geometry và camera từ source. Sau compile phải
so object pose/camera reprojection với source tolerance. Scene không ổn định vật
lý được giữ ở chế độ observation replay, nhưng không dùng làm dynamic rollout
cho tới khi có settle/reconciliation record.

### 5.2 `drop_and_settle`

Chọn object assets bằng seed, spawn trên table/bin/shelf với separation ban đầu,
thả bằng MuJoCo, chờ kinetic energy và pose delta dưới threshold liên tục. Scene
không settle trong bounded steps bị reject; không tăng damping tùy scene để ép
pass.

### 5.3 `pose_compose`

Tạo scene trực tiếp từ pose templates hoặc imported layout rồi chạy collision
repair giới hạn. Mọi object bị di chuyển phải ghi delta so với template; repair
không được âm thầm đổi scene benchmark.

### 5.4 `sequential_declutter`

Từ một scene settled, chọn target, sinh/validate grasp, lấy object ra, settle lại
và lặp. Mỗi bước tạo scene-state mới liên kết parent hash. Một grasp pass ở state
trước không tự động pass ở state sau.

## 6. Pipeline sinh grasp trong scene

```text
Scene source/builder
  → canonicalize + settle/reconcile
  → choose target and visible observation
  → target-local contact proposals
  → hand IK/transmission command
  → whole-scene static + swept collision
  → multi-object MuJoCo squeeze/lift/perturb
  → scene-aware label and rendered evidence
  → immutable scene shard + manifest
```

### 6.1 Target và observation

- Target selection có module `uniform_visible`, `difficulty_weighted` và
  `declutter_ordered`; policy ID được ghi vào sample.
- Visibility tính từ depth/instance mask thật của camera đã chọn.
- Network-ready point cloud có cả target mask và context points; object-frame
  crop chỉ là một view, không được làm mất obstacle geometry cho collision gate.

### 6.2 Scene-aware feasibility

- Target self-collision/contact dùng contract P3.1.
- Obstacle collision kiểm mọi hand link, wrist proxy và approach sweep.
- Support contact của target được ghi riêng; floor/bin/shelf không được tính như
  finger contact hoặc grasp wrench.
- Candidate bị reject nếu cần xuyên non-target object để tới pregrasp/grasp.

### 6.3 Dynamic scene label

Rollout gồm pregrasp, squeeze, lift và perturbation. Ngoài tiêu chí P3.1/P3.2,
scene success phải thỏa:

- đúng `target_object_id` được nâng;
- target đạt lift ratio và giữ qua disturbance;
- non-target object không bị nâng nhầm;
- displacement/rotation/impulse của non-target dưới threshold đã pin;
- không dùng support/floor/non-target làm contact giả;
- scene và robot finite, không MuJoCo warning;
- positive có stage snapshots/hashes thật, không fabricated validation.

Negative reasons tối thiểu: `target_occluded`, `approach_blocked`,
`hand_scene_collision`, `wrong_object_contact`, `non_target_disturbed`,
`target_not_lifted`, `scene_unstable`, `source_frame_invalid`.

## 7. Rendering và kiểm tra trực quan

Mỗi known-positive và known-negative canonical fixture xuất ảnh pin camera ở:

- initial/settled scene;
- pregrasp;
- squeeze;
- lift;
- perturbation/final.

Ảnh phải overlay target ID, active fingers, contact points, approach path và
failure reason. Render được tạo từ đúng `MjData`/scene-state đã sinh label, không
dựng lại pose bằng script khác. Release manifest liên kết image hash với outcome.

## 8. Dataset mới

### 8.1 `QDGrasp-Scene-Tiny`

- 12 native canonical scenes: tabletop, bin và shelf/container.
- Ba clutter tiers: `single` (1 object), `sparse` (2–4), `dense` (5–10).
- Tối thiểu hai camera views/scene và đủ RGB-D/point cloud/instance mask.
- LEAP, Allegro và Shadow đi qua cùng public pipeline.
- Có positive thật và negative cho collision/occlusion/non-target disturbance.
- Split theo object family **và scene template**; không để cùng layout/seed rò
  sang train/validation.

CI chỉ chạy micro fixtures và một scene/hand; tạo đủ 12 scene là bounded release
job, không nằm trong unit test.

### 8.2 Schema phát hành

- `scene_manifest.json`: source/build recipe, object/camera/environment hashes.
- `scene_state` shards: initial, settled và state lineage.
- `observation` shards: references tới packed RGB/depth/mask/point cloud.
- `grasp` shards: target, robot solution, labels và trajectory metrics.
- `dataset_manifest.json`: split, counts, class/robot/layout coverage, source
  licenses, protocol hashes và shard SHA-256.

Metadata dùng sharding có index; không tạo hàng triệu file nhỏ. Images có thể
được pack theo scene shard. Storage backend phải deterministic và không làm
loader cần source dataset sau khi một native release hợp lệ đã được tạo.

## 9. License và provenance boundary

- Mỗi adapter có `SourceDatasetManifest` pin URL/DOI, version, file layout,
  license/terms, download timestamp và expected checksums khi upstream cung cấp.
- Mặc định `redistributable=false` cho external RGB, depth, mesh và labels cho
  tới khi review license xác nhận ngược lại.
- Repository chỉ lưu adapter code, tiny synthetic fixtures tự tạo và provenance;
  không commit raw GraspNet/DexGraspNet2/GraspClutter6D.
- Derived QDGrasp labels trên external scene phải giữ source scene ID/hash và có
  release policy riêng; “derived” không tự động có nghĩa được phép phân phối.
- Import phải hoạt động bằng user-provided root và không tải network ngầm.

## 10. Work breakdown

| ID | Module | Output chính | Phụ thuộc |
| --- | --- | --- | --- |
| P3.3-00 | Genuine-positive entry gate | xác minh P3.1 đã bỏ fabricated label; thêm scene writer regression | P3.1-12 |
| P3.3-01 | Scene contracts và frame convention | `qdgrasp/scenes/contracts.py` | 00 |
| P3.3-02 | Native adapter + allowlist registry | `scenes/adapters/native.py` | 01 |
| P3.3-03 | GraspNet-1Billion adapter | `adapters/graspnet1b.py` | 01 |
| P3.3-04 | DexGraspNet 2.0 adapter | `adapters/dexgraspnet2.py` | 01 |
| P3.3-05 | GraspClutter6D adapter | `adapters/graspclutter6d.py` | 01 |
| P3.3-06 | Environment modules | table/bin/shelf support specs | 01 |
| P3.3-07 | Replay/compose/drop-settle builders | `scenes/builders/` | 02/06 |
| P3.3-08 | Camera renderer + observation builder | `scenes/observations/` | 01/07 |
| P3.3-09 | Target selection + scene crop modules | `scenes/targets/` | 07/08 |
| P3.3-10 | Scene clearance + swept collision | `scenes/validators/clearance.py` | 07/09 |
| P3.3-11 | Multi-object dynamic validator | scene rollout + evidence | P3.1/P3.2/10 |
| P3.3-12 | Sequential declutter generator | state lineage pipeline | 11 |
| P3.3-13 | Scene shards/manifest/loader | `qdgrasp/dataset/scene_*` | 01/08/11 |
| P3.3-14 | `QDGrasp-Scene-Tiny` generation + rendered QA | dataset and image evidence | 12/13 |
| P3.3-15 | Gates, docs, license review và handoff | `scripts/check_phase3_3.py`, revision | 14 |

Thứ tự bắt buộc:
`00 → 01 → (02 || 03 || 04 || 05 || 06) → 07 → 08 → 09 → 10 → 11 →
(12 || 13) → 14 → 15`.

Mỗi source và mỗi strategy là module riêng. Orchestrator chỉ gọi contract; không
chứa branch theo tên dataset, hand hoặc environment.

### Trạng thái thực thi tại checkpoint `a6d067e` (2026-08-25)

| Work package | Trạng thái | Bằng chứng / blocker |
| --- | --- | --- |
| P3.3-00 | complete | no-positive-substitution regression pass; P3.2.1 measured full flow 3/3 hand |
| P3.3-01 | complete | canonical dataclasses/interface được thực thi qua bốn adapter và 35 scene tests |
| P3.3-02 | complete | native adapter đọc scene manifest/shards đã verify, kiểm SceneSpec/calibration/reference/hash và fail closed tại `a6d067e` |
| P3.3-03…05 | implementation complete; source-scale audit pending | GraspNet-1Billion, DexGraspNet2 và GraspClutter6D load scene/pose/camera/observation/external label thật; source manifest/license/hash và audit fail closed tại `a6d067e`; chưa claim full-dataset replay |
| P3.3-06 | implemented, audit pending | table/bin/shelf support specs đã có test; transform compile được builder kiểm fail-closed |
| P3.3-07 | complete | verified object manifests được compile thành MuJoCo geoms; replay giữ pose; settle finite/velocity/timeout gate tại `3a2c6a9` |
| P3.3-08 | remediation required | camera render RGB/depth/segmentation có thật nhưng visibility vẫn map geom ID thô và được ghi chú mock; chưa pack observation evidence |
| P3.3-09 | implemented, audit pending | ba target selector deterministic có micro tests; context crop module theo plan chưa tách riêng |
| P3.3-10 | complete | swept translation/rotation samples, target-at-goal policy, whole-scene rejection, state restore và MuJoCo micro fixtures tại `5276232` |
| P3.3-11 | validator complete; rollout integration pending | stage/hash/load/lift/contact/non-target displacement/rotation/impulse fail-closed tại `fd5f46e`; chưa instrument multi-object hand rollout để sinh evidence đầu vào |
| P3.3-12 | complete | sequential revalidation, exact-target removal và parent/child lineage hash tại `e266ed7` |
| P3.3-13 | complete | canonical JSONL shards, manifest, loader, positive admission và cross-shard audit tại `6639c5f` |
| P3.3-14…15 | pending | bị chặn bởi renderer 08, multi-object rollout 11, external source-scale smoke và genuine rendered release evidence |

Checkpoint adapter: 35/35 test trong `tests/scenes` pass tuần tự, giới hạn một
BLAS/OpenMP thread. Bốn adapter dùng fixture theo layout nguồn, kiểm pose/frame,
visibility, external-label boundary, file provenance và negative fail-closed.
Kết quả xác nhận implementation 02–05 nhưng không thay thế smoke trên dataset
nguồn đầy đủ, multi-object rollout hoặc evidence `QDGrasp-Scene-Tiny`.

## 11. Test matrix bắt buộc

### Adapter và frame

- Tiny synthetic fixture cho mỗi adapter; unit test không cần tải dataset thật.
- Missing root/file/camera/pose/license record fail closed với source key rõ.
- Source→canonical→source transform round-trip trong tolerance.
- Depth unprojection và object mask align với known 3D points.
- Scene index deterministic, split/limit hoạt động mà không scan toàn root.
- External grasp giữ đúng `external_label`; không thể tự đổi thành
  `dynamic_valid`.

### Scene builder

- Cùng seed/config tạo cùng object IDs, initial poses và settled state hash.
- Spawn không interpenetration vượt tolerance; settle có bounded timeout.
- Imported replay giữ camera/object pose hoặc báo reconciliation delta.
- Parent/child hash đúng qua một bước sequential removal.

### Scene grasp và physics

- Candidate target-valid nhưng đụng obstacle phải fail trước rollout.
- Swept approach bắt được collision mà endpoint-only check bỏ sót.
- Known-positive thật cho ba hand trong scene sparse.
- Negative: wrong target, non-target support, blocked approach, excessive
  neighbor displacement, floor support và unstable scene.
- Không positive nào thiếu measured contacts, trajectory stages hoặc state hash.
- Xóa nhánh genuine validation phải làm generator test fail; monkeypatch
  `DynamicValidation(passed=True)` không đủ để writer chấp nhận release sample.

### Dataset/reproducibility

- Hai clean staging runs tạo cùng relative paths và hashes.
- Split không rò object family, imported source scene hoặc native template.
- Audit class/robot/layout/clutter/source coverage.
- Loader trả target-local và full-scene observations trong frame đã khai báo.
- Clean wheel import adapter registry mà không có external dataset SDK.

## 12. Resource và I/O safety

Do workstation từng hard-freeze vì I/O:

- Mọi CLI có `--scene-limit`, `--frame-limit`, `--worker-count` mặc định 1,
  timeout hữu hạn và dry-run index summary.
- Không recursive-hash toàn dataset root; chỉ hash file thuộc selected records.
- Không copy external dataset vào cache nếu chỉ cần reference/index.
- Không preload image/point cloud của nhiều scene; iterator lazy và bounded.
- Không chạy nhiều camera render/MuJoCo process song song mặc định.
- Full external adapter smoke dùng đúng một scene, một frame và không viết output
  ngoài temporary directory.
- Release job ghi checkpoint tiến độ nguyên tử để resume, không khởi động lại từ
  đầu sau lỗi; không retry vô hạn scene không settle.

## 13. Gate đóng P3.3

```bash
timeout 60s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  .venv/bin/python scripts/check_docs.py --root .

timeout 180s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  .venv/bin/python scripts/check_phase3_3.py --profile micro

.venv/bin/python -m pytest tests/scenes/ -q
git diff --check
git status --short
```

External full-dataset checks là opt-in và ghi evidence riêng; chúng không chạy
trong CI. `check_phase3_3.py` tối thiểu kiểm contracts, adapter micro fixtures,
frame parity, deterministic native scene, swept collision, dynamic positives/
negatives ba hand, manifest audit và absence của fabricated labels.

## 14. Điều kiện hoàn tất

P3.3 chỉ complete khi:

1. Ba external adapters đọc được official-layout micro/real smoke mà không đổi
   source files và có license/provenance record.
2. Native builders tạo/replay scene deterministic cho ba environment modules.
3. Ba hand sinh grasp trong cùng scene contract và whole-scene gates.
4. Mọi positive trong `QDGrasp-Scene-Tiny` có measured physical evidence thật.
5. Dataset có scene/object/camera/robot/split hashes và regenerate parity.
6. Rendered stage images cho ba hand được maintainer kiểm tra trực quan.
7. Resource gate chứng minh không full-root scan/copy và worker mặc định bằng 1.
8. Independent review xác nhận frame math, collision semantics, license boundary
   và không có synthetic-positive bypass.

P3.3 tạo dữ liệu scene và target-local views tương thích với P4 hiện tại. Việc
đổi P4 sang full scene-conditioned architecture phải có revision P4 riêng; plan
này không âm thầm thay model objective.
