---
document_id: ROADMAP-P2-001
document_type: plan
title: Kế hoạch thực thi Phase 2 — Robot layer
version: 1.0.0
status: active
date: 2026-08-22
revises: none
related_plan: PLAN-V2
---

# Kế hoạch thực thi Phase 2 — Robot layer

Tài liệu này chỉ diễn giải cách thực thi P2 đã được `docs/roadmap/PROJECT_PHASES.md`
(P2), `PLAN.md` §3/§6 và `docs/decisions/0003-cross-embodiment-flow-architecture.md`
§1 quy định. Nó không sửa, không mở rộng và không đánh giá lại quyết định của
phase trước. Người thực hiện là maintainer; kế hoạch viết để chạy được mà không
cần suy đoán.

## 1. Ranh giới phạm vi

`PLAN.md` gộp **M2 = data generator + robot layer**, còn `PROJECT_PHASES.md`
tách thành **P2 = robot layer** và **P3 = data layer** với hai gate riêng. Vì
`PROJECT_PHASES.md` là tài liệu định nghĩa ranh giới và gate của phase, kế hoạch
này chỉ nhận nửa robot của M2.

Thuộc P2:

- `RobotSpec`, named joints, joint limits, mimic/coupling policy và `HandGraph`.
- URDF/MJCF importer, mesh resolver, frame transforms, FK và batch kinematics.
- MuJoCo adapter cho fixture grasp/squeeze/lift.
- Compatibility matrix bắt buộc cho LEAP, Allegro và Shadow Hand.

**Không** thuộc P2, để nguyên cho P3: `GraspBatch`, dataset shards/splits/manifest,
procedural/CC0 object generator, candidate sampler, collision filter, physics
label pipeline và `DGN-Open-Tiny`.

Artifact cuối phase: LEAP, Allegro và Shadow cùng chạy qua một `RobotSpec`.
Gate chuyển phase: parse/mesh/FK/MuJoCo fixtures pass cho cả ba hand.

## 2. Điều kiện tiên quyết

1. `feature/phase1-framework` chưa merge vào `develop`, mà `scripts/git/start_feature.sh`
   bắt buộc đứng trên `develop`. Chọn một trong hai và ghi vào session report:
   merge P1 trước rồi `start_feature.sh phase2-robot-layer`, hoặc tạo branch từ
   commit P1 và ghi rõ phụ thuộc theo `GIT_WORKFLOW.md` §1.
2. `REV-20260822-010` đang `in_review`. Trạng thái `complete` của P1 — và do đó
   thẩm quyền bắt đầu P2 — là tạm thời cho tới khi reviewer không phải tác giả
   xác nhận. Đây là caveat cần ghi, không phải blocker.
3. Bốn checkout robot asset đã có sẵn dưới `.references/robot-assets/`; không cần
   fetch lại. Chúng là read-only.

### Bước 0 — xác minh trước khi thiết kế

Trường `compatibility` trong `robot_assets.lock.yaml` được ghi lúc tạo lock, có
thể dưới build MuJoCo khác. Trước khi thiết kế theo nó, chạy lại với
`mujoco 3.12.0` đang cài và ghi kết quả thật vào session report:

```bash
.venv/bin/python -c "
import mujoco
for name in ('shadow_hand', 'wonik_allegro', 'leap_hand'):
    path = f'.references/robot-assets/mujoco-menagerie/{name}/right_hand.xml'
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(name, 'nq', model.nq, 'nu', model.nu, 'nbody', model.nbody)
"
```

Nếu số liệu lệch lock, dừng và xử lý theo `docs/governance/REFERENCE_INTAKE.md`
trước khi viết code.

## 3. Ràng buộc kế thừa (không thay đổi)

- Không sửa `PLAN.md`, `PROJECT_PHASES.md`, ADR, `qdgrasp/runtime.py`, environment
  lock, train-argument registry và cây legacy Ultralytics-derived.
- **Không thêm dependency mới.** Dep đã khai báo: torch, lightning, numpy, scipy,
  mujoco, trimesh, safetensors, pydantic, PyYAML, einops, rich, typer. Không có
  thư viện URDF nào trong đó, nên URDF parse bằng `xml.etree` của stdlib; MJCF đi
  qua `mujoco`; mesh load qua `trimesh`. Kéo thêm `yourdfpy`, `urdfpy` hay `lxml`
  buộc phải regenerate lock và chạy lại toàn bộ gate CPU/CUDA theo bàn giao P0.
- Mọi đường dẫn là relative path; `RunConfig.project_dir` đã từ chối absolute path.
- RH56E2 bị loại hoàn toàn. `scripts/check_phase0.py` quét mọi tracked file theo
  regex `rh56[\s_-]*e2` với allowlist theo path, nên bất kỳ file robot mới nào
  nhắc tên này sẽ làm fail gate P0.
- Version package giữ `0.1.0a1` cho tới khi có release branch.

### Ánh xạ thư mục

Khác P1: `qdgrasp/robot/` và `qdgrasp/sim/` đều trống, không đụng legacy, nên P2
dùng thẳng tên của `PLAN.md` §3. Không cần bảng ánh xạ như P1.

| Thành phần | Đường dẫn |
| --- | --- |
| Schema profile, parser, RobotSpec, HandGraph, FK | `qdgrasp/robot/` |
| MuJoCo adapter và fixtures | `qdgrasp/sim/` |
| Profile YAML của ba hand | `qdgrasp/presets/robots/` |
| Normalization transform và manifest | `qdgrasp/robot/normalize.py` |

## 4. Corpus và vai trò từng asset

Nguồn và hash chuẩn nằm trong `robot_assets.lock.yaml`; bảng dưới chỉ nêu vai trò.

| Asset | Đường dẫn | License | Vai trò trong P2 |
| --- | --- | --- | --- |
| Menagerie Shadow E3M5 | `mujoco-menagerie/shadow_hand/right_hand.xml` | Apache-2.0 | MJCF chuẩn; ca kiểm coupling (`nq 24`, `nu 20`) |
| Menagerie Wonik Allegro | `mujoco-menagerie/wonik_allegro/right_hand.xml` | BSD-2-Clause | MJCF chuẩn (`nq 16`, `nu 16`) |
| Menagerie LEAP | `mujoco-menagerie/leap_hand/right_hand.xml` | MIT | MJCF chuẩn (`nq 16`, `nu 16`) |
| LEAP official URDF | `leap-hand-sim/assets/leap_hand/robot.urdf` | MIT | URDF chuẩn, 17 link, 16 movable joint |
| Allegro ROS2 URDF | `wonik-allegro-ros2/.../allegro_hand_description_right_A.urdf` | BSD-2-Clause | URDF chuẩn, 22 link, 16 movable joint; **raw MuJoCo forward fail** |
| dex-urdf Shadow URDF | `dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf` | Apache-2.0 quan sát được | URDF **ngoài distribution** để kiểm parser, 33 link, 24 movable joint |
| dex-urdf Barrett | `dex-urdf/robots/hands/barrett_hand/bhand_model.urdf` | không đủ SPDX | Fixture nghiên cứu, **bị chặn phát hành** |

Hai điểm dễ sai:

- `dexsuite_dex_urdf` có `release_status: reference_only_not_for_redistribution`.
  Shadow URDF của nó chỉ là parser fixture, **không bao giờ** trở thành profile
  được phát hành.
- Barrett có `distribution_status: blocked_pending_rightsholder_license`. Nếu dùng
  làm fixture, profile phải mang cờ release-blocked và gate phải cưỡng chế cờ đó.

## 5. Hạng mục công việc

| ID | Hạng mục | Output | Ghi chú |
| --- | --- | --- | --- |
| P2-01 | Schema `qdgrasp/robot/v2` cạnh v1, cùng migration test | `qdgrasp/robot/schema.py` | v1 giữ nguyên nghĩa |
| P2-02 | URDF importer bằng `xml.etree` | `qdgrasp/robot/urdf.py` | link/joint/axis/origin/limit/mimic/mesh ref |
| P2-03 | MJCF importer qua `mujoco` | `qdgrasp/robot/mjcf.py` | body/joint/actuator/geom |
| P2-04 | Mesh resolver `package://`, relative và `trimesh` load | `qdgrasp/robot/meshes.py` | zero missing mesh |
| P2-05 | Normalization transform tái lập | `qdgrasp/robot/normalize.py` | **hạng mục lớn nhất**, xem §6 |
| P2-06 | `RobotSpec` và `HandGraph` biến độ dài | `qdgrasp/robot/spec.py`, `graph.py` | không tạo `N×N` |
| P2-07 | Frame transforms, FK và batch kinematics bằng torch | `qdgrasp/robot/kinematics.py` | CPU FP32 là reference |
| P2-08 | MuJoCo adapter và fixture grasp/squeeze/lift | `qdgrasp/sim/mujoco.py`, `fixtures.py` | deterministic seed |
| P2-09 | Compatibility matrix ba hand cộng parser fixture thứ tư | `qdgrasp/presets/robots/*.yaml`, `tests/` | |
| P2-10 | Provenance/license manifest cho mỗi profile công khai và cờ release-blocked | `qdgrasp/robot/provenance.py` | |
| P2-11 | Checker cho `robot_assets.lock.yaml` | `scripts/check_robot_assets.py` | xem §7 |
| P2-12 | Gate script CPU của P2 | `scripts/check_phase2.py` | |
| P2-13 | Test suite | `tests/test_robot_*.py`, `tests/test_sim_*.py` | |
| P2-14 | Contract doc và session report | `docs/configuration/ROBOT_PROFILE.md`, `docs/sessions/` | `document_type: registry` |
| P2-15 | CUDA FK parity qua harness Kaggle | `scripts/phase2_cuda_fk_parity.py`, `kaggle-phase2/` | ngoài gate roadmap, xem §10 |

Thứ tự đề xuất: P2-01 → P2-02/03 → P2-04 → P2-05 → P2-06 → P2-07 → P2-08 →
P2-09/10 → P2-11/12/13/14 → P2-15. P2-02 và P2-03 chạy song song được. P2-15
chạy sau khi P2-07 ổn định vì nó cần FK đã có kết quả CPU tham chiếu.

## 6. Hai hạng mục dễ bị đánh giá thấp

### P2-01 — schema v2 phải sống cạnh v1, không thay thế

`docs/configuration/RUN_CONFIG.md` §3 và §7 đã khóa quy tắc: Phase 2 mở rộng robot
schema bằng **version mới**, không sửa nghĩa `v1` tại chỗ. Hệ quả cụ thể:

- `qdgrasp/robot/v2` bổ sung: khai báo link base/wrist/palm/contact/fingertip,
  joint order, frame transform, actuator/squeeze settings, mesh/package resolution
  và mimic/coupling policy.
- `dummy-hand.yaml` **giữ nguyên v1** để lifecycle dummy của P1 và test của nó
  tiếp tục pass. Ba hand thật phát hành dưới v2.
- `load_public_bundle` đã có sẵn cơ chế từ chối theo profile hash — mở rộng, không
  viết lại. Bundle v1 không load được với profile v2 là **đúng thiết kế**, không
  phải regression.
- Migration test: document v1 vẫn parse, v2 round-trip, và hash v1 khác hash v2.

### P2-05 — normalization là deliverable hạng nhất

`robot_assets.lock.yaml` ghi hai asset không load thô được:

- `wonik_allegro_ros2_v5`: `raw_mujoco_forward: fail_invalid_fullinertia`,
  `simulation_path: use_paired_menagerie_mjcf_or_recorded_normalization`.
- `dexsuite_dex_urdf` `allegro_hand`: `raw_mujoco_forward: fail_balanceinertia`.

`PLAN.md` §3 cấm đường tắt: “Raw asset không được sửa để làm test pass; mọi
normalization phải là transform tái lập, có hash nguồn/đầu ra và cờ `modified`.”
§6 nhắc lại: transform sửa inertia/package URI phải tái lập và không ghi đè asset
gốc.

Nên P2-05 là một pipeline đầy đủ: đọc source read-only → áp transform khai báo →
ghi artifact dẫn xuất ra ngoài `.references/` → ghi manifest gồm source hash,
transform id, output hash và cờ `modified`. Chạy hai lần trên cùng input phải cho
cùng output hash. Đây là hạng mục tốn công nhất của P2; đừng xếp nó cuối.

## 7. Lỗ hổng governance cần vá trong P2 (P2-11)

`scripts/check_references.py` chỉ khóa `references.lock.yaml`. Thứ duy nhất chạm
tới `robot_assets.lock.yaml` hiện nay là một lệnh grep marker trong
`scripts/check_phase0.py`. Nhưng P2 chính là phase phát hành robot profile công
khai mà provenance dựa hoàn toàn vào lock đó.

`scripts/check_robot_assets.py` phải kiểm, theo mẫu của `check_references.py`:
origin/commit/tree của từng checkout, `selected_manifest_sha256`,
`entrypoint_sha256`, `license_sha256`, checkout sạch, và exclusion RH56E2. Chỉ
stdlib. Sau đó thêm nó vào danh sách gate trong session report cạnh
`scripts/check_phase2.py`.

## 8. Tiêu chí nghiệm thu

Mỗi mục dưới đây phải kiểm được bằng lệnh, không phải bằng nhận định.

**Parse và mesh**

- Cả ba MJCF Menagerie và cả ba URDF (LEAP, Allegro ROS2, dex-urdf Shadow) parse
  từ raw XML, không sửa file nguồn.
- Zero missing mesh trên mọi profile phát hành; mesh load được bằng `trimesh`.
- LEAP URDF cho 17 link / 16 movable joint; Allegro ROS2 URDF cho 22 link / 16
  movable joint; dex-urdf Shadow cho 33 link / 24 movable joint.

**Joint và coupling**

- Mọi actuated revolute/prismatic joint có limit hữu hạn; continuous hoặc thiếu
  limit bắt buộc override tường minh, không im lặng gán mặc định.
- **Shadow là ca kiểm coupling**: từ `nq 24` phải rút ra đúng 20 actuated named
  joint, và quan hệ coupling phải được **khai báo**, không phải suy đoán. Allegro
  và LEAP đều 16/16.
- Fixed và mimic joint được xử lý tự động và có test riêng.

**Semantic link — kiểm bằng test phủ định**

- Parser **từ chối** phát ra profile khi palm/wrist/fingertip/contact chỉ được suy
  đoán. `PLAN.md` §3: “semantic links không được suy đoán im lặng.” Test phải
  chứng minh parser fail chứ không phải đoán đúng.

**FK và HandGraph**

- FK/frame fixture pass cho ba hand; batch kinematics cho cùng kết quả với vòng
  lặp từng mẫu trong tolerance.
- `HandGraph` biến độ dài, và memory profile chứng minh **không tạo `N×N`**: tăng
  số link 2× không làm memory tăng gần 4×.

**MuJoCo**

- `mj_forward` pass cho ba hand qua đường MJCF chuẩn.
- Fixture grasp/squeeze/lift kiểm geometry, frames, joint order, actuator mapping
  và repeatability với seed cố định.
- Allegro ROS2 URDF chỉ vào được MuJoCo qua normalization đã ghi manifest, hoặc
  qua MJCF Menagerie ghép cặp — không bao giờ bằng cách sửa file gốc.

**Normalization**

- Chạy hai lần cùng input cho cùng output hash.
- Manifest có source hash, output hash và cờ `modified`; asset gốc không đổi byte.

**CPU/CUDA parity của FK (P2-15)**

- FK và batch kinematics cho ba hand chạy được trên CUDA thật, không hard-code
  `.cuda()` và không fallback CPU khi CUDA được yêu cầu.
- Sai khác lớn nhất giữa FP32 CPU và FP32 CUDA `<= 1e-4` trên cùng profile hash,
  cùng seed và cùng joint vector.
- Evidence JSON có environment fingerprint, profile hash của ba hand và deviation
  từng hand; lưu theo mẫu `evidence/phase2-run-NNN-pass/`.

**Provenance và release**

- Mỗi URDF/mesh/profile công khai có provenance và license manifest.
- Profile dựng từ `dexsuite_dex_urdf` hoặc Barrett mang cờ release-blocked và gate
  từ chối phát hành chúng.
- Checkpoint bị từ chối khi profile hash hoặc joint schema không khớp.

## 9. Gate phải chạy trước khi đóng phiên

```bash
python3 scripts/check_references.py --lock-only
python3 scripts/check_references.py --source-root .references
python3 scripts/check_train_args.py --registry-only
python3 scripts/check_train_args.py --source .references/ultralytics
python3 scripts/check_docs.py --root .
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
python3 scripts/check_phase0.py
python3 scripts/check_phase1.py
python3 scripts/check_robot_assets.py
python3 scripts/check_phase2.py
python3 -m pytest tests/ -q
```

P1 phải tiếp tục pass nguyên vẹn; bất kỳ regression nào ở `check_phase1.py` là lỗi
của P2, không phải lý do sửa gate P1.

P2-15 không nằm trong danh sách trên vì nó chạy trên GPU ngoài máy phát triển;
kết quả của nó vào bảng test của session report như `T-NN` riêng.

## 10. CUDA trong P2 (P2-15)

Gate chuyển phase của P2 theo `PROJECT_PHASES.md` là parse/mesh/FK/MuJoCo, toàn
bộ chạy CPU. Kế hoạch này **không** sửa định nghĩa gate đó: P2-15 là hạng mục bổ
sung của kế hoạch thực thi, không phải điều kiện chuyển phase. P2 vẫn đóng được
khi P2-15 fail, miễn là kết quả fail được ghi lại và xử lý.

Lý do vẫn đưa vào: FK và batch kinematics là torch op mà P4 trở đi chạy trên
CUDA, và `PLAN.md` §6 đòi CPU/CUDA FP32 parity `<= 1e-4`. Bug resume của P1 là
bằng chứng trực tiếp rằng code chạm device chỉ verify trên CPU sẽ bỏ sót một lớp
lỗi thật: FP32 train và AMP train đều pass trên T4, chỉ bước đầu tiên sau resume
mới lộ ra optimizer state nằm sai device. FK có cùng đặc tính rủi ro đó. Phát
hiện ở P2 rẻ hơn nhiều so với phát hiện giữa P4.

Cách làm, theo đúng khuôn đã chạy được ở P1:

1. Viết `scripts/phase2_cuda_fk_parity.py` fail-closed: gọi
   `qdgrasp.require_cuda()`, dựng `RobotSpec` cho ba hand, chạy FK trên CPU và
   CUDA với cùng seed/joint vector, so deviation, ghi evidence JSON kèm hash.
   Script từ chối chạy khi `--device` không phải CUDA.
2. Tạo `kaggle-phase2/` trong repository `ninicom/qdgrasp-cuda-kaggle` với
   `kernel-metadata.json` riêng; **không** sửa `kaggle/` của P0 hay
   `kaggle-phase1/`.
3. Notebook cài `qdgrasp` từ exact public commit, tải script gate từ đúng commit
   đó qua `raw.githubusercontent.com` và assert SHA-256 trước khi chạy. Cách này
   giữ script gate chỉ có một nguồn duy nhất trong repository library.
4. Lưu evidence pass vào `evidence/phase2-run-NNN-pass/` và ghi hash vào session
   report. Run fail vẫn giữ trong bảng test của session report, không xóa.

Ràng buộc: notebook cần commit library đã public, nên bước này chỉ chạy được sau
khi branch P2 được push. Nếu chưa muốn public code P2, hoãn P2-15 tới lúc đó và
ghi là `Bị chặn` với điều kiện gỡ chặn rõ ràng, thay vì bỏ im lặng.

## 11. Ngoài phạm vi P2

- Toàn bộ data layer (P3) và model (P4).
- Retarget làm đường mặc định; IK chỉ là baseline/adapter, không nằm trên đường
  inference.
- Backend simulator ngoài MuJoCo/MJX.
- Train, AMP hoặc resume trên CUDA; P2-15 chỉ so parity của FK, không train.
- Phát hành asset Barrett hoặc bất cứ thứ gì dẫn xuất từ `dexsuite_dex_urdf`.
- Sửa hoặc rút cây legacy Ultralytics-derived.
