---
document_id: ROADMAP-P3-001
document_type: plan
title: Kế hoạch thực thi Phase 3 — Data layer
version: 1.0.0
status: active
date: 2026-08-22
revises: none
related_plan: PLAN-V2
---

# Kế hoạch thực thi Phase 3 — Data layer

Tài liệu này chỉ diễn giải cách thực thi P3 đã được `docs/roadmap/PROJECT_PHASES.md`
(P3), `PLAN.md` §4/§5 (M2, nửa data) và `docs/configuration/RUN_CONFIG.md` §7 quy
định. Nó không sửa, không mở rộng và không đánh giá lại quyết định của phase
trước. Người thực hiện là maintainer; kế hoạch viết để chạy được mà không cần
suy đoán.

## 1. Ranh giới phạm vi

`PLAN.md` gộp **M2 = data generator + robot layer**. `PHASE2_EXECUTION_PLAN.md`
§1 đã nhận nửa robot; P3 nhận đúng nửa còn lại theo phân chia của
`PROJECT_PHASES.md`.

Thuộc P3:

- Procedural object generator và asset manifest (CC0 bổ sung nếu có manifest).
- Candidate palm/contact sampling, IK/qpos theo named limits, collision filtering.
- Physics validation bằng MuJoCo: replay squeeze/lift, lưu success/quality.
- Render depth/point cloud với camera/frame metadata và deterministic seed.
- `GraspBatch` schema, shards, splits, seed, hashes và immutable manifest.
- `DGN-Open-Tiny` đủ nhỏ cho CI/overfit và tái tạo được từ đầu.

**Không** thuộc P3, để nguyên cho phase sau: object encoder, HandGraph
conditioning, flow backend và mọi thứ thuộc model (P4); training loop, quality
head, evaluator và benchmark protocol (P5); `DGN-Open-v1` quy mô lớn và
distributed generation (P6).

Artifact cuối phase: `DGN-Open-Tiny` có manifest và deterministic regeneration.
Gate chuyển phase: regenerate cùng seed cho cùng hash; dataset audit và tiny
loader pass.

### Ranh giới với giới hạn đã khai báo của P2

`SESSION-20260822-021` ghi fixture grasp/squeeze/lift "chưa nắm được vật ở cấu
hình mặc định" và hoãn việc tune sang M5. Đó **không** phải xung đột phạm vi với
P3, vì hai việc khác nhau:

- Cái bị hoãn: tune **default arguments** của `evaluate_grasp_fixture` để nó tự
  nắm được vật khi không ai truyền gì. P3 không đụng vào default đó.
- Cái thuộc P3: sinh ra **argument thật** — palm pose, joint targets, object
  geometry — rồi truyền vào fixture. `PLAN.md` §4 bước 2 và bước 3 giao đúng việc
  này cho pipeline `DGN-Open` (`sample wrist/contact candidates và giải IK/qpos
  theo named limits`, `replay squeeze/lift bằng MuJoCo/MJX, lưu success/quality`).

Đo thăm dò trực tiếp trên corpus đã pin cho thấy fixture chạy được khi có argument
thật. Với `joint_targets` dựng từ joint range của từng hand và object đặt trong
workspace của ngón, `palm_pos=(0, 0, 0.1)`, `seed=0`:

| Hand | Contact tay–vật lớn nhất (lưới A) | Số cấu hình `stable_lift=True` (lưới B) |
| --- | ---: | ---: |
| LEAP | 32 | 0 |
| Allegro | 12 | 24 |
| Shadow | 18 | 6 |

Hai cột đến từ **hai lưới khác nhau**, không phải một sweep:

- **Lưới A** — closing fraction `f ∈ {0.6, 0.8, 1.0}`, object size `0.02`, offset
  vị trí vật `dx, dy ∈ {-0.05, 0, 0.05}`, `dz ∈ {-0.04, -0.02, 0}`. Lưới này sắp
  hạng theo `contact_count` trước nên **không** dùng để kết luận gì về
  `stable_lift`: một cấu hình ít contact nhưng lift ổn định sẽ không nổi lên.
- **Lưới B** — `f ∈ {0.5, 0.7, 0.9}`, object size `∈ {0.015, 0.022}`,
  `dx, dy ∈ {-0.03, 0, 0.03}`, `dz ∈ {-0.02, 0}`; đếm mọi cấu hình có
  `stable_lift=True`.

Cả hai là script thăm dò chạy trong scratchpad, không commit. Người thực hiện P3
dựng lại chúng như một phần của P3-05/P3-08 với evidence ghi vào session report.

Con số `0 contact` của `SESSION-20260822-021` là kết quả của `joint_targets=None`
tại pose mặc định, nơi vật nằm ngoài tầm ngón — không phải bằng chứng fixture
hỏng. Kết luận: P3 làm việc thông thường của mình, không cần remediation item cho
`qdgrasp/sim/`.

Hai cảnh báo rút ra từ chính hai lưới đó, phải xử lý trong P3 chứ không bỏ qua:

- LEAP không có hit nào trong lưới B. Sampler phải đủ tốt cho **cả ba** hand,
  không được coi LEAP là ca ngoại lệ.
- Ba cấu hình Allegro khác nhau về vị trí vật cho cùng `contact_count=4` và cùng
  `lift_height=0.0434` tới bốn chữ số. Trùng khớp mức đó **có thể** là vật tựa
  lên link thay vì bị nắm, nhưng cũng có thể là ba grasp thật đều đạt gần đúng
  độ cao lift được lệnh. Chưa phân biệt được; đó chính là lý do tiêu chí `success`
  của P3 phải kiểm force closure/lực tiếp xúc chứ không chỉ đếm contact và độ cao
  (xem §6.2).

## 2. Điều kiện tiên quyết

1. `feature/phase2-robot-layer` đã merge vào `develop` tại `ece9703`. Remediation
   packaging/status `REV-20260822-012`/`013` phải được commit trên `develop` và
   worktree phải sạch trước khi chạy
   `scripts/git/start_feature.sh phase3-data-layer`.
2. `REV-20260822-010`, `011`, `012` và `013` vẫn `in_review`.
   `SESSION-20260822-021` §Bàn giao đặt independent review là điều kiện trước khi
   mở P3, nhưng P2 đã merge trước khi review xong. Ghi đây là **caveat** trong
   session report của P3, giống cách `PHASE2_EXECUTION_PLAN.md` §2 ghi
   `REV-20260822-010`; trạng thái `complete` của P2 là tạm thời cho tới khi
   reviewer không phải tác giả xác nhận. Không phải blocker.
3. Bốn checkout robot asset dưới `.references/robot-assets/` là read-only và đã
   qua `scripts/check_robot_assets.py`. Đặt
   `QDGRASP_ROBOT_ASSETS_ROOT=.references/robot-assets`; ba preset P2 dùng URI
   `asset://…`, không còn mang đường dẫn workspace trong profile hash.
4. `scripts/check_wheel.py` pass từ một install target ngoài source tree. Đây là
   điều kiện để P3 thêm preset dưới `qdgrasp/presets/data/` mà không lặp defect
   package-data của P2.

### Bước 0 — xác minh trước khi thiết kế

**Môi trường phát triển đã được khôi phục về lock CPU trước khi mở P3.** Trước
remediation, `scripts/check_environment.py` exit `1`:

```text
QDGrasp environment: FAIL (cpu)
- torch: cần 2.11.0, hiện tại 2.13.0
- trimesh: cần 4.12.2, hiện tại 5.0.0
- rich: cần 14.3.4, hiện tại 15.0.0
- CPU profile cần torch wheel +cpu, hiện tại 2.13.0+cu130
```

Gate của P3 là *bit-level reproducibility*. Sinh dataset trên môi trường lệch
lock rồi công bố hash của nó là tự tạo ra một lớp defect không thể debug về sau.
Vì vậy đây là **điều kiện tiên quyết**, không phải hạng mục công việc. Môi trường
đã được sync từ `environments/requirements/dev-cpu.lock`, cài lại package editable
không dependency, rồi xác minh:

```bash
.venv/bin/python scripts/check_environment.py --profile cpu
# QDGrasp environment: PASS (cpu)
```

Lệnh này phải tiếp tục exit `0` trước khi commit bất kỳ artifact dataset nào. Nếu
môi trường lại drift, hai đường xử lý hợp lệ là khôi phục venv về lock CPU tương
ứng (`cpu.lock` cho runtime, `dev-cpu.lock` cho development/test), hoặc regenerate
lock theo bàn giao P0 rồi chạy lại toàn bộ gate CPU/CUDA. Không có đường thứ ba.

Xác minh thứ hai — backend CSG và MJX, chạy trước khi thiết kế object generator:

```bash
.venv/bin/python -c "
import trimesh
try:
    trimesh.boolean.union([trimesh.creation.box(), trimesh.creation.icosphere()])
    print('csg available')
except Exception as exc:
    print('csg unavailable:', exc)
try:
    import mujoco.mjx; print('mjx available')
except ModuleNotFoundError as exc:
    print('mjx unavailable:', exc)
"
```

Kết quả đo được tại thời điểm viết: `No boolean backend: pip install manifold3d
or install blender` và `No module named 'mujoco.mjx'`. Cả hai đều đúng như thiết
kế của §3; nếu kết quả khác đi (backend bỗng có mặt), dừng lại và kiểm xem có
dependency nào đã bị cài ngoài lock.

## 3. Ràng buộc kế thừa (không thay đổi)

- Không sửa `PLAN.md`, `PROJECT_PHASES.md`, ADR, `qdgrasp/runtime.py`,
  environment lock, train-argument registry và cây legacy Ultralytics-derived.
- Không sửa nghĩa `qdgrasp/data/v1` và `qdgrasp/robot/v2` đã phát hành.
  `RUN_CONFIG.md` §7: version mới thêm bằng schema identifier mới cùng test
  round-trip riêng.
- **Không thêm dependency mới.** Dep đã khai báo: torch, lightning, numpy, scipy,
  mujoco, trimesh, safetensors, pydantic, PyYAML, einops, rich, typer. Hệ quả
  trực tiếp cho P3:

  | Thứ không có | Hệ quả bắt buộc |
  | --- | --- |
  | `manifold3d`/blender | Không dùng `trimesh.boolean`. Compound object dựng bằng union nhiều convex geom trong MJCF, không bằng mesh boolean (§6.1) |
  | `mujoco.mjx` (cần `jax`) | Label pipeline chạy MuJoCo CPU; song song bằng `multiprocessing`. `PLAN.md` §4 cho phép "MuJoCo/MJX"; nhánh MuJoCo thỏa mãn |
  | Thư viện IK | IK viết bằng damped least squares trên `RobotSpec.forward_kinematics` của P2 (torch), có projection về joint limit |
  | Open3D/PyTorch3D | Point cloud sinh bằng sampler tự viết trên numpy, không mượn RNG của trimesh |

- Mọi đường dẫn là relative path; `RunConfig.project_dir` đã từ chối absolute path.
- RH56E2 bị loại hoàn toàn; `scripts/check_phase0.py` quét mọi tracked file.
- Version package giữ `0.1.0a1` cho tới khi có release branch.

### Ánh xạ thư mục

`qdgrasp/data/` là legacy Ultralytics **đang có import sống** (`from .base import
BaseDataset`, …), khác `qdgrasp/robot/` và `qdgrasp/sim/` vốn trống lúc P2 bắt
đầu. Nên P3 quay lại cách của P1: đặt code mới ở tên chưa bị chiếm.

| PLAN §3/§4 | Đường dẫn P3 | Lý do |
| --- | --- | --- |
| `data/` | `qdgrasp/dataset/` | `qdgrasp/data/` là legacy có import sống |
| object generator | `qdgrasp/objects/` | tên trống |
| candidate/IK/label pipeline | `qdgrasp/dataset/pipeline/` | thuộc data layer |
| MuJoCo label replay | `qdgrasp/sim/labeling.py` | `qdgrasp/sim/` đã là của P2, thêm file mới |
| dataset preset YAML | `qdgrasp/presets/data/` | `qdgrasp/cfg/` là legacy; xem cảnh báo package-data bên dưới |
| artifact dataset sinh ra | `datasets/dgn-open-tiny/` (repo root) | không nằm trong package/reference; phải mở allowlist hẹp trong `.gitignore` (P3-22) |

Ánh xạ này là ghi chú triển khai, không phải đề nghị sửa `PLAN.md`.

**Cảnh báo package-data — đã xử lý trước khi mở P3.** `pyproject.toml` cũ khai
báo `presets/*.yaml` và `assets/*`; glob `*` của setuptools **không vượt qua `/`**,
nên thư mục con không được đóng gói. Wheel dựng từ `develop` tại `ece9703` đã
chứng minh điều đó:

```text
qdgrasp/presets/dummy-hand.yaml
qdgrasp/presets/robots/__init__.py      # chỉ file .py lọt vào, vì nó là module
```

`qdgrasp/presets/robots/*.yaml` (ba profile robot của P2) và
`qdgrasp/assets/derived/*` (URDF chuẩn hóa cùng manifest của P2-05) **không có
trong wheel**. Đây là defect của P2 đối với gate P0 "clean wheel import/CLI";
đã được sửa trước khi mở P3 bằng `REV-20260822-012` và các glob đệ quy hiện tại.
P3-15/P3-20 vì vậy có thể đặt file dưới `qdgrasp/presets/data/` mà không còn
thừa hưởng lỗi này; P3-21 được coi là hoàn tất sớm, không còn là backlog P3.

Ba profile chỉ mang cấu hình và URI logic; raw MJCF/mesh vẫn nằm ngoài wheel theo
`THIRD_PARTY.yml`. `scripts/check_wheel.py` dựng `RobotSpec` bằng checkout đã pin
qua `QDGRASP_ROBOT_ASSETS_ROOT`. Artifact `assets/derived/*.urdf` có mặt để giữ
normalization/provenance nhưng không phải robot bundle tự chứa; mesh path của nó
vẫn cần external asset checkout như `ROBOT_PROFILE.md` đã khai báo.

**Cảnh báo output bị ignore.** `.gitignore` hiện có `/datasets/`, nên output P3-20
sẽ bị Git bỏ qua âm thầm. P3-22 phải mở đúng `datasets/dgn-open-tiny/` và tiếp tục
ignore mọi dataset runtime khác; gate dùng `git check-ignore` cùng
`git ls-files` để chứng minh manifest/shard phát hành được track. Không dùng
`git add -f` như một ngoại lệ khó tái lập.

## 4. Nguồn asset và license

| Nguồn | Vai trò trong P3 | License artifact sinh ra |
| --- | --- | --- |
| Procedural primitives/superquadrics do dự án sinh | Nguồn object chính của `DGN-Open-Tiny` | `CC0-1.0`, dự án sở hữu |
| MJCF Menagerie ba hand (P2) | Hand model cho label replay | Giữ license gốc; chỉ tham chiếu, không phân phối lại trong dataset |
| Profile robot v2 của P2 | `robot_profile_hash` trong manifest | Cờ release-blocked phải lan xuống shard (§6.3) |
| CC0 asset ngoài (nếu dùng) | Bổ sung đa dạng | Mỗi mesh cần manifest license riêng |

GSO/YCB `CC-BY-4.0` và Objaverse **không** vào `DGN-Open-Tiny`. `PLAN.md` §4 xếp
chúng thành dataset/weight track riêng có attribution; mở track đó không thuộc P3.

Dataset do dự án phát hành mang `CC0-1.0`, tách khỏi AGPL của code
(`PLAN.md` §4 mục 6). Manifest phải ghi license của dataset chứ không thừa kế
license của repository.

## 5. Hạng mục công việc

| ID | Hạng mục | Output | Ghi chú |
| --- | --- | --- | --- |
| P3-01 | Deterministic RNG contract: `numpy.random.Generator(PCG64(seed))` phân nhánh theo (split, object_id, candidate_id) | `qdgrasp/dataset/rng.py` | không RNG nào chảy qua trimesh/torch default, xem §6.1 |
| P3-02 | Procedural object generator: primitives + superquadrics + compound convex | `qdgrasp/objects/generate.py` | không CSG |
| P3-03 | Object asset manifest: mesh hash, tham số sinh, license, physical properties | `qdgrasp/objects/manifest.py` | |
| P3-04 | Collision representation guard: geom va chạm phải khớp mesh hiển thị trong tolerance | `qdgrasp/objects/collision.py` | xem §6.1, đây là defect class của `SESSION-20260822-021` |
| P3-05 | Candidate palm/contact sampler theo `RobotSpec` và object surface | `qdgrasp/dataset/pipeline/sample.py` | phải phủ cả ba hand, gồm LEAP |
| P3-06 | IK damped least squares trên FK của P2, projection về named limits | `qdgrasp/dataset/pipeline/ik.py` | công cụ sinh dữ liệu, không nằm trên đường inference |
| P3-07 | Collision filter tay–vật và self-collision trước khi tốn bước simulate | `qdgrasp/dataset/pipeline/filter.py` | |
| P3-08 | Physics label: replay squeeze/lift, tiêu chí success có force closure | `qdgrasp/sim/labeling.py` | dùng `evaluate_grasp_fixture` với argument thật, không sửa default |
| P3-09 | Camera model và point cloud sampler analytic, có frame metadata | `qdgrasp/dataset/render.py` | rasterizer chỉ là cross-check tùy chọn, xem §6.1 |
| P3-10 | Schema `qdgrasp/data/v2` cạnh v1, cùng migration test | `qdgrasp/dataset/schema.py` | v1 giữ nguyên nghĩa |
| P3-11 | Schema `qdgrasp/dataset-manifest/v1`: seed, hash nguồn/đầu ra, split, license, cờ release-blocked | `qdgrasp/dataset/manifest.py` | document schema riêng, không nhét vào run config |
| P3-12 | `GraspBatch` và shard writer/reader immutable | `qdgrasp/dataset/batch.py`, `shards.py` | |
| P3-13 | Split theo object/shape family, không leak giữa train/val | `qdgrasp/dataset/split.py` | |
| P3-14 | Loader đăng ký qua `register_dataset`, nối vào runner của P1 | `qdgrasp/dataset/loader.py` | dummy dataset của P1 vẫn phải chạy |
| P3-15 | CLI `qdgrasp data generate/audit` | `qdgrasp/cli.py` | thêm subcommand, không đổi command cũ |
| P3-16 | Checker manifest dataset, stdlib-only | `scripts/check_dataset_manifest.py` | xem §7 |
| P3-17 | Gate script CPU của P3 | `scripts/check_phase3.py` | |
| P3-18 | Test suite | `tests/test_objects_*.py`, `tests/test_dataset_*.py`, `tests/test_sim_labeling.py` | |
| P3-19 | Contract doc và session report | `docs/configuration/DATASET_SCHEMA.md`, `docs/sessions/` | `document_type: registry` |
| P3-20 | `DGN-Open-Tiny` phát hành cùng manifest | `datasets/dgn-open-tiny/` | |
| P3-21 | Sửa glob package-data để thư mục con vào được wheel, cùng test kiểm nội dung wheel | `pyproject.toml`, `scripts/check_wheel.py` | Hoàn tất trước khi mở P3; `REV-20260822-012` |
| P3-22 | Mở allowlist Git hẹp cho artifact tiny và gate chống ignored output | `.gitignore`, `scripts/check_dataset_manifest.py` | làm trước P3-20; không unignore dataset khác |

Thứ tự đề xuất: P3-01 → P3-02/03/04 → P3-05/06/07 → P3-08 → P3-09 →
P3-10/11/12/13 → P3-14/15 → P3-16/17/18/19/22 → P3-20. P3-05 và P3-09 chạy song
song được sau khi P3-04 xong. P3-21 đã hoàn tất trước khi mở phase
(`REV-20260822-012`), nên không nằm trong thứ tự trên; hệ quả của nó là preset
dataset của P3-15/P3-20 được `scripts/check_wheel.py` bảo vệ ngay từ đầu.

## 6. Ba hạng mục dễ bị đánh giá thấp

### 6.1 Determinism là toàn bộ gate của P3

Gate chuyển phase là "regenerate cùng seed/hash". Mọi byte phát hành phải đến từ
nguồn mà dự án kiểm soát. Bốn nguồn phá hash đã xác định:

1. **trimesh drift.** Lock pin `trimesh==4.12.2`; trước remediation dev từng ở
   5.0.0, và
   `SESSION-20260822-021` đã ghi `trimesh.sample_surface` không tái lập giữa hai
   version. Quy tắc: không RNG nào đi qua trimesh, không dùng hàm sampling của
   trimesh trong đường sinh dữ liệu. Mọi ngẫu nhiên đi qua P3-01.
2. **torch drift.** Lock pin `torch==2.11.0+cpu`; trước remediation dev từng ở
   `2.13.0+cu130`. Đã xử lý ở Bước 0, không phải hạng mục công việc.
3. **Rasterized depth.** `mujoco.Renderer` khởi tạo được trên máy này, nhưng qua
   software rasterizer. Depth do driver sinh ra không ổn định giữa các máy. Nguồn
   chân lý được hash phải là **analytic sampling/raycast với camera model ghi rõ
   trong manifest**; renderer chỉ là cross-check tùy chọn, không bao giờ là dữ
   liệu phát hành.
4. **Convex hull mismatch.** Không có backend CSG, nên compound object dựng bằng
   nhiều convex geom. Bẫy: MuJoCo va chạm mesh geom lõm bằng **convex hull** của
   nó, nên vật lý nhìn thấy một hình khác với mesh mà point cloud lấy mẫu — label
   sai một cách im lặng. Đây đúng là lớp defect đã làm hỏng `SESSION-20260822-020`
   (báo cáo kết quả cho một scene không có vật). Bắt buộc: object lồi theo cấu
   trúc, hoặc phần lõm dựng bằng union các convex geom; P3-04 assert biểu diễn va
   chạm khớp mesh hiển thị trong tolerance và gate cưỡng chế.

### 6.2 Tiêu chí `success` phải kiểm được, không chỉ đếm

`FixtureResult` hiện có `success`, `stable_lift`, `contact_count`,
`max_penetration`, `lift_height`. Lưới B ở §1 cho ba cấu hình Allegro khác nhau
về vị trí vật nhưng cùng `contact_count=4` và cùng `lift_height=0.0434`. Các
trường hiện có **không đủ** để phân biệt vật bị nắm với vật tựa lên link, nên
nhận thẳng `stable_lift` làm label là chấp nhận rủi ro nhãn dương giả mà không có
cách phát hiện.

P3-08 phải bổ sung, trên tầng của mình chứ không sửa fixture của P2:

- Contact phải là cặp tay–vật với lực pháp tuyến khác 0, phân bố trên ít nhất hai
  link không cùng ngón.
- Kiểm nhiễu loạn: sau khi lift, tác động một shaking/wrench nhỏ đã khai báo; vật
  vẫn bám mới tính dương.
- Penetration vượt ngưỡng khai báo là loại, không phải cảnh báo.
- Ngưỡng ghi vào manifest; đổi ngưỡng đổi dataset hash.

`PLAN.md` §4 bước 3 và ưu tiên "correctness → reproducibility" của
`PROJECT_PHASES.md` không cho nới tiêu chí để số liệu đẹp lên. Cấm hạ ngưỡng để
tăng tỉ lệ dương.

### 6.3 Tiêu chí nghiệm thu mà roadmap chưa nêu: tỉ lệ nhãn dương

Gate chữ nghĩa của P3 (regenerate cùng hash, audit, loader) **pass được trên một
dataset toàn nhãn âm**. Nhưng khi đó `DGN-Open-Tiny` vô dụng cho gate overfit của
P4, và defect chỉ lộ ra giữa P4. Vì vậy P3 tự thêm một tiêu chí kiểm được bằng
lệnh:

- Tỉ lệ nhãn dương **khác 0 trên cả ba hand** trong `DGN-Open-Tiny`, in ra bởi
  `scripts/check_phase3.py` và ghi vào manifest.
- LEAP là ca khó đã biết (0 hit trong lưới thô ở §1). Không được đóng phase bằng
  cách bỏ LEAP khỏi tiny dataset.

Đây là bổ sung của kế hoạch thực thi, không phải sửa định nghĩa gate của
`PROJECT_PHASES.md`.

### 6.4 Nợ kỹ thuật thừa kế từ P2 mà P3 vấp phải

`SESSION-20260822-021` ghi **surface anchor để rỗng**: `RobotSpec.from_config`
có `sample_anchors=True` nhưng không sinh anchor, vì anchor thật cần pose offset
mức geom và `trimesh.sample_surface` không tái lập giữa version. Contact sampler
của P3-05 cần điểm trên bề mặt tay, nên phải chọn một trong hai và ghi vào session
report:

- **A.** Viết deterministic geom-level anchor sampler trong P3-01/P3-05 (không
  dùng trimesh RNG), rồi nạp ngược vào `RobotSpec`. Đắt hơn, nhưng gỡ nợ.
- **B.** Giới hạn contact candidate ở `RobotSpec.fingertip_positions` trong P3.
  Rẻ, nhưng thu hẹp không gian contact và phải ghi là giới hạn đã khai báo.

Khuyến nghị **A**, vì P4 dùng contact head trên link chứ không chỉ fingertip; trả
nợ ở P3 rẻ hơn trả ở P4. Nếu chọn B, phải ghi điều kiện gỡ chặn rõ ràng.

## 7. Lỗ hổng governance cần vá trong P3 (P3-16)

P2 đã vá phần asset robot bằng `scripts/check_robot_assets.py`. P3 phát hành một
loại artifact mới — dataset shard — mà chưa có checker nào chạm tới.

`scripts/check_dataset_manifest.py` phải kiểm, theo mẫu của
`check_robot_assets.py`, chỉ stdlib:

- Mỗi shard có trong manifest và mọi mục manifest có shard thật.
- Manifest và shard của `datasets/dgn-open-tiny/` không bị `.gitignore` loại và
  đều xuất hiện trong `git ls-files`; dataset runtime khác vẫn bị ignore.
- `sha256` của từng shard khớp.
- `seed`, `generator_version`, `environment_fingerprint` và `robot_profile_hash`
  có mặt và khớp profile đang cài.
- Split không leak: không object family nào xuất hiện ở hai split.
- License field có mặt; shard tham chiếu profile **release-blocked** thì thừa kế
  cờ đó và gate từ chối phát hành.
- Ngưỡng success của P3-08 được ghi trong manifest.

Sau đó thêm nó vào danh sách gate ở §9 cạnh `scripts/check_phase3.py`.

## 8. Tiêu chí nghiệm thu

Mỗi mục phải kiểm được bằng lệnh, không phải bằng nhận định.

**Object generation**

- Sinh hai lần cùng seed cho cùng mesh hash trên cùng environment lock.
- Mọi object có manifest gồm tham số sinh, mesh hash, license và physical
  properties.
- Biểu diễn va chạm khớp mesh hiển thị trong tolerance đã khai báo (P3-04); vi
  phạm là fail, không phải cảnh báo.
- Không object nào phụ thuộc `trimesh.boolean` hay backend CSG.

**Candidate, IK và filter**

- Mọi qpos sinh ra nằm trong named joint limit của profile v2; vi phạm bị loại
  trước khi simulate.
- IK hội tụ có tiêu chí dừng ghi rõ; ca không hội tụ bị đánh dấu, không bị âm
  thầm bỏ.
- Collision filter loại đúng ca tay–vật xuyên nhau và self-collision, có test
  phủ định chứng minh nó thật sự từ chối.

**Physics label**

- Label replay dùng `evaluate_grasp_fixture` với argument sinh ra, không sửa
  default của P2.
- Tiêu chí success gồm force closure và perturbation test (§6.2); ngưỡng nằm
  trong manifest.
- Chạy lại cùng seed cho cùng label trên mọi sample.
- Tỉ lệ nhãn dương khác 0 trên cả LEAP, Allegro và Shadow (§6.3).

**Point cloud và camera**

- Point cloud sinh bằng đường analytic; camera intrinsics/extrinsics và frame nằm
  trong metadata của từng sample.
- Sinh hai lần cùng seed cho cùng hash, không phụ thuộc driver đồ họa.

**Schema và shard**

- `qdgrasp/data/v2` round-trip; `qdgrasp/data/v1` vẫn parse với nghĩa cũ; hash v1
  khác hash v2.
- Shard immutable: ghi xong không sửa; sửa nghĩa là bump version dataset.
- Split theo object/shape family, không leak; kiểm bằng lệnh.
- Tiny loader nạp được qua `register_dataset` và chạy hết một epoch trên runner
  của P1 mà dummy lifecycle không regression.

**Reproducibility**

- `qdgrasp data generate` hai lần cùng seed cho cùng manifest hash toàn bộ.
- Manifest có seed, environment fingerprint, robot profile hash, generator
  version, hash từng shard và license.
- `scripts/check_environment.py` exit `0` tại lần chạy sinh ra dataset phát hành.

**Provenance và release**

- Dataset phát hành mang `CC0-1.0` ghi trong manifest, tách khỏi AGPL của code.
- Shard dựng từ profile release-blocked thừa kế cờ và bị gate từ chối phát hành.
- Không asset GSO/YCB/Objaverse nào lọt vào `DGN-Open-Tiny`.
- Manifest và shard phát hành của `DGN-Open-Tiny` được Git track; gate fail nếu
  output chỉ tồn tại trên đĩa hoặc được thêm bằng ngoại lệ `git add -f` không có
  allowlist trong `.gitignore`.

## 9. Gate phải chạy trước khi đóng phiên

```bash
python3 scripts/check_environment.py
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
python3 scripts/check_dataset_manifest.py
python3 scripts/check_phase3.py
python3 -m pytest tests/ -q
```

P0, P1 và P2 phải tiếp tục pass nguyên vẹn; bất kỳ regression nào ở
`check_phase1.py` hoặc `check_phase2.py` là lỗi của P3, không phải lý do sửa gate
cũ.

## 10. CUDA trong P3

Gate chuyển phase của P3 theo `PROJECT_PHASES.md` là regenerate/audit/loader,
toàn bộ chạy CPU; sinh dữ liệu là MuJoCo CPU và numpy nên **không có hạng mục
CUDA bắt buộc** trong P3.

Một hạng mục bổ sung nên làm, ngoài gate, theo mẫu P2-15: thêm **một cell** vào
`kaggle-phase1/qdgrasp_phase1_cuda.ipynb` nạp `DGN-Open-Tiny` và chạy một
train-step thật trên GPU với dataset thật thay vì dummy. Lợi ích giống P2-15: mỗi
lần chạy đồng thời chạy lại gate CUDA của P1 và FK parity của P2, nên P3 không
thể âm thầm làm hỏng hai phase trước trên GPU. Repin `QDGRASP_COMMIT` sang commit
P3 và cập nhật hằng SHA-256 của các script gate. Nếu chưa public code P3, ghi là
`Bị chặn` với điều kiện gỡ chặn rõ ràng, không bỏ im lặng.

Đây không phải điều kiện chuyển phase: P3 vẫn đóng được khi hạng mục này fail,
miễn là kết quả fail được ghi lại và xử lý.

## 11. Ngoài phạm vi P3

- Toàn bộ model (P4) và training/evaluator (P5).
- `DGN-Open-v1` quy mô lớn, distributed generation và tối ưu throughput (P6).
- MJX và mọi backend simulator ngoài MuJoCo CPU.
- Track dataset GSO/YCB/Objaverse và attribution đi kèm.
- Benchmark protocol, metrics report và ablation (P5).
- Retarget làm đường mặc định; IK của P3-06 chỉ là công cụ sinh dữ liệu.
- Sửa default của `evaluate_grasp_fixture`, sửa hoặc rút cây legacy
  Ultralytics-derived.
