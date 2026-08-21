# DexGraspNet Custom — hợp nhất với Ultralytics

Lộ trình dài hạn. Ước lượng **6–14 tháng**, không phải vài tuần. Tài liệu này chia
nhỏ để mỗi mốc tự nó đã có giá trị, kể cả khi dừng giữa chừng.

---

## 1. Vì sao

`PKU-EPIC/DexGraspNet2` là code nghiên cứu. Đo được trên `upstream/main`
(`26ecd761`), 49 file `.py` / 7180 dòng:

- **Không một `__init__.py` nào.** `src/` là namespace package ngầm, chỉ import
  được nhờ mọi script gọi `os.chdir(<repo>)` rồi `sys.path.append('.')`.
- **18 file `os.chdir()` ở module scope**, trong đó 4 file thư viện
  (`utils/dataset.py:4`, `vis_plotly.py:4`, `width_mapper.py:4`, `render.py:4`).
  Import là đổi cwd cả tiến trình.
- **Chỉ chạy được trên GPU NVIDIA.** MinkowskiEngine là CUDA-only và đã ngừng
  phát triển (0.5.4, 2021). IsaacGym cũng GPU-only và đã deprecated.
- **Không có `pyproject.toml`, `setup.py`, `requirements.txt`.**
- Cấu hình nửa vời: `load_config` dùng `eval(f'args.{key}')`, và
  `DotDict.__getattr__` trả `None` cho khóa thiếu nên gõ sai khóa **không báo lỗi**.

Ultralytics giải xong đúng những bài này: package cài được, YAML điều khiển mọi
thứ, `Model().train()/.val()/.predict()`, chạy CPU/CUDA/MPS, callback, CLI. Nhiều
module của nó dùng lại được nguyên vẹn.

**Mục tiêu:** một package `dexgrasp` mang kiến trúc Ultralytics, chạy được ngoài
NVIDIA, cấu hình thuần YAML, và **theo kịp** upstream Ultralytics vốn cập nhật rất
nhanh.

---

## 2. Ba quyết định nền

### 2.1 Không vendor Ultralytics — phụ thuộc nó

Cám dỗ là clone cả Ultralytics vào rồi trộn. **Đừng.**

Ultralytics ra bản mới gần như hàng tuần. Vendor một bản chép là tự nhận nợ merge
vĩnh viễn: mỗi lần muốn lấy sửa lỗi hay module mới lại phải hòa tay ba giữa bản
chép đã sửa, bản chép gốc, và bản mới.

Thay vào đó:

```
dexgrasp/  ──phụ thuộc──►  ultralytics (pip, ghim phiên bản)
```

- **Kế thừa** `BaseTrainer`, `BaseValidator`, `BasePredictor`, `Model` từ
  `ultralytics.engine.*` khi nghĩa của chúng khớp.
- **Dùng lại** `ultralytics.utils` nguyên vẹn: `LOGGER`, `IterableSimpleNamespace`,
  `yaml_load/yaml_save`, `colorstr`, `callbacks`, `checks`, `files.increment_path`,
  `torch_utils.select_device/ModelEMA/init_seeds/strip_optimizer`, `TQDM`.
- **Chỉ chép** khi thật sự phải rẽ nhánh — chép vào `dexgrasp/_vendor/` kèm ghi
  chú nêu commit gốc và lý do rẽ.

Ghim `ultralytics==<X.Y.Z>` trong `pyproject.toml`. Nâng phiên bản là một PR
riêng, có test.

> **Rủi ro cần chấp nhận:** Ultralytics là AGPL-3.0. Phụ thuộc nó khiến `dexgrasp`
> cũng phải AGPL nếu phát hành. Nếu dự án cần giấy phép khác thì **quyết định này
> phải đảo ngay từ đầu** — lúc đó chỉ mô phỏng kiến trúc, không import. Chốt việc
> này trước khi viết dòng code nào.

### 2.2 YAML là nguồn sự thật duy nhất

Theo đúng `ultralytics/cfg/`:

```
dexgrasp/cfg/
├── default.yaml              # mọi khóa + giá trị mặc định, phẳng
├── models/
│   ├── graspness_diffusion.yaml
│   ├── graspness_isa.yaml
│   └── graspness_cvae.yaml
├── robots/
│   ├── leap_hand.yaml        # urdf, meta, keypoint_links, fingertip, joint limits
│   └── rh56e2.yaml
├── datasets/
│   ├── graspnet.yaml
│   └── acronym.yaml
└── sim/
    └── isaacgym.yaml
```

`get_cfg(cfg, overrides)` gộp và **validate**; khóa lạ là lỗi, không im lặng.
CLI: `dexgrasp train model=graspness_diffusion.yaml data=graspnet.yaml robot=rh56e2.yaml`.

Cái này thay `configs/network/*.yaml` hiện tại (4 file × 153 dòng, chép đè lẫn
nhau) và toàn bộ `arg_mapping` với `eval()`.

### 2.3 Kiến trúc mới — bỏ hẳn sparse conv và mọi op CUDA tùy biến

Không dùng MinkowskiEngine. Cũng không dùng Point Transformer với KNN/FPS CUDA.

```
                  RGB-D / Point Cloud
                          │
                          ▼
              CPU preprocessing
          voxel sample + spatial sort
                          │
                    4096 points
                          │
                          ▼
               ┌──────────────────┐
               │ Point Patch      │
               │ Encoder          │
               │ Linear/MLP       │
               │ + Transformer    │
               └────────┬─────────┘
                        │
                contextual point features
                        │
                        ▼
             object + graspness
                        │
                        ▼
                  Top-K seeds
                        │
                        ▼
             local geometry tokens
                        │
                        ▼
             ┌───────────────────┐
       t ───►│    Grasp DiT      │
   noise ───►│ self-attention    │
geometry ───►│ cross-attention   │
             └─────────┬─────────┘
                       │
                       ▼
            wrist + 12 joint DoF
```

**Toàn bộ đường đi chỉ gồm `Linear`, `LayerNorm`, `softmax`, `gather`.** Không
kernel tùy biến, không sparse hash, không KNN, không ball query. Chạy CPU / CUDA /
MPS / bất kỳ backend nào PyTorch hỗ trợ. Đây là cách đạt "chạy mọi môi trường"
triệt để hơn cả PTv3 — bản chính thức của PTv3 vẫn kéo theo `spconv` cho xCPE và
`flash-attn`.

#### Vì sao spatial sort là mấu chốt

Sắp điểm theo đường cong lấp không gian (Z-order hoặc Hilbert) trên CPU, **một
lần, lúc tiền xử lý**. Sau khi sắp:

- **Điểm gần nhau trong không gian thì gần nhau trong chuỗi.**
- Chia patch = `reshape(4096, C) → (256, 16, C)`. Không KNN.
- Lân cận của một seed = **lát cắt liên tiếp** của chuỗi. Không ball query.

Toàn bộ chi phí tìm lân cận biến mất, đổi lấy một lần sort `O(N log N)` trên CPU.
Đó chính là ý tưởng của PTv3, nhưng lấy phần rẻ và bỏ phần phải cài kernel.

#### 40000 → 4096 điểm

Giảm 10 lần. Nhờ vậy attention dày đặc mới khả thi: 256 token patch → 65k cặp,
rẻ. Nếu để 40000 điểm thì 1.6 tỷ cặp, không chạy nổi.

**Rủi ro phải đo:** graspness là điểm số **theo từng điểm**. Ít điểm hơn nghĩa là
độ phân giải vị trí grasp thô hơn. Với `voxel_size: 0.005` hiện tại, 4096 điểm
trên một scene bàn cho khoảng cách điểm thưa hơn đáng kể.

Sơ đồ đã tự xử lý: **"local geometry tokens" sau Top-K seeds** — ngữ cảnh thô ở
4096 điểm, hình học tinh tại chính các seed. Nhưng phải **đo**, không được giả
định. Thử 4096 / 8192 / 16384 và so sim success rate.

#### Bốn khối, đặc tả cụ thể

**1. CPU preprocessing** → `dexgrasp/data/preprocess.py`

```
voxel_downsample(pc, voxel_size) -> (M, 3)      torch.unique trên toạ độ voxel
spatial_sort(pc, order='hilbert'|'zorder')       -> chỉ số đã sắp
pad_or_sample(pc, n=4096)                        -> (4096, 3) cố định
```

Đầu ra **kích thước cố định**. Hệ quả lớn: `minkowski_collate_fn` biến mất,
`coors`/`feats`/`quantize2original`/`original2quantize` biến mất, collate mặc
định `torch.stack` là đủ. Sáu điểm gọi ME trong `dataset.py` biến mất theo.

Phải đo: tiền xử lý chạy trong `num_workers` của DataLoader, đừng để nó thành nút
cổ chai. `O(N log N)` sort trên 40k điểm là nhanh, nhưng **đo thật**.

**2. Point Patch Encoder** → `dexgrasp/nn/encoder.py`

```
(B, 4096, 3)
  → chia patch      (B, 256, 16, 3)        reshape, nhờ đã sort
  → MLP trên patch  (B, 256, C)            toạ độ tương đối trong patch
  → + pos embed     tâm patch
  → Transformer     L tầng, self-attn trên 256 token
  → feature theo điểm (B, 4096, C)          token patch + MLP điểm cục bộ
```

Dòng họ Point-BERT / Point-MAE, đã được chứng minh. `feature_dim` giữ 512 để các
head phía sau không phải đổi.

**3. object + graspness** → giữ nguyên `nn.Linear(C, 3)` như hiện tại: 2 logit
objectness + 1 graspness.

**Top-K seeds**: `torch.topk(graspness, k)` trên các điểm objectness dương.
Hiện tại đang dùng `sample_farthest_points` của pytorch3d (op CUDA tùy biến) —
`topk` bỏ được nó, và đúng tinh thần sơ đồ.

**local geometry tokens**: với mỗi seed, lấy lát cắt liên tiếp quanh nó trong
chuỗi đã sort + token patch chứa nó. Ra `(B, k, T, C)`.

**4. Grasp DiT** → `dexgrasp/nn/dit.py`

Thay `MLPWrapper` + `GaussianDiffusion1D` hiện tại.

Token hoá grasp thay vì một vector phẳng:
```
token 0     wrist translation (3)
token 1     wrist rotation (6D)
token 2..6  5 ngón, mỗi ngón 1 token   (thumb 4 khớp, 4 ngón còn lại 2 khớp)
```
DiT block: self-attn trên các token grasp → cross-attn sang local geometry
tokens → adaLN-Zero điều biến theo timestep `t`.

**Vì sao token hoá theo ngón là thắng lợi thật, không phải trang trí:** đã đo
trên đầu ra model hiện tại — `ring1/ring2/pinky1/pinky2` tương quan **0.926–0.994**,
PC1 chiếm **81.4%** phương sai, participation rank **1.48/12**. Tay thật vận động
theo khớp đồng vận. Kiến trúc hiện tại là một MLP phẳng, phải học cấu trúc đó một
cách ngầm. Self-attention giữa các token ngón **mô hình hoá nó tường minh**.

Giữ nguyên `bounded_logit`: decode qua sigmoid vào giới hạn URDF. Trực giao với
kiến trúc, không đụng.

#### Cái giá — nói thẳng

**Mất toàn bộ checkpoint.** `leap_ours_ckpt_50000.pth` (67.19% success) là nền
của cả chiến lược RH56E2: v1 warm-start từ nó, LoRA 12-DoF warm-start từ nó,
`freeze_backbone` đóng băng chính nó. Kiến trúc mới không có gì tương thích.

Phải **huấn luyện lại mô hình nền trên toàn bộ GraspNet** — đúng cái run 50k
iteration của tác giả. Đó là phần tốn nhất, và không có đường tắt.

Đổi lại: bỏ được `MinkowskiEngine`, `spconv`, `flash-attn`, và cả
`pytorch3d.sample_farthest_points`/`ball_query`. Phụ thuộc còn lại của đường
train/infer chỉ là **PyTorch thuần**.

#### Mốc so sánh bắt buộc

Không có tương thích checkpoint để đối chiếu, nên **giao thức đo là thứ duy nhất
giữ được sự trung thực**:

- So với `leap_ours_ckpt_50000` (MinkUNet, 67.19% trên scene_0004) trên **giao
  thức 30 scene** (`grasp_eval/kaggle/*_ms_sim_30`, 7680 grasp, ≥2 seed).
- Không so bằng loss. Loss diffusion không đo được chất lượng grasp — v1 iter20000
  có loss ổn hơn best-14000 nhưng sinh 266 joint ngoài giới hạn.
- IsaacGym không tất định: cùng một file re-sim ra 15/17/21 (±1.2pp). Một scene
  256 grasp **không** phân giải nổi 2pp.

"Kiến trúc mới bằng hoặc hơn MinkUNet trên 30 scene" là tiêu chí nghiệm thu. Chưa
đạt thì chưa xong, dù code có đẹp tới đâu.

## 3. Cấu trúc đích

```
dexgrasp/
├── __init__.py                  # __version__, GraspModel
├── cfg/                         # xem 2.2
├── data/
│   ├── base.py                  # BaseDataset
│   ├── dataset.py               # GraspSceneDataset
│   ├── build.py                 # build_dataloader, InfiniteDataLoader, collate
│   ├── split.py                 # split manifest
│   └── preprocess/              # từ src/preprocess/, gỡ os.chdir
├── engine/
│   ├── trainer.py               # GraspTrainer(ultralytics BaseTrainer)
│   ├── validator.py             # GraspValidator(ultralytics BaseValidator)
│   ├── predictor.py             # GraspPredictor(ultralytics BasePredictor)
│   ├── model.py                 # GraspModel(ultralytics Model)
│   └── results.py               # GraspResults
├── nn/
│   ├── tasks.py                 # GraspnessSample, DiffusionSample
│   ├── encoder.py               # Point Patch Encoder
│   ├── dit.py                   # Grasp DiT
│   ├── heads.py                 # objectness + graspness
│   └── modules/                 # head, diffusion, mlp, condition
├── sim/
│   ├── base.py                  # Simulator ABC
│   ├── mujoco.py                # ĐƯỜNG CHÍNH — tất định, chạy CPU
│   ├── mjcf.py                  # URDF -> MJCF, giữ capsule nguyên bản
│   ├── rollout.py               # batch đa luồng qua mujoco.rollout
│   └── isaacgym.py              # giữ để hiệu chuẩn, rồi thành tuỳ chọn
├── robot/
│   ├── model.py                 # RobotModel
│   ├── urdf.py                  # setup_custom_urdf  ← đã có, đã test
│   └── keypoints.py
├── utils/                       # CHỈ những gì ultralytics.utils không có
│   ├── metrics.py               # GraspMetrics
│   └── geometry.py              # rot6d, pc, edge
└── pyproject.toml
```

Cái gì `ultralytics.utils` đã có thì **import, đừng viết lại**.

---

## 4. Lộ trình

Mỗi mốc là một release dùng được. Ước lượng cho một người làm chính.

### M1 — Đóng gói được (4–6 tuần)

Mục tiêu: `pip install -e .` rồi `import dexgrasp` từ bất kỳ đâu, cwd không đổi.

1. Clone upstream `26ecd761`, áp diff RH56E2 846 dòng
   (`colab_issagym/notebook_train/patches/upstream_to_rh56e2.diff` — đã kiểm áp sạch).
2. **Cắt chuỗi import độc trước tiên.** `network/graspness_sample.py:13` import
   `Loader` từ `dataset.py`, `:15` import `Vis` từ `vis_plotly.py`; cả hai file đó
   `os.chdir` lúc import. Cả `Loader` lẫn `Vis` đều không cần cho forward/loss.
3. Bỏ `os.chdir` + `sys.path.append` khỏi 4 file thư viện.
4. Thêm `__init__.py` khắp nơi, viết `pyproject.toml`.
5. Mọi đường dẫn nhận từ tham số, không từ cwd. Riêng `dataset.py` có ~13 đường
   dẫn cwd-tương đối.

**Xong khi:** `cd /tmp && python -c "import dexgrasp"` không đổi cwd, không lỗi.

### M2 — YAML + Model API (4–6 tuần)

`dexgrasp/cfg/` theo 2.2. `GraspModel(Model)` với `task_map`. CLI entrypoint.
`GraspTrainer` kế thừa `ultralytics.engine.trainer.BaseTrainer`.

**Điểm lệch phải xử lý:** Ultralytics đếm theo **epoch**, DGN2 đếm theo
**iteration** (`max_iter`), và `dataset.__len__` trả `100000` khi train để
DataLoader không cạn. Giữ ngữ nghĩa iteration; đặt "một epoch" = một chu kỳ giữa
hai lần validate, để `on_train_epoch_end` / `save_model()` rơi đúng nhịp cũ.

**Xong khi:** `dexgrasp train model=... data=... max_iter=100` chạy, và
`GraspModel(...).train()` cho cùng loss với bản gốc ở cùng seed (chênh < 1e-5).

### M3 — Kiến trúc mới (12–18 tuần)

Việc lớn nhất. Xem 2.3. Không còn M3A/M3B — không giữ MinkUNet nữa nên không phải
viết lại 60 điểm gọi ME; xoá thẳng.

- **Tuần 1–2 — tiền xử lý CPU.** `voxel_downsample`, `spatial_sort` (Z-order +
  Hilbert), `pad_or_sample`. Đầu ra kích thước cố định. Xoá `minkowski_collate_fn`
  và bốn khoá `coors`/`feats`/`quantize2original`/`original2quantize`.
  *Đo ngay:* thời gian mỗi mẫu trong worker, đừng để thành nút cổ chai.
- **Tuần 3–5 — Point Patch Encoder.** Patch bằng reshape, MLP, transformer, trả
  feature theo điểm. Giữ `feature_dim: 512` để head phía sau không đổi.
- **Tuần 6 — head + seed.** `graspable` giữ nguyên. Đổi `sample_farthest_points`
  sang `torch.topk`. Cắt local geometry tokens bằng lát cắt liên tiếp.
- **Tuần 7–10 — Grasp DiT.** Token hoá theo ngón, self-attn + cross-attn +
  adaLN-Zero. Giữ `bounded_logit`.
- **Tuần 11–14 — train mô hình nền** trên toàn GraspNet. Không có warm-start.
- **Tuần 15–18 — đo và tinh chỉnh.** Quét số điểm 4096 / 8192 / 16384. So với
  MinkUNet trên giao thức 30 scene.

**Xong khi:**
1. Không còn `import MinkowskiEngine` ở đâu; `pip install` không cần CUDA toolkit.
2. Train và infer chạy được trên máy **không GPU** (chậm là chấp nhận được).
3. Có bảng so sánh sim success 30 scene giữa kiến trúc mới và MinkUNet, kèm
   khoảng tin cậy.
4. Đo được cả `rel_dis_joint` lẫn số vi phạm joint limit — với `bounded_logit`
   thì vi phạm phải bằng 0.

**Bẫy:** đừng dùng "0 vi phạm joint limit" làm bằng chứng chất lượng. Với
`bounded_logit` thì sigmoid làm vi phạm **bất khả thi** — phép đếm đó chỉ hỏng
được khi có NaN. Phải chạy thêm cổng kiểm sụp chế độ (modal bin, PC1,
participation rank), vì checkpoint `table_only` từng vượt phép đếm limit sạch sẽ
trong khi thực tế sinh gần như một tư thế tay cố định.

### M4 — MuJoCo làm sim đánh giá chính (8–12 tuần)

Không phải "thêm một backend". **Thay IsaacGym bằng MuJoCo làm đường chính.**

#### Vì sao — lý do số một là tất định

IsaacGym **không tất định**. Đã đo: cùng một file dự đoán
(`grasps_sha256 498c365d…`) sim lại nhiều lần cho **15 / 17 / 17 / 17 / 17 / 21**
lần thành công — tức 5.86%–8.20%, **±1.2pp trên đầu vào không đổi**.

Hệ quả: mọi so sánh checkpoint trong dự án này đều nằm dưới sàn nhiễu. Con số
5.86% từng bị gán nhầm cho hai run khác nhau suốt nhiều tháng chính vì không ai
phân biệt nổi.

MuJoCo tất định với cùng seed và timestep. Sửa được cái đó là sửa được **toàn bộ
vòng đo**, không chỉ đổi thư viện.

Cộng thêm:
- `pip install mujoco` — không đăng ký, không tải thủ công, không ràng buộc driver.
- Chạy CPU. Bỏ được ràng buộc "phải có GPU NVIDIA mới eval được".
- DeepMind bảo trì tích cực. IsaacGym Preview đã **ngừng phát triển**.
- Không còn ràng buộc "phải `import isaacgym` trước `import torch`" — thứ đã đẻ
  ra thứ tự import mong manh trong `eval/evaluate_dexterous.py:7,12`.

#### Phải giữ nguyên nghĩa, không được "cải tiến"

Mọi con số success rate lịch sử đều tính theo quy trình này. Đổi bất kỳ chỗ nào
là mất khả năng so sánh.

Từ `simulation_evaluator.py` và `configs/simulator/IsaacGymSimulator.yaml`:

| | giá trị |
|---|---|
| waypoint | 5 chặng: pregrasp / cover / grasp / squeeze / lift |
| `waypoint_steps` | `[40, 20, 20, 60]` |
| tần số điều khiển | `env_hz: 60`, `control_freq_inv: 1` |
| siết ngón | `-0.025 / -0.025`, rồi `+0.03 / +0.03` với `keep_z=True` |
| lùi tay | `[-0.1, 0, 0]` rồi `[-0.2, 0, 0]` |
| nâng | `[0, 0, +0.2]` |
| trọng lực | tắt cho tay; bật `[0, 0, -1]` ở chặng nâng |
| vật thể | `density: 500`, `override_com`, `override_inertia` |
| mặt bàn | `table_height` ép về **0** (`simulation_evaluator.py:69`) |
| thành công | `object_height_final > object_height_init + 0.03` |
| ngưỡng xuyên | `scene_pen_threshold: 0.0`, `table_pen_threshold: 0.0` |

Và công thức siết ngón, đã đo tác dụng thật:
```
q ← q + (upper − q) · min(0.15 · d/0.03, 1)
```
Ở mode "đỗ" `u = 0.07` mà model hiện sinh ra, siết hết biên đẩy khớp lên `u = 0.21`
— **thêm ~0.20 rad**. Nghĩa là tư thế được chấm điểm **không phải** tư thế model
dự đoán. Port sai chỗ này là mọi con số vô nghĩa.

#### URDF → MJCF: dùng capsule gốc, đừng dùng bản đã gộp

Đây là chỗ dễ sai, và **bản hiện tại đang sai**.

`setup_custom_urdf` gộp capsule (sphere + trụ + sphere cùng bán kính) thành một
trụ dài thêm `2r`. Lý do chính đáng: `RobotModel._build_geometry` chỉ nhận
Box/Cylinder/Mesh, gặp `<sphere>` là raise.

Nhưng IsaacGym lại đặt `replace_cylinder_with_capsule: True`. Nên đường đi thật là:

```
capsule(r, L) gốc  →  [gộp]  →  trụ(r, L+2r)  →  [IsaacGym]  →  capsule(r, L+2r)
```

Capsule cuối **dài hơn capsule gốc `2r`**. Với ngón r=0.008 là dư **16 mm**; với
`right_thumb_4` r=0.011 là dư **22 mm**. Đầu ngón bị phình ra trong sim.

Bài học: **hai bên tiêu thụ hình học khác nhau, cần hình học khác nhau.**
- `RobotModel` (kiểm va chạm trong mạng) → cần primitive không sphere → dùng bản gộp.
- Sim → nhận capsule gốc trực tiếp.

MuJoCo hỗ trợ `<geom type="capsule">` **nguyên bản**. Nên đường MuJoCo phải xuất
capsule thật từ URDF gốc, không đi qua bản gộp. Vừa đúng hơn vừa đơn giản hơn.

`robot/urdf.py` phải có hai chế độ xuất: `target='robotmodel'` (gộp capsule) và
`target='mjcf'` (giữ capsule). Cùng một FK, khác hình học va chạm.

#### Đế tự do: `freejoint` hay giữ 6 khớp ảo

- **`<freejoint/>` nguyên bản của MuJoCo** — sạch, nhưng `qpos` là 7 (3 vị trí +
  4 quaternion) còn `qvel` là 6. Chỉ số **khác** IsaacGym, và
  `simulation_evaluator.py:153` đang dùng `dof_names[6:]` để lấy khớp ngón.
- **Giữ 6 khớp ảo** (3 prismatic + 3 revolute) như `rh56e2_free.urdf` đang có —
  chậm hơn chút, nhưng giữ nguyên chỉ số và ngữ nghĩa waypoint.

**Chọn giữ 6 khớp ảo ở M4.** Đổi sang `freejoint` là tối ưu hoá, để sau khi đã
có bản chạy đúng. Đừng đổi hai thứ cùng lúc.

#### Chạy song song

IsaacGym chạy 100 env một lượt. MuJoCo CPU là một env.

- **`mujoco.rollout`** — module rollout đa luồng có sẵn, chạy nhiều trạng thái ban
  đầu trên nhiều luồng CPU. Đây là công cụ đúng cho bài này: cùng một model, khác
  trạng thái đầu.
- **MJX** (MuJoCo trên JAX) nếu cần batch trên GPU. Thêm phụ thuộc JAX — chỉ làm
  khi `rollout` không đủ nhanh, và đo trước rồi mới quyết.

Khối lượng: 30 scene × 256 grasp = **7680 lần rollout**. Mỗi rollout 140 bước ở
60 Hz. Đo thông lượng trước khi chọn `rollout` hay MJX.

#### Hiệu chuẩn — không được bỏ

MuJoCo và IsaacGym **sẽ không cho cùng số**. Bộ giải khác, xử lý tiếp xúc khác,
tham số ma sát khác. Đừng kỳ vọng trùng.

Việc phải làm là **hiệu chuẩn**, theo thứ tự:

1. Chạy cả hai trên **cùng một bộ grasp cố định** (`leap_ours_ckpt_50000`,
   scene_0004, 256 grasp) — bộ có số tham chiếu đáng tin nhất (67.19%).
2. Báo cáo **ma trận nhầm lẫn theo từng grasp**, không chỉ tỉ lệ tổng. Grasp nào
   IsaacGym cho đậu mà MuJoCo cho rớt, và ngược lại. Xem chúng có mẫu chung không.
3. Chỉnh ma sát / độ cứng tiếp xúc / `solref`-`solimp` của MuJoCo cho tới khi
   **tương quan theo grasp** cao, chứ không phải cho tới khi tỉ lệ tổng bằng nhau.
   Khớp tỉ lệ tổng mà xếp hạng grasp khác nhau là tệ hơn cả không khớp.
4. Chốt tham số, ghi vào `cfg/sim/mujoco.yaml`, **đóng băng**. Sau đó mọi số đều
   là số MuJoCo; số IsaacGym cũ chỉ để tham khảo lịch sử.

**Xong khi:**
1. `pip install -e .` rồi eval chạy được trên máy **không GPU**.
2. Chạy hai lần cùng seed cho **kết quả trùng từng bit** — đây là điểm ăn tiền
   so với IsaacGym.
3. Có bảng hiệu chuẩn theo từng grasp so với IsaacGym trên bộ tham chiếu.
4. Thông lượng đủ cho 7680 rollout trong thời gian chấp nhận được.

#### Giữ IsaacGym bao lâu

Giữ `sim/isaacgym.py` cho tới khi bước hiệu chuẩn xong và mọi số lịch sử đã được
chạy lại bằng MuJoCo. Sau đó chuyển nó thành tuỳ chọn, và bỏ ràng buộc thứ tự
import khỏi đường chính.

### M5 — Theo kịp Ultralytics (liên tục)

- Ghim phiên bản, nâng theo PR có test.
- Một bộ test đối chiếu: mỗi API của Ultralytics mà `dexgrasp` kế thừa đều có một
  test khẳng định chữ ký còn nguyên. Ultralytics đổi chữ ký thì **test đỏ ngay**,
  không phải phát hiện lúc chạy thật.
- Danh sách module đáng theo dõi để mượn: `utils/benchmarks.py`,
  `utils/autobatch.py`, `utils/tuner.py` (Ray Tune), `utils/export/`,
  `utils/callbacks/*` (tensorboard, wandb, mlflow, clearml, comet đều có sẵn).

---

## 5. Cạm bẫy đã đo được

Từ khảo sát upstream và từ việc đưa RH56E2 vào thực tế.

**Hỏng im lặng — nguy hiểm nhất:**
- `dataset.py` tìm graspness `.npy`; dữ liệu Stage 5f ship `.npz`; nhánh `else`
  gán zeros **không cảnh báo** → train với graspness rỗng mà loss vẫn đẹp.
- `__getitem__` bọc `try/except` rồi **đệ quy sang index ngẫu nhiên** khi có bất
  kỳ lỗi nào (4 chỗ: `:90`, `:113`, `:183`, `:217`).
- `DotDict.__getattr__` trả `None` cho khóa thiếu → gõ sai khóa không báo.
- `config.data.grasp_data` là **khóa chết**: `dataset.py:181,188` hard-code
  `dex_grasps_new` trong khi yaml khai `poses_gn`.
- Trỏ `palm` vào link không có `<collision>` → `mean(rỗng)` = NaN → graspness NaN,
  không lỗi.

**Hard-code theo leap:**
- `choices=['leap_hand']` ở 7 argparse.
- `network/cvae.py:142` hard-code `robot_name='leap_hand'` **bên trong class model**.
- Ba tên link định nghĩa nón graspness: `hand_base_link`, `thumb_fingertip`,
  `fingertip_2` (`dex_graspness.py:203,207,213`).
- `simulator.py:213` `get_squeeze_params` chỉ rẽ nhánh cho `leap_hand`.
- `simulation_evaluator.py:153` `dof_names[6:]` giả định đúng 6 DOF đế tự do.

**Đường dẫn & môi trường:**
- `robot_model.py:230` `base_path = dirname×3(urdf_path)` → với đường dẫn tương
  đối chuẩn thì `base_path == ''`, nạp mesh theo cwd.
- URDF phải **không còn joint `fixed`**: `_compute_joint_order()` duyệt DFS mọi
  joint và đọc `joint.limit.upper` trên `None`. Dùng `robot/urdf.py`.
- `utils/ik.py` nạp `ur5.urdf` + `ur5_full.urdf` — **cả hai không có trong repo**,
  mà `vis_plotly.py:22` import nó vô điều kiện.
- `configs/network/train.yaml` — mặc định của `--yaml` — **không tồn tại**.
- Vài file **chạy việc thật lúc import**: `refine_dataset.py:63` gọi `main()`;
  `dex_graspness.py:30` dựng `PoseRefine()` — hàm này **ghi vào `data/fps/`**.

**Đo lường:**
- IsaacGym không tất định: cùng một file re-sim ra 15/17/21 (±1.2pp). Wilson CI ở
  n=256 khoảng ±3pp. **Không xếp hạng checkpoint bằng một scene 256 grasp.**
  M4 sửa tận gốc bằng MuJoCo tất định.
- **Hình học sim khác hình học kiểm va chạm.** `replace_cylinder_with_capsule:
  True` của IsaacGym, cộng với việc `setup_custom_urdf` gộp capsule thành trụ dài
  thêm 2r, làm đầu ngón phình ra 16–22 mm trong sim. Xuất MJCF phải giữ capsule
  gốc.
- Loss diffusion không đo được chất lượng grasp. Chọn checkpoint theo
  `rel_dis_joint` (phần hành trình khớp), không theo total loss.

**Frame triển khai:**
- Model dự đoán pose của **root = `base_link`**. Controller thật điều khiển theo
  `right_hand`, lệch **105 mm + xoay 180°**. Ghi transform ra `<robot>_frames.json`.

---

## 6. Dùng lại nguyên vẹn

Đã kiểm: không side effect lúc import, không phụ thuộc cwd. **Chép thẳng, chỉ sửa
đường dẫn import. Đừng viết lại.**

`utils/config.py`, `utils/pc.py`, `utils/rot6d.py`, `utils/edge.py`,
`utils/robot_info.py`, `utils/util.py`, `network/mlp.py`, `network/pointnet.py`,
`network/condition.py`, `network/diffusion.py`, `network/diffusion_sample.py`,
`utils/collision_checker.py`, `utils/simulator/simulator.py`,
`utils/data_evaluator/data_evaluator.py`.

`network/backbones/*` **xoá thẳng** — MinkUNet không còn được dùng.
`network/diffusion.py` cũng bị thay bởi `nn/dit.py`; chỉ giữ lại lịch nhiễu
(noise schedule) nếu nó độc lập với kiến trúc.

---

## 7. Tiêu chí nghiệm thu xuyên suốt

Chạy sau **mỗi** mốc. Lệch ở đâu thì dừng ở đó.

**M1–M2 (chưa đổi kiến trúc)** — vẫn phải tương đương từng bit:
1. `sorted(model.state_dict())` trùng bản gốc, mọi shape trùng.
2. `final_merged_rh56e2_12dof.pth` và `rh56e2_table_only_ckpt_20000.pth` nạp
   `strict=True`.
3. Một bước train, cùng seed cùng batch: `loss`, `loss_joint`, `abs_dis_joint`,
   `rel_dis_joint` chênh < 1e-5 so với bản gốc.
4. Smoke 20 iteration, so đường loss.
5. Infer `scene_0004`, `grasp_num=256`, `seed=0`: chênh < 1e-4.

**Từ M3 (đổi kiến trúc)** — 1–5 không còn nghĩa, thay bằng:
6. Không còn `import MinkowskiEngine`; cài được không cần CUDA toolkit.
7. Train + infer chạy trên máy không GPU.
8. Sim success trên **30 scene** (7680 grasp, ≥2 seed) kèm khoảng tin cậy, so với
   `leap_ours_ckpt_50000`.
9. Cổng kiểm sụp chế độ: modal bin < 25%, PC1 < 70%, participation rank > 2.5.
10. `bounded_logit`: 0 vi phạm joint limit (bắt buộc, nhưng **không** phải bằng
    chứng chất lượng).

---

## 8. Việc phải chốt trước khi bắt đầu

1. **Giấy phép.** Ultralytics là AGPL-3.0. Phụ thuộc nó ⇒ `dexgrasp` cũng AGPL nếu
   phát hành. Không chấp nhận được thì đổi sang "mô phỏng kiến trúc, không import",
   và mục 2.1 phải viết lại.
2. **Ngân sách train lại mô hình nền.** Kiến trúc mới làm
   `leap_ours_ckpt_50000.pth` (67.19%) thành vô dụng. Phải train lại trên toàn
   GraspNet — đúng run 50k iteration của tác giả. Cần bao nhiêu GPU-giờ, chạy ở
   đâu? Đây là ràng buộc thật, không phải chi tiết.
3. **Thông lượng MuJoCo.** 7680 rollout × 140 bước. Đo trước bằng
   `mujoco.rollout` đa luồng; không đủ thì mới tính MJX (thêm phụ thuộc JAX).
4. **Ai bảo trì.** M5 là việc liên tục, không có điểm kết.
