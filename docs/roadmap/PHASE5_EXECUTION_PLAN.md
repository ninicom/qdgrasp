---
document_id: ROADMAP-P5-001
document_type: plan
title: Kế hoạch thi công Phase 5 — Training & evaluation
version: 1.1.0
status: active
date: 2026-08-31
revises: ROADMAP-P5-001@1.0.0
related_plan: PLAN-V2
revision_reason: P5-01/02 đo được số positive thật của DGN-Open-Tiny; nó chặn P5-03 trở đi và phải nằm trong §0.1.
necessity: N2
impact: Thêm một blocker dữ liệu vào §0.1 và một cổng đọc được bằng máy; không nới điều kiện hoàn tất nào.
depends_on:
  - ROADMAP-P3.1-001
  - ROADMAP-P3.5-001
  - ROADMAP-P4-001
active_hands:
  - leap_hand
  - wonik_allegro
paused_hands:
  - shadow_hand
release_class: not_a_release
---

# Kế hoạch Phase 5 — Training & evaluation

`PLAN.md` §M4/§M5 và `PROJECT_PHASES.md` mô tả P5 nhưng chưa có execution plan.
Tài liệu này là cái đó: khóa phạm vi, contract, protocol, work breakdown, test
matrix và điều kiện đóng **trước** khi chạy lần train nào.

Thứ tự đó không phải hình thức. P5 là phase đầu tiên sinh ra con số mà người
ngoài có thể trích dẫn, và một protocol khóa sau khi nhìn kết quả thì không còn
là protocol.

## 0. Trạng thái đầu vào

| Thứ P5 cần | Đã có ở đâu | Trạng thái |
| --- | --- | --- |
| Kiến trúc QDGrasp-Flow | `qdgrasp/models/` | complete (P4), overfit pass trên CPU cho hai tay |
| Config/registry `qdgrasp-flow-n/s/m` | `qdgrasp/models/config.py` | complete (P4-08) |
| Engine `fit`/`validate`/resume | `qdgrasp/engine/runner.py`, `trainer.py` | complete (P1) |
| `GraspBatch` | `qdgrasp/dataset/batch.py` | complete (P3.1) |
| `DGN-Open-Tiny` | `datasets/dgn-open-tiny/` | complete: 13 train / 5 val object |
| Môi trường replay vật lý | `qdgrasp/rl/envs/`, `qdgrasp/scenes/settle.py` | in_progress (P3.5, 17/19) |
| Seeding tất định | `qdgrasp/engine/seeding.py` | complete (P1) |

### 0.1 Ba thứ P5 **chưa** có và phải chờ

| Chặn | Thuộc | Ảnh hưởng tới P5 |
| --- | --- | --- |
| CUDA evidence của model | `P4-11b` | Không được công bố số train nào trước khi forward/backward trên CUDA thật được đo. `ADR-0006`. |
| Backend decision | `P3.5-15` | Quyết định throughput của đánh giá vật lý ở `P5-08`. Không chặn phần CPU, chặn quy mô. |
| Contact-rich input | `P3.4.3` | `release_blocked`. P5 chỉ dùng dữ liệu static/offline. |
| **Số grasp thành công trong dataset** | `DGN-Open-Tiny` | **Chặn `P5-03` trở đi.** Đo được ở §0.2. |

P5 **được phép** thi công song song phần không phụ thuộc các mục trên: adapter
dữ liệu, protocol split, metric và ablation harness đều chạy và kiểm được trên
CPU. Cái không được phép là **công bố** kết quả.

### 0.2 Dataset chưa đủ positive để train — đo được, không phải phỏng đoán

`P5-01` và `P5-02` đã thi công, và việc đầu tiên chúng cho phép làm là đếm.
Dưới protocol đã khóa (`phase5-dgn-open-tiny-v1`):

| Split | Hand | Sample | Positive | Tỉ lệ |
|---|---|---:|---:|---:|
| train | `leap_hand` | 32 | **1** | 0.031 |
| train | `wonik_allegro` | 42 | **2** | 0.048 |
| val | `leap_hand` | 12 | 1 | 0.083 |
| val | `wonik_allegro` | 22 | 1 | 0.045 |

**Ba** grasp thành công trong toàn bộ train split của cả hai active hand.

Điều đó không phải "dataset hơi nhỏ". Nó đổi bản chất việc train:

- Generator được regress về label mà 96% là **proposal thất bại**. Học đúng
  label nghĩa là học sinh ra grasp rơi. Loss giảm vẫn giảm; nó chỉ không đo cái
  ta tưởng.
- Quality head có ba positive. Nó sẽ học prior và đạt accuracy cao bằng cách
  luôn trả `fail`.
- Không cái nào trong hai điều trên hiện ra trong loss curve.

Vì vậy `P5-03` trở đi **blocked on the data layer**, và blocker này được đo bằng
máy chứ không ghi trong chú thích:

```bash
python scripts/check_phase5_inputs.py
```

Cổng trả `1` khi bất kỳ active hand nào có dưới 25 positive trong train.
Con số 25 không phải ngưỡng thống kê — ở quy mô này không có ngưỡng nào trung
thực — mà là vạch dưới đó chữ "train" bị dùng sai.

Đường ra là ở data layer, không ở P5: regenerate `DGN-Open-Tiny` bằng recipe cho
ra positive (proposal/solver hiện tại cho tỉ lệ ~3%), hoặc thay bằng corpus lớn
hơn. Cho tới lúc đó, phần P5 làm được là adapter, protocol, metric và ablation
harness — tức là làm sẵn chỗ để đổ dữ liệu vào khi có.

## 1. Quyết định thiết kế

### 1.1 Không chọn checkpoint bằng total loss

P4 đo được rằng `flow_velocity` có sàn không rút được: với `(x_t, t, cond)` cho
trước, velocity target `target - noise` vẫn ngẫu nhiên, nên MSE của bộ dự đoán
kỳ vọng có điều kiện không về 0. Total loss vì thế trộn một hằng số không ý
nghĩa vào tín hiệu chọn model. Checkpoint chọn bằng pose/joint/quality error và
physics success, đúng như `PLAN.md` §M6 yêu cầu.

### 1.2 Protocol khóa bằng hash trước khi train

Split, seed, ablation và metric được ghi vào một file config và khóa bằng
`content_hash`, giống `configs/mvp/dexacquire-mvp-v0.yaml` của MVP-T. Mỗi run
ghi hash đó vào artifact. Đổi protocol thì hash đổi, và một kết quả cũ không thể
được trình bày như thể nó chạy dưới protocol mới.

### 1.3 Held-out embodiment với hai tay là `n=1`

`ADR-0008` giữ corpus active ở LEAP và Allegro. Held-out embodiment vì thế chỉ
có hai chiều: train LEAP → test Allegro, và ngược lại. Đó là hai điểm dữ liệu,
không phải một phân phối. P5 vẫn chạy protocol này vì nó bắt được lỗi "model
học thuộc một hình dạng tay", nhưng mọi báo cáo phải ghi `n=1 per direction` và
không được viết thành "generalizes across embodiments".

### 1.4 Physics success là nguồn sự thật, không phải pose error

Một grasp có pose error nhỏ vẫn có thể rơi. Đánh giá cuối cùng là replay trong
MuJoCo qua đường P3.5: đặt tay theo output, đóng theo prior, nâng, đo giữ được
hay không. Pose error là proxy dùng để chọn nhanh giữa các checkpoint; nó không
được trích dẫn như success.

### 1.5 Hard negative lấy từ rollout thất bại, không sinh giả

Quality head học từ `success` thật. Negative được khai thác từ chính grasp mà
model sinh ra rồi rơi trong replay, chứ không phải bằng cách nhiễu loạn một
positive — một positive bị nhiễu vẫn có thể là grasp tốt, và dán nhãn nó là
`fail` dạy model điều sai.

## 2. Phạm vi

### 2.1 Thuộc P5

- Adapter `DGN-Open-Tiny` → batch mà `QDGraspFlow` nhận, chọn graph theo robot
  của từng sample.
- Protocol split khóa bằng hash: train/val, held-out object family, held-out
  embodiment; kiểm rò rỉ bằng test chứ không bằng mắt.
- Vòng train đầy đủ qua `qdgrasp.engine.Runner`: AMP, grad clip, LR schedule,
  seed tất định.
- Exact resume: một run bị cắt rồi resume phải trùng bit với run không cắt.
- Sampling `K` grasp mỗi quan sát; diversity và coverage.
- Quality head với hard negative khai thác từ replay.
- Đánh giá vật lý: success, collision, penetration.
- Metric module có confidence interval; latency đo riêng warm-up và steady state.
- Ablation khóa trước khi chạy: graph/no-graph, direct/contact-first, FK
  consistency on/off, quality-guided sampling on/off.
- Gate CUDA cho train/eval/resume, và packet review.

### 2.2 Không thuộc P5

- Không `DGN-Open-v1`, không scale dữ liệu — đó là P6.
- Không export mới, không ONNX/TorchScript parity — P1 đã có, P6 mở rộng.
- Không model card công bố, không paper — P7.
- Không contact-rich input; nó vẫn `release_blocked` theo P3.4.3.
- Không three-hand coverage; Shadow vẫn `paused_by_ADR-0008`.
- Không so sánh với DGN2 làm baseline hay nguồn nhãn.

## 3. Contract

### 3.1 Protocol document

```text
qdgrasp/protocol/v1
  splits:
    train_objects        [str]      object_id
    val_objects          [str]
    heldout_family       [str]      họ hình dạng không xuất hiện trong train
    heldout_embodiment   {train_hand: str, test_hand: str}
  seeds                  [int]      mỗi seed là một run độc lập
  ablations              [str]      tên đã đăng ký, không phải mô tả tự do
  metrics                [str]
  selection              str        tiêu chí chọn checkpoint
  protocol_hash          str        SHA-256 canonical, ghi vào mọi artifact
```

Rò rỉ là lỗi cứng: một `object_id` xuất hiện ở cả train và bất kỳ held-out nào
thì loader **từ chối**, không cảnh báo.

### 3.2 Kết quả một run

```text
RunReport
  protocol_hash, model_hash, robot_hash, dataset_hash, commit
  seed, device, cuda (bool)
  steps, wall_s
  selection_metric, selected_step
  metrics_per_split: {split: {success, collision, penetration, diversity, coverage, latency_ms}}
  confidence: {metric: [lower, upper]}   Wilson cho tỉ lệ
  ablation: str | null
```

Một report có `cuda: false` không được dùng làm kết quả chất lượng.

### 3.3 Metric

| Metric | Định nghĩa | Nguồn |
| --- | --- | --- |
| success | grasp giữ được vật qua lift và hold, đo trong MuJoCo | replay P3.5 |
| collision | có tiếp xúc bị cấm ở tư thế trước khi đóng | scene state |
| penetration | độ xuyên sâu tối đa, ngưỡng ổn định và ngưỡng tức thời tách nhau | như MVP-T |
| diversity | phân tán của `K` grasp trên cùng quan sát | pose distance |
| coverage | tỉ lệ vùng bề mặt vật có ít nhất một grasp thành công | mesh sampling |
| latency | ms mỗi grasp, warm-up tách khỏi steady state | timer |

## 4. Work breakdown

| ID | Công việc | Output | Phụ thuộc |
| --- | --- | --- | --- |
| P5-00 | Khóa scope/contract/protocol | tài liệu này | — |
| P5-01 | Verified dataset → canonical Runner batch | `qdgrasp/dataset/loader.py`, `qdgrasp/engine/sampling.py` | 00, P3.1 |
| P5-02 | Protocol split khóa bằng hash + kiểm rò rỉ | `configs/phase5/protocol-v1.yaml`, `qdgrasp/models/protocol.py` | 01 |
| P5-03 | Vòng train qua `Runner`, AMP, schedule, seed | `qdgrasp/models/train.py` | 02 |
| P5-04 | Exact resume trùng bit | `qdgrasp/models/train.py` + test | 03 |
| P5-05 | Chọn checkpoint bằng pose/quality/physics | `qdgrasp/models/selection.py` | 03 |
| P5-06 | Sampling `K`, diversity, coverage | `qdgrasp/models/sampling.py` | 03 |
| P5-07 | Quality head + hard negative từ replay | `qdgrasp/models/quality_training.py` | 06, 08 |
| P5-08 | Đánh giá vật lý trong MuJoCo | `qdgrasp/models/physics_eval.py` | 06, P3.5 |
| P5-09 | Metric + confidence interval | `qdgrasp/models/metrics.py` | 08 |
| P5-10 | Ablation harness, khóa trước khi chạy | `scripts/run_phase5_ablations.py` | 02–09 |
| P5-11 | CUDA train/eval/resume gate | `scripts/phase5_cuda_gate.py`, notebook | 10, **P4-11b** |
| P5-12 | Independent review | packet + verdict | 11 |

Critical path `00 → 01 → 02 → 03 → {04,05,06} → 08 → 07 → 09 → 10 → 11 → 12`.

`P5-11` **không được bắt đầu** trước khi `P4-11b` đóng: một gate train trên CUDA
đặt lên một kiến trúc chưa từng chạy trên CUDA thì đo hai thứ chưa biết cùng lúc.

## 5. Test matrix bắt buộc

### Dataset adapter
- Mỗi sample ra đúng graph của robot của nó; trộn hai robot trong một batch thì
  từ chối hoặc nhóm theo robot, không im lặng dùng graph của sample đầu.
- Số sample vào bằng số sample ra; không sample nào bị bỏ lặng lẽ.

### Protocol
- Một `object_id` ở cả train và held-out làm loader raise, không warn.
- `protocol_hash` đổi khi bất kỳ trường nào đổi, và ổn định khi chỉ đổi thứ tự
  khóa trong YAML.
- Held-out embodiment: không sample nào của tay test lọt vào train.

### Train loop
- Cùng seed, cùng device, cùng protocol → cùng loss curve.
- Grad clip thực sự chặn: một batch dựng để nổ gradient không làm NaN checkpoint.
- AMP không đổi kết quả quá dung sai FP32/AMP đã pin.

### Resume
- Cắt ở bước `k`, resume, chạy tới `n`: trạng thái ở `n` trùng **bit** với run
  không cắt. Optimizer state, RNG và vị trí dataloader đều nằm trong checkpoint.

### Selection
- Một checkpoint có total loss thấp hơn nhưng pose/physics tệ hơn **không** được
  chọn. Đây là test viết trực tiếp từ phát hiện của P4 §1.1.

### Metric
- Wilson interval khớp giá trị tham chiếu đã tính tay.
- `success` chỉ đến từ replay; không đường nào cho phép pose error thành success.
- Latency tách warm-up khỏi steady state.

### Ablation
- Tên ablation phải nằm trong registry; tên lạ bị từ chối.
- Ablation chạy trên đúng protocol_hash của baseline, nếu không thì từ chối so.

### CPU/CUDA
- CPU chạy correctness. CUDA chạy train/eval/resume thật; không fallback.
- FP32 CPU/CUDA parity `atol/rtol ≤ 1e-4` theo `PLAN.md` §6.

## 6. Gate đóng P5

```bash
python scripts/check_phase5_inputs.py     # phải pass trước khi P5-03 có nghĩa
python scripts/check_phase5.py --profile micro
python -m pytest tests/model_training -q
python scripts/run_phase5_ablations.py --dry-run
git diff --check
```

GPU gate chạy ở notebook riêng:

```bash
python scripts/phase5_cuda_gate.py --device cuda:0 --evidence phase5_cuda_evidence.json
```

## 7. Điều kiện hoàn tất

0. `scripts/check_phase5_inputs.py` pass — mỗi active hand có đủ positive để
   train. Chừng nào còn fail thì mọi điều kiện dưới đây đều vô nghĩa.
1. Protocol khóa bằng hash trước run đầu tiên, và mọi artifact mang hash đó.
2. Train/val/held-out object family/held-out embodiment đều chạy, không rò rỉ,
   với ít nhất ba seed mỗi cấu hình.
3. Exact resume trùng bit.
4. Checkpoint chọn bằng pose/quality/physics, không bằng total loss.
5. Metric có confidence interval; `success` chỉ từ replay vật lý.
6. Ablation khóa trước khi chạy và báo cáo đủ cả cấu hình thua.
7. Train/eval/resume chạy trên CUDA thật; CPU/CUDA FP32 parity trong dung sai.
8. Independent reviewer ký trên packet bất biến, zero S0/S1.

P5 kết thúc ở **một checkpoint multi-hand có số đo tái lập được**, không phải ở
"model tốt nhất có thể". Mọi con số phải ghi kèm `n`, seed và protocol_hash; một
kết quả không có ba thứ đó thì không được trích dẫn.
