---
document_id: SESSION-20260831-003
document_type: session_report
title: Thi công Phase 4 — kiến trúc QDGrasp-Flow, và bàn giao phiên làm việc
status: complete
date: 2026-08-31
session_id: SESSION-20260831-003
author: claude-agent
revises: none
related_plan: ROADMAP-P4-001
---

# SESSION-20260831-003 — Phase 4 (P4-00 … P4-10) và bàn giao

## Mục tiêu phiên

- P3.5 đã cạn phần việc làm được trên máy phát triển: hai package còn lại cần
  một GPU NVIDIA thật và một người kiểm tra không phải tác giả. Vì vậy phiên
  chuyển sang phase kế tiếp theo `PROJECT_PHASES.md`: **P4 — Model MVP**.
- P4 chưa có execution plan trong khi mọi phase trước đều có. Việc đầu tiên là
  viết nó, rồi thi công theo đúng work breakdown của nó.
- Tiêu chí dừng giữ nguyên như hai phiên trước: mỗi package hoặc có artifact
  chạy được kèm test, hoặc được ghi rõ `open`/`blocked` cùng lý do.
- Yêu cầu cuối phiên của người dùng là **bàn giao**; §"Bàn giao" ghi trạng thái
  của cả ba mạch việc trong phiên (MVP-T, P3.5, P4), không riêng P4.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| P4-00 | Khóa scope, contract, work breakdown, test matrix và cổng đóng cho P4 | `docs/roadmap/PHASE4_EXECUTION_PLAN.md` (`ROADMAP-P4-001@1.0.0`) | verified |
| P4-01 | Point tokenizer thuần Torch: key int64 **đơn ánh** theo vị trí, không hash; từ chối grid mịn quá khả năng pack | `qdgrasp/models/tokenizer.py` | verified |
| P4-02 | Serialized point encoder với windowed/shifted attention; chi phí tuyến tính theo số token | `qdgrasp/models/encoder.py` | verified |
| P4-03 | HandGraph encoder biến độ dài: message passing trên edge list, edge đối xứng hóa kèm cờ hướng | `qdgrasp/models/hand_graph.py` | verified |
| P4-04 | Cross-attention conditioning `[L, T]` (hand query × object key) | `qdgrasp/models/flow.py` (`CrossAttentionBlock`) | verified |
| P4-05 | Rectified-flow head sinh palm pose + joint state; rotation 9D → SO(3) bằng Gram-Schmidt, joint kẹp vào limit có tên | `qdgrasp/models/flow.py` | verified |
| P4-06 | FK consistency: fingertip tính từ palm+joint bằng FK của P2, không dự đoán song song | `qdgrasp/models/losses.py`, `RobotSpec.fingertip_positions` | verified |
| P4-07 | Quality head trên conditioning | `qdgrasp/models/flow.py` (`GraspFlowModel.quality`) | verified |
| P4-09 | Loss assembly: tổng **bằng** tổng các term được log, term lạ bị từ chối, rotation error là geodesic | `qdgrasp/models/losses.py` | verified |
| P4-08 | Scale table `n`/`s`/`m` là nơi duy nhất ghi width/depth; preset đặt tên scale và override năm scalar, tham số lạ bị **từ chối** chứ không bị bỏ qua; model chạy được qua facade P1 của cả hai tay | `qdgrasp/models/config.py`, `qdgrasp/presets/qdgrasp-flow-{n,s,m}.yaml` | verified |
| P4-10 | Tiny overfit trên CPU cho **cả hai** active hand: kiến trúc học được, gradient phủ 186/186 tham số | `scripts/overfit_qdgrasp_flow.py`, `evidence/phase4/overfit-{leap,allegro}-cpu.json` | verified |
| P4-11a | Harness CUDA gate và notebook pin commit; harness từ chối `--device cpu` và từ chối nhãn CUDA khi không có CUDA | `scripts/phase4_cuda_gate.py`, `notebooks/phase4_cuda_gate.ipynb` | verified |
| P4-12a | Packet review bất biến và hướng dẫn reviewer; packet không mang verdict | `scripts/phase4_review_packet.py`, `docs/roadmap/PHASE4_REVIEWER_GUIDE.md` | verified |
| — | Cổng đọc được bằng máy cho P4, báo từng package delivered/open/blocked | `scripts/check_phase4.py` | verified |

Lệch so với work breakdown: P4-04, P4-06 và P4-07 được viết **trong**
`qdgrasp/models/flow.py` và `losses.py` thay vì `conditioning.py`, `fk_head.py`,
`quality.py` như §4 của plan liệt kê. Ba phần này chia chung một forward pass và
một khối conditioning; tách file làm chúng phải truyền qua lại conditioning đã
tính rồi. Đây là lệch về vị trí file, không phải về phạm vi; plan sẽ được sửa
tên file khi P4-08 chạm vào registry.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | plan | `docs/roadmap/PHASE4_EXECUTION_PLAN.md` | `ROADMAP-P4-001@1.0.0` |
| E-02 | code | `qdgrasp/models/{tokenizer,encoder,hand_graph,flow,losses}.py` | commit `ff5d816` |
| E-03 | test | `tests/model_flow/` (63 test) | commit `9583faf` |
| E-04 | evidence | `evidence/phase4/overfit-{leap,allegro}-cpu.json` | commit `9583faf` |
| E-05 | gate | `scripts/check_phase4.py --profile micro` | 12/14 delivered, exit 1 |
| E-06 | harness | `scripts/phase4_cuda_gate.py`, `notebooks/phase4_cuda_gate.ipynb` | pin `9583faf` |
| E-07 | review | `evidence/phase4/review/review-packet.json` | digest `24556a1849607349ebf2b0e25b5f082673be47a3b5f006f2e8d0b129e993d83b` tại `93b330f` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `python -m pytest tests/model_flow -q` | 0 | 63 passed |
| T-02 | `python -m pytest -q` | 0 | 1252 passed, 1 skipped, 93 subtests |
| T-03 | `python scripts/overfit_qdgrasp_flow.py --report evidence/phase4/overfit-leap-cpu.json` | 0 | converged; xem số đo bên dưới |
| T-04 | `python scripts/check_docs.py` | 0 | 145 file pass |
| T-05 | `ruff check` + `ruff format --check` trên file mới | 0 | sạch |
| T-06 | `python scripts/check_phase4.py --profile micro` | 1 | 12/14 delivered; hai package còn mở |
| T-07 | `python scripts/phase4_cuda_gate.py --device cpu` | 1 | từ chối: `verdict=refused` |
| T-08 | `python scripts/phase4_cuda_gate.py --device cuda:0` | 1 | từ chối: không có CUDA |
| T-09 | `python scripts/overfit_qdgrasp_flow.py --robot wonik_allegro.yaml` | 0 | palm 0.0462 m, rot 0.0270 rad, joint 0.0372 rad, tip 0.0461 m |

Số đo của T-03 (LEAP, 8 sample, 256 điểm, 1200 bước, CPU, noise cố định,
2 503 821 tham số, 279 s). Allegro (T-09) hội tụ cùng ngưỡng, xem bảng T:

| Đại lượng | Bước 0 | Bước 1199 | Ngưỡng |
|---|---:|---:|---:|
| palm translation | 1.8709 m | 0.0403 m | 0.06 m |
| palm rotation | 2.6172 rad | 0.0213 rad | 0.05 rad |
| joint (trị tuyệt đối) | 0.8636 rad | 0.0526 rad | 0.10 rad |
| fingertip | 1.8381 m | 0.0413 m | 0.06 m |
| gradient coverage | — | 186/186 | 186/186 |

Đây là bằng chứng rằng **kiến trúc học được**. Nó không nói gì về chất lượng
grasp, và `ROADMAP-P4-001` §7 cấm trích nó như thể có.

## Bốn hiệu chỉnh dựa trên đo đạc

1. **Overfit phải hỏi một câu tất định.** Lần chạy đầu báo fail (loss 8.04 →
   3.70, palm 0.317 m). Nguyên nhân thứ nhất: mỗi lần gọi lấy một noise mới, nên
   bài toán đặt ra thực chất là "ánh xạ **mọi** vector noise về cùng một grasp".
   `sample_state` nay nhận `noise` để ghim điểm xuất phát; chỉ riêng thay đổi
   này đã kéo palm về 0.077 m.
2. **Tiêu chí phải đọc trên pose, không trên tổng loss.** Nguyên nhân thứ hai:
   term `flow_velocity` có sàn không rút được — với `(x_t, t, cond)` cho trước,
   velocity target `target - noise` vẫn ngẫu nhiên, nên bộ dự đoán kỳ vọng có
   điều kiện giữ MSE khác 0 vĩnh viễn. Đo được nó dừng quanh 1.0 trong khi mọi
   term pose giảm một đến hai bậc. Một ngưỡng đặt trên tổng sẽ gọi một lần chạy
   khỏe mạnh là lỗi đấu dây. Verdict nay đặt trên bốn ngưỡng pose.
3. **Solver lúc chưa train là ánh xạ đồng nhất.** Test "đổi số bước Euler thì
   mẫu phải đổi" fail, và fail đúng: `VelocityField.output` được khởi tạo bằng 0
   (cố ý, để field chưa train không đẩy state vào vùng mà clamp phải cứu và làm
   một gradient chết trông như bình thường), nên sample **bằng** noise bất kể số
   bước. Test được tách đôi: một test ghim tính chất đồng nhất lúc khởi tạo, một
   test nhấc head khỏi 0 rồi mới đo ảnh hưởng của số bước.

4. **Một probe im lặng vì lý do sai thì tệ hơn không có probe.** Test "không có
   tensor `N×N`" ban đầu đọc saved tensor của autograd. Nó pass — nhưng negative
   control cho thấy nó cũng pass trên một model dùng attention toàn cục: kernel
   attention hợp nhất **tính lại** ma trận ở backward nên autograd không bao giờ
   thấy nó. Test được viết lại trên `TorchDispatchMode`, đọc chính lời gọi và độ
   dài chuỗi của nó, và negative control nằm **trong** test: nếu probe không
   flag được model attention toàn cục thì test fail, vì lúc đó sự im lặng của nó
   với model thật không có giá trị gì.

## Việc chưa hoàn tất

- **P4-11b — CUDA gate evidence:** `blocked`. Harness và notebook đã sẵn sàng và
  đã pin commit `9583faf`, nhưng theo `ADR-0006` chỉ một lần chạy trên NVIDIA
  thật mới thành bằng chứng. Harness từ chối cả `--device cpu` lẫn `--device
  cuda:0` khi không có CUDA, nên không có đường nào để một lần chạy CPU trở
  thành CUDA evidence do nhầm lẫn.
- **P4-12 — independent review:** `blocked`. Packet và hướng dẫn reviewer đã có;
  verdict thì không, vì tác giả artifact không được tự ký.
- **Nhánh chưa được push.** Notebook pin `9583faf`; Kaggle/Colab clone từ
  `origin` nên commit đó phải nằm trên origin trước khi notebook chạy được.
  Việc push là quyết định của người dùng, không phải của phiên này.
- Model chưa được train trên `DGN-Open-Tiny`. Fixture của overfit là label sinh
  bằng FK của chính profile — đủ để một lần không hội tụ có nghĩa là "kiến trúc
  sai", không đủ để nói gì về dữ liệu thật.

## Sửa đổi phiên trước

Không. Phiên này không đụng tới kết luận của MVP-T hay P3.5; cả hai giữ nguyên
trạng thái mà `SESSION-20260831-001` và `-002` ghi.

## Bàn giao

### Trạng thái ba mạch việc

| Mạch | Trạng thái | Còn lại |
|---|---|---|
| `ROADMAP-MVP-001` (MVP-T) | **complete**, `experimental_non_release` | không; kết luận âm đã ghi: policy học được **không** vượt controller prior |
| `ROADMAP-P3.5-001` | **in_progress**, 17/19 | P3.5-15 (cần GPU), P3.5-18 (cần reviewer) |
| `ROADMAP-P4-001` | **in_progress**, 12/14 | P4-11b (cần GPU), P4-12 (cần reviewer) |

### Việc cần người, không cần thêm code

1. Chạy `notebooks/phase3_5_rl_readiness.ipynb` trên Kaggle/Colab GPU, lưu kết
   quả dưới `evidence/phase3_5/`. Đó là đầu vào hợp lệ duy nhất cho P3.5-15.
   Harness đã từ chối đúng khi không có CUDA (`verdict=refused`), nên một lần
   chạy CPU **không** thể trở thành bằng chứng CUDA do nhầm lẫn.
2. Chỉ định reviewer độc lập ký packet P3.5, digest
   `e1062c476a679515e54c25140b81e3c4a8ea515503b004aac7c93e9e02ca0c05`
   tại commit `dc0d08a` (`scripts/phase3_5_review_packet.py`).
3. Push nhánh, rồi chạy `notebooks/phase4_cuda_gate.ipynb` trên Kaggle/Colab GPU
   và lưu kết quả vào `evidence/phase4/cuda-<gpu>-<ngày>.json` — đó là chỗ
   `check_phase4.py` tìm nó. Đầu vào duy nhất hợp lệ cho P4-11b.
4. Chỉ định reviewer độc lập cho P4-12, theo
   `docs/roadmap/PHASE4_REVIEWER_GUIDE.md`. Packet đã sinh ở worktree sạch:
   digest `24556a1849607349ebf2b0e25b5f082673be47a3b5f006f2e8d0b129e993d83b`
   tại commit `93b330f`, 22 artifact, `complete: true`, `verdict: null`.

### Việc tiếp theo trong repo

Không còn package P4 nào làm được trên máy này. Hai mục còn lại đều cần thứ
không tồn tại ở đây: một GPU NVIDIA thật và một người kiểm tra không phải tác
giả. Việc tiếp theo trong repo là **P5 — Training & evaluation**, và nó cần
`ROADMAP-P5-001` trước khi viết dòng code nào, đúng như P4 đã cần
`ROADMAP-P4-001`.

### Điều kiện phiên sau phải kiểm trước khi tin trạng thái

- `python -m pytest -q` phải còn trả 0 với ít nhất 1252 test pass.
- `python scripts/check_phase3_5.py --profile micro` phải còn trả `1` với đúng
  hai package mở. Nếu nó trả `0` mà chưa có GPU evidence và review thì cổng đã
  bị nới, không phải phase đã xong.
- `python scripts/check_phase4.py --profile micro` phải còn trả `1` với đúng hai
  package mở. Nếu nó trả `0` mà chưa có CUDA evidence và review thì cổng đã bị
  nới, không phải phase đã xong.
- `python scripts/overfit_qdgrasp_flow.py` phải còn hội tụ dưới bốn ngưỡng pose
  đã pin. Nếu ai đó nới ngưỡng thay vì sửa model, evidence cũ trong
  `evidence/phase4/overfit-*-cpu.json` ghi lại số đo gốc để đối chiếu.
- `python scripts/phase4_cuda_gate.py --device cpu` phải còn **fail** với
  `verdict: refused`. Nếu nó trả `measured` trên máy không có GPU thì harness đã
  bị phá và mọi evidence sau đó vô giá trị.

### Nhánh và commit

- Nhánh `feature/mvp-grasp-policy`, chưa push, 237 commit trước `main`.
- Commit của P4: `ff5d816` (kế hoạch, năm module model, script overfit, test và
  evidence LEAP), `d21525d` (bản ghi phiên và revision record), `9583faf`
  (config/registry, hai cổng, packet review, evidence Allegro), cùng commit pin
  notebook và commit chứa bản cập nhật này.
