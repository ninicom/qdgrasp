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
tính rồi. Đây là lệch về vị trí file, không phải về phạm vi, và §4 của plan đã
được sửa cho khớp code (`ROADMAP-P4-001@1.1.0`) thay vì để lệch lại đó.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | plan | `docs/roadmap/PHASE4_EXECUTION_PLAN.md` | `ROADMAP-P4-001@1.0.0` |
| E-02 | code | `qdgrasp/models/{tokenizer,encoder,hand_graph,flow,losses}.py` | commit `ff5d816` |
| E-03 | test | `tests/model_flow/` (65 test) | commit `2c532ac` |
| E-04 | evidence | `evidence/phase4/overfit-{leap,allegro}-cpu.json` | commit `9583faf` |
| E-05 | gate | `scripts/check_phase4.py --profile micro` | 12/14 delivered, exit 1 |
| E-06 | harness | `scripts/phase4_cuda_gate.py`, `notebooks/phase4_cuda_gate.ipynb` | pin `9583faf` |
| E-07 | review | `evidence/phase4/review/review-packet.json` | `packet_digest` và `commit` nằm trong chính packet |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `python -m pytest tests/model_flow -q` | 0 | 63 passed |
| T-02 | `python -m pytest -q` | 0 | 1263 passed, 1 skipped, 93 subtests |
| T-03 | `python scripts/overfit_qdgrasp_flow.py --report evidence/phase4/overfit-leap-cpu.json` | 0 | converged; xem số đo bên dưới |
| T-04 | `python scripts/check_docs.py` | 0 | 145 file pass |
| T-05 | `ruff check` + `ruff format --check` trên file mới | 0 | sạch |
| T-06 | `python scripts/check_phase4.py --profile micro` | 1 | 12/14 delivered; hai package còn mở |
| T-07 | `python scripts/phase4_cuda_gate.py --device cpu` | 1 | từ chối: `verdict=refused` |
| T-08 | `python scripts/phase4_cuda_gate.py --device cuda:0` | 1 | từ chối: không có CUDA |
| T-09 | `python scripts/overfit_qdgrasp_flow.py --robot wonik_allegro.yaml` | 0 | palm 0.0462 m, rot 0.0270 rad, joint 0.0372 rad, tip 0.0461 m |
| T-10 | `python -m pytest tests/contactrich_active -q` | 0 | 408 passed sau khi sửa `audit_closure` |
| T-11 | `python -m pytest tests/model_flow -q` (sau hai test cuối của §5) | 0 | 65 passed |
| T-12 | `python -m pytest tests/model_flow -q` (sau khi siết cổng CUDA evidence) | 0 | 74 passed |
| T-13 | probe phần cứng: `nvidia-smi`, `/dev/nvidia*`, `lspci`, `torch.version.cuda` | — | không có thiết bị NVIDIA nào |
| T-14 | diễn tập notebook: clone commit đã pin vào thư mục rỗng, chạy stage 1/2/3 | 0/0/1 | stage 1 chạy đủ cổng micro, stage 3 từ chối đúng |

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

## Sáu hiệu chỉnh dựa trên đo đạc

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

5. **Cổng của chính phiên này đếm file thay vì đọc nó.** `check_phase4.py` đánh
   dấu `P4-11b` là delivered nếu tồn tại bất kỳ `evidence/phase4/cuda-*.json`
   nào. Nhưng harness ghi record ở **mọi** lần chạy, kể cả lần bị từ chối — cố ý,
   vì một lần từ chối cũng là kết quả đáng giữ — nên cổng lẽ ra đã được thỏa mãn
   bởi chính cái máy không chạy được nó. Record nay chỉ được tính khi
   `verdict=measured`, `device.cuda=true`, mọi hand đo được đều pass, và đủ cả
   hai active hand; mọi trường hợp khác bị **nêu tên** trong dòng detail chứ
   không bị bỏ qua.

6. **Notebook export sai tên biến môi trường, và chỉ một lần diễn tập mới thấy.**
   Không đóng được `P4-11b` từ máy này, nên việc gần nhất với chạy thật là
   clone commit đã pin vào một thư mục rỗng rồi chạy đúng những lệnh notebook
   chạy. Stage 1 chết ngay: ba notebook (`mvp_grasp_policy`,
   `phase3_5_rl_readiness`, `phase4_cuda_gate`) export
   `QDGRASP_ROBOT_ASSETS` trong khi `qdgrasp/robot/assets.py` đọc
   `QDGRASP_ROBOT_ASSETS_ROOT`. Trên máy phát triển không ai thấy vì source
   checkout có fallback `.references/robot-assets`; trên Kaggle thì không.
   Nghĩa là **cả hai** lần chạy GPU mà dự án đang chờ — P3.5 và P4 — đều sẽ
   chết ngay sau cell cài đặt, và lỗi trông như lỗi code chứ không như một chữ
   sai. Đã sửa ở ba notebook và hai script sinh chúng;
   `tests/model_flow/test_notebook_environment.py` kiểm mọi notebook và builder
   theo đúng tên mà code đọc. Sau khi sửa, ba stage chạy sạch từ clone trắng.

## Việc chưa hoàn tất

- **P4-11b — CUDA gate evidence:** `blocked`, và không phải vì thiếu công việc.
  Máy phát triển được probe ở mức phần cứng: không có `nvidia-smi`, không có
  `/dev/nvidia*`, GPU là Intel Iris Plus G1 tích hợp, torch là `2.11.0+cpu`
  không có CUDA build. Probe đó được lưu ở
  `evidence/phase4/cuda-refused-devmachine-20260831.json` để phiên sau không phải
  suy luận lại, và một test khẳng định chính file đó **không** được cổng tính là
  evidence. Harness từ chối cả `--device cpu` lẫn `--device cuda:0` khi không có
  CUDA. Chỉ một runtime GPU thật mới đóng được mục này.
- **P4-12 — independent review:** `blocked`. Packet và hướng dẫn reviewer đã có;
  verdict thì không, vì tác giả artifact không được tự ký.
- Nhánh đã được push theo yêu cầu của người dùng trong phiên
  (`origin/feature/mvp-grasp-policy`), nên commit mà notebook pin đã fetch được.
  Không còn rào cản kỹ thuật nào cho P4-11b ngoài việc có một runtime GPU.
- Model chưa được train trên `DGN-Open-Tiny`. Fixture của overfit là label sinh
  bằng FK của chính profile — đủ để một lần không hội tụ có nghĩa là "kiến trúc
  sai", không đủ để nói gì về dữ liệu thật.

## Sửa đổi phiên trước

Một, và nó không thuộc P4.

`tests/contactrich_active/test_closure_trust_chain.py::test_the_ledger_rejects_a_test_path_that_does_not_exist`
fail lần đầu tiên trong phiên này — không phải vì P4 làm hỏng nó, mà vì đây là
lần đầu suite chạy trên một worktree **sạch**. Test dựng một manifest giả trong
đó một requirement `passed` trỏ tới file test không tồn tại, rồi đòi có
violation. Violation nó vẫn nhận được là `passed claimed on a dirty worktree` —
đúng với mọi requirement khi cây bẩn và không liên quan gì tới đường dẫn bị giả
mạo. Cây sạch thì cái cớ đó biến mất và test lộ ra điều nó lẽ ra phải bắt:
`audit_closure` kiểm tồn tại cho `implementation_refs` và `evidence_refs` nhưng
**bỏ qua** `test_ids`.

`test_ids` nay đi qua cùng `_missing_paths`. Trên ledger P3.4.3 thật với cây
sạch, thay đổi này cho **0 violation** — mọi file test được trích dẫn đều có
thật — nên verdict giữ nguyên `FAIL` và `release_blocked` giữ nguyên `true` vì
đúng những lý do cũ. Không kết luận nào của P3.4.3 bị đổi; cổng chỉ chặt hơn.

Ngoài mục đó, phiên này không đụng tới kết luận của MVP-T hay P3.5; cả hai giữ
nguyên trạng thái mà `SESSION-20260831-001` và `-002` ghi.

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
   `docs/roadmap/PHASE4_REVIEWER_GUIDE.md`. Packet đã sinh ở worktree sạch với
   22 artifact, `complete: true`, `verdict: null`. Digest và commit của nó nằm
   trong chính `evidence/phase4/review/review-packet.json`, và chỉ ở đó: chép
   digest sang tài liệu khác thì mỗi lần sinh lại packet sẽ để lại một bản sao
   sai ở nơi khác. Sinh lại bằng `python scripts/phase4_review_packet.py` và ký
   trên digest của bản có `worktree_clean: true`.

### Việc tiếp theo trong repo

Không còn package P4 nào làm được trên máy này. Hai mục còn lại đều cần thứ
không tồn tại ở đây: một GPU NVIDIA thật và một người kiểm tra không phải tác
giả.

`ROADMAP-P5-001` đã được viết trong phiên này
(`docs/roadmap/PHASE5_EXECUTION_PLAN.md`): scope, protocol khóa bằng hash, work
breakdown P5-00…P5-12, test matrix và cổng đóng. Ba điều nó khóa trước, đúng vì
P4 đã dạy:

1. Checkpoint **không** được chọn bằng total loss — `flow_velocity` có sàn không
   rút được, nên total trộn một hằng số vô nghĩa vào tín hiệu chọn model.
2. Protocol khóa bằng hash **trước** run đầu tiên. Khóa sau khi nhìn kết quả thì
   không còn là protocol.
3. `P5-11` không được bắt đầu trước khi `P4-11b` đóng: đặt một gate train CUDA
   lên một kiến trúc chưa từng chạy CUDA là đo hai ẩn số cùng lúc.

Held-out embodiment với hai tay là `n=1` mỗi chiều, và plan bắt mọi báo cáo phải
ghi như vậy thay vì viết thành "generalizes across embodiments".

### Điều kiện phiên sau phải kiểm trước khi tin trạng thái

- `python -m pytest -q` phải còn trả 0 với ít nhất 1263 test pass, **chạy trên
  worktree sạch**. Trên cây bẩn, ledger test của P3.4.3 pass vì lý do sai.
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
