---
document_id: SESSION-20260901-003
document_type: session_report
title: Đóng mười chuỗi lỗi corrective và sửa lại suite quanh chúng
status: complete
date: 2026-09-01
session_id: SESSION-20260901-003
author: claude-agent
related_plan: PLAN-V2@4.6.0
revises: none
---

# SESSION-20260901-003 — R2–R7: đóng mười finding và sửa suite

## Mục tiêu phiên

- Tiếp nhận phần thi công đang dở trong worktree (semantics của `R2`–`R5`,
  `R7`), xác minh nó, và sửa lại toàn bộ test cũ đang đo semantics đã bị thay.
- Đóng những finding thực sự đã có regression evidence, và chỉ những finding đó.
- Thi công tiếp phần còn thiếu của `R3` (protocol view chưa đi vào public path)
  và toàn bộ `R6` (bundle identity và export).

## Việc đã hoàn tất

| ID | Kết quả | Output | Trạng thái |
|---|---|---|---|
| `S3-01` | Xác minh phần thi công đang dở bằng chính characterization suite; 24 test chuyển từ expected-fail sang pass | `qdgrasp/dataset/artifact.py`, `artifact_io.py`, `models/protocol.py`, `engine/*`, `mvp/*` | verified |
| `S3-02` | Sửa 26 test cũ đang đo semantics đã thay đổi | `tests/model_training/*`, `tests/test_dataset_*`, `tests/mvp/*` | complete |
| `S3-03` | Thêm fixture corpus có provenance tự nhất quán, tách "mở được" khỏi "là artifact đã phát hành" | `tests/conftest.py` | complete |
| `S3-04` | Đóng `COR-00`, `COR-01`, `COR-03`, `COR-05`, `COR-06`, `COR-07`, `COR-08`, `COR-11` trong registry | `qdgrasp/corrective/registry.py` | complete |
| `S3-05` | Nối `protocol_file` vào public loader; train view là protocol view | `qdgrasp/dataset/loader.py` | complete |
| `S3-06` | `qdgrasp/bundle/v2`: gate theo model config và preprocess, tách training/runtime robot | `qdgrasp/engine/checkpoint.py`, `api/*` | complete |
| `S3-07` | `EmbodimentBinding`: transfer là một đối tượng được khai báo, có protocol làm căn cứ | `qdgrasp/engine/compatibility.py` | complete |
| `S3-08` | `FlowExportAdapter` và tokenizer dạng dense; một trace đúng cho mọi cloud size đã khai báo | `qdgrasp/export/flow.py`, `models/tokenizer.py` | complete |
| `S3-09` | Đóng `COR-09`, `COR-10` | `qdgrasp/corrective/registry.py` | complete |
| `S3-10` | Ghi lại rằng packet Phase 4 nay phải bị từ chối, kèm lý do | `tests/model_flow/test_phase4_gate.py` | complete |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/kết quả | Tham chiếu |
|---|---|---|---|
| `E-01` | trạng thái bàn giao | 52 test hỏng trên tổng 1363 khi phần thi công dở được tiếp nhận | `scratchpad/handoff-suite.log` |
| `E-02` | protocol view trên public path | LEAP train: 28 sample / 7 object đúng protocol, 16 exclusion được đếm | `tests/corrective/test_cor02_protocol_view.py` |
| `E-03` | held-out embodiment | train view của `wonik_allegro` rỗng vì view từ chối, không phải vì thiếu shard | cùng file |
| `E-04` | bundle identity | `flow_steps` lệch một đơn vị làm load fail dù mọi tensor vừa khít | `tests/corrective/test_cor09_bundle_binding.py` |
| `E-05` | cross-embodiment | LEAP→Allegro bị từ chối khi không có protocol; có protocol thì result ghi cả hai hash | cùng file |
| `E-06` | export parity | eager/traced khớp tuyệt đối tại 1, 129, 200, 512, 1024 điểm trên cả hai active hand | `tests/corrective/test_cor10_flow_export.py` |
| `E-07` | giới hạn export được khai báo | cloud vượt `max_points` bị từ chối tại adapter, không rơi vào số học của tracer | cùng file |
| `E-08` | evidence cũ bị supersede | packet Phase 4 nay fail vì `flow.py`/`tokenizer.py` đã đổi so với lúc ký | `tests/model_flow/test_phase4_gate.py` |

## Kiểm tra đã chạy

| Test ID | Lệnh/phương pháp | Kết quả | Exit/đánh giá |
|---|---|---|---|
| `T-01` | `.venv/bin/pytest -q tests/corrective` | 62 pass, 3 xfail | pass |
| `T-02` | `.venv/bin/pytest -q tests/corrective --runxfail` | chỉ `COR-04` và `COR-12` còn tái hiện | expected |
| `T-03` | `.venv/bin/pytest -q` trên worktree sạch, sau khi commit toàn bộ thay đổi code | 1391 pass, 1 skip, 3 xfail, 93 subtest, không failure | pass |
| `T-04` | `.venv/bin/ruff check` trên các file đã sửa | không thêm finding mới | pass |
| `T-05` | `.venv/bin/python scripts/check_docs.py --root .` | hợp lệ | pass |

## Việc chưa hoàn tất

- `COR-04` (target-validity mask) và `COR-12` (bề mặt thực thi code trong wheel)
  còn mở. `COR-04` cần generator ghi `kinematics_valid`, `pose_target_valid`,
  `joint_target_valid`, `fk_target_valid` rồi loss chỉ hồi quy sample hợp lệ;
  phần dữ liệu của nó thuộc `R8`.
- `COR-02` có đủ code nhưng gate `G2` vẫn chờ `R8`: canonical audit còn fail vì
  source drift, và positive floor vẫn chưa đạt. Registry ghi điều đó ở trường
  `blocked_on` thay vì để "không còn test nào fail" bị đọc nhầm thành "gate đã
  đóng".
- `qdgrasp/models/data.py` không còn được sản phẩm dùng nhưng vẫn tồn tại như
  contract thứ hai; nó thuộc `R9`.
- Chưa regenerate DGN, chưa chạy lại Phase 4, chưa có CUDA/AMP gate mới, chưa có
  independent review. Không evidence release nào được tạo trong phiên này.
- ONNX parity chưa chạy được ở đây vì `onnxruntime` không có trong môi trường;
  TorchScript parity đã chạy đủ trên cả hai active hand.

## Sửa đổi phiên trước

Không. Phiên này thi công tiếp `SESSION-20260901-002` theo đúng dependency của
`PLAN.md` §9.11. Hai điều chỉnh trong chính công cụ của phiên trước được ghi rõ
ở đây vì chúng làm thay đổi cách đọc registry, không phải cách đọc một verdict
cũ:

- `satisfied_by` cho phép một phần của chuỗi còn mở trở thành regression test
  ngay khi nó được giao, thay vì tiếp tục bị đánh dấu là expected-fail.
- `blocked_on` buộc một finding còn mở mà không còn test nào fail phải nói nó
  đang chờ gì.

Bảng `PLANNED_SCHEMA_BUMPS` được sửa hai chỗ: manifest bump chuyển từ `COR-01`
sang `COR-04`, vì thứ làm shard cũ không đọc được là validity flag chứ không
phải việc hợp nhất reader; và hai hằng số MVP trỏ sang alias đang hoạt động.

## Bàn giao

1. `R8` là bước kế tiếp và nó mở khoá phần còn lại: regenerate DGN từ commit
   sạch với validity flag, rồi chạy lại Phase 4 và packet của nó.
2. Khi generator ghi validity flag, `COR-04` đóng bằng chính hai test đang có;
   không cần viết test mới.
3. Bundle cũ dưới `qdgrasp/bundle/v1` không đọc được nữa. Đó là chủ ý: chúng
   được sinh dưới joint parameterization và quality head khác.
4. Export chỉ được công bố kèm `max_points` và `token_capacity` trong metadata;
   một cloud lớn hơn phải bị chặn ở nơi người đọc thấy được, không phải trong
   runtime.
5. Không nới hard-stop, không đổi manifest để hợp thức hoá byte cũ.
