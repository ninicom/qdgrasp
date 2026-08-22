---
document_id: SESSION-20260822-009
document_type: session_report
title: Phân loại nn/backends/ và data/*.py cho scope-trim M1 (Phase 1 kế hoạch)
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-009
author: codex-primary-agent
revises: none
related_plan: PLAN-V2
---

# SESSION-20260822-009 — Phân loại nn/backends/ và data/*.py

## Mục tiêu phiên

Phase 1 của kế hoạch M1 đã duyệt (`/home/quyen/.claude/plans/sleepy-booping-map.md`):
phân loại phần còn lại chưa được investigate kỹ — `nn/backends/`, `data/*.py`,
xác nhận `utils/events.py` vào nhóm HUB/Platform, và audit import-time side
effect. Phiên này CHỈ đọc và ghi nhận bằng chứng (grep/đọc trực tiếp), KHÔNG
sửa code — output là quyết định cho Phase 2 thực thi.

## Việc đã hoàn tất

### 1. `dexgrasp/nn/backends/` (20 file, dispatch qua `AutoBackend._BACKEND_MAP`)

Xác nhận cấu trúc: MỘT dict dispatch duy nhất (`nn/autobackend.py:147-170`),
MỘT block import duy nhất (`nn/autobackend.py:16-34`) — khác hẳn
`utils/export/`'s ~20 import rải rác trong từng method, nên trim
`nn/backends/` đơn giản hơn nhiều (xóa file + xóa dòng import + xóa entry
dict tương ứng, không cần sửa nhiều method riêng lẻ).

| File | Verdict | Bằng chứng |
|---|---|---|
| `base.py` | **KEEP** | `BaseBackend(ABC)` — lớp trừu tượng nền, mọi backend khác kế thừa |
| `pytorch.py` | **KEEP** | `PyTorchBackend`, `TorchScriptBackend` — khớp định dạng giữ lại theo PLAN §4/M4 |
| `onnx.py` | **KEEP một phần** | `ONNXBackend` (giữ) + `ONNXIMXBackend(ONNXBackend)` (Sony IMX500-specific, cùng dạng "mixed-purpose file" như `utils/export/engine.py` — tách khi thực thi, không xóa cả file) |
| `ascend.py`, `axelera.py`, `coreml.py`, `deepx.py`, `executorch.py`, `hailo.py`, `litert.py`, `mnn.py`, `ncnn.py`, `openvino.py`, `paddle.py`, `qnn.py`, `rknn.py`, `tensorflow.py`, `tensorrt.py`, `triton.py` (16 file) | **REMOVE** | Mỗi file một `XBackend(BaseBackend)` cho một hardware/format cụ thể, tất cả nằm trong danh sách loại trừ PLAN §6 ("TensorRT/OpenVINO/CoreML và toàn bộ exporter của YOLO") |

`AutoBackend._BACKEND_MAP` giữ lại: `"pt"→PyTorchBackend`,
`"torchscript"→TorchScriptBackend`, `"onnx"→ONNXBackend`,
`"dnn"→ONNXBackend` (đặc biệt hóa, dùng chung class). 17 entry còn lại xóa
cùng import tương ứng.

### 2. `dexgrasp/data/*.py` (11 file) đối chiếu call site thật trong engine/

`grep` xác nhận các hàm sau ĐƯỢC dùng thật bởi engine spine — không thể xóa
file chứa chúng mà không sửa `predictor.py`/`trainer.py`/`exporter.py`/`validator.py`:

- `engine/predictor.py:53-54` → `from ultralytics.data import load_inference_source` (từ `data/__init__.py`/`data/loaders.py`), `from ultralytics.data.augment import LetterBox` (`data/augment.py`).
- `engine/trainer.py:29`, `engine/validator.py:42` → `check_cls_dataset, check_det_dataset, convert_ndjson_to_yolo_if_needed` (`data/utils.py`).
- `engine/exporter.py:81-83` → `build_dataloader, build_yolo_dataset` (`data/__init__.py`/`data/build.py`), `ClassificationDataset` (`data/dataset.py`), `check_cls_dataset, check_det_dataset` (`data/utils.py`).

| File | Verdict | Bằng chứng |
|---|---|---|
| `augment.py`, `base.py`, `build.py`, `dataset.py`, `loaders.py`, `utils.py` | **GIỮ, nhưng gắn nhãn "kế thừa, chưa dùng cho grasp tới M2"** | Được import thật bởi engine spine (danh sách trên); chưa có pipeline điểm-đám-mây nào thay thế — đó là scope M2 ("Depth-to-point-cloud preprocessing thuần NumPy/Torch"), không phải lỗ hổng cần đóng ngay |
| `split_dota.py`, `split.py`, `converter.py` | **REMOVE** | `grep` toàn `dexgrasp/` xác nhận KHÔNG có nơi nào khác import các file này ngoài chính chúng |
| `annotator.py` | **REMOVE** | Không ai import; bản thân nó `from ultralytics import SAM, YOLO` (dòng 7) — phụ thuộc hai model family đã/sắp bị xóa, tự nó cũng hỏng |
| `scripts/` (4 shell script: `download_weights.sh`, `get_coco.sh`, `get_coco128.sh`, `get_imagenet.sh`) | **REMOVE** | Tải weight/dataset YOLO — không liên quan grasp. Có 1 call site còn sót: `data/utils.py:683` gọi `get_imagenet.sh` qua `subprocess.run` bên trong logic xử lý dataset đặc biệt tên "imagenet" — vì `utils.py` được GIỮ, dòng này sẽ thành dangling reference sau khi xóa `scripts/`; **không sửa `utils.py` ở phiên này** (nằm ngoài phạm vi "chỉ phân loại"), ghi lại để commit xóa `data/scripts/` xử lý luôn dòng 683 hoặc để lại lỗi rõ ràng (không silent-fail) — quyết định cụ thể thuộc về commit thực thi |

### 3. `utils/events.py` — xác nhận thuộc nhóm HUB/Platform

`utils/callbacks/base.py:185,187` — `from ultralytics.utils.events import
callbacks as events_cb` rồi `callbacks_list = [events_cb]`, **không có gate**
theo loại instance (Trainer/Validator/Predictor/Exporter đều dính), khác hẳn
8 integration khác vốn chỉ gắn cho `Trainer`. Xác nhận đúng như phát hiện
trước đó của investigation ban đầu — bổ sung vào nhóm xóa cùng
`utils/uploads.py`/`utils/callbacks/platform.py`.

### 4. Audit side-effect lúc import (đã có từ investigation trước, không đọc lại)

`utils/__init__.py`: `ONLINE = is_online()` (DNS probe tới `one.one.one.one`/
`dns.google` lúc import) và `SETTINGS = SettingsManager()` (ghi
`settings.json` vào `USER_CONFIG_DIR` nếu chưa có) — cả hai chạy ở
module-import time, không đổi cwd nhưng là side-effect I/O/network không nên
xảy ra âm thầm. Để nguyên phạm vi quyết định cho Phase 6 (`import không đổi
cwd` + side-effect hardening), không xử lý ở phiên phân loại này.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | Ghi chú |
|---|---|---|---|
| E-01 | grep dispatch table | `dexgrasp/nn/autobackend.py:16-34,147-170` | 1 import block, 1 dict — xác nhận cấu trúc đơn giản hơn `utils/export/` |
| E-02 | grep data/ call sites | `dexgrasp/engine/{predictor,trainer,exporter,validator}.py` | liệt kê đầy đủ ở mục 2 |
| E-03 | grep data/ orphan check | toàn `dexgrasp/` | `split_dota.py`/`split.py`/`converter.py`/`annotator.py` không có caller ngoài chính chúng |
| E-04 | grep events.py wiring | `dexgrasp/utils/callbacks/base.py:185-187` | không có gate theo instance type |
| E-05 | kế hoạch đã duyệt | `/home/quyen/.claude/plans/sleepy-booping-map.md` | Phase 1 |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `python3 scripts/check_docs.py --root .` | 0 | pass (session report mới) |
| T-02 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | 0 | 47/47 pass, không hồi quy (phiên này không sửa code) |

Không chạy `check_dexgrasp_imports.py` với kỳ vọng đổi số — phiên này không
sửa `dexgrasp/`, số lỗi vẫn giữ nguyên 10/154 file như baseline
`SESSION-20260822-008`.

## Việc chưa hoàn tất

- Chưa xóa file nào — đây là investigation, thực thi thuộc Phase 2 kế tiếp.
- Quyết định cụ thể cho dòng `data/utils.py:683` (dangling `get_imagenet.sh`
  reference sau khi xóa `data/scripts/`) chưa chốt — để lại cho commit xóa
  `data/scripts/` xử lý.
- Import-time side effect (`is_online()`, `SettingsManager`) chưa xử lý —
  thuộc Phase 6.

## Sửa đổi phiên trước

Không.

## Bàn giao

Đủ căn cứ để Phase 2 thực thi các nhóm xóa đã sẵn sàng theo đúng thứ tự kế
hoạch: `models/{sam,fastsam,nas,rtdetr,llm.py,utils}` trước, rồi callback
integrations, rồi HUB/Platform telemetry (`uploads.py`, `callbacks/platform.py`,
`utils/events.py`), rồi `utils/export/` + `nn/backends/` (cùng cấu trúc, có
thể làm liền nhau), rồi `data/{split_dota,split,converter,annotator}.py` +
`data/scripts/`. `nn/tasks.py` (quyết định "coupled" với `exporter.py`) vẫn
là commit rủi ro cao nhất, làm riêng và cẩn thận nhất trong Phase 2.
