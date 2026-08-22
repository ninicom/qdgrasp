---
document_id: SESSION-20260822-014
document_type: session_report
title: Xóa 18 export_format method + split utils/export/engine.py
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-014
author: codex-primary-agent
revises: none
related_plan: PLAN-V2
---

# SESSION-20260822-014 — Trim `engine/exporter.py`, tách `utils/export/engine.py`

## Mục tiêu phiên

Phase 2, nhóm 4 (phần 2/2, bắt buộc) của kế hoạch M1 đã duyệt
(`/home/quyen/.claude/plans/sleepy-booping-map.md`): hoàn tất việc
`SESSION-20260822-013` đã bàn giao — sửa `engine/exporter.py` (rủi ro cao nhất
trong Phase 2, có khả năng phá đường ONNX/TorchScript đang giữ nếu làm ẩu),
tách `utils/export/engine.py`.

**Phát hiện cấu trúc quan trọng trước khi sửa** (khác với giả định ban đầu ở
`SESSION-20260822-013`'s "Bàn giao"): `exporter.py` không có dispatch sạch
theo method — nó có MỘT method `__call__` (~415 dòng) chứa hàng chục điều
kiện `if fmt == "X"`/`fmt in {...}` đan xen (device default, quantize
requirement, format alias, Hailo/RKNN-specific validation...), rồi dispatch
động `getattr(self, f"export_{fmt}")()` ở cuối. Cố sửa từng nhánh rải rác
này mà không chạy được code (không có torch/cv2 trong môi trường) đúng là
rủi ro "rushed" mà kế hoạch đã cảnh báo trước. Đã tham vấn advisor để chọn
chiến lược an toàn hơn: **chặn tại điểm validate format duy nhất**
(`export_formats()`), xóa nguyên method cho format bị loại (rủi ro thấp vì
mỗi method độc lập, tự chứa), chỉ sửa nhánh nội tuyến khi nhánh đó thật sự
BẮT BUỘC (import trỏ vào file đã xóa) — không đuổi theo toàn bộ 34 nhánh
`fmt ==` còn sót, để lại làm dead code có đếm số, không phải bỏ sót không
ghi nhận.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-00 | `grep -rn "export_formats"` xác nhận 3 caller khác ngoài `exporter.py` (`nn/autobackend.py`, `engine/model.py`, `utils/benchmarks.py`) — cả ba đều ĐỌC ĐỘNG bảng (`export_formats()["Argument"]`/`["Suffix"]`), không hardcode danh sách format nào → thu hẹp bảng an toàn, không cần sửa 3 file này | console | grep trực tiếp |
| W-01 | Rút gọn `export_formats()`'s bảng `x` từ 20 hàng xuống 3 hàng (PyTorch/TorchScript/ONNX); mọi format khác giờ fail ngay tại `if fmt not in fmts: raise ValueError(...)` (dòng ~379 trong `__call__`) — cơ chế lỗi rõ ràng có sẵn của chính codebase, không phải no-op im lặng | `dexgrasp/engine/exporter.py:144-166` | đọc trực tiếp + T-02 |
| W-02 | Rút gọn `EXPORT_ENVS` xuống 1 entry `"base"` (khớp Env duy nhất 3 format còn lại dùng); xác nhận dict này không được `grep` thấy tham chiếu ở BẤT KỲ đâu khác trong repo kể cả trước khi trim — không phải regression, là dead metadata từ upstream | `dexgrasp/engine/exporter.py:168-177` | grep xác nhận 0 caller |
| W-03 | Xóa nguyên 18 method `export_openvino`...`export_hailo` + `_add_tflite_metadata` (dòng cũ 954-1544, xác nhận ranh giới chính xác bằng `grep -n "^    def "` trước/sau) — mỗi method tự chứa (`@try_export` decorator riêng), xóa nguyên method là thao tác rủi ro thấp, khác hẳn sửa nhánh nội tuyến | `dexgrasp/engine/exporter.py` (1926→1100 dòng) | `grep -n "^    def "` trước/sau khớp danh sách dự kiến |
| W-04 | Xác nhận `_add_tflite_metadata` chỉ được gọi từ các method vừa xóa (`export_saved_model`, `export_edgetpu`) qua `grep -n "self\._add_tflite_metadata"` trước khi xóa — không phải dependency của `export_onnx`/`export_torchscript` | console | grep trực tiếp trước xóa |
| W-05 | Xóa nhánh RKNN nội tuyến trong `export_onnx()` (gọi `_NormalizeCoords` từ `engine.py`) — nhánh `if` khép kín, điều kiện rõ ràng (`self.args.format == "rknn"`), là caller cuối cùng của `_NormalizeCoords` nên xóa nhánh này là bắt buộc để tách `engine.py` an toàn | `dexgrasp/engine/exporter.py` (method `export_onnx`) | đọc trực tiếp |
| W-06 | Chạy checker sau W-01→W-05: phát sinh THÊM 3 lỗi mới trong chính `exporter.py` (import nội tuyến `utils.export.{imx,tensorflow,executorch}` trong đoạn "Update model" của `__call__`) — cùng hình dạng với nhánh RKNN (nhánh `if` khép kín, tự chứa, import trỏ file đã xóa) nên xóa luôn cả 3, không để sót trong phạm vi file đang sửa | `dexgrasp/engine/exporter.py` | `check_dexgrasp_imports.py` bắt đúng, xóa xong về 0 lỗi trong `exporter.py` |
| W-07 | Tách `utils/export/engine.py`: giữ `best_onnx_opset`/`torch2onnx` (logic ONNX export thật, `exporter.py` gọi trực tiếp), xóa `_NormalizeCoords`/`modelopt_quantize_onnx`/`onnx2engine` (TensorRT/RKNN-only, không còn caller nào sau W-05); dọn theo import không dùng (`LOGGER`, `check_requirements`, `check_tensorrt`, `check_version`, `IS_JETSON`, `is_dgx`, `is_jetson`, `json`) | `dexgrasp/utils/export/engine.py` (460→90 dòng) | `grep -rn "onnx2engine\|modelopt_quantize_onnx\|_NormalizeCoords" dexgrasp` rỗng sau xóa |
| W-08 | Sửa `utils/export/__init__.py`: bỏ `onnx2engine` khỏi import/`__all__` (dòng vừa thêm ở `SESSION-20260822-013`, giờ dangling vì W-07 xóa hàm) | `dexgrasp/utils/export/__init__.py` | `compileall` pass |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | Ghi chú |
|---|---|---|---|
| E-01 | `check_dexgrasp_imports.py` trước phiên | 26 lỗi/85 file | baseline `SESSION-20260822-013` |
| E-02 | `check_dexgrasp_imports.py` sau W-01→W-06 (exporter.py) | 7 lỗi/85 file | 0 lỗi từ `exporter.py`; 7 lỗi còn lại đều đã biết trước (xem dưới) |
| E-03 | `check_dexgrasp_imports.py` sau W-07/W-08 (engine.py split) | 7 lỗi/85 file | không đổi — `engine.py`/`__init__.py` không phải nguồn lỗi nào |
| E-04 | Phân loại 7 lỗi còn lại | — | `cfg/__init__.py:797` (handle_yolo_solutions, Phase 5) + `engine/model.py:572,642` (trackers/depth.calibrate, Phase 5+) + `nn/backends/{qnn,tensorflow}.py` (2 lỗi, đã ghi nhận ở `SESSION-20260822-013`, chờ nhóm `nn/backends/`) + `nn/modules/head.py:1652` + `nn/tasks.py:991` (2 lỗi, "coupled decision" đã ghi nhận ở `SESSION-20260822-010`, chờ nhóm 5) — KHÔNG có lỗi mới nào ngoài 7 lỗi đã biết |
| E-05 | Đếm nhánh `fmt ==`/`fmt in {` còn sót trong `exporter.py` sau phiên | 34 nhánh | `grep -c 'fmt == "\|fmt in {' dexgrasp/engine/exporter.py` — dead code vô hại (không thể chạm tới vì `export_formats()` chỉ còn 3 format), để lại có chủ đích, có đếm số, KHÔNG phải bỏ sót; dọn tiếp là việc tùy chọn ngoài phạm vi phiên này |
| E-06 | Khối `is_tf_format` chết không bị bắt bởi checker/compileall | `dexgrasp/engine/exporter.py` (`if is_tf_format: f, keras_model = self.export_saved_model()...`) | `is_tf_format = fmt in {"saved_model","pb","edgetpu"}` luôn `False` sau W-01 (3 format đó không còn hợp lệ) nên nhánh không bao giờ chạy — nhưng đây là truy cập thuộc tính (`self.export_saved_model`), không phải `import`, nên KHÔNG bị `check_dexgrasp_imports.py` hay `compileall` phát hiện dù method đã bị xóa ở W-03. An toàn về mặt hành vi (không thể chạm tới), ghi nhận rõ ràng thay vì im lặng bỏ qua |
| E-07 | 3 caller `export_formats()` ngoài `exporter.py` không bị ảnh hưởng | `dexgrasp/nn/autobackend.py`, `dexgrasp/engine/model.py`, `dexgrasp/utils/benchmarks.py` | đọc trực tiếp — cả 3 chỉ đọc động (`export_formats()["Argument"]` v.v.), không hardcode format nào; hành vi thu hẹp xuống 3 format là hệ quả đúng đắn, không phải side effect ẩn |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `python3 -m compileall dexgrasp/engine dexgrasp/utils/export -q` | 0 | pass |
| T-02 | `python3 scripts/check_dexgrasp_imports.py` | 1 | 7 lỗi/85 file — tất cả đã biết trước, xem E-04 |
| T-03 | `python3 scripts/check_references.py --lock-only` | 0 | pass |
| T-04 | `python3 scripts/check_references.py --source-root .references` | 0 | pass |
| T-05 | `python3 scripts/check_train_args.py --registry-only` | 0 | pass |
| T-06 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | pass |
| T-07 | `python3 scripts/check_docs.py --root .` | 0 | `55 file đã kiểm tra` |
| T-08 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | 0 | 47/47 pass |
| T-09 | `bash -n .githooks/* scripts/git/*.sh scripts/release_gate.sh` | 0 | pass |
| T-10 | `git diff --check` | 0 | pass |

**Giới hạn xác minh (nói rõ, không mập mờ):** tất cả kiểm tra trên đều là mức
"structural" — cú pháp hợp lệ, mọi import nội bộ resolve được. Đây KHÔNG phải
bằng chứng đường ONNX/TorchScript export chạy đúng — không có torch/cv2
trong môi trường này, không thể `import dexgrasp` hay chạy `model.export()`
thật. Gọi đúng tên: "đường import ONNX/TorchScript nguyên vẹn về mặt cấu
trúc, rung 1" — không phải "đã xác minh đường ONNX hoạt động."

## Việc chưa hoàn tất

- 34 nhánh `fmt == "X"`/`fmt in {...}` rải rác trong `__call__` (device
  default, quantize requirement, Hailo/RKNN/Ascend-specific validation...)
  vẫn còn nguyên trong file — dead code vô hại (không thể chạm tới), dọn dẹp
  là việc tùy chọn, KHÔNG bắt buộc cho M1, để lại có đếm số (E-05).
- Khối `is_tf_format` (dòng ~415 trong `__call__`) vẫn gọi
  `self.export_saved_model()`/`export_pb()`/`export_edgetpu()` — 3 method đã
  xóa — nhưng không bao giờ chạy được (E-06). Không sửa trong phiên này vì
  không bắt buộc và không có công cụ nào bắt được lỗi này để xác nhận việc
  sửa đúng; để nguyên có ghi nhận rõ hơn là sửa mà không xác minh được.
- 7 lỗi còn lại của checker — phân bổ đúng các nhóm kế hoạch tiếp theo (Phase
  5 cho `cfg/__init__.py`/`model.py`; nhóm `nn/backends/` cho
  `qnn.py`/`tensorflow.py`; nhóm 5 "coupled decision" cho
  `nn/tasks.py`/`head.py`) — không phải việc bỏ sót của phiên này.
- Import ONNX/TorchScript CHƯA được xác minh chạy thật (torch chưa cài) —
  giới hạn đã nêu ở trên, thuộc Phase 4/rung 2 của kế hoạch.

## Sửa đổi phiên trước

Không.

## Bàn giao

Phase 2 nhóm 4 (export/backend trim) đã xong hoàn toàn. Bước hợp lệ tiếp
theo theo kế hoạch: Phase 2 nhóm 5 — "coupled decision" `nn/tasks.py` (xóa
11 class YOLO-task, giữ `BaseModel`/`Ensemble`/`_SafeLoad`/`torch_safe_load`/
`parse_model`/`guess_model_*`), xử lý cùng lúc 2 lỗi đã biết
`nn/modules/head.py:1652`/`nn/tasks.py:991`. Sau đó Phase 2 nhóm 6 — trim
`nn/backends/` (giải quyết cùng lúc `nn/backends/{qnn,tensorflow}.py`) và
`data/*.py`.
