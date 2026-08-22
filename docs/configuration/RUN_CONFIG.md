---
document_id: QDGRASP-CONFIG-V1
document_type: registry
title: Registry schema cấu hình QDGrasp v1
version: 1.0.0
status: active
date: 2026-08-22
revises: none
related_plan: PLAN-V2
---

# Registry schema cấu hình QDGrasp v1

Tài liệu này mô tả contract cấu hình do Phase 1 tạo ra trong `qdgrasp/config/`.
Nó độc lập với `TRAIN_ARGUMENTS.md`: tài liệu kia khóa surface Ultralytics đã pin
để audit, còn tài liệu này là schema riêng của QDGrasp. Không key nào ở đây được
sao chép từ default/grammar upstream.

## 1. Nguyên tắc

- Mỗi document YAML bắt buộc có trường `schema` mang version tường minh.
- Unknown key và dead key là lỗi (`extra="forbid"`), không có alias layer.
- Parser dùng `yaml.safe_load` cộng registry allowlist theo tên. Không `eval`,
  không `globals()` và không import module theo nội dung file.
- Document đã validate là immutable và có `content_hash()` SHA-256 ổn định; hash
  này đi vào public bundle, export sidecar và `GraspResults`.
- Giá trị người dùng yêu cầu (`requested`) và giá trị thực thi (`effective`)
  được lưu tách biệt cùng danh sách `adjustments`.

## 2. Document `qdgrasp/model/v1`

| Key | Type | Default | Ràng buộc |
| --- | --- | --- | --- |
| `schema` | str | bắt buộc | phải là `qdgrasp/model/v1` |
| `name` | str | bắt buộc | tên hiển thị của model |
| `type` | str | bắt buộc | phải nằm trong model registry |
| `params` | map[str, int\|float\|bool\|str] | `{}` | builder từ chối param lạ |

## 3. Document `qdgrasp/robot/v1`

| Key | Type | Default | Ràng buộc |
| --- | --- | --- | --- |
| `schema` | str | bắt buộc | phải là `qdgrasp/robot/v1` |
| `name` | str | bắt buộc | tên profile |
| `palm_link` | str | bắt buộc | link gốc của palm pose |
| `frame` | str | `palm` | frame của translation/rotation |
| `joints` | list[str] | bắt buộc | không rỗng, không trùng, có thứ tự |
| `joint_limits` | map[str, [float, float]] | bắt buộc | phủ đúng tập `joints`, hữu hạn, `lower < upper` |

Phase 1 chỉ khóa named-joint contract. URDF/MJCF, mesh resolver và FK thuộc
Phase 2 và sẽ mở rộng schema này bằng version mới, không sửa `v1` tại chỗ.

## 4. Document `qdgrasp/data/v1`

| Key | Type | Default | Ràng buộc |
| --- | --- | --- | --- |
| `schema` | str | bắt buộc | phải là `qdgrasp/data/v1` |
| `name` | str | bắt buộc | tên dataset |
| `type` | str | bắt buộc | phải nằm trong dataset registry |
| `params` | map[str, int\|float\|bool\|str] | `{}` | builder từ chối param lạ |

## 5. Document `qdgrasp/run/v1`

| Key | Type | Default | Range | Device policy |
| --- | --- | --- | --- | --- |
| `schema` | str | `qdgrasp/run/v1` | enum một giá trị | — |
| `device` | str | `cpu` | `cpu`, `cuda`, `cuda:<index>` | CUDA fail-closed, không fallback |
| `amp` | bool | `false` | — | CPU ép về `false` và ghi adjustment |
| `seed` | int | `0` | `[0, 2^32)` | áp cho Python/NumPy/torch/CUDA |
| `deterministic` | bool | `true` | — | bật deterministic algorithms |
| `max_steps` | int | `100` | `>= 1` | độ dài schedule của toàn run |
| `stop_after_steps` | int \| null | `null` | `>= 1` | ngân sách step của riêng phiên chạy |
| `val_interval` | int | `0` | `>= 0` | `0` nghĩa là chỉ val ở cuối |
| `batch_size` | int | `4` | `>= 1` | — |
| `learning_rate` | float | `0.001` | — | — |
| `ema_decay` | float | `0.0` | `[0.0, 1.0)` | `0.0` tắt EMA |
| `workers` | int | `0` | `>= 0` | clamp theo số CPU và ghi adjustment |
| `grad_clip` | float | `0.0` | `>= 0.0` | `0.0` tắt clipping |
| `project_dir` | str | `runs` | relative path | cấm đường dẫn tuyệt đối |
| `run_name` | str | `phase1` | — | thư mục con của `project_dir` |
| `resume` | str \| null | `null` | file hoặc run directory | thiếu artifact là lỗi |

`max_steps` là độ dài schedule; `stop_after_steps` là số step tối đa chạy trong
phiên hiện tại. Tách hai khái niệm này cho phép dừng giữa chừng rồi resume mà
scheduler vẫn giữ đúng horizon ban đầu.

## 6. Artifact schema

| Schema | Nơi dùng | Nội dung |
| --- | --- | --- |
| `qdgrasp/bundle/v1` | `bundle.json` | model/robot config, preprocess, data manifest, hash |
| `qdgrasp/resume/v1` | `resume.pt` | optimizer, scheduler, scaler/stream, EMA, RNG, global step |
| `qdgrasp/results/v1` | `results.json` | step, loss, metrics, runtime, artifact path, hash |
| `qdgrasp/export/v1` | sidecar `.json` | format, artifact hash, joint names, round-trip deviation |
| `qdgrasp/evidence/phase1-cuda/v1` | evidence JSON | environment và kết quả CUDA smoke |

Public bundle chỉ chứa `weights.safetensors` và `bundle.json`; không pickle
module. Resume state là artifact riêng và không thuộc public release.

## 7. Migration

Version mới được thêm bằng schema identifier mới cùng test round-trip riêng.
Không giữ alias để mô phỏng cú pháp của framework khác và không sửa nghĩa của
`v1` sau khi đã phát hành.
