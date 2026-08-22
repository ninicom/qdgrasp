---
document_id: ROADMAP-P1-001
document_type: plan
title: Kế hoạch thực thi Phase 1 — Core framework
version: 1.0.0
status: active
date: 2026-08-22
revises: none
related_plan: PLAN-V2
---

# Kế hoạch thực thi Phase 1 — Core framework

Tài liệu này chỉ diễn giải cách thực thi P1 đã được `PLAN.md` (M1) và
`docs/roadmap/PROJECT_PHASES.md` (P1) quy định. Nó không sửa, không mở rộng và
không đánh giá lại bất kỳ quyết định nào của các phase/chốt trước.

## 1. Phạm vi cố định

Từ `PROJECT_PHASES.md` (P1) và `PLAN.md` (M1):

- Public façade `QDGrasp` và CLI tối thiểu; không mở DSL tùy ý.
- YAML khai báo chọn preset/module có schema; unknown/dead key bị từ chối.
- Runner trên Lightning Fabric: deterministic seed, logging, callbacks,
  checkpoint/resume và result bundle.
- Dummy model/data để kiểm toàn bộ lifecycle trước khi có model thật.
- Chỉ PyTorch/TorchScript/ONNX; không telemetry, account, HUB hay SaaS hook.
- Device policy CPU/CUDA, AMP, EMA, deterministic resume, import không đổi cwd.

Gate chuyển phase: API/config round-trip, CPU smoke và CUDA dummy train-step.

## 2. Ràng buộc kế thừa từ P0 (không thay đổi)

- Package name/version `qdgrasp 0.1.0a1`; `scripts/check_phase0.py` khóa version.
- `qdgrasp/runtime.py` là nguồn duy nhất của device contract fail-closed.
- Cây legacy Ultralytics-derived trong `qdgrasp/` được giữ nguyên; P1 chỉ thêm
  module mới, không xóa và không sửa file legacy.
- Không thêm dependency mới: chỉ dùng torch, lightning, numpy, pydantic,
  PyYAML, safetensors, typer, rich đã khai báo ở P0.
- Mọi đường dẫn là relative path.

### Ánh xạ thư mục

`PLAN.md` §3 đặt tên `data/`, `models/` và `engine/`. Trong cây hiện tại
`qdgrasp/data/` và `qdgrasp/models/` vẫn thuộc legacy, còn
`qdgrasp/engine/__init__.py` đã rỗng. Vì P1 không được sửa legacy, code mới dùng:

| PLAN §3 | Đường dẫn P1 | Lý do |
| --- | --- | --- |
| `api/` | `qdgrasp/api/` | tên trống, dùng đúng PLAN |
| `config/` | `qdgrasp/config/` | tên trống, dùng đúng PLAN |
| `engine/` | `qdgrasp/engine/runner.py`, `callbacks.py`, `checkpoint.py`, `seeding.py` | `__init__.py` rỗng nên thêm file mới không đụng legacy |
| `export/` | `qdgrasp/export/` | tên trống, dùng đúng PLAN |
| `data/` + `models/` (dummy) | `qdgrasp/dummy/` | hai tên legacy đang chiếm chỗ; dummy là vật liệu tạm của P1 |
| preset YAML | `qdgrasp/presets/` | `qdgrasp/cfg/` là legacy |

Ánh xạ này là ghi chú triển khai, không phải đề nghị sửa `PLAN.md`. Khi legacy
được rút ở phase sau, code mới chuyển về đúng tên của `PLAN.md` §3.

## 3. Hạng mục công việc

| ID | Hạng mục | Output |
| --- | --- | --- |
| P1-01 | Config schema, registry allowlist, loader, requested/effective | `qdgrasp/config/` |
| P1-02 | `GraspResults` và façade `QDGrasp` | `qdgrasp/api/` |
| P1-03 | Seeding, callbacks, checkpoint bundle/resume, Fabric runner | `qdgrasp/engine/*.py` |
| P1-04 | Dummy model và dummy dataset | `qdgrasp/dummy/` |
| P1-05 | Export TorchScript/ONNX bundle | `qdgrasp/export/` |
| P1-06 | CLI `train/val/predict/export` | `qdgrasp/cli.py` |
| P1-07 | Preset YAML model/robot/data | `qdgrasp/presets/` |
| P1-08 | Test lifecycle CPU và config contract | `tests/` |
| P1-09 | Gate script CPU và CUDA smoke script | `scripts/check_phase1.py`, `scripts/phase1_cuda_smoke.py` |
| P1-10 | Session report và schema documentation | `docs/` |

## 4. Tiêu chí nghiệm thu

- Config round-trip: YAML → schema → serialize → YAML cho kết quả bằng nhau;
  unknown/dead key raise lỗi; schema version lạ bị từ chối.
- Requested vs effective config được lưu; CPU ép `amp=False` và ghi log tường minh.
- Public bundle không pickle module: safetensors + JSON metadata + hash.
- Resume tái lập optimizer, scheduler, scaler, RNG và global step; loss sau resume
  khớp bit-for-bit với run liên tục trên CPU.
- `GraspResults` đủ trường và `.cpu()/.to()/.numpy()/.save()/.summary()`.
- `import qdgrasp` không đổi `os.getcwd()`.
- TorchScript round-trip giữ số grasp, ordering, rotation, translation, named
  joints và score trong tolerance.
- CUDA dummy train-step chạy trên GPU NVIDIA thật; CPU fallback không được tính.

## 5. Giới hạn đã biết

Máy phát triển hiện tại là `torch 2.11.0+cpu`, `cuda_available=false`. Vì
`docs/decisions/0006-cuda-hardware-required.md` cấm CPU fallback làm bằng chứng
CUDA, hạng mục CUDA của gate P1 được giao cho repository notebook riêng
`ninicom/qdgrasp-cuda-kaggle` chạy `scripts/phase1_cuda_smoke.py`. Trước khi có
evidence đó, P1 không được ghi là hoàn tất.
