---
document_id: ADR-0006
document_type: decision
title: CUDA workload bắt buộc chạy trên GPU NVIDIA thật
status: accepted
date: 2026-08-22
revises: ADR-0005
supersedes: none
necessity: N2
impact: Loại bỏ CPU fallback khỏi train, AMP, distributed, CUDA parity và GPU benchmark; CUDA gate phải có hardware evidence.
revision_reason: Maintainer chốt phần CUDA phải được thực hiện trên CUDA, không dùng resolution hoặc CPU execution thay thế.
---

# ADR-0006 — CUDA phải chạy trên CUDA

## Quyết định

Mọi workload, metric hoặc kết quả được gắn nhãn CUDA phải chạy trên GPU NVIDIA
thật trong profile `torch==2.11.0+cu128`. Phạm vi bắt buộc gồm:

- model train-step và full training;
- FP32/AMP, distributed và resume;
- full evaluation, CPU/CUDA parity và GPU memory/latency benchmark;
- checkpoint được dùng cho model zoo hoặc paper.

CPU chỉ dùng cho CI, schema, FK, unit tests, numerical correctness reference và
inference/export smoke. Khi người dùng yêu cầu CUDA mà GPU/driver/VRAM không đáp
ứng, chương trình phải fail sớm; không tự chuyển sang CPU.

## Bằng chứng bắt buộc

CUDA gate cần lưu GPU model, driver, `torch.__version__`,
`torch.version.cuda`, device capability, environment-lock hash, model/data/robot
hash và raw log. `uv sync --dry-run`, CUDA wheel resolution hoặc mock device
không phải hardware evidence.

## Trạng thái hiện tại

Profile cu128 đã resolve nhưng chưa hardware-verified vì host hiện tại không có
`nvidia-smi` hoặc GPU NVIDIA nhìn thấy được. Vì vậy P0 vẫn `in progress` ở CUDA
gate và chưa có CUDA benchmark claim.

## Hệ quả

- Không đóng P0/P1 CUDA gates hoặc P4–P7 model gates bằng CPU-only evidence.
- CI CPU vẫn bắt buộc nhưng không thay thế GPU CI/runner.
- Nếu cần backend ROCm/MPS/XPU, phải có profile và ADR riêng; không gọi chúng là
  CUDA.
