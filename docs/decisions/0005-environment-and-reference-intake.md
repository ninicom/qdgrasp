---
document_id: ADR-0005
document_type: decision
title: Khóa môi trường research và reference-on-demand
status: accepted
date: 2026-08-22
revises: none
supersedes: none
necessity: N2
impact: Khóa Python/PyTorch/CPU-CUDA profiles và dừng mở rộng baseline trừ khi có vấn đề kỹ thuật tái hiện được.
revision_reason: Baseline đã đủ đa dạng để bắt đầu triển khai; môi trường và quy trình tham khảo cần ổn định trước implementation.
---

# ADR-0005 — Environment freeze và reference-on-demand

## Quyết định

DexGrasp dùng Python 3.11, PyTorch 2.11.0, Lightning Fabric 2.6.5 và MuJoCo
3.12.0. CPU wheel và CUDA 12.8 wheel dùng cùng minor PyTorch để giữ parity.
Direct dependencies được khóa bằng `uv pip compile` trong `environments/`.

CPU là baseline bắt buộc cho CI và correctness. CUDA là baseline train/benchmark
nhưng chỉ được xác nhận sau smoke/parity test trên phần cứng NVIDIA. Python 3.14
system của host không được dùng làm project environment.

Bộ architecture reference hiện tại được xem là đủ. Reference mới chỉ được thêm
khi có issue kỹ thuật cụ thể và phải tuân thủ
`docs/governance/REFERENCE_INTAKE.md`.

## Hệ quả

- Không cài dependency của GraspDiffuser, DexDiffuser, CEDex hay GraspGenX vào
  core chỉ để chạy code tham chiếu.
- PyTorch3D, pointops, GraspQP, Isaac Gym và cuRobo không thuộc base environment.
- Mọi nâng Python/PyTorch/MuJoCo là thay đổi N2; đổi backend/license hoặc phá
  checkpoint/schema compatibility là N3.
- Paper result phải lưu environment lock hash, GPU/driver và simulator version.
