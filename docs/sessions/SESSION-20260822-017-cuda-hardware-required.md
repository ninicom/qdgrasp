---
document_id: SESSION-20260822-017
document_type: session_report
title: Chốt CUDA hardware execution
status: complete
date: 2026-08-22
session_id: SESSION-20260822-017
author: codex-primary-agent
revises: SESSION-20260822-016
related_plan: PLAN-V2
revision_reason: Maintainer yêu cầu mọi phần CUDA phải thực hiện trên CUDA thật.
necessity: N2
impact: CPU fallback bị cấm cho train/AMP/distributed/parity/benchmark; CUDA gate cần NVIDIA hardware evidence.
---

# SESSION-20260822-017 — CUDA hardware required

## Mục tiêu phiên

Chuyển yêu cầu CUDA từ lock có thể resolve thành execution gate bắt buộc trên
GPU NVIDIA thật, đồng thời giữ đúng trạng thái chưa xác minh của host hiện tại.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
| --- | --- | --- | --- |
| W-01 | Ghi quyết định fail-closed, không CPU fallback | ADR-0006 | accepted |
| W-02 | Cập nhật environment protocol và fingerprint | `docs/installation/ENVIRONMENT.md`, `environments/environment.lock.yaml` | docs/hash pass |
| W-03 | Đưa CUDA hardware gate vào PLAN và roadmap | `PLAN.md`, `docs/roadmap/PROJECT_PHASES.md` | docs gate pass |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- | --- |
| E-01 | PLAN 3.3 | `PLAN.md` | `b9c74d6b47adc63813912bdf734a2589dce54fb481c2ab9b119cfaeffd20ec18` |
| E-02 | CUDA decision | `docs/decisions/0006-cuda-hardware-required.md` | `25525b4f1fe931d33e1a1855fa54adde3a1f3db2501293fe5148a06c17def837` |
| E-03 | Environment fingerprint | `environments/environment.lock.yaml` | `0af971e4e0d15dd6b2cdcf0472b7383d500bc59f5b8197bd92f887c0a8cc5db9` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| T-01 | `python3 scripts/check_docs.py --root .` | 0 | 66 files pass trước khi thêm session/revision |
| T-02 | `git diff --check` | 0 | pass |
| T-03 | CUDA hardware validator | not_run | Host không có GPU NVIDIA; gate giữ chưa pass |

## Việc chưa hoàn tất

- Chưa chạy CUDA hardware smoke, train-step, AMP, resume, parity hoặc benchmark.
- P0 giữ `in progress`; không có CUDA claim được tạo trong phiên này.

## Sửa đổi phiên trước

`REV-20260822-008` làm chặt environment contract của phiên 016. CPU evidence cũ
vẫn đúng nhưng không thể đại diện cho CUDA.

## Bàn giao

Trên máy NVIDIA, cài `cu128.lock`, chạy validator CUDA và lưu GPU/driver/device,
environment/model/data/robot hashes trước khi bắt đầu model training.
