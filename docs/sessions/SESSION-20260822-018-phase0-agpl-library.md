---
document_id: SESSION-20260822-018
document_type: session_report
title: Hoàn tất Phase 0 AGPL library-first
status: draft
date: 2026-08-22
session_id: SESSION-20260822-018
author: codex-primary-agent
revises: SESSION-20260822-017
related_plan: PLAN-V2
revision_reason: Maintainer chốt tên QDGrasp, AGPL, library-first, Kaggle GPU repo riêng và loại RH56E2.
necessity: N3
impact: Thay repository/license strategy và bổ sung package plus external CUDA evidence trước khi đóng P0.
---

# SESSION-20260822-018 — Phase 0 AGPL library

## Mục tiêu phiên

Đóng Phase 0 bằng public package AGPL, provenance manifests, exclusion RH56E2 và
CUDA/cu128 evidence từ notebook repository tách biệt trên Kaggle.

## Việc đã hoàn tất

| ID | Kết quả | Artifact | Trạng thái |
| --- | --- | --- | --- |
| W-01 | Public package skeleton, runtime API và CLI | `pyproject.toml`, `qdgrasp/` | verified CPU/wheel |
| W-02 | AGPL/provenance pivot | PLAN 4.0, ADR-0007, NOTICE, THIRD_PARTY | docs pass |
| W-03 | RH56E2 active exclusion | robot lock, checker, archive sidecar | static pass |
| W-04 | Kaggle notebook repo riêng | `ninicom/qdgrasp-cuda-kaggle` | pending run |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256/tham chiếu |
| --- | --- | --- | --- |
| E-01 | PLAN 4.0 | `PLAN.md` | `b3618c00b9dd5a8ea0af092a3f6ad3c5e3169d3d23a97110d9431d7354f54be5` |
| E-02 | Package | `pyproject.toml` | version `0.1.0a1` |
| E-03 | CUDA evidence | external Kaggle output | pending |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| T-01 | build wheel/sdist + install wheel ngoài source tree | 0 | import/CLI pass |
| T-02 | `check_environment --profile cpu`; runtime pytest | 0 | pass; 3 tests |
| T-03 | docs validator + legacy unittest | 0 | pass; 69 files + 47 tests |
| T-04 | `scripts/check_phase0.py` | 0 | pass trước public push |
| T-05 | Kaggle cu128 run | pending | chưa chạy |

## Việc chưa hoàn tất

- Public push của library và notebook repository.
- Kaggle hardware run, evidence download/hash và environment lock update.
- Chuyển roadmap/session/revision từ in-progress/draft sang complete.

## Sửa đổi phiên trước

`REV-20260822-009` supersede pivot Apache ở REV-006 nhưng giữ nguyên CUDA
fail-closed của phiên 017.

## Bàn giao

Push library trước, pin exact commit trong notebook, chạy Kaggle và chỉ đóng P0
khi evidence xác nhận CUDA 12.8, GPU NVIDIA, AMP train step và MuJoCo forward.
