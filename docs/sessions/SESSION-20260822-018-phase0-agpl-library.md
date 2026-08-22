---
document_id: SESSION-20260822-018
document_type: session_report
title: Hoàn tất Phase 0 AGPL library-first
status: complete
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
| W-04 | Kaggle notebook repo riêng | `ninicom/qdgrasp-cuda-kaggle` | public run 8 verified |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256/tham chiếu |
| --- | --- | --- | --- |
| E-01 | PLAN 4.0 | `PLAN.md` | `f1d4b9eb1692f229704593502afe088b73ae7f769367f7d9e6a515cc0cfe245c` |
| E-02 | Package | `pyproject.toml` | version `0.1.0a1` |
| E-03 | CUDA evidence | `https://github.com/ninicom/qdgrasp-cuda-kaggle/blob/main/evidence/run-008-pass/phase0_cuda_evidence.json` | `00e75d727393ff7dae8c2c893362a6df34ba5abd265cb346823058af86b69291` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| T-01 | build wheel/sdist + install wheel ngoài source tree | 0 | import/CLI pass |
| T-02 | `check_environment --profile cpu`; runtime pytest | 0 | pass; 3 tests |
| T-03 | docs validator + legacy unittest | 0 | pass; 71 files + 47 tests |
| T-04 | `scripts/check_phase0.py` | 0 | Phase 0 foundation pass |
| T-05 | Kaggle cu128 run 8 | 0 | Tesla T4; CUDA 12.8; AMP, resume và MuJoCo pass |

## Việc chưa hoàn tất

Không còn hạng mục mở trong gate P0. Independent license/release review vẫn là
gate của P7, không phải điều kiện để bắt đầu P1.

## Sửa đổi phiên trước

`REV-20260822-009` supersede pivot Apache ở REV-006 nhưng giữ nguyên CUDA
fail-closed của phiên 017.

## Bàn giao

P0 đã đóng. P1 tiếp tục từ public package commit đã được CUDA-verified; mọi thay
đổi dependency hoặc runtime contract phải regenerate lock và chạy lại CPU/CUDA
gate tương ứng.
