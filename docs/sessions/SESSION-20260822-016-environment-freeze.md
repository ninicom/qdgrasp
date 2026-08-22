---
document_id: SESSION-20260822-016
document_type: session_report
title: Khóa môi trường research và reference intake
status: complete
date: 2026-08-22
session_id: SESSION-20260822-016
author: codex-primary-agent
revises: SESSION-20260822-015
related_plan: PLAN-V2
revision_reason: Baseline đã đủ để triển khai; maintainer yêu cầu chốt môi trường/tài liệu và chỉ bổ sung reference khi gặp ca khó.
necessity: N2
impact: Khóa Python 3.11, PyTorch CPU/cu128, Lightning Fabric, MuJoCo và quy trình reference-on-demand.
---

# SESSION-20260822-016 — Environment freeze

## Mục tiêu phiên

Khóa một môi trường cài đặt tái lập được trước implementation và định nghĩa cách
bổ sung source tham chiếu khi có blocker cụ thể mà không làm trôi kiến trúc,
license hoặc dependency surface.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
| --- | --- | --- | --- |
| W-01 | Chọn Python 3.11, PyTorch 2.11, CPU/cu128 parity, Lightning 2.6.5 và MuJoCo 3.12 | `docs/installation/ENVIRONMENT.md` | accepted ADR-0005 |
| W-02 | Tạo direct requirement inputs và bốn transitive lock có hash | `environments/requirements/` | uv resolution pass |
| W-03 | Cài CPU lock vào `.venv` và chạy Torch/Fabric/MuJoCo smoke | `scripts/check_environment.py` | pass |
| W-04 | Ghi fingerprint và trạng thái CPU/CUDA | `environments/environment.lock.yaml` | hash verified |
| W-05 | Đóng baseline và quy định reference-on-demand | `docs/governance/REFERENCE_INTAKE.md` | documentation gate |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- | --- |
| E-01 | Plan 3.2 | `PLAN.md` | `b1b5e28b031ac2f0f5a820fcf90265aff9c4cbb784b36db14d295f7bce12dbd4` |
| E-02 | Environment manifest | `environments/environment.lock.yaml` | `1ac53f3441412c79aa8f716a004dbe66845ee2dc7c9e611a831d925adfd41242` |
| E-03 | CPU lock | `environments/requirements/cpu.lock` | `5d8c65b043eb39b2452f4765e54639acf6e859567198f2354d31213c0aa094e1` |
| E-04 | CUDA lock | `environments/requirements/cu128.lock` | `43536551e9ecbf4f525eb08f77567378a1531c2a24f1b7275a65751b0b95b4ef` |
| E-05 | Environment validator | `scripts/check_environment.py` | `ea80c179cc1f631033c0f044bee900b6a63f25580b2da359a7bc9495b25517ed` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| T-01 | `.venv/bin/python scripts/check_environment.py --profile cpu` | 0 | Python 3.11.15, Torch CPU op, Fabric import và MuJoCo forward pass |
| T-02 | `uv pip check --python .venv/bin/python` | 0 | 48 packages compatible |
| T-03 | `uv pip sync --dry-run --torch-backend cu128 environments/requirements/cu128.lock` | 0 | CUDA 12.8 profile resolves; hardware smoke chưa chạy |
| T-04 | `python3 scripts/check_docs.py --root .` | 0 | 65 documentation files pass |
| T-05 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | 0 | 47/47 pass |
| T-06 | `git diff --check` | 0 | pass |

## Việc chưa hoàn tất

- CUDA hardware smoke/parity chưa chạy vì host không có `nvidia-smi` hoặc GPU
  NVIDIA nhìn thấy được. Không được dùng trạng thái resolution thay cho benchmark
  evidence.
- Package implementation mới chưa bắt đầu; `.venv` chỉ xác minh research
  environment contract.

## Sửa đổi phiên trước

`REV-20260822-007` cập nhật PLAN từ 3.1 lên 3.2 để khóa environment và
reference-on-demand. Không thay đổi boundary DGN2, robot asset pins hoặc license
decision của phiên 015.

## Bàn giao

Implementation tiếp theo dùng `.venv` CPU cho correctness và một environment
cu128 riêng cho train. Khi gặp ca khó, mở issue tái hiện trước khi thêm entry mới
vào `architecture_references.lock.yaml`.
