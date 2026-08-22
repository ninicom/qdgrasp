---
document_id: SESSION-20260822-019
document_type: session_report
title: Phase 1 core framework trên CPU
status: complete
date: 2026-08-22
session_id: SESSION-20260822-019
author: claude-primary-agent
revises: none
related_plan: PLAN-V2
---

# SESSION-20260822-019 — Phase 1 core framework

## Mục tiêu phiên

- Lập kế hoạch thực thi Phase 1 theo `PLAN.md` M1 và `PROJECT_PHASES.md` P1, không
  sửa bất kỳ chốt nào của phase trước.
- Xây façade, CLI, config schema, Fabric runner, checkpoint contract, dummy
  model/data và export path.
- Tiêu chí dừng: phần CPU của gate P1 pass và phần CUDA có script chạy được kèm
  điều kiện gỡ chặn rõ ràng.

## Việc đã hoàn tất

| ID | Kết quả | Output | Trạng thái |
| --- | --- | --- | --- |
| W-01 | Kế hoạch thực thi P1 và ánh xạ thư mục tránh legacy | `docs/roadmap/PHASE1_EXECUTION_PLAN.md` | Hoàn tất |
| W-02 | Config layer: schema có version, allowlist registry, loader `safe_load`, requested/effective policy | `qdgrasp/config/` | Hoàn tất |
| W-03 | Façade `QDGrasp` và `GraspResults` đủ trường/phương thức public | `qdgrasp/api/` | Hoàn tất |
| W-04 | Engine: seeding, callbacks, EMA, deterministic batch stream, checkpoint bundle/resume, Fabric runner | `qdgrasp/engine/*.py` | Hoàn tất |
| W-05 | Dummy model/dataset và ba preset YAML | `qdgrasp/dummy/`, `qdgrasp/presets/` | Hoàn tất |
| W-06 | Export TorchScript và ONNX kèm sidecar và round-trip verification | `qdgrasp/export/` | Hoàn tất |
| W-07 | CLI subcommand `train/val/predict/export` cạnh `env` | `qdgrasp/cli.py` | Hoàn tất |
| W-08 | Registry schema cấu hình QDGrasp v1 | `docs/configuration/RUN_CONFIG.md` | Hoàn tất |
| W-09 | Gate script CPU và script CUDA smoke | `scripts/check_phase1.py`, `scripts/phase1_cuda_smoke.py` | Hoàn tất (CPU); script CUDA chưa chạy |
| W-10 | Bộ test lifecycle/contract | `tests/` (7 file mới) | Hoàn tất |

Ràng buộc đã giữ: không sửa `PLAN.md`, `PROJECT_PHASES.md`, ADR, `runtime.py`,
lock môi trường, registry train-argument và toàn bộ cây legacy Ultralytics-derived.
Không thêm dependency mới; version package vẫn là `0.1.0a1`.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- | --- |
| E-01 | Kế hoạch phase | `docs/roadmap/PHASE1_EXECUTION_PLAN.md` | `9ab87fe4a938b27ee9a1e55b2b133c61e9e1a2b7fa46384f70c1fbb351d31f59` |
| E-02 | Config registry | `docs/configuration/RUN_CONFIG.md` | `5d68b4c8fc76104d8da199dcb74655b6f972fc687c70628826a1fa201abe3562` |
| E-03 | CPU gate script | `scripts/check_phase1.py` | `e987bbbdf5f57d2bce1b0d33412454439a5dc9c01c59983998dfd5f27bc782d3` |
| E-04 | CUDA gate script | `scripts/phase1_cuda_smoke.py` | `ff9da25b9008af8c24c7f034197595fdbbda5fbcbdfb89b4e8a1046fb54ecaa3` |
| E-05 | Runner | `qdgrasp/engine/runner.py` | `9899e747f2184dd034a7cab95dc087a8e51886d421317ff0068ace2fb55946ab` |
| E-06 | Wheel dựng lại từ cây phiên này | `qdgrasp-0.1.0a1-py3-none-any.whl` | `228d5c0572323001a2be4dd5db4e5cefed572f133e77cef3cb9c0e78cd81dce2` |
| E-07 | PLAN không đổi trong phiên | `PLAN.md` | `f1d4b9eb1692f229704593502afe088b73ae7f769367f7d9e6a515cc0cfe245c` |
| E-08 | CUDA evidence Phase 1 (Kaggle T4, cu128) | `ninicom/qdgrasp-cuda-kaggle` `evidence/phase1-run-002-pass/phase1_cuda_evidence.json` | `e2285f6005fa9c48535d199f5a8e1b3b2e30421e559846c8593fbcecadc329ca` |
| E-09 | Commit library được gate CUDA xác minh | `github.com/ninicom/qdgrasp` | `67a40724d3521d1447a5234974a36f70504f7f18` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| T-01 | `git diff --check` | 0 | không có lỗi whitespace |
| T-02 | `python3 scripts/check_references.py --lock-only` | 0 | immutable lock pass |
| T-03 | `python3 scripts/check_references.py --source-root .references` | 0 | 2 required checkouts sạch |
| T-04 | `python3 scripts/check_train_args.py --registry-only` | 0 | 127 tên, zero missing/extra |
| T-05 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | khớp source `329682a29d27` |
| T-06 | `python3 scripts/check_docs.py --root .` | 0 | 72 file pass |
| T-07 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | 0 | 47 tests OK |
| T-08 | `python3 scripts/check_phase0.py` | 0 | Phase 0 foundation vẫn PASS |
| T-09 | `python3 scripts/check_phase1.py` | 0 | config round-trip, unknown-key reject, CPU lifecycle, gradient coverage, resume bit-exact, TorchScript deviation `0.0` |
| T-10 | `python3 -m pytest tests/ -q` | 0 | 126 passed, 1 skipped (ONNX extra vắng trong venv dev) |
| T-13 | Kaggle kernel `qdgrasp-phase-1-cuda-framework-gate` version 1 (T4, cu128) | 1 | FAIL — bước optimizer đầu tiên sau resume trộn CPU state với CUDA gradient |
| T-14 | Cùng kernel version 2 ở commit `67a4072` | 0 | PASS — FP32 và AMP train 8 step, resume bit-exact, joints trong limit, TorchScript deviation `0.0`, CPU/CUDA FP32 parity `2.98e-08` |
| T-11 | `uv build` rồi cài wheel vào venv sạch ngoài source tree, chạy `qdgrasp train/export` | 0 | import, preset packaged và CLI pass |
| T-12 | `pytest tests/test_export.py` trong venv có `export-cpu.lock` | 0 | 5 passed; ONNX Runtime CPU deviation lớn nhất `2.68e-07` |

Chi tiết nghiệm thu T-09: `max_steps=8` liên tục cho cùng dãy loss với
`stop_after_steps=4` rồi `--resume`, so khớp bit-for-bit; joints dự đoán nằm trong
limit khai báo; `amp=True` trên CPU bị ép về `false` kèm adjustment được ghi lại;
mọi trainable parameter đều nhận gradient (`parameters_without_gradient: []`).

Gradient coverage lúc đầu phát hiện `rotation_head` không nhận gradient vì
`training_step` chưa giám sát rotation. Đã sửa trong cùng phiên: dummy sample có
thêm `target_rotation` và loss có thành phần rotation; test và gate script kiểm
điều kiện này để head bị ngắt kết nối không thể lọt qua.

## Việc chưa hoàn tất

- **Không còn hạng mục kỹ thuật mở trong gate P1.** Máy phát triển chỉ có
  `torch 2.11.0+cpu` nên phần CUDA chạy trên Kaggle T4 cu128 qua notebook riêng.
  Version 1 fail (T-13): `RuntimeError: Expected all tensors to be on the same
  device, but found at least two devices, cuda:0 and cpu!` ở bước optimizer đầu
  tiên sau resume, vì `ResumeState.load` đọc bằng `map_location="cpu"` còn
  `_restore` chạy trước khi `Fabric.setup` chuyển model lên accelerator. Đã sửa
  bằng `align_optimizer_state()` và `ModelEma.to()`; version 2 pass (T-14). Lớp
  lỗi này không tái hiện được trên host chỉ có CPU, nên gate CUDA chính là
  regression test của nó.
- **Cần quyết định của maintainer:** cập nhật trạng thái P1 trong
  `PROJECT_PHASES.md` từ `pending` sang `complete` là sửa một active normative
  document và theo `DOCUMENTATION_POLICY.md` §4 phải đi kèm revision record.
  Phiên này không tự ý sửa; bằng chứng gate đã đủ để maintainer ra quyết định.
- `scripts/check_qdgrasp_imports.py` vẫn báo 7 lỗi resolve trong cây legacy
  (`qdgrasp/engine/model.py`, `qdgrasp/nn/*`). Đây là trạng thái có sẵn từ trước
  phiên này, không phải regression: mọi module mới của P1 resolve sạch. Việc rút
  legacy thuộc phase sau và không nằm trong phạm vi được phép của phiên.
- ONNX chỉ được xác minh trong venv có `export` extra; base install vẫn không kéo
  `onnx`/`onnxruntime`.

## Sửa đổi phiên trước

Không.

## Bàn giao

- `feature/phase1-framework` chứa toàn bộ output P1 và đã được push lên
  `origin`; `develop` đã có P0 sau khi merge `feature/ultralytics-fork-import`
  trong phiên này. Branch P1 chưa merge vào `develop`.
- Toàn bộ tiêu chí gate P1 (API/config round-trip, CPU smoke, CUDA dummy
  train-step) đã có bằng chứng. Bước hợp lệ tiếp theo: chạy
  `scripts/git/finish_feature.sh`, cập nhật trạng thái roadmap kèm revision
  record, rồi mở Phase 2 (robot layer).
- Trước khi mở rộng schema, đọc `docs/configuration/RUN_CONFIG.md`: version mới
  phải là schema identifier mới, không sửa nghĩa `v1`.
