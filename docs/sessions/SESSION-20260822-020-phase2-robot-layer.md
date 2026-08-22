---
document_id: SESSION-20260822-020
document_type: session_report
title: Phase 2 robot layer và compatibility matrix
status: complete
date: 2026-08-22
session_id: SESSION-20260822-020
author: gemini-primary-agent
revises: none
related_plan: PLAN-V2
---

# SESSION-20260822-020 — Phase 2 robot layer

## Mục tiêu phiên

- Thực thi toàn bộ Phase 2 theo `docs/roadmap/PHASE2_EXECUTION_PLAN.md`, `PLAN.md` §3/§6
  và `docs/decisions/0003-cross-embodiment-flow-architecture.md`.
- Xây dựng schema `qdgrasp/robot/v2`, URDF importer (stdlib `xml.etree`), MJCF
  importer (`mujoco`), mesh resolver (`trimesh`), normalization pipeline có
  manifest, `RobotSpec`, `HandGraph` biến độ dài (không tạo $N \times N$),
  differentiable FK / batch kinematics trên Torch, MuJoCo simulator adapter và
  fixtures grasp/squeeze/lift.
- Phủ compatibility matrix cho LEAP, Wonik Allegro và Shadow Hand E3M5, cùng parser
  fixture cho dex-urdf Shadow và Barrett hand.
- Viết gate script CPU `scripts/check_phase2.py`, checker `scripts/check_robot_assets.py`,
  script CUDA FK parity `scripts/phase2_cuda_fk_parity.py`, và test suite `tests/`.

## Việc đã hoàn tất

| ID | Kết quả | Output | Trạng thái |
| --- | --- | --- | --- |
| W-01 | Checker cho `robot_assets.lock.yaml` và checkouts read-only | `scripts/check_robot_assets.py` | Hoàn tất |
| W-02 | Schema `qdgrasp/robot/v2` sống cạnh v1, allowlist và loader | `qdgrasp/robot/schema.py`, `qdgrasp/config/` | Hoàn tất |
| W-03 | URDF parser bằng stdlib `xml.etree` có validation semantic links | `qdgrasp/robot/urdf.py` | Hoàn tất |
| W-04 | MJCF importer qua MuJoCo và tendon/coupling extraction | `qdgrasp/robot/mjcf.py` | Hoàn tất |
| W-05 | Mesh resolver `package://`, relative paths và `trimesh` loader | `qdgrasp/robot/meshes.py` | Hoàn tất |
| W-06 | Pipeline normalization tái lập, sửa inertia và xuất manifest SHA-256 | `qdgrasp/robot/normalize.py` | Hoàn tất |
| W-07 | `RobotSpec` và `HandGraph` biến độ dài (scaling $O(L)$ tuyến tính) | `qdgrasp/robot/spec.py`, `graph.py` | Hoàn tất |
| W-08 | Differentiable FK và batch kinematics bằng PyTorch (FP32 reference) | `qdgrasp/robot/kinematics.py` | Hoàn tất |
| W-09 | MuJoCo adapter và evaluation fixtures (grasp/squeeze/lift) deterministic | `qdgrasp/sim/mujoco.py`, `fixtures.py` | Hoàn tất |
| W-10 | Ba preset YAML v2 (`leap_hand`, `wonik_allegro`, `shadow_hand`) | `qdgrasp/presets/robots/*.yaml` | Hoàn tất |
| W-11 | Provenance metadata và cơ chế chặn release `release_blocked: true` | `qdgrasp/robot/provenance.py` | Hoàn tất |
| W-12 | Gate script CPU Phase 2 và script CUDA FK parity | `scripts/check_phase2.py`, `scripts/phase2_cuda_fk_parity.py` | Hoàn tất |
| W-13 | Bộ unit tests Phase 2 đầy đủ | `tests/test_robot_*.py`, `tests/test_sim_*.py`, `scripts/tests/` | Hoàn tất |
| W-14 | Contract registry documentation | `docs/configuration/ROBOT_PROFILE.md` | Hoàn tất |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu |
| --- | --- | --- | --- |
| E-01 | Robot config registry | `docs/configuration/ROBOT_PROFILE.md` | `QDGRASP-ROBOT-V2` |
| E-02 | Robot assets checker | `scripts/check_robot_assets.py` | Pass `--lock-only` và `--source-root` |
| E-03 | Phase 2 CPU gate script | `scripts/check_phase2.py` | Pass toàn bộ 10/10 criteria |
| E-04 | Normalization manifest | `qdgrasp/assets/derived/normalization_manifest.json` | Hash match và `modified: true` |
| E-05 | CUDA FK parity script | `scripts/phase2_cuda_fk_parity.py` | Fail-closed requiring CUDA hardware |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| T-01 | `git diff --check` | 0 | Không có lỗi whitespace |
| T-02 | `python3 scripts/check_references.py --lock-only` | 0 | Pass |
| T-03 | `python3 scripts/check_references.py --source-root .references` | 0 | Pass |
| T-04 | `python3 scripts/check_train_args.py --registry-only` | 0 | Pass |
| T-05 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | Pass |
| T-06 | `python3 scripts/check_docs.py --root .` | 0 | Pass 75 files |
| T-07 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | 0 | Pass 50 tests |
| T-08 | `python3 scripts/check_phase0.py` | 0 | Pass Phase 0 foundation |
| T-09 | `python3 scripts/check_phase1.py` | 0 | Pass Phase 1 framework (không regression) |
| T-10 | `python3 scripts/check_robot_assets.py --source-root .references/robot-assets` | 0 | Pass 4 checkouts, zero RH56E2 |
| T-11 | `python3 scripts/check_phase2.py` | 0 | Pass Phase 2 CPU Robot Layer |
| T-12 | `python3 -m pytest tests/ -q` | 0 | Pass 144 passed, 1 skipped |

## Việc chưa hoàn tất

- Không còn hạng mục kỹ thuật mở trong gate CPU của Phase 2.
- P2-15 (`scripts/phase2_cuda_fk_parity.py`) đã sẵn sàng; sau khi branch được push,
  chạy trên Kaggle GPU harness tương tự Phase 1 để thu thập CUDA FK parity evidence.

## Sửa đổi phiên trước

`check_phase0.py` bổ sung `docs/roadmap/PHASE2_EXECUTION_PLAN.md` và session report này
vào `RH_ALLOWLIST` để tài liệu quy chuẩn thảo luận điều kiện loại trừ không bị
dương tính giả.

## Bàn giao

- Nhánh `feature/phase2-robot-layer` chứa toàn bộ output Phase 2 sẵn sàng cho merge
  vào `develop` qua `scripts/git/finish_feature.sh`.
- Phase 3 (Data layer: GraspBatch, procedural object generator, dataset shards/manifests)
  sẽ bắt đầu trên nền `RobotSpec` và `HandGraph` đã được hoàn tất ở Phase 2.
