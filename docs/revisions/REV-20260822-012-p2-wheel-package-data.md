---
document_id: REV-20260822-012
document_type: revision_record
revision_schema: 2
title: Sửa package-data thiếu asset đệ quy của Phase 2
status: in_review
date: 2026-08-22
record_id: REV-20260822-012
session_id: SESSION-20260822-022
created_at: 2026-08-22T23:10:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-020
    artifact: pyproject.toml
    revision: 7e050dbff2eee04d3a15e605241b011d558a3c048d1e79131540a06ad8f98186
  - session_id: SESSION-20260822-020
    artifact: qdgrasp/config/loader.py
    revision: b86c8e009026cf83cb9a2c7f4b18b639449c9071a77c670328b631f834ca4826
  - session_id: SESSION-20260822-020
    artifact: qdgrasp/presets/robots/leap_hand.yaml
    revision: d80b44fa09dc81dfa91525fbc8bd293dca0baa92be8ae85575ea4e3e22322fa7
  - session_id: SESSION-20260822-020
    artifact: qdgrasp/presets/robots/wonik_allegro.yaml
    revision: 78964db851863ae1be3d210717e6c079ca462221c0d523683e9b506e3809bdc3
  - session_id: SESSION-20260822-020
    artifact: qdgrasp/presets/robots/shadow_hand.yaml
    revision: a2286767c1dfc1b3a754586f6ce2730806276fa07e32fa7fb1cb208bdc725713
  - session_id: SESSION-20260822-020
    artifact: qdgrasp/robot/__init__.py
    revision: 9c832f24ef9eb3fdb27f0cb193233732590413a54c2e968c0d3dd9768bc3a709
  - session_id: SESSION-20260822-020
    artifact: qdgrasp/robot/spec.py
    revision: 2c49bd29aa36d56b114b7e78bdf21df73e8fa5d2e8d7f9722dac89c00507b6af
  - session_id: SESSION-20260822-018
    artifact: scripts/check_phase0.py
    revision: 377e9ad86ec739bc4f8a5b7b2e4238e554b84b7e87afb8df1d2bf473b9846f81
  - session_id: SESSION-20260822-018
    artifact: README.md
    revision: cfb34e9ff7c84311ff979322007bca9789ca23bf31d18e1d5a974a65f4bfa7ff
  - session_id: SESSION-20260822-020
    artifact: docs/configuration/ROBOT_PROFILE.md
    revision: 169597c98f030eeceec62732229c9131dd79d2b6634656831c72abdbd54279a4
  - session_id: SESSION-20260822-022
    artifact: qdgrasp/robot/assets.py
    revision: absent@ece9703
  - session_id: SESSION-20260822-022
    artifact: scripts/check_wheel.py
    revision: absent@ece9703
  - session_id: SESSION-20260822-022
    artifact: tests/test_robot_assets.py
    revision: absent@ece9703
reason: "Setuptools package-data dùng glob một cấp nên wheel từ ece9703 bỏ các robot presets và derived normalization assets của P2; điều này phá clean wheel import/CLI gate của P0 và sẽ làm preset dataset của P3 tiếp tục bị thiếu."
necessity: N2
impact: "Sửa cấu hình đóng gói và đường dẫn asset của profile, không đổi schema; wheel mới discover preset con và RobotSpec resolve asset từ external root đã pin thay vì đường dẫn .references."
---

# REV-20260822-012 — Sửa package-data thiếu asset đệ quy

## 1. Liên kết truy vết

- Phát hiện: `docs/roadmap/PHASE3_EXECUTION_PLAN.md` §3.
- Baseline: commit `ece9703`.
- Artifact và hash trước/sau:

| Artifact | SHA-256 trước (`ece9703`) | SHA-256 sau |
|---|---|---|
| `pyproject.toml` | `7e050dbf…98186` | `fcff7a50…aa47` |
| `qdgrasp/config/loader.py` | `b86c8e00…4826` | `55ec1fbe…e8444` |
| `qdgrasp/presets/robots/leap_hand.yaml` | `d80b44fa…2fa7` | `ad75fc1c…aab2` |
| `qdgrasp/presets/robots/wonik_allegro.yaml` | `78964db8…bdc3` | `daad1774…c690` |
| `qdgrasp/presets/robots/shadow_hand.yaml` | `a2286767…5713` | `c4e8bd70…0d05` |
| `qdgrasp/robot/__init__.py` | `9c832f24…a709` | `bd2af59c…cdac` |
| `qdgrasp/robot/spec.py` | `2c49bd29…b6af` | `086d3580…6610` |
| `scripts/check_phase0.py` | `377e9ad8…6f81` | `affedbba…726e` |
| `README.md` | `cfb34e9f…a7ff` | `16a013b0…732c` |
| `docs/configuration/ROBOT_PROFILE.md` | `169597c9…79a4` | `6615177b…3700` |
| `qdgrasp/robot/assets.py` | absent | `fc9ea4a4…9fb5` |
| `scripts/check_wheel.py` | absent | `e732d3b1…3f61` |
| `tests/test_robot_assets.py` | absent | `03845852…097f` |

## 2. Lý do chỉnh sửa

`presets/*.yaml` và `assets/*` không khớp file nằm trong thư mục con. Vì vậy
wheel từ `ece9703` thiếu `qdgrasp/presets/robots/*.yaml`, dù source tree và CPU
gate nhìn thấy chúng. Sau khi thêm glob, profile vẫn không dùng được từ wheel vì
`source_asset` trỏ thẳng vào `.references/`; thư mục này theo policy không nằm
trong distribution. Đây là defect phát hành của P2 và đồng thời chặn các preset
data của P3.

## 3. Mức độ cần thiết

- Mức: `N2` — artifact wheel không đầy đủ và gate P0 không còn đáng tin.
- Hành động: hoàn tất trước khi phát hành hoặc sinh artifact P3 phụ thuộc
  package data.

## 4. Phạm vi và tác động

Thay đổi giữ nguyên boundary phân phối: raw MJCF/mesh không vào main wheel. API
`RobotSpec` giờ nhận profile shipped trong wheel và resolve asset từ checkout đã
pin; profile hash thay đổi vì `source_asset` đổi từ workspace path sang URI.
Bundle/checkpoint dùng hash profile cũ phải được coi là không tương thích.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact |
|---|---|---|
| CH-001 | Thêm glob đệ quy cho preset YAML và assets | `pyproject.toml` |
| CH-002 | Đổi profile sang `asset://…` và thêm resolver external-root fail-closed | `qdgrasp/robot/assets.py`, `qdgrasp/robot/spec.py`, presets robot |
| CH-003 | Đệ quy preset resolver; thêm wheel gate vào Phase 0 | `qdgrasp/config/loader.py`, `scripts/check_wheel.py`, `scripts/check_phase0.py` |
| CH-004 | Ghi revision record riêng, không sửa session P2 cũ | Record này |

## 6. Xác minh

| Verification ID | Phương pháp | Kết quả |
|---|---|---|
| V-001 | `git diff --check` | pass |
| V-002 | `./.venv/bin/python scripts/check_wheel.py` | pass; wheel chứa nested robot presets và derived URDF/manifest, rồi `RobotSpec` tải LEAP qua asset root đã pin |
| V-003 | `./.venv/bin/python scripts/check_phase0.py` | pass; Phase 0 gọi wheel gate mới |
| V-004 | `./.venv/bin/python -m pytest tests/ -q` | pass; 204 passed, 1 skipped, 93 subtests passed |
| V-005 | `./.venv/bin/python -m unittest discover -s scripts/tests -p 'test_*.py'` | pass; 50 tests |

Test suite đầy đủ đã chạy; `scripts/check_wheel.py` là verification bắt buộc
riêng cho defect này và được gọi từ `check_phase0.py`.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo bị ảnh hưởng: `SESSION-20260822-020` vì phát hành P2 không còn tự
  chứa asset từ bare wheel.
- Loại kiểm tra lại: `delta_review` cho packaging và compatibility của profile
  hash.
- `PLAN.md` và policy “robot assets không nằm trong wheel” không thay đổi.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-22 Asia/Bangkok.
- Người kiểm tra: chưa chỉ định; cần independent review.
- Kết luận: implementation và verification pass, record ở `in_review`.
