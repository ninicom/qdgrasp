---
document_id: REV-20260821-005
document_type: revision_record
title: Bổ sung hợp đồng đầy đủ cho train arguments Ultralytics
status: complete
date: 2026-08-21
record_id: REV-20260821-005
session_id: SESSION-20260821-006
created_at: 2026-08-21T23:33:18+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260821-001
    artifact: PLAN.md
    revision: 797c3b9ae17f8eb5975f6018b377e3b30c8bc5419920000502055271cf38384d
  - session_id: SESSION-20260821-004
    artifact: docs/README.md
    revision: 4b04687ccae82eebbe0aeab01b0abd7f8ec7f4e0cadc347dd93121a619af54fb
  - session_id: SESSION-20260821-004
    artifact: docs/schemas/README.md
    revision: 9f2163d00f07b332f42c2a65df229b456e36a6fdb55ba6933a099562ca410b50
  - session_id: SESSION-20260821-004
    artifact: docs/governance/GIT_WORKFLOW.md
    revision: 371ea69cd1064f495cee4c985bd4732650826e6ebc7151969498f4bb856cfd58
  - session_id: SESSION-20260821-002
    artifact: scripts/check_docs.py
    revision: b1df42bff93c453baec4a67e88c2a4edf86820c2a5ed07d3c2835d770dc0564a
  - session_id: SESSION-20260821-004
    artifact: .githooks/pre-commit
    revision: 8f994e285895403390a5592f434dc53b6d5d517e2afe4002b55d50366d4b9c2c
  - session_id: SESSION-20260821-004
    artifact: scripts/git/common.sh
    revision: 7bdebe764970e8ab818b5dc89c2456360b6ffa6f75d20d7b20505e7342428cfa
reason: Kế hoạch và gate trước chỉ yêu cầu config/YAML tổng quát, chưa kiểm kê toàn bộ public train/config surface của đúng source Ultralytics đã pin nên có nguy cơ bỏ sót hoặc tạo dead key.
necessity: N2
impact: Thay đổi contract cấu hình/API và gate kiểm tra; không thay đổi số liệu model, runtime, checkpoint hoặc verdict release alpha.1.
---

# REV-20260821-005 — Hợp đồng train arguments

## Lý do chỉnh sửa

Phiên bootstrap đã yêu cầu façade tương đương Ultralytics và unknown/dead key là
lỗi, nhưng chưa có danh mục exhaustive hoặc phép đối chiếu source. Chỉ liệt kê
các key thường dùng sẽ bỏ qua config chung, custom kwargs, alias cũ và tham số
điều khiển riêng của `Model.train`. Yêu cầu mới cần biến tập đầu vào này thành
contract có máy kiểm tra.

## Mức độ cần thiết

Mức `N2` — bắt buộc cho tính đúng của config/API. Thiếu một key hoặc âm thầm
no-op có thể làm run khác cấu hình người dùng nghĩ rằng đã chạy, ảnh hưởng
reproducibility, CPU/CUDA behavior và exact resume. Đây không chỉ là chỉnh câu
chữ nên không dùng `N0/N1`.

## Phạm vi và tác động

| Artifact đã có | SHA-256 trước | SHA-256 sau | Lý do sửa |
|---|---|---|---|
| `PLAN.md` | `797c3b…384d` | `7fc224…42b0` | Thêm compatibility manifest và acceptance 127 tên |
| `docs/README.md` | `4b0468…54fb` | `22c74a…1d62` | Chỉ mục registry/checker mới |
| `docs/schemas/README.md` | `9f2163…0b50` | `021521…a8ad` | Mở phạm vi validator và schema registry |
| `GIT_WORKFLOW.md` | `371ea6…fd58` | `4950b7…a4d` | Thêm registry/full-source gate |
| `scripts/check_docs.py` | `b1df42…564a` | `5baffb…e061` | Quản lý `docs/configuration/*.md` |
| `.githooks/pre-commit` | `8f994e…9c2c` | `91fe66…9222` | Chạy registry-only checker |
| `scripts/git/common.sh` | `7bdebe…8cfa` | `1c3dc0…68b8` | Chạy registry-only ở feature/release gate |

Artifact mới gồm `references.lock.yaml`, hai file registry, checker standard
library, sáu regression tests và raw check log. Không sửa SESSION/REV/release
report đã đóng; không thay tag `v0.1.0-alpha.1`.

Tác động kỹ thuật là khóa 115 canonical key + 2 extra kwargs + 9 legacy names +
1 API control. Disposition `reject/defer` là quyết định explicit; không tuyên bố
runtime cho các key đó đã được implement.

## Xác minh

| Verification ID | Lệnh/artifact | Kết quả |
|---|---|---|
| V-001 | `python3 scripts/check_train_args.py --registry-only` | PASS; 127 tên, zero missing/extra |
| V-002 | `python3 scripts/check_train_args.py --source .references/ultralytics` | PASS tại `329682a…` |
| V-003 | `python3 scripts/check_docs.py --root .` | PASS; 28 docs trước khi thêm hồ sơ này |
| V-004 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` | PASS; 10/10 |
| V-005 | `bash -n ...` và `git diff --check` | PASS |
| V-006 | `TRAIN-ARGS-20260821-checks.txt` | SHA-256 `86a051467e3db6f2f53d8021e5623ee57361a4aab0324c5d1adafeee9670df0e` |

Independent review phải đối chiếu feature commit đã khóa trước khi release
alpha.2. Nếu review tìm mismatch, sửa bằng commit/revision kế tiếp; không sửa
im lặng record này.
