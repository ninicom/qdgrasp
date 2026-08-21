---
document_id: SESSION-20260821-006
document_type: session_report
title: Kiểm kê và khóa train arguments Ultralytics
status: complete
date: 2026-08-21
session_id: SESSION-20260821-006
author: codex-primary-agent
revises:
  - SESSION-20260821-001
  - SESSION-20260821-002
  - SESSION-20260821-004
revision_reason: Các phiên trước chưa có exhaustive train-argument compatibility manifest hoặc source-drift checker.
necessity: N2
impact: Mở rộng plan, documentation schema và Git gates cho contract config/API; không thay đổi kết luận alpha.1 hay tuyên bố model runtime.
related_plan: PLAN-V2
---

# SESSION-20260821-006 — Train-argument registry

## Mục tiêu phiên

- Lấy dữ liệu trực tiếp từ clone chuẩn, không suy diễn danh sách tham số.
- Ghi rõ toàn bộ train/config surface Ultralytics và disposition DexGrasp.
- Chặn tự động mọi thiếu/thừa/default/type/hash/commit drift.
- Ghi riêng hành vi CPU/CUDA và phần chưa được implement.

## Việc đã hoàn tất

| ID | Việc đã làm | Output/xác minh |
|---|---|---|
| W-01 | Clone read-only Ultralytics và checkout đúng source | `v8.4.125`, `329682a…` |
| W-02 | Clone read-only DexGraspNet2 và checkout đúng source | `26ecd761…` |
| W-03 | Lập provenance/license state | `references.lock.yaml` |
| W-04 | Kiểm kê 115 canonical key | `TRAIN_ARGUMENTS.yaml`; source check zero missing/extra |
| W-05 | Ghi 2 extra, 9 legacy và `trainer` | Tổng public names 127 |
| W-06 | Phân loại từng key retain/adapt/defer/reject và device policy | Registry YAML + bảng Markdown |
| W-07 | Ghi CPU/CUDA behavior | `TRAIN_ARGUMENTS.md` mục 9 |
| W-08 | Thêm checker standard library và sáu regression tests | 10/10 toàn suite pass |
| W-09 | Nối checker vào hook/feature/release gate | registry-only không cần `.references` |
| W-10 | Mở docs validator cho configuration registries | discovery regression pass |

## Bằng chứng

- Ultralytics `default.yaml` SHA-256:
  `eb5e9ab6825a5d55076f8b38aed00953dec722ed5d5368a6584df35f50f32839`.
- Config validator SHA-256:
  `ea9a98b498686f44561e9ce9892aa0e7de52226429496b08a20475ebbfa4eed8`.
- `Model.train` source SHA-256:
  `69cb4c9c6f572c50bd35328e5154e6b596ed50dc10d218584da8aed3f7c9b32a`.
- Trainer source SHA-256:
  `d02bfd82d2af38fb58a6fe7903ef0cb9a93633a75b3db778768896d6c57a32e6`.
- Raw check log: `docs/reports/evidence/TRAIN-ARGS-20260821-checks.txt`,
  2,193 bytes, SHA-256
  `86a051467e3db6f2f53d8021e5623ee57361a4aab0324c5d1adafeee9670df0e`.
- Revision accountability: `REV-20260821-005`.

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit | Kết quả |
|---|---|---:|---|
| T-01 | `python3 scripts/check_train_args.py --registry-only` | 0 | 127 names; zero missing/extra |
| T-02 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | HEAD/hash/default/group/type/API pass |
| T-03 | `python3 scripts/check_docs.py --root .` | 0 | 28 managed docs trước session/revision |
| T-04 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` | 0 | 10/10 pass |
| T-05 | `bash -n .githooks/* scripts/git/*.sh scripts/release_gate.sh` | 0 | Shell syntax pass |
| T-06 | `git diff --check` | 0 | Whitespace pass |

## Việc chưa hoàn tất

- Registry là design/config contract; package, parser và trainer DexGrasp chưa
  được implement, do đó chưa có train-smoke CPU/CUDA.
- 19 DexGrasp extension vẫn `required_design`; default/range chưa được chấp nhận.
- GraspGenX và rectified-flow trong `references.lock.yaml` còn `pending`.
- Ultralytics distribution/porting còn bị chặn tới khi có commercial license;
  pinned DexGraspNet2 tree không có root license file nên license vẫn unresolved.
- Independent review và release `0.1.0-alpha.2` thực hiện sau khi khóa feature
  commit; không được ghi pass trước khi review chạy.

## Sửa đổi phiên trước

Có. `REV-20260821-005` ghi lý do, SHA trước/sau và mức `N2`. Việc sửa là cần
thiết vì contract API/config không thể được xác minh chỉ bằng plan tổng quát.
Không sửa các SESSION/REV/TPR/release alpha.1 đã hoàn tất và không thay verdict
cũ; chỉ mở rộng normative plan/schema/gate cho công việc từ phiên này trở đi.

## Bàn giao

- Commit feature bằng Conventional Commit và giữ working tree sạch.
- Giao exact feature commit cho agent độc lập, tạo `TPR-20260821-003` từ kết quả.
- Chỉ merge vào `develop` sau khi checker/docs/tests và review không có S0/S1.
- Prerelease alpha.2 chỉ được claim source/config registry, không claim model,
  runtime, license hoặc benchmark.
