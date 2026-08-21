---
document_id: SESSION-20260822-003
document_type: session_report
title: Đóng fsmonitor-valid và raw front-matter scalar bypass
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-003
author: codex-primary-agent
revises:
  - SESSION-20260822-002
  - REV-20260822-002
revision_reason: TPR-20260822-003 chứng minh fsmonitor-valid và scalar normalization vẫn cho source/metadata khác semantic đi qua checker.
necessity: N2
impact: Siết hai lớp xác minh hiện hữu; không đổi inventory 127 tên, normative body hoặc model/runtime.
related_plan: PLAN-V2
---

# SESSION-20260822-003 — Raw scalar/fsmonitor hardening

## Mục tiêu phiên

Đóng F-010/F-011 và phần còn mở của F-005 sau TPR-003: kiểm đồng thời hai
Git index views và so front-matter raw lexeme. Feature vẫn bị cấm merge cho tới
khi exact committed snapshot nhận independent verdict `pass/NONE`.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Lưu TPR-003 fail/S2 bất biến trước khi sửa | review + raw evidence | commit `3b1c0d0` |
| W-02 | Kiểm cả `ls-files -v` và `ls-files -f` | source checker | fsmonitor-valid regression pass |
| W-03 | Bỏ scalar normalization; giữ raw lexeme | Markdown parser | 11 + 5 field matrix pass |
| W-04 | Thêm regression cho fsmonitor và 11 + 5 field | unit tests | 38/38 pass |
| W-05 | Bump Markdown contract 1.4.0 và pointer REV-003 | registry document | normative body không đổi |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | failed review | `TPR-20260822-003` | reviewed commit `42914f5`; fail/S2 |
| E-02 | failed review log | `docs/reviews/evidence/TPR-20260822-003-delta-checks.txt` | `05c992a54e1bec6b6f9ee0022ebab5d9ce5b7e7507df91e5567511fbcffe3095` |
| E-03 | revision record | `REV-20260822-003` | before/after identity và necessity N2 |
| E-04 | checker | `scripts/check_train_args.py` | `975edb8ea49577a8ef4f7f9369de18059706622ac4bfa44a6008c16bc2ecd62a` |
| E-05 | tests | `scripts/tests/test_check_train_args.py` | `c3005835daae23357e2921a903150e3ca0ce633ff773baaa49fe6866bb7aa685` |
| E-06 | Markdown | `docs/configuration/TRAIN_ARGUMENTS.md` | `3baf998fdc21c9c8d7a718fe08432d01a7edfa4458d146f401d9e0be48777356` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | TPR-003 independent destructive probes | 0 | Hai bypass được tái hiện; verdict fail/S2 |
| T-02 | `python3 scripts/check_references.py --lock-only` | 0 | immutable lock pass |
| T-03 | `python3 scripts/check_references.py --source-root .references` | 0 | 2 clones exact/clean |
| T-04 | `python3 scripts/check_train_args.py --registry-only` | 0 | exact registry/doc metadata pass |
| T-05 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | origin/HEAD/index/status/hash pass |
| T-06 | `python3 scripts/check_docs.py --root .` | 0 | 42 managed docs pass |
| T-07 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` | 0 | 38/38 pass |
| T-08 | `bash -n .githooks/* scripts/git/*.sh scripts/release_gate.sh` và `git diff --check` | 0 | pass |

## Việc chưa hoàn tất

- Commit implementation, khóa exact tree/replay evidence rồi independent delta
  review lần bốn.
- Model/runtime/CPU-CUDA smoke vẫn ngoài claim và chưa được implement.

## Sửa đổi phiên trước

Có, mức `N2`. `REV-20260822-003` ghi rõ nguyên nhân, độ cần thiết, before
hashes và tác động. SESSION-002/REV-002/TPR-003 được giữ nguyên làm bằng chứng
của snapshot thất bại; không sửa lại kết luận cũ.

## Bàn giao

Local implementation/gates đã pass. Commit clean snapshot, tạo immutable replay
evidence và giao reviewer; chỉ complete/merge sau verdict `pass/NONE`.
