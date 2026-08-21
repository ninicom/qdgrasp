---
document_id: SESSION-20260822-002
document_type: session_report
title: Đóng index-flag và front-matter bypass của train registry
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-002
author: codex-primary-agent
revises:
  - SESSION-20260822-001
  - REV-20260822-001
revision_reason: TPR-20260822-002 chứng minh assume-unchanged/skip-worktree che source drift và front matter quản trị nhận pointer/claim sai.
necessity: N2
impact: Siết standalone source identity và exact metadata/revision pointer; không đổi 127 names, YAML body contract hoặc model/runtime.
related_plan: PLAN-V2
---

# SESSION-20260822-002 — Source/metadata hardening

## Mục tiêu phiên

Đóng F-010/F-011 và phần còn mở của F-005 bằng kiểm Git index flags, exact
Markdown front-matter schema/value và revision pointer resolution. Không merge
trước khi independent reviewer trả `pass/NONE` trên exact committed snapshot.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Reject mọi `git ls-files -v` prefix khác `H` | source checker | `h` và `S` regressions pass |
| W-02 | Khóa exact 11-field front matter/value set | train checker | missing/extra/wrong probes pass |
| W-03 | Bắt duplicate/malformed front matter | train checker | negative tests pass |
| W-04 | Resolve pointer trong project và xác minh REV/session/status | REV-002 | identity/missing-target tests pass |
| W-05 | Bump Markdown version 1.3.0; giữ body contract | `TRAIN_ARGUMENTS.md` | body hash không đổi |
| W-06 | Mở rộng full suite từ 30 lên 35 tests | unit suite | 35/35 pass |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | finding | `TPR-20260822-002` | reviewed commit `363a671` |
| E-02 | checker | `scripts/check_train_args.py` | `99b1b8dde330e011cf0e9e3fb0693ac25aa7c6fd672e69249b72baa1c78a2f29` |
| E-03 | tests | `scripts/tests/test_check_train_args.py` | `76f3480f483963b6d318fbb60ce72e25aa98123259867ecc3b8900e3e3c8fdd9` |
| E-04 | Markdown | `docs/configuration/TRAIN_ARGUMENTS.md` | `6911f367e8d068200ecaa4de4410922712614805a3f83971c7c98c88de841b17` |
| E-05 | replay log | `docs/reports/evidence/TRAIN-ARGS-20260822-source-metadata-hardening.txt` | 5.540 byte; `15b540f7dbf4a1be3b173882d02488e93b7638915ceba82cf4e58f67ecd1de17` |

Implementation commit: `62111ed4b0c31e85e47da02c33831f2cf5a32894`, tree
`a52d4782654d8b5c5b2228392965beff1b4da817`, committed at
`2026-08-22T03:11:26+07:00`.

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `python3 scripts/check_references.py --lock-only` | 0 | immutable lock pass |
| T-02 | `python3 scripts/check_references.py --source-root .references` | 0 | 2 clones exact/clean |
| T-03 | `python3 scripts/check_train_args.py --registry-only` | 0 | exact registry/doc metadata pass |
| T-04 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | origin/HEAD/index/status/hash pass |
| T-05 | `python3 scripts/check_docs.py --root .` | 0 | 39 managed docs pass |
| T-06 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` | 0 | 35/35 pass |
| T-07 | `bash -n ...` và `git diff --check` | 0 | pass |

## Việc chưa hoàn tất

- Independent delta review lần ba trên final feature snapshot chứa evidence.
- Model/runtime/CPU-CUDA smoke vẫn ngoài claim và chưa được implement.

## Sửa đổi phiên trước

Có, mức `N2`. `REV-20260822-002` ghi nguyên nhân, before/after hashes và phạm vi.
SESSION-001/REV-001/TPR-002 đã đóng, không sửa lại.

## Bàn giao

Implementation/replay evidence đã khóa. Giao final snapshot cho reviewer; chỉ
complete/merge sau verdict `pass/NONE` và zero open finding.
