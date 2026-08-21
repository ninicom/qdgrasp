---
document_id: SESSION-20260822-001
document_type: session_report
title: Sửa S1 bằng train-argument registry dialect v2
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-001
author: codex-primary-agent
revises:
  - SESSION-20260821-007
  - REV-20260821-006
revision_reason: TPR-20260822-001 chứng minh remediation fdb60eb vẫn nhận scalar/cú pháp YAML sai, Markdown drift và standalone dirty source.
necessity: N2
impact: Thay contract parser/fingerprint/evidence gate sang schema v2; giữ nguyên 127 tên và upstream revision.
related_plan: PLAN-V2
---

# SESSION-20260822-001 — Registry dialect v2

## Mục tiêu phiên

Đóng F-001/F-002/F-004/F-005/F-010 từ `TPR-20260822-001` bằng contract máy
đọc không làm mất kiểu, khóa toàn bộ tài liệu quy phạm và kiểm source checkout
sạch trong standalone full mode. Không merge trước independent delta review pass.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Dialect cấm quote/tab/trailing whitespace/sai separator | `scripts/check_train_args.py` | mutation regressions pass |
| W-02 | Fingerprint ghi kind + exact lexeme, bao phủ mọi section | schema v2 fingerprint `a43bf2…7a10` | fingerprint tests pass |
| W-03 | Khóa toàn bộ normative Markdown body và type rows | body SHA-256 `b10f58…13ae` | 5 doc mutations bị reject |
| W-04 | Ghi structured canonical mapping cho `quantize` | YAML + Markdown | registry-only pass |
| W-05 | Full mode tự kiểm origin/HEAD/clean tracked+untracked | source checker | clean/dirty tests pass |
| W-06 | Mở rộng regression suite từ 24 lên 30 | unit suite | 30/30 pass |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | finding | `TPR-20260822-001` | reviewed commit `fdb60eb` |
| E-02 | registry | `docs/configuration/TRAIN_ARGUMENTS.yaml` | `5fe682f1d84351564edf858c36bd1f6ac07005295234ad9b23f97d3081c48626` |
| E-03 | human contract | `docs/configuration/TRAIN_ARGUMENTS.md` | `a77ab9466ddf44fecca5d8d59410ab315b7423652f7ef5f42e4f07b55dad5480` |
| E-04 | checker | `scripts/check_train_args.py` | `a77e57e26b277b777585dddb99aea7e0f8bb5b7b8764cb744ca440c85bf73c8d` |
| E-05 | tests | `scripts/tests/test_check_train_args.py` | `71c6ede6799c7e6011a342ffc6747820d28d07f0714317afbc402ce824321b4d` |
| E-06 | replay log | `docs/reports/evidence/TRAIN-ARGS-20260822-dialect-v2.txt` | 4.725 byte; `49b29aa755890422863cc2821a293abed612a4981e93d384ac112edc480cdda1` |

Implementation commit: `8dc0e2e0a122e46df95f2f4f7a3cfbfee9dec58c`, tree
`60794bd3930713bde84a8707325517b05bce2ea2`, committed at
`2026-08-22T00:25:08+07:00`.

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `python3 scripts/check_train_args.py --registry-only` | 0 | 127 names; typed manifest/doc body pass |
| T-02 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | exact origin/HEAD/clean source pass |
| T-03 | `python3 scripts/check_references.py --lock-only` | 0 | immutable lock pass |
| T-04 | `python3 scripts/check_references.py --source-root .references` | 0 | 2 pinned clones sạch |
| T-05 | `python3 scripts/check_docs.py --root .` | 0 | 36 managed docs pass |
| T-06 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` | 0 | 30/30 pass |
| T-07 | `bash -n .githooks/* scripts/git/*.sh scripts/release_gate.sh` và `git diff --check` | 0 | pass |

## Việc chưa hoàn tất

- Independent delta review lần hai trên clean exact commit.
- Model/runtime/CPU-CUDA smoke ngoài claim và vẫn chưa được implement.

## Sửa đổi phiên trước

Có, mức `N2`. `REV-20260822-001` ghi exact before/after hashes, nguyên nhân và
tác động. SESSION-007/REV-006/TPR-001 đã đóng và không bị sửa lại.

## Bàn giao

Implementation commit và replay evidence đã khóa. Giao final feature snapshot
chứa evidence này cho reviewer không tham gia sửa code; chỉ merge sau pass/NONE.
