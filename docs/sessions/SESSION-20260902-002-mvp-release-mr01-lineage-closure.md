---
document_id: SESSION-20260902-002
document_type: session_report
title: MVP release MR-01 — đóng implementation lineage
status: complete
date: 2026-09-02
session_id: SESSION-20260902-002
author: claude-agent
revises: none
related_plan: ROADMAP-MVP-RELEASE-001
scope: mujoco_cpu_leap_state_cuboid_only
candidate_branch: feature/mvp-grasp-policy
candidate_commit: c8e4337f204c8ef167d4080c1e0debad40ba5e2f
---

# SESSION-20260902-002 — MVP release MR-01: đóng implementation lineage

## Mục tiêu phiên

- Đóng lineage implementation đang dở trên tám file MVP: demonstration
  schema/index/manifest v1 và aggregate lineage của checkpoint.
- Bổ sung negative coverage tối thiểu mà `ROADMAP-MVP-RELEASE-001` §5 MR-01
  yêu cầu, để mỗi trường identity đều có một test làm hỏng đúng một thứ.
- Chạy full gate trên worktree sạch, commit theo Conventional Commits và đưa
  candidate commit lên remote.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
| --- | --- | --- | --- |
| `MR01-01` | Review diff tám file; toàn bộ nằm trong chuỗi `COR-11`, không có thay đổi ngoài phạm vi cần tách | `git diff` trên `90b3acc` | verified |
| `MR01-02` | Giữ demonstration schema v1, index v1, content manifest v1 và aggregate checkpoint lineage | `qdgrasp/mvp/expert.py`, `qdgrasp/mvp/policy.py` | verified |
| `MR01-03` | Bổ sung negative tests cho array byte, ledger seed/variant/accepted, index identity, training config, dataset hash, parent path, parent bytes và archive thừa/thiếu/pickle | `tests/mvp/test_check_mvp_gate.py`, `tests/mvp/test_mvp_learning.py` | verified |
| `MR01-04` | Chạy full suite, static core, docs, wheel và Phase 0 gate trên worktree sạch | bảng §Kiểm tra đã chạy | verified |
| `MR01-05` | Commit theo Conventional Commits và push feature candidate |  `729d403` (plan/baseline), `c8e4337` (implementation) | verified |

Negative coverage mới ánh xạ một-một vào danh sách bắt buộc của MR-01:

| Yêu cầu MR-01 §5.3 | Test |
| --- | --- |
| đổi một byte observation/action/episode index | `test_a_doctored_demonstration_array_fails_the_gate` (3 subtest) |
| sửa ledger, seed hoặc variant ID | `test_a_doctored_demonstration_ledger_fails_the_gate` (3 subtest) |
| sửa scope/prior/environment fingerprint trong index | `test_a_doctored_demonstration_index_identity_fails_the_gate` (4 subtest) |
| sửa training config | `test_doctored_checkpoint_training_config_fails_to_load` |
| sửa dataset hash | `test_a_doctored_checkpoint_dataset_hash_fails_to_load`, `test_a_training_report_that_cites_other_demonstrations_fails_the_gate` |
| thay parent path | `test_a_ppo_parent_that_is_not_the_reported_bc_checkpoint_fails_the_gate` |
| thay parent checkpoint bytes | `test_ppo_refuses_a_parent_checkpoint_whose_bytes_changed` |
| archive có array thừa/thiếu | `test_a_demonstration_archive_with_the_wrong_array_set_fails_the_gate` (2 subtest) |
| archive có object pickle | `test_a_pickled_demonstration_archive_fails_the_gate` |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- | --- |
| `MR01-E01` | commit plan/baseline | `docs/roadmap/MVP_RELEASE_EXECUTION_PLAN.md`, `docs/sessions/SESSION-20260902-001-*.md` | `729d40384c62acab4e9991a9918bd8ef83528eac` |
| `MR01-E02` | commit lineage implementation | tám file MVP | `c8e4337f204c8ef167d4080c1e0debad40ba5e2f` |
| `MR01-E03` | parent commit để rollback | baseline MR-00 | `90b3accb76ae1417199167661c58cd937b00d9b9` |
| `MR01-E04` | MVP artifact gate v0 dưới contract v1 | `scripts/check_mvp.py --root . --runs runs/mvp` | 50/75, `blocked_with_evidence`, 25 FAIL |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| `MR01-T01` | `.venv/bin/python -m pytest -q` | 0 | 1423 passed, 1 skipped, 105 subtests (baseline 1415/93) |
| `MR01-T02` | `.venv/bin/python scripts/check_static_core.py` | 0 | Ruff 135 file, Mypy 32 contract |
| `MR01-T03` | `.venv/bin/python scripts/check_docs.py --root .` | 0 | 156 tài liệu |
| `MR01-T04` | `.venv/bin/python scripts/check_wheel.py` | 0 | Wheel gate PASS |
| `MR01-T05` | `.venv/bin/python scripts/check_phase0.py` | 0 | Phase 0 foundation PASS |
| `MR01-T06` | `.venv/bin/ruff check` trên tám file đã sửa | 0 | không phát sinh finding mới |
| `MR01-T07` | `.venv/bin/python scripts/check_mvp.py --root . --runs runs/mvp` | 1 | expected: artifact v0 bị contract v1 chặn |

`MR01-T01`–`MR01-T05` được chạy lại sau khi commit, trên worktree sạch, vì các
governance test chỉ có hiệu lực khi cây làm việc không còn thay đổi chưa commit.

## Việc chưa hoàn tất

- `.venv/bin/ruff check .` trên toàn repository báo 357 finding có sẵn, không
  finding nào thuộc tám file của MR-01 (`qdgrasp/` 72, `scripts/` 37, `tests/`
  43, `notebooks/` 6, `evidence/` 6, `kaggle-*` 18). Gate tĩnh mà dự án thi
  hành là `scripts/check_static_core.py` và nó pass. Ma trận release ở MR-07
  liệt kê `ruff check .` với yêu cầu exit 0; khoảng cách này phải được giải
  quyết trong MR-07 và không được sửa ở MR-01.
- MVP scope v1, eval manifest v1, release-mode checker và version identity
  contract chưa tồn tại: đó là MR-02.
- Chưa sinh release evidence, chưa train candidate, chưa có independent review.

## Sửa đổi phiên trước

Không. Phiên này không sửa artifact, kết luận hay model card của MVP v0.
`evidence/mvp/round-1`…`round-3` giữ nguyên là `experimental_non_release`.

## Bàn giao

MR-01 đạt exit gate: worktree sạch, full gate pass, candidate commit
`c8e4337` có trên remote, rollback là parent commit `90b3acc`. Work package kế
tiếp là MR-02: tạo `configs/mvp/dexacquire-mvp-v1.yaml`, eval manifest v1,
release class đọc được bằng máy, release-mode checker và version identity
`0.1.0a2` / `v0.1.0-alpha.2` — tất cả phải khóa trước khi train.
