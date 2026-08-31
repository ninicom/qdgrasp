---
document_id: SESSION-20260831-001
document_type: session_report
title: Thi công và đóng Grasp Policy MVP (LEAP vertical slice)
status: complete
date: 2026-08-31
session_id: SESSION-20260831-001
author: claude-agent
revises: ROADMAP-MVP-001@1.0.0
related_plan: ROADMAP-MVP-001
revision_reason: Thi công MVP-00–07, sửa success predicate cho đúng §4 và thêm low-pass vào interface residual sau ba vòng tune đo được.
necessity: N2
impact: ROADMAP-MVP-001 chuyển sang complete với artifact experimental_non_release; không gate release nào đổi.
latest_revision_record: docs/revisions/REV-20260831-001-grasp-policy-mvp-closure.md
---

# SESSION-20260831-001 — Grasp Policy MVP

## Mục tiêu phiên

- Thực thi phase tiếp theo theo `PLAN.md` và `docs/roadmap/PROJECT_PHASES.md`.
  Phase đó là `ROADMAP-MVP-001`, đường thực thi ưu tiên tạm thời từ 2026-08-29,
  và chưa có work package nào được thi công.
- Tiêu chí dừng: MVP-00–07 đều có artifact thật và `ROADMAP-MVP-001` §7 pass sau
  khi reload checkpoint, hoặc — nếu ba vòng tune không đạt — trạng thái
  `blocked_with_evidence` với ledger failure đầy đủ.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Scope, randomization và eval protocol được khóa dưới một content hash | `configs/mvp/dexacquire-mvp-v0.yaml`, `configs/mvp/dexacquire-mvp-v0.eval-manifest.json` | verified |
| W-02 | Môi trường `QDGrasp-DexAcquire-MVP-v0` với `reset/step`, predicate §4 và residual có biên | `qdgrasp/mvp/{env,scene}.py` | verified |
| W-03 | Pinch prior LEAP fit trên sáu train width, loại trừ held-out | `qdgrasp/mvp/prior.py`, `configs/mvp/leap-pinch-prior-v0.json` | verified |
| W-04 | Expert minimum-intervention với demo có nhiễu và generator ledger | `qdgrasp/mvp/expert.py`, `evidence/mvp/round-3/demonstrations/` | verified |
| W-05 | Behaviour cloning và residual PPO, checkpoint mang fingerprint và reload probe | `qdgrasp/mvp/{bc,ppo,policy}.py`, `evidence/mvp/round-3/policy/` | verified |
| W-06 | Locked evaluation ba tier với ledger từng episode và Wilson bound | `scripts/evaluate_mvp.py`, `evidence/mvp/round-3/evaluation/` | verified |
| W-07 | Runner Kaggle/Colab có resume, pin commit public | `notebooks/mvp_grasp_policy.ipynb`, `scripts/build_mvp_notebook.py` | verified |
| W-08 | Cổng đóng MVP đọc artifact và áp ngưỡng của plan | `scripts/check_mvp.py`, `evidence/mvp/round-3/closure.json` | verified |
| W-09 | Model card ghi rõ kết quả âm của phần học | `docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md` | verified |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | manifest | `evidence/mvp/round-3/MANIFEST.json` | manifest có SHA-256 của 17 artifact |
| E-02 | checkpoint | `evidence/mvp/round-3/policy/ppo.pt` | `498d03ae6f062ed2…` |
| E-03 | checkpoint | `evidence/mvp/round-3/policy/bc.pt` | `2e470a384239c6d5…` |
| E-04 | report | `evidence/mvp/round-3/evaluation/ppo.json` | `75513aec16f7a76e…` |
| E-05 | report | `evidence/mvp/round-3/evaluation/controller_prior.json` | `4b2ff9b7d4698c4b…` |
| E-06 | report | `evidence/mvp/round-3/closure.json` | `ccc244aedad3fbcc…` |
| E-07 | ledger | `evidence/mvp/round-3/evaluation/ppo/tier-{a,b,c}.jsonl` | 600 episode thô |
| E-08 | artifact vòng hỏng | `evidence/mvp/round-1/`, `evidence/mvp/round-2/` | giữ nguyên, không xóa |
| E-09 | config | `configs/mvp/dexacquire-mvp-v0.yaml` | `scope_hash a897e36084c9ab11…` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `python -m pytest tests/mvp -q` | 0 | 49 test của scope, prior, môi trường, học và cổng đóng |
| T-02 | `python -m pytest tests/mvp/test_check_mvp_gate.py -q` | 0 | 10 test, mỗi test bẻ đúng một artifact và đòi cổng đóng bắt được |
| T-03 | `python scripts/check_mvp.py --runs evidence/mvp/round-3` | 0 | 38/38 check pass |
| T-04 | `python scripts/lock_mvp_scope.py --check` | 0 | eval manifest khớp scope |
| T-05 | `python scripts/check_docs.py --root .` | 0 | 141 file hợp lệ |
| T-06 | `python -m pytest tests -q` | 0 | không regression ngoài phạm vi MVP |

Kết quả đo của candidate cuối, trên seed đã khóa, sau khi reload checkpoint:
Tier A `100/100 = 100.0%`, Tier B `284/300 = 94.7%` (Wilson `91.5%`),
Tier C `187/200 = 93.5%` (Wilson `89.2%`), với `invalid_state=0`,
`safety_violation=0`, `checkpoint_reload_mismatch=0`.

Controller prior đo trên cùng seed: `100/100`, `284/300`, `188/200`.

## Việc chưa hoàn tất

- Phần học **không** cải thiện được controller prior. Tier A và B trùng đến từng
  episode, Tier C kém prior một episode, residual trung bình là `0.0039` đơn vị
  action (khoảng 32 micromet palm). Gate đạt được là nhờ controller.
- Không có independent review. Tác giả artifact không được tự ký verdict, và
  artifact giữ `experimental_non_release` nên không đi qua cổng release nào.
- Ngân sách ba vòng tune của §8 đã dùng hết. Không còn vòng nào cho scope này mà
  không bump version MVP.

## Sửa đổi phiên trước

Có. `docs/revisions/REV-20260831-001-grasp-policy-mvp-closure.md`, mức `N2`.
Hai sửa đổi đáng kể: success predicate được cài chặt hơn §4 (gộp tiếp xúc vào bộ
đếm liên tục về độ cao) và interface residual thiếu làm mượt. Cả hai làm đổi
`scope_hash` và `eval_manifest_hash`; tier, cỡ mẫu, miền randomization và dẫn
xuất seed không đổi.

## Bàn giao

- Trạng thái hiện tại: `ROADMAP-MVP-001` `complete`; `QDGrasp-Leap-Grasp-MVP` là
  artifact `experimental_non_release`. Mọi blocker của P3.4.3, P3.5, P4 và P5
  giữ nguyên; `QDGrasp-ContactRich-Active-Tiny` vẫn `release_blocked=true`.
- Bước hợp lệ tiếp theo, theo `ROADMAP-MVP-001` §10 và đọc từ failure evidence
  của MVP: mở rộng miền cho tới khi controller prior thực sự hỏng trước khi đầu
  tư thêm vào kiến trúc policy. Cụ thể là bỏ giả định privileged pose, ra ngoài
  họ cuboid, rồi thêm Allegro. Trong phạm vi hiện tại phần học không có gì để
  học.
- Điều kiện phiên sau phải kiểm trước: `python scripts/check_mvp.py --runs
  evidence/mvp/round-3` phải còn trả `0` trên worktree sạch, và `scope_hash`
  phải còn khớp hằng số đã pin trong `tests/mvp/test_mvp_scope.py`.
