---
document_id: SESSION-20260822-006
document_type: session_report
title: Vendor Ultralytics pinned commit vào vendor/ (chưa đổi tên, chưa strip)
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-006
author: codex-primary-agent
revises: none
related_plan: PLAN-V2
---

# SESSION-20260822-006 — Vendor import Ultralytics (bước 1 của M1 fork)

## Mục tiêu phiên

Bắt đầu M1 (`PLAN.md` §4: "Tạo product branch từ Ultralytics pinned commit,
đổi namespace thành `dexgrasp`") sau khi maintainer xác nhận rõ ràng
("bắt đầu fork luôn") và căn cứ pháp lý đã đúng (`REV-20260822-005`, public
AGPL-3.0). Bước đầu tiên: vendor nguyên trạng package Python `ultralytics/`
tại commit đã pin `329682a`, KHÔNG đổi tên, KHÔNG strip bất kỳ file nào — để
mọi commit sau (đổi namespace, loại YOLO tasks/trackers/HUB/solutions) có
baseline hash rõ ràng, diff được.

Không sửa `.references/ultralytics` (nguồn read-only, bị hash-lock bởi
`scripts/check_references.py`); trích xuất bằng `git archive` (read-only) rồi
`tar -x` sang thư mục mới, không checkout/switch trên clone gốc.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Tạo `feature/ultralytics-fork-import` từ `develop` (sau khi merge `feature/train-argument-registry`) theo đúng `scripts/git/start_feature.sh` | branch `feature/ultralytics-fork-import` | `git branch --show-current` |
| W-02 | Trích xuất `ultralytics/`, `LICENSE`, `pyproject.toml` từ `.references/ultralytics` tại commit `329682a29d27203582ba30e519340f95abccc6a6` bằng `git archive` (không checkout) | `vendor/ultralytics-329682a/` (365 file) | `diff -rq` với `.references/ultralytics` |
| W-03 | Xác nhận nội dung byte-for-byte giống clone pinned; `LICENSE` khớp hash pin trong `references.lock.yaml` | — | `sha256sum` khớp `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` |
| W-04 | Xác nhận `.references/ultralytics` không bị đụng | — | `scripts/check_references.py --source-root .references` pass sau khi vendor |
| W-05 | Tạo manifest SHA-256 của toàn bộ 365 file vendor để làm baseline cho commit đổi tên/strip sau này | `docs/reports/evidence/VENDOR-ULTRALYTICS-20260822-329682a-manifest.sha256.txt` | `wc -l` = 365 |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | căn cứ pháp lý | `REV-20260822-005` | public AGPL-3.0, không cần văn bản cấp phép thương mại riêng |
| E-02 | nguồn pinned | `.references/ultralytics` @ `329682a29d27203582ba30e519340f95abccc6a6` | khớp `references.lock.yaml.sources.ultralytics.commit` |
| E-03 | vendor manifest (365 file) | `docs/reports/evidence/VENDOR-ULTRALYTICS-20260822-329682a-manifest.sha256.txt` | `853e93126ca4a247d457b4907a61e6e181d5c942a8a6321956dc2dbbd7d9ee1b` |
| E-04 | LICENSE vendor == LICENSE pinned | `vendor/ultralytics-329682a/LICENSE` | `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `diff -rq .references/ultralytics/ultralytics vendor/ultralytics-329682a/ultralytics` | 0 | không khác biệt |
| T-02 | `diff -q .references/ultralytics/{LICENSE,pyproject.toml} vendor/ultralytics-329682a/{LICENSE,pyproject.toml}` | 0 | không khác biệt |
| T-03 | `git diff --check` (365 file mới staged) | 0 | không có whitespace/EOF error |
| T-04 | `python3 scripts/check_references.py --lock-only` | 0 | pass |
| T-05 | `python3 scripts/check_references.py --source-root .references` | 0 | pass — `.references/ultralytics` vẫn sạch, không bị đụng |
| T-06 | `python3 scripts/check_train_args.py --registry-only` | 0 | pass |
| T-07 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | pass |
| T-08 | `python3 scripts/check_docs.py --root .` | 0 | `46 file đã kiểm tra` (vendor/ không thuộc phạm vi checker) |
| T-09 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | 0 | `38/38 pass` |
| T-10 | `bash -n .githooks/* scripts/git/*.sh scripts/release_gate.sh` | 0 | pass |

## Việc chưa hoàn tất

- **Chưa đổi namespace** `ultralytics` → `dexgrasp`; `vendor/ultralytics-329682a/`
  vẫn giữ nguyên tên/đường dẫn upstream. Đây là commit kế tiếp, dựa trên
  manifest `E-03` làm baseline diff.
- **Chưa loại bỏ** YOLO tasks, trackers, HUB, solutions, integrations ngoài
  scope theo `PLAN.md` §4/M1 — toàn bộ 365 file vendor còn nguyên, kể cả phần
  sẽ bị xóa.
- **Chưa dựng** façade `DexGrasp`, CLI `dexgrasp`, `task_map`, `GraspResults`
  hay bất kỳ phần nào của `PLAN.md` §2/§3 — session này chỉ là vendor drop.
- **`references.lock.yaml`** vẫn còn `distribution_status` cũ, mâu thuẫn với
  quyết định public AGPL-3.0; chưa sửa (cần maintainer quyết định giá trị mới,
  field bị hash-lock).
- Chưa cập nhật `references.lock.yaml`/`PLAN.md` §1 để ghi provenance của
  `vendor/ultralytics-329682a/` như một "phần đã port" — nên làm ở commit đổi
  namespace, khi biết chính xác phần nào thực sự được giữ lại.

## Sửa đổi phiên trước

Không. Đây là bổ sung mới (vendor import), không sửa lại bất kỳ artifact nào
đã được ghi "hoàn tất" ở phiên trước.

## Bàn giao

`vendor/ultralytics-329682a/` là baseline pristine, đã xác minh khớp
`.references/ultralytics` byte-for-byte và có manifest SHA-256 bất biến. Bước
hợp lệ tiếp theo: `git mv` đổi `vendor/ultralytics-329682a/ultralytics` →
`dexgrasp/`, xóa phần ngoài scope (YOLO tasks/trackers/HUB/solutions), mỗi
việc một commit riêng, diff so với `E-03` để chứng minh không đổi nội dung
ngoài dự kiến. Không claim bất kỳ phần nào của `dexgrasp` "hoạt động" cho tới
khi có façade/CLI/test thật.
