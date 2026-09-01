---
document_id: SESSION-20260901-001
document_type: session_report
title: Audit lỗi liên mô-đun và bổ sung corrective remediation vào kế hoạch chính
status: complete
date: 2026-09-01
session_id: SESSION-20260901-001
author: codex-agent
revises: PLAN-V2@4.5.0
revision_reason: Audit toàn dự án tái hiện các lỗi an toàn và correctness chỉ xuất hiện khi ghép dataset, protocol, model, runner và artifact lifecycle; PLAN-V2@4.5.0 chưa có corrective track để chặn các đường này.
necessity: N3
impact: PLAN-V2 tăng lên 4.6.0 và thêm §9; DGN-Open-Tiny cùng Flow evidence cũ không còn đủ điều kiện làm release evidence, P4/P5 phải qua corrective gates mới.
related_plan: PLAN-V2@4.6.0
---

# SESSION-20260901-001 — Audit liên mô-đun và corrective plan

## Mục tiêu phiên

- Quét lại toàn bộ dự án theo chuỗi dữ liệu và runtime thay vì chỉ kiểm module
  riêng lẻ.
- Tái hiện các lỗi chỉ xuất hiện khi ghép generator → manifest → protocol →
  adapter/collator → model/loss → Runner → resume/bundle/export/evaluation.
- Ghi toàn bộ hướng xử lý vào `PLAN.md` theo đúng chính sách revision, không sửa
  code sản phẩm trong phiên lập kế hoạch.

## Việc đã hoàn tất

| ID | Kết quả | Output | Trạng thái |
|---|---|---|---|
| `A-01` | Audit data/protocol/supervision bằng agent chuyên trách và probe trên public path | Finding `COR-00`…`COR-04`, `COR-12` | verified |
| `A-02` | Audit Flow/loss/Runner/resume/bundle/export | Finding `COR-05`…`COR-10` | verified |
| `A-03` | Audit MVP fingerprint/checkpoint/action và packaged legacy surface | Finding `COR-00`, `COR-11`, `COR-12` | verified |
| `A-04` | Đối chiếu full suite, canonical dataset audit, Phase 5 input gate, Ruff và Mypy | Baseline trong `PLAN.md` §9.1 | verified |
| `A-05` | Phân loại artifact/evidence bị ảnh hưởng và dependency order | `PLAN.md` §9.1, §9.11 | complete |
| `A-06` | Bổ sung corrective track G0–G7, closure gates và interaction-test matrix | `PLAN-V2@4.6.0` §9 | complete |
| `A-07` | Tạo revision record N3 cho active plan | `docs/revisions/REV-20260901-001-cross-component-remediation-plan.md` | complete |

Không file code, config, dataset, checkpoint hoặc raw evidence nào bị sửa trong
phiên này.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/kết quả | Tham chiếu |
|---|---|---|---|
| `E-01` | active plan trước sửa | `PLAN-V2@4.5.0` | SHA-256 `0395aba6688bd0a8479cf17d2c33e3fcbe45a8fcf6aae2761cf44d43cb0b69f0` |
| `E-02` | active plan sau sửa | `PLAN-V2@4.6.0` | SHA-256 `97b916d0e119b1cb0ebc8a380505f6084a264364ddd56bc114955396b2f44c7b` |
| `E-03` | immutable Git base | repository state trước sửa tài liệu | commit `159f8607ad72295528e9051ff3cccc3486e052a1` |
| `E-04` | dataset audit | `python scripts/check_dataset_manifest.py --root datasets/dgn-open-tiny` | fail trên ba generator source hash drift |
| `E-05` | Phase 5 gate | `python scripts/check_phase5_inputs.py` | LEAP `1/32`, Allegro `2/42`, exit non-zero |
| `E-06` | protocol leakage probe | public `DgnOpenDataset` train so với protocol view | `12` sample `comp_*` ngoài protocol trên mỗi active hand |
| `E-07` | security probes | temporary malicious shard/checkpoint/path fixtures | absolute shard load, pickle reducer execution và sibling-prefix escape đều tái hiện |
| `E-08` | Flow/Runner probes | joint round-trip, quality ranking, validation/resume | joint error `0.636822 rad`; 6 grasp/1 score; validation đổi RNG và split-resume không bit-exact |
| `E-09` | lifecycle probes | bundle/from_bundle/export/cross-robot resume | semantic bundle mismatch được nhận; robot/v2 self-load và Flow export fail; LEAP resume được Allegro nhận |
| `E-10` | MVP probes | guarded load so với evaluator worker | foreign fingerprint bị direct guard từ chối nhưng worker nhận rồi report stamp fingerprint hiện tại |

Các security probe chỉ dùng thư mục tạm và marker vô hại; không payload hoặc
artifact tạm nào được giữ trong repository.

## Kiểm tra đã chạy

| Test ID | Lệnh/phương pháp | Kết quả | Exit/đánh giá |
|---|---|---|---|
| `T-01` | `.venv/bin/pytest -q` | `1306 passed, 1 skipped, 93 subtests`, 23 warning | pass |
| `T-02` | `.venv/bin/python scripts/check_dataset_manifest.py --root datasets/dgn-open-tiny` | fail tại source provenance drift | expected blocker |
| `T-03` | `.venv/bin/python scripts/check_phase5_inputs.py` | 3 positive toàn logical train view; dưới floor | expected blocker |
| `T-04` | `.venv/bin/ruff check qdgrasp scripts tests --statistics` | 514 finding, 329 auto-fixable | debt recorded |
| `T-05` | `.venv/bin/mypy qdgrasp --no-error-summary` | 1480 finding | debt recorded |
| `T-06` | Mypy active-core slice với `--ignore-missing-imports` | 112 finding | debt recorded |
| `T-07` | `.venv/bin/python scripts/check_docs.py --root .` sau cập nhật | 151 file được kiểm tra | pass |

## Việc chưa hoàn tất

- Chưa sửa implementation của `COR-00`…`COR-12`; đây là phiên audit và lập kế
  hoạch, không phải phiên remediation.
- Chưa regenerate DGN, chưa retrain Flow/MVP và chưa tạo evidence thay thế.
- Chưa chạy lại CUDA/AMP gate hoặc independent review; các blocker cũ giữ nguyên.
- Các work package G0–G7 trong `PLAN.md` §9 phải được triển khai theo dependency
  và PR map tại §9.11.

## Sửa đổi phiên trước

Có: `docs/revisions/REV-20260901-001-cross-component-remediation-plan.md`.

- Claim cũ: PLAN-V2@4.5.0 cho phép tiếp tục P4/P5 theo các module gate hiện hữu
  và không có corrective track cho artifact trust, protocol-to-trainer binding,
  Flow semantics hay resume/bundle identity.
- Claim mới: PLAN-V2@4.6.0 chặn full training/evaluation/release cho tới khi
  `COR-00`…`COR-12` và G0–G7 có regression evidence.
- Mức `N3`: có đường thực thi pickle từ input không tin cậy và nhiều đường có
  thể tạo release artifact mang sai data/protocol/model/robot identity.

## Bàn giao

1. Bắt đầu bằng `R1`/G0: characterization tests và hard-stop, không sửa semantic
   lẫn vào cùng commit.
2. `R2` đóng artifact I/O trước khi bất kỳ loader/protocol/data regeneration nào
   được mở lại.
3. Chỉ regenerate DGN sau khi unified schema, protocol view và validity mask đã
   merge; nếu làm trước, dataset sẽ phải sinh lại lần nữa.
4. Joint/quality fix phải đi trước deterministic Runner/resume vì nó làm đổi
   model/checkpoint semantics.
5. Không dùng evidence Flow/DGN cũ làm baseline release; MVP round 3 chỉ cần
   revalidation nếu action distribution không đổi.
6. Khi mọi G0–G7 pass, tạo revision/evidence mới và yêu cầu independent reviewer
   ký scope security, protocol isolation và artifact identity.
