---
document_id: SESSION-20260821-007
document_type: session_report
title: Sửa findings audit train-argument registry
status: complete
date: 2026-08-21
session_id: SESSION-20260821-007
author: codex-primary-agent
revises:
  - SESSION-20260821-006
  - REV-20260821-005
revision_reason: TPR-20260821-003 chứng minh checker và evidence của snapshot 7fb01a7 có false-negative S1 và các contract/governance gap S2/S3.
necessity: N2
impact: Siết exact schema/semantic/source/reference gates và sửa documentation contract; không thay đổi danh sách 127 tên hay tuyên bố model runtime.
related_plan: PLAN-V2
review_outcome: TPR-20260822-001
---

# SESSION-20260821-007 — Audit remediation

## Mục tiêu phiên

Đóng F-001–F-009 của `TPR-20260821-003` mà không rewrite snapshot `7fb01a7`,
thêm negative tests và chuẩn bị exact commit cho delta review.

## Việc đã hoàn tất

| ID | Việc đã triển khai | Bằng chứng hiện tại |
|---|---|---|
| W-01 | Exact top-level/section/field schema và safe scalar subset | Invalid YAML-like mutation bị reject |
| W-02 | Semantic fingerprint khóa canonical/custom/legacy/API/extensions | Default/group/target/resume/merge mutations bị reject |
| W-03 | Đồng bộ Markdown rows/default/disposition với YAML | Checker chạy trong cả hai mode |
| W-04 | Ghi đúng hai nhánh merge `cfg=` và special constraints | Registry + Markdown |
| W-05 | Thêm immutable reference lock checker | Lock-only và hai checkout full pass |
| W-06 | Ghi DGN2 README CC BY-NC evidence, giữ distribution block | README SHA-256 đã pin |
| W-07 | Conditional full source checks trong hook/Git gates | `.references/` hiện diện thì full bắt buộc |
| W-08 | Quy định Git-versioned active living contracts | Policy + version/latest revision pointers |
| W-09 | Revision schema 2 đủ tám mục | Template, validator và regression test |
| W-10 | Mở rộng regression suite | 24/24 pass trước khi thêm hồ sơ này |
| W-11 | Gửi exact snapshot cho delta reviewer | `fdb60eb`, verdict `fail/S1` |

## Bằng chứng

- Finding source: `TPR-20260821-003`, verdict fail/S1.
- Failed snapshot được giữ ở commit `7fb01a7`, tree `8a6b6cc1`.
- Semantic fingerprint mới:
  `2a7d96fbf6140b196b9e5deae364e4ac34f2fc04ea66bdf7b62a143ec843febc`.
- Reference lock SHA-256:
  `921e798163abace643c8a825559425e1bbd16d19476e1464436615fa00b193b7`.
- Exact remediation commit:
  `5d86a426ff7f250bc7a77e80e241610d10a9958a`, tree
  `9020e17aae346d71233f501db658314c28b3a065`, committed at
  `2026-08-21T23:56:27+07:00`.
- Replay log: `docs/reports/evidence/TRAIN-ARGS-20260821-audit-remediation.txt`,
  4.178 byte, SHA-256
  `fd7888018bfa8e691dd8a6e236ffbca9fbfb6e864f483ed97522abec0ced037f`.
- Delta review: `TPR-20260822-001`, evidence SHA-256
  `739837d1dd3f559594cb68a99ae8111311d1b53e4755907f6ba5dd606693a29c`,
  verdict `fail`, max severity `S1`.

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit | Kết quả |
|---|---|---:|---|
| T-01 | `python3 scripts/check_references.py --lock-only` | 0 | Immutable lock pass |
| T-02 | `python3 scripts/check_references.py --source-root .references` | 0 | 2 required clones clean/pinned |
| T-03 | `python3 scripts/check_train_args.py --registry-only` | 0 | Exact semantic + Markdown manifest |
| T-04 | `python3 scripts/check_train_args.py --source .references/ultralytics` | 0 | Full source pass |
| T-05 | `python3 scripts/check_docs.py --root .` | 0 | 33 tài liệu pass tại remediation commit |
| T-06 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` | 0 | 24/24 pass |

## Việc chưa hoàn tất

- Typed/lexical registry fingerprint, normative Markdown pin, structured
  quantize aliases và standalone clean-source gate chưa được triển khai.
- F-001/F-002/F-004/F-005/F-010 còn mở; snapshot này không được merge.
- Model/runtime/CPU-CUDA smoke vẫn chưa được implement và không thuộc claim.

## Sửa đổi phiên trước

Có, mức `N2`. `REV-20260821-006` ghi chi tiết artifact/hash và thay thế mức độ
đầy đủ của REV-005; TPR-003 giữ nguyên làm bằng chứng failure. Danh sách 127 tên
không đổi, nhưng guarantee checker/evidence/governance được sửa đáng kể.

## Bàn giao

Phiên kết thúc với kết quả audit thất bại, không phải acceptance. Giữ nguyên
`TPR-20260822-001` và evidence; mở session/revision N2 mới để sửa dialect v2,
sau đó delta review lại exact commit trước merge.
