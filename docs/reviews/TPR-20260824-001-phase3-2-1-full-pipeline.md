---
document_id: TPR-20260824-001
document_type: third_party_review
title: Independent review Phase 3.2.1 full-pipeline correctness
status: complete
date: 2026-08-24
revises: none
report_id: TPR-20260824-001
reviewed_session: SESSION-20260824-P3.2.1
reviewed_revision: uncommitted-worktree-with-pinned-provenance
reviewer: codex-independent-review
reviewer_organization: codex-internal-independent-review
review_date: 2026-08-24
independence: internal_independent
verdict: pass
max_severity: none
---

# TPR-20260824-001 — Independent review Phase 3.2.1

## 1. Đối tượng và phạm vi

Reviewer read-only kiểm toàn bộ staged, unstaged và untracked worktree của
P3.2.1, không viết implementation hay evidence. Hai review session độc lập là
`01a0326a-5865-7c01-b9d9-2282fdb856a7` và
`01a0326e-34fb-78d2-a723-20727af0226f`. Đây là review nội bộ độc lập, không phải
external/human review.

Phạm vi gồm solver math và active mask, palm hypotheses, proposal identity,
exact collision admission, task-space transmission, dynamic predicate,
generated-reachable fixtures, canonical-independent matrix, mutation tests và
release scope.

## 2. Tuyên bố độc lập

Reviewer không viết implementation, không sửa project và chỉ trả finding cho
primary agent xử lý. Mức độc lập là `internal_independent`.

## 3. Môi trường và phương pháp

Review kiểm worktree, chạy full test và các test task-space liên quan, sau đó
đối chiếu source contact-state, proposal normals và fail-closed rollout paths.

## 4. Bằng chứng

Raw manifests nằm tại `evidence/phase3_2_1/p10-p13-release-gate/`; mỗi manifest
pin source, robot, object và rollout-protocol provenance. Hai review session và
toàn bộ gate outputs được định danh trong tài liệu này.

## 5. Kết quả đối chiếu

P00–P13 đáp ứng gate cục bộ và release gate; P14 đáp ứng sau khi finding vòng
đầu được sửa và vòng review thứ hai không còn correctness finding xác định.

## 6. Phát hiện

Vòng đầu tìm thấy một finding xác định mức P2: khi `active_fingers` toàn false
nhưng command có task displacement, rollout gọi `np.concatenate` với danh sách
rỗng thay vì fail closed. Implementation đã được sửa để reject mọi active count
nhỏ hơn `min_active_fingers` bằng stage `insufficient_active_fingers` trước
`mj_step`; regression được tham số hóa cho active count 0 và 1.

Vòng thứ hai chạy lại inspection và test liên quan, kết luận: “No definite
correctness issues were identified in the current staged, unstaged, or untracked
changes.” Không còn finding mở.

## 7. Evidence xác minh cuối

| Gate | Kết quả |
| --- | --- |
| `scripts/check_phase3_2_1.py` | PASS; 100 contract/mutation tests; generated positive 3/3 hand; deterministic run A/B |
| Generated payload SHA-256 | `5a34f9d8e7f6568c7dc28e1b5f70c0421b1910ef9fe038d75e33efec55b3c563` |
| Canonical payload SHA-256 | `20450c16e2bc74ca6079a7407b9d2573c05784335dcdac3c88f0a6531c3a8eaa` |
| Canonical matrix | 12/12 cells measured; 0 accepted |
| Full regression | 366 passed, 1 skipped, 0 xfail |
| Environment/docs/Phase 2/Phase 3.2 | PASS |

## 8. Điều kiện còn lại

Không còn điều kiện bắt buộc để đóng P3.2.1 trong claim scope. Canonical yield
và các gate còn lại của P3 là work ngoài verdict này.

## 9. Kết luận

Verdict `pass` cho claim hẹp sau:

- full pipeline tự sinh ít nhất một generated-reachable positive cho LEAP,
  Allegro và Shadow mà không nhận oracle q/palm/contact;
- mutation và fail-closed contracts bắt các lỗi solver, collision, command và
  dynamic predicate đã liệt kê trong plan;
- regeneration generated là deterministic trên protocol đã pin;
- có thể gỡ `release_blocked` của Shadow cho scope này.

Verdict không chứng minh canonical yield/generalization: ma trận độc lập hiện là
`0/12`. Nó cũng không đóng P3, không cho phép gọi fixture generated là canonical
dataset và không thay các gate ablation/regeneration còn lại.

## 10. Chữ ký

- Reviewer: `codex-independent-review`.
- Ngày: 2026-08-24, Asia/Bangkok.
- Verdict: pass; không còn finding correctness xác định trong scope.
