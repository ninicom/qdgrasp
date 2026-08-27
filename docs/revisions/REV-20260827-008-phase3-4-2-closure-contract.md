---
document_id: REV-20260827-008
document_type: revision_record
revision_schema: 2
title: Hoàn thiện P3.4.2 thành closure contract đầy đủ cho P3.4
status: complete
date: 2026-08-27
record_id: REV-20260827-008
session_id: SESSION-20260827-P3.4.2-CLOSURE-CONTRACT
created_at: 2026-08-27T21:25:22+07:00
author: codex-primary-agent
revises:
  - session_id: ROADMAP-P3.4.2-001
    artifact: docs/roadmap/PHASE3_4_2_CORRECTNESS_RECOVERY_PLAN.md
    revision: 10b5ca96de2c42d42e23b128aba67ade97b9490a43d3d14fb049906c8e2bb18d
reason: "P3.4.2 v1.0 xử lý đủ ba blocker hiện tại nhưng chưa ánh xạ tường minh toàn bộ parent §10/§11/§15/§16, chưa khóa full dataset/ablation/command matrix và còn một scope-revision escape hatch cho Shadow."
necessity: N2
impact: "Nâng ROADMAP-P3.4.2-001 lên v1.1.0 với parent gate coverage matrix, full CUDA/dataset/review gates, executable closure checker contract và quy tắc Shadow không thể bị scope-out để tính là pass."
---

# REV-20260827-008 — P3.4.2 closure contract

## 1. Liên kết truy vết

- Phiên thực hiện: `SESSION-20260827-P3.4.2-CLOSURE-CONTRACT`.
- Artifact trước sửa: `ROADMAP-P3.4.2-001@1.0.0`, checksum trong metadata.
- Artifact sau sửa: `ROADMAP-P3.4.2-001@1.1.0`, SHA-256
  `102e47c7e3b039809ff735195179852dbf5304e20b56fd31d7634e10a3a22c35`.
- Requirement nguồn: P3.4 parent §9–§16 và yêu cầu người dùng làm plan đủ để
  đóng exact P3.4 gate.

## 2. Lý do chỉnh sửa

P3.4.2 v1.0 đã có đường xử lý GPU correctness, Shadow control semantics và
independent review, nhưng gate cuối chỉ nhắc các blocker mới. Parent P3.4 còn
yêu cầu paired static-fail/dynamic-pass cho ba hand, hai positive class mỗi hand,
negative coverage, ba tầng CPU/GPU parity, P1/P2 CUDA replay, deterministic
regeneration/render evidence, controlled ablation và exact command matrix.

Nếu không đưa chúng vào closure contract, executor hoặc reviewer có thể đóng
P3.4.2 sau một Shadow positive và một GPU benchmark dù parent §16 chưa pass.
V1.1 loại ambiguity đó bằng machine-checkable coverage matrix.

## 3. Mức độ cần thiết

- Mức: `N2` vì thay đổi điều kiện nghiệm thu và release decision.
- Thời hạn: trước P3.4.2 execution hoặc tạo release candidate.
- Nếu trì hoãn: plan không đủ làm nguồn duy nhất cho closure; P3.4 tiếp tục
  blocked và review packet không được phát hành.

## 4. Phạm vi và tác động

| Lĩnh vực | V1.0 | V1.1 | Tác động |
| --- | --- | --- | --- |
| CUDA gate | correctness/speed/VRAM chính | full §10 gồm P1/P2, 3 hand, 3 parity tier, per-hand finalist | cao |
| Dataset | Shadow positive + ContactRich v2 tổng quát | full §11 modes/scenes/positive/negative/pair/regeneration/render | cao |
| Ablation | disclosure chung | rerun controlled ablation + yield/compute trade-off | vừa |
| Review | independent exact RC | scope force math/parity/damage/3-hand/determinism/ablation rõ | cao |
| Closure | sáu successor conditions | parent coverage matrix + exact commands + nine fail-closed conditions | cao |

Không runtime/data/evidence nào được sửa trong revision này; plan chỉ định công
việc và gate tương lai.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Evidence |
| --- | --- | --- |
| `CH-001` | Thêm §8.2 matrix cho toàn bộ CUDA/Kaggle parent gate | P3.4.2 v1.1 §8.2 |
| `CH-002` | Thêm full ContactRich v2 contract, gồm hai positive class/hand và negative fixtures | §8.3 |
| `CH-003` | Ánh xạ tám điều kiện parent §16 tới owner/evidence/verdict | §8.4 |
| `CH-004` | Khóa controlled ablation, disclosure và post-validation invalidation | §8.5 |
| `CH-005` | Thêm exact local/Kaggle/post-review command matrix | §8.6 |
| `CH-006` | Xóa khả năng scope-out Shadow để tính P3.4 pass | §4.4/§9 |
| `CH-007` | Giới hạn post-review diff vào verdict/governance record | §5.2/§8.6 |

## 6. Xác minh

| Verification ID | Phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái |
| --- | --- | --- | --- | --- |
| `V-001` | manual parent §10/§11/§15/§16 → successor matrix | không parent condition bị orphan | đủ owner/evidence/verdict | pass |
| `V-002` | `git diff --check` | không whitespace error | không có output, exit 0 | pass |
| `V-003` | `python3 scripts/check_docs.py --root .` | document graph/front matter pass | 110 file checked, pass | pass |
| `V-004` | simulation/CUDA workload | ngoài plan-only revision | không chạy | not_run_nonblocking |

- Runtime regression không chạy vì không sửa runtime.
- Closure commands trong plan là acceptance cho execution tương lai, không được
  ghi pass trong revision tài liệu này.
- Rollback: kế hoạch có thể trở lại v1.0 bằng successor revision; record này và
  parent evidence không bị xóa.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `REV-20260827-007` còn đúng cho P3.4.2 v1.0 nhưng không phải closure contract
  cuối; revision này là delta bắt buộc.
- Không có review verdict cũ cần thu hồi vì packet P3.4.1 chưa được ký.
- P3.4 vẫn `pending`; ContactRich v1 vẫn `release_blocked=true`.
- Loại review tương lai: full independent review trên exact P3.4.2 RC.

## 8. Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-27, Asia/Bangkok.
- Người kiểm: chưa có independent reviewer; đây là plan revision.
- Kết luận: revision closure-contract hoàn tất; P3.4.2 execution vẫn `active`.
- Bản ghi: `REV-20260827-008`.
