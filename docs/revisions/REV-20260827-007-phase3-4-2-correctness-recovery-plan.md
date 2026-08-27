---
document_id: REV-20260827-007
document_type: revision_record
revision_schema: 2
title: Tạo P3.4.2 từ findings hậu báo cáo P3.4/P3.4.1
status: complete
date: 2026-08-27
record_id: REV-20260827-007
session_id: SESSION-20260827-P3.4.2-PLAN
created_at: 2026-08-27T21:17:45+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260827-002
    artifact: docs/sessions/SESSION-20260827-P3.4-AND-P3.4.1.md
    revision: 692ccc7706aa506759cb64ed75da1d1bab9f1fbb0248d32484539fe5e6339a37
  - session_id: ROADMAP-P3.4.1-001
    artifact: docs/roadmap/PHASE3_4_1_FIX_PLAN.md
    revision: c03c11981fcfbc279fe5d8f5ab7d3b96aa5c6067e53cb34159e87f007c481c09
  - session_id: ROADMAP-P3.4-001
    artifact: docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md
    revision: d2c197b88ff06c9d633c061bd861a66ee5070b69e2d1add9620eb3aeb6ef0931
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: b2a9eeb0eb7a65e5e348ca7fa964e5566e2df41697b47ca4d0c54459dd17827e
reason: "Báo cáo hậu P3.4.1 còn package-count mâu thuẫn, gọi packet cũ là sẵn sàng, thu hẹp GPU root cause quá sớm và chưa tách joint tracking khỏi actuator/tendon tracking của Shadow."
necessity: N2
impact: "Tạo ROADMAP-P3.4.2-001, supersede execution path P3.4.1 nhưng giữ diagnostic history; bổ sung fail-closed GPU triage, Shadow controllable-space decision tree, corrected ledger và rebuild independent-review packet."
---

# REV-20260827-007 — Kế hoạch Phase 3.4.2

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260827-P3.4.2-PLAN`.
- Bản ghi hoàn tất: `REV-20260827-007`.
- Finding nguồn: `SESSION-20260827-002`, source code validator/sanitizer và
  `evidence/phase3_4_1/review-packet/`.
- Artifact mới: `docs/roadmap/PHASE3_4_2_CORRECTNESS_RECOVERY_PLAN.md`, SHA-256
  `10b5ca96de2c42d42e23b128aba67ade97b9490a43d3d14fb049906c8e2bb18d`.
- Checksum sau sửa của P3.4.1 plan:
  `863809fc81e381283b043493db6ecbaf7cfbf0ad039c63f5af2e81230535f020`.
- Checksum sau sửa của parent P3.4 plan:
  `0257b6732cf5c4404d169c2a4599dd37c2ccde2730aad3b68a33f4800aea8e3c`.
- Checksum sau sửa của project roadmap:
  `f696f30727f370160c87d027b7b45802000e2b139996502387cd575a5915ebdc`.
- Session report giữ nguyên checksum; revision này đính chính bằng successor,
  không sửa báo cáo đã hoàn tất.

## 2. Lý do chỉnh sửa

Session report là handoff hữu ích nhưng chưa thể dùng trực tiếp làm release
ledger. Headline ghi P3.4 `14/18`, bảng liệt kê output cho 16 package và project
roadmap trước đó ghi `15/18`. Report cũng gọi review packet là sẵn sàng dù packet
khóa commit cũ, tự ghi worktree bẩn, manifest count không khớp và còn disclosure
Shadow đã bị rút lại.

Hai kết luận kỹ thuật cần thêm bước phân loại. GPU identical-world divergence
chứng minh output chưa tái lập nhưng chưa loại trừ đủ để gọi root cause duy nhất
là race/uninitialized memory. Shadow failure label `actuator_tracking` trong code
là phép AND giữa actuator-coordinate và joint tracking; chưa có attribution theo
dimension/stage nên chưa đủ căn cứ đổi controller hoặc nới predicate.

P3.4.2 được tạo để sửa đúng protocol ra quyết định trước khi sửa runtime: ledger
reconciliation, sanitizer fail-closed, first-bad-tick/upstream isolation,
controllable-space Shadow semantics và packet regeneration sau exact RC.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: thay đổi ảnh hưởng trực tiếp protocol validation, release evidence,
  controller decision và independent review; không phải biên tập.
- Thời hạn: trước khi tiếp tục tuyên bố P3.4 release candidate, phát hành
  ContactRich v2 hoặc dùng dynamic contact data làm P4/RL release input.
- Nếu trì hoãn: P3.4 giữ `pending`, packet P3.4.1 giữ `invalid_for_final_review`,
  ContactRich v1 giữ `release_blocked=true`.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | Chưa đổi; GPU divergence và Shadow blocker còn nguyên | Fix được chọn sau classification gate | cao, chưa triển khai | P3.4.2-01–08 |
| API/schema/config | Tracking label gộp; chưa có requested/projected/actuator split | Plan yêu cầu controllable-space evidence và conditional safe-set controller | vừa | version schema/controller nếu classification yêu cầu |
| Dữ liệu/checkpoint | ContactRich v1 release-blocked | Chỉ tạo v2 sau Shadow/GPU regression pass | cao | P3.4.2-08 |
| CPU/GPU | Speed pass, correctness/VRAM fail | zero invalid + outcome parity + allocator-correct VRAM | cao | P3.4.2-01–04 |
| Số liệu/kết luận | package count/packet readiness không nhất quán | successor ledger và packet invalidation bắt buộc | cao | P3.4.2-00/09 |

Tóm tắt tác động khớp metadata: phiên này chỉ thay living plans/revision record,
không sửa runtime, raw evidence, threshold hoặc dataset.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Tạo plan P3.4.2 với ba nhánh GPU/Shadow/review | `ROADMAP-P3.4.2-001@1.0.0` | SHA-256 mục 1 |
| `CH-002` | Chuyển P3.4.1 thành diagnostic history `superseded` và liên kết successor | `ROADMAP-P3.4.1-001@1.1.0` | plan §9 |
| `CH-003` | Liên kết parent P3.4 tới correctness-recovery successor | `ROADMAP-P3.4-001@1.2.0` | parent §18 |
| `CH-004` | Cập nhật project roadmap và khóa ContactRich/RL release input | `ROADMAP-001@1.23.0` | project Phase 3.4.2 entry |
| `CH-005` | Ghi packet cũ không hợp lệ cho final review và yêu cầu rebuild exact clean RC | `ROADMAP-P3.4.2-001` §1.4/§5 | packet JSON/checklist local |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `git diff --check` | không whitespace error | không có output, exit 0 | pass | terminal phiên |
| `V-002` | manual baseline → classification → conditional fix → gate review | không có đường nới gate hoặc tự ký verdict | speed/safety/Shadow/reviewer gate giữ fail-closed | pass | P3.4.2 §1–§8 |
| `V-003` | `python3 scripts/check_docs.py --root .` | document graph/front matter pass | 109 file checked, pass | pass | terminal phiên |
| `V-004` | simulation/test workload | không thuộc plan-only revision | không chạy theo phạm vi yêu cầu | not_run_nonblocking | không đổi runtime/data |

- Regression runtime đã chạy lại: không có; phiên chỉ thay tài liệu kế hoạch.
- Kiểm tra chưa chạy: mô phỏng/CUDA/Shadow; chính các workload này là backlog
  P3.4.2, không được ghi pass trong revision plan.
- Rollback: successor revision có thể đưa P3.4.1 về `active` và gỡ link P3.4.2;
  không xóa revision/session/evidence cũ.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `SESSION-20260827-002` giữ nguyên như historical report; package count,
  root-cause strength và packet-readiness được successor ledger đính chính.
- Packet P3.4.1 không có verdict nên không có third-party decision để thu hồi;
  packet vẫn được giữ nhưng không hợp lệ cho final review.
- `TPR-20260825-001` chỉ còn hiệu lực cho exact P3.3 candidate của nó, không mở
  rộng sang Shadow controller/validator hoặc ContactRich v2.
- Loại kiểm lại cần thiết: full independent review sau P3.4.2-09.
- Không có risk acceptance: mọi gate chưa đạt tiếp tục block release.

## 8. Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-27, Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer; đây là plan revision, không phải
  release verdict.
- Kết luận: hồ sơ plan hoàn tất; P3.4.2 execution `active`, P3.4 vẫn `pending`.
- Liên kết bản ghi hoàn tất: `REV-20260827-007`.
