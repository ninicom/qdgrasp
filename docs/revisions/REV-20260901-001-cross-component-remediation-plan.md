---
document_id: REV-20260901-001
document_type: revision_record
revision_schema: 2
title: Thêm corrective remediation bắt buộc sau audit liên mô-đun
status: complete
date: 2026-09-01
record_id: REV-20260901-001
session_id: SESSION-20260901-001
created_at: 2026-09-01T07:14:56+07:00
author: codex-agent
revises:
  - session_id: PLAN-V2
    artifact: PLAN.md
    revision: PLAN-V2@4.5.0
reason: Audit liên mô-đun tái hiện đường thực thi pickle, protocol/split không đi vào public trainer, supervision và Flow latent mâu thuẫn, validation/resume không quyết định, bundle/export sai identity và MVP evaluator bypass fingerprint; plan cũ chưa chặn các chuỗi này.
necessity: N3
impact: PLAN-V2 tăng lên 4.6.0 và thêm corrective track G0–G7; DGN-Open-Tiny cùng Flow release evidence cũ bị supersede cho mục đích release, P4/P5 giữ blocked tới khi corrective gates pass, MVP round 3 cần revalidation nhưng chưa bị invalidated vì fingerprint hiện khớp.
---

# REV-20260901-001 — Corrective remediation sau audit liên mô-đun

## 1. Liên kết truy vết

- Phiên audit/lập kế hoạch: `SESSION-20260901-001` tại
  `docs/sessions/SESSION-20260901-001-cross-component-audit-plan.md`.
- Active document bị sửa: `PLAN.md` (`PLAN-V2`).
- Revision trước sửa: `PLAN-V2@4.5.0`, SHA-256
  `0395aba6688bd0a8479cf17d2c33e3fcbe45a8fcf6aae2761cf44d43cb0b69f0`.
- Revision sau sửa: `PLAN-V2@4.6.0`, SHA-256
  `97b916d0e119b1cb0ebc8a380505f6084a264364ddd56bc114955396b2f44c7b`.
- Git base trước sửa tài liệu:
  `159f8607ad72295528e9051ff3cccc3486e052a1`.
- Finding registry và execution/closure gates mới: `PLAN.md` §9.

## 2. Lý do chỉnh sửa

PLAN-V2@4.5.0 mô tả đúng mục tiêu cấp cao nhưng giả định rằng các contract đã
có — dataset audit, protocol helper, mixed-robot collator, resume, bundle và MVP
fingerprint — nằm trên public execution path. Audit này chứng minh giả định đó
sai trong nhiều chuỗi:

1. Manifest-controlled path đi tới `torch.load(weights_only=False)` và có thể
   thực thi reducer; Phase 5 còn dùng `verify=False`.
2. Protocol held-out không được materialize trước public trainer; physical train
   chứa 12 sample `comp_*` ngoài protocol cho mỗi active hand.
3. Mixed-robot guard và fingerprint guard tồn tại nhưng Runner/evaluator gọi
   helper khác nên bypass được.
4. Missing kinematics được serialise thành zero/identity rồi hồi quy như target;
   joint encode/decode của Flow dùng hai parameterization khác nhau.
5. Quality không phụ thuộc candidate; validation tiêu RNG train; resume và bundle
   dựa vào tensor shape thay vì full identity; Flow public export không chạy.

Đây không phải một danh sách lint. Từng chuỗi trên đã được tái hiện qua public
hoặc gần-public path trong thư mục tạm, trong khi full unit suite vẫn pass. Plan
phải chuyển từ “có helper/test” sang “public path giữ invariant end-to-end”.

## 3. Mức độ cần thiết

- Mức: `N3`.
- Lý do an toàn: artifact dataset/checkpoint không tin cậy có đường thực thi
  pickle và path escape. Hash do cùng người cung cấp manifest khai báo không tạo
  authentication.
- Lý do phát hành: public training có thể dùng object/hand ngoài protocol, model
  có thể load/resume weights từ semantic config hoặc robot khác nhưng artifact
  vẫn được gắn nhãn hiện tại. Đây là sai identity có thể lan vào model card,
  benchmark và release bundle.
- Lý do không hạ xuống `N2`: vấn đề không chỉ làm sai một metric cũ; nó ảnh
  hưởng security boundary và điều kiện cho phép tạo/phát hành artifact mới.
- Biện pháp giảm thiểu tức thời: giữ DGN/Flow non-release, chặn P5 và không nhận
  artifact bên ngoài qua unsafe loader cho tới G0/G1.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước revision | Sau revision | Hành động bắt buộc |
|---|---|---|---|
| Active plan | Không có corrective track liên mô-đun | §9 có `COR-00`…`COR-12`, G0–G7, PR map và closure gates | Thi công theo dependency, không mega-PR |
| DGN-Open-Tiny | Manifest ghi unblocked nhưng canonical audit hiện fail | Non-release input cho tới khi regenerate/audit pass | Sửa schema/protocol/masks trước regenerate |
| Phase 4 | CPU overfit evidence được giữ | Evidence cũ không đủ cho semantics mới | Chạy lại sau G3 |
| Phase 5 | Đã blocked chủ yếu vì thiếu positive/CUDA | Bị chặn thêm bởi security, protocol, data, Runner và artifact identity | Chỉ mở sau G1–G5 và các gate cũ |
| Flow checkpoint/bundle | Có thể load theo tensor shape | Superseded for release dưới schema cũ | Retrain/rebundle, không migrate im lặng |
| MVP round 3 | Experimental non-release, fingerprint hiện khớp | Giữ lịch sử; cần guard revalidation | Retrain chỉ khi action semantics đổi |
| P3.4/P3.5/CUDA/reviewer gates | Các blocker cũ còn hiệu lực | Không đổi và không được nới | Tiếp tục theo roadmap hiện hành |
| Code/runtime | Chưa sửa trong revision này | Chưa sửa; chỉ thêm plan và audit trail | G0–G7 còn pending |

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã hoàn tất | Output |
|---|---|---|
| `CH-001` | Tăng active plan version và revision pointer | `PLAN-V2@4.6.0` front matter |
| `CH-002` | Ghi baseline audit và artifact disposition | `PLAN.md` §9.1 |
| `CH-003` | Đăng ký toàn bộ root-cause chain cùng target state | `PLAN.md` §9.2, `COR-00`…`COR-12` |
| `CH-004` | Chia remediation thành G0–G7 có work package và acceptance gate | `PLAN.md` §9.3–§9.10 |
| `CH-005` | Khóa dependency/PR order để tách security, semantics, regeneration và cleanup | `PLAN.md` §9.11 |
| `CH-006` | Thêm interaction-test matrix trên đúng public path | `PLAN.md` §9.12 |
| `CH-007` | Ghi audit session và revision record | `SESSION-20260901-001`, tài liệu này |

Không implementation fix nào được đánh dấu hoàn tất trong revision này.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái |
|---|---|---|---|---|
| `V-001` | `.venv/bin/python scripts/check_docs.py --root .` | PLAN/session/revision hợp lệ | 151 file được kiểm tra | pass |
| `V-002` | `sha256sum PLAN.md` | khớp hash revision mới | `97b916d0e119b1cb0ebc8a380505f6084a264364ddd56bc114955396b2f44c7b` | pass |
| `V-003` | `git diff --check` | không whitespace error | không có output/lỗi | pass |
| `V-004` | `git status --short` và review diff | chỉ thay đổi tài liệu audit/plan | `PLAN.md` và hai record mới; không code/config/data/evidence | pass |

Baseline kỹ thuật của audit nằm trong `SESSION-20260901-001` §"Kiểm tra đã
chạy"; revision này không tuyên bố implementation remediation đã pass.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Phase 4 CPU overfit vẫn là bằng chứng lịch sử rằng code cũ tối ưu được fixture
  cũ, nhưng không còn đủ để phê duyệt Flow sau khi joint parameterization,
  target masks và quality objective đổi. Nó phải được chạy lại, không rewrite.
- DGN-Open-Tiny hiện tại không còn đủ provenance cho release vì ba source hash
  drift; byte/shard cũ được giữ để audit, không đổi manifest hash để hợp thức hóa.
- Không số Phase 5 nào được invalidated vì P5 chưa vượt input gate; thay vào đó
  phase nhận thêm blocker bắt buộc trước run đầu tiên.
- MVP round 3 fingerprint đã được đối chiếu và hiện khớp world hiện tại, nên
  chưa có bằng chứng report đó bị relabel sai. Guard path vẫn phải sửa; nếu PPO
  action distribution đổi thì checkpoint mới thay checkpoint cũ.
- Không verdict P3.4/P3.5, ADR-0006 hoặc ADR-0008 nào bị nới hoặc đảo ngược.
- Loại review cần thiết: full independent review cho security boundary,
  protocol isolation và artifact identity trước khi mở release.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-agent, 2026-09-01, UTC+07.
- Người kiểm tra độc lập: chưa có; revision N3 này đặt thêm gate và không tự ký
  bất kỳ release verdict nào.
- Kết luận: audit và việc cập nhật active plan đã hoàn tất. Implementation
  `COR-00`…`COR-12` chưa hoàn tất và được bàn giao theo G0–G7/R1–R9.
- Bản ghi phiên: `docs/sessions/SESSION-20260901-001-cross-component-audit-plan.md`.
