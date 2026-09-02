---
document_id: REV-20260902-003
document_type: revision_record
revision_schema: 2
title: Attempt 2 bị locked evaluation từ chối trên Tier C; mở attempt 3 trên scope v2
status: complete
date: 2026-09-02
record_id: REV-20260902-003
session_id: SESSION-20260902-007
created_at: 2026-09-02T23:30:00+07:00
author: claude-agent
revises:
  - session_id: SESSION-20260902-006
    artifact: evidence/mvp/release-v1/attempt-2-locked-no-go/contribution.json
    revision: 7c46d57fb1a2f956e38036bc565c221935be4fb1
reason: "Locked evaluation của attempt 2 cho Tier D +6.00 pp với paired CI [+2.67, +9.33] và ablation hoàn hảo (tắt residual trả về đúng 187/300 của prior), nhưng Tier C tụt còn 188/200 so với 192/200 của prior — được 2, mất 6, bucket drop tăng từ 5 lên 10. §7 nói rõ: candidate tốt hơn D nhưng kém prior ở A/B/C là NO-GO, không đổi reliability lấy novelty. Locked seed của scope v1 đã bị tiêu cho candidate đó, nên attempt kế phải chạy trên seed mới."
necessity: N3
impact: "Không ngưỡng nào bị đổi: scope v2 sao chép nguyên văn mọi tier size, gate, uplift threshold, paired estimator/seed, ablation bound và safety budget của v1, và có test khẳng định điều đó. Thay đổi duy nhất là seed root, cộng thêm một phép đo held-out trên development seed để bắt regression kiểu Tier C trước khi tiêu locked evaluation. Artifact attempt 2 giữ nguyên làm lịch sử."
---

# REV-20260902-003 — Attempt 2 bị từ chối trên Tier C

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260902-007`
- Bản ghi hoàn tất của phiên hiện tại:
  `docs/sessions/SESSION-20260902-007-mvp-release-attempt-three.md`
- Phiên bị sửa: `SESSION-20260902-006` (attempt 2)
- Artifact bị sửa: không artifact nào bị sửa. Locked evidence của attempt 2 nằm
  nguyên tại `evidence/mvp/release-v1/attempt-2-locked-no-go/`.
- Revision trước sửa: candidate commit `7c46d57fb1a2f956e38036bc565c221935be4fb1`
- Revision sau sửa: scope v2 `2419b70cb82fc94dc5841ea77356c7c2a1f32802650a707e5a51c77c0b5a3485`
- Issue/finding/yêu cầu nguồn: `ROADMAP-MVP-RELEASE-001` §2.2, §2.3.2, §5 MR-05,
  §7

## 2. Lý do chỉnh sửa

Locked evaluation của attempt 2 chạy đúng một lần trên bốn tier và trả về:

| Tier | Prior | Candidate | Paired | 95% CI |
| --- | ---: | ---: | ---: | --- |
| A | 100/100 | 100/100 | `+0.00 pp` | `[+0.00, +0.00]` |
| B | 288/300 | 290/300 | `+0.67 pp` | `[-0.67, +2.00]` |
| C | 192/200 | 188/200 | `-2.00 pp` | `[-5.00, +0.50]` |
| D | 187/300 | 205/300 | `+6.00 pp` | `[+2.67, +9.33]` |

Tier D đạt gate với biên rõ, và ablation là bằng chứng sạch nhất có thể: tắt
learned residual trả về đúng `187/300`, tức con số của chính prior, nên toàn bộ
phần cải thiện thuộc về model. Residual không suy biến (magnitude `0.0440`,
saturation `0.0000`) và không có safety violation hay invalid state nào trên bất
kỳ tier nào.

Tier C là chỗ hỏng. Candidate được 2 episode mà prior thua, mất 6 episode mà
prior thắng, và bucket `drop` tăng từ 5 lên 10. Trên kích thước held-out —
`w075` hẹp hơn mọi variant train và `w255` rộng hơn mọi variant train — residual
đóng chặt làm rơi vật nhiều hơn là cứu được.

`ROADMAP-MVP-RELEASE-001` §7 đã lường trước đúng tình huống này và trả lời sẵn:
“Candidate tốt hơn D nhưng kém prior ở A/B/C → `NO-GO`; không đổi reliability
lấy novelty.” Ngưỡng không được đổi sau khi thấy kết quả, và không được đổi.

## 3. Mức độ cần thiết

- Mức đã chọn: `N3`
- Lập luận: quyết định phát hành. Một candidate làm giảm độ tin cậy trên kích
  thước chưa từng thấy không được phát hành, kể cả khi nó thắng ở tier mới.
- Thời hạn/độ khẩn: trước bất kỳ locked evaluation nào tiếp theo.
- Biện pháp giảm thiểu nếu trì hoãn: không áp dụng; không có artifact công khai.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Ngưỡng và tier | v1 | **sao chép nguyên văn sang v2** | không | test khẳng định từng field |
| Seed | locked seed v1 đã tiêu | seed root mới, rời hoàn toàn | vừa | test disjointness |
| Quy trình chọn candidate | chỉ đo challenge domain | thêm held-out development | thấp | regression bị bắt trước locked run |
| Dữ liệu/checkpoint | attempt 2 | attempt 3 sinh bộ mới | không | attempt 2 giữ nguyên bytes |
| Số liệu đã công bố | không có | không có | không | chưa phát hành gì |

Tóm tắt tác động: điều duy nhất được phép đổi giữa hai attempt là tập seed và
quy trình huấn luyện. Thước đo giữ nguyên từng byte, và
`tests/mvp/test_mvp_release_contract.py` biến điều đó thành một test thay vì một
lời hứa.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Bảo tồn locked evidence của attempt 2 nguyên vẹn | `evidence/mvp/release-v1/attempt-2-locked-no-go/` | `V-001` |
| `CH-002` | Tạo scope v2: seed root mới, mọi ngưỡng sao chép nguyên văn | `configs/mvp/dexacquire-mvp-v2.yaml` `2419b70c…` | `V-002` |
| `CH-003` | Sinh eval manifest v2 bằng locker và rebind challenge domain sang v2 | `…v2.eval-manifest.json` `403c9c11…`, `…v2.challenge.json` | `V-003` |
| `CH-004` | Thêm held-out development measurement vào selection | `qdgrasp/mvp/challenge.py`, `scripts/train_mvp_policy.py` | `V-004` |
| `CH-005` | Trỏ release gate sang scope v2 | `scripts/check_mvp.py` | `V-004` |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `scripts/check_mvp.py --release --runs runs/release-v1` | attempt 2 bị chặn ở Tier C | 108/112, `tier_c_no_paired_regression` FAIL | `pass` | log phiên |
| `V-002` | `pytest tests/mvp/test_mvp_release_contract.py -q` | v2 giữ nguyên mọi ngưỡng của v1 | 40 passed | `pass` | log phiên |
| `V-003` | `scripts/lock_mvp_scope.py --scope …v2.yaml --check` | manifest v2 khớp scope v2 | OK `2419b70c…` | `pass` | log phiên |
| `V-004` | `pytest tests/mvp -q` | không regression | 131 passed, 12 subtests | `pass` | log phiên |

- Regression đã chạy lại: toàn bộ `tests/mvp`, static core, scope lock.
- Kiểm tra chưa chạy: locked evaluation của scope v2 — cố ý chưa chạy cho tới
  khi selection trên development cho thấy cả uplift lẫn không-regression.
- Khả năng rollback: revert commit; artifact attempt 2 không bị chạm.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có.
- Verdict cũ còn hiệu lực cho revision mới: `NO-GO` của attempt 2 là chung
  thẩm cho candidate đó. Nó không nói gì về candidate khác.
- Loại kiểm tra lại cần thiết: `full_review` cho artifact attempt 3 ở MR-06.
- Đính chính số liệu/tài liệu cần phát hành: không có số liệu nào sai.
- Người chấp nhận rủi ro/ngoại lệ: maintainer đã giao toàn quyền thi công; quyết
  định N3 này được ghi để kiểm chứng được.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-agent, 2026-09-02, UTC+07:00
- Người kiểm tra: chưa có; independent review là MR-06.
- Kết luận: đã hoàn tất. Attempt 3 chạy trên seed chưa từng dùng, với thước đo
  không đổi và một phép đo mới đứng trước locked evaluation thay vì sau nó.
- Liên kết bản ghi hoàn tất phiên hiện tại:
  `docs/sessions/SESSION-20260902-007-mvp-release-attempt-three.md`
