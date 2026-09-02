---
document_id: REV-20260902-002
document_type: revision_record
revision_schema: 2
title: Mở attempt 2 cho MVP release với giả thuyết đã đo, candidate version v2
status: complete
date: 2026-09-02
record_id: REV-20260902-002
session_id: SESSION-20260902-006
created_at: 2026-09-02T20:00:00+07:00
author: claude-agent
revises:
  - session_id: SESSION-20260902-005
    artifact: evidence/mvp/release-v1/mr04-no-go/policy/training-report.json
    revision: 0180d0da24173ad0f8e85db9f694e0a612b343512308f9c6bfe94477586ff45b
reason: "Attempt 1 kết thúc NO-GO tại MR-04 với uplift -1.0 pp. §8.8 cấm lặp lại cùng một run mà không đổi giả thuyết, và probe development đo được rằng action space đã khóa CÓ chứa lời giải: một residual hằng số đóng chặt ở segment carry cho +7.00 pp trên challenge domain với zero safety violation. Nguyên nhân attempt 1 thất bại là hai lỗi thi công cụ thể chứ không phải giới hạn của bài toán, nên attempt 2 được mở với giả thuyết đã đo."
necessity: N3
impact: "Không ngưỡng, tier, scope hay eval manifest nào bị đổi: scope v1 giữ nguyên scope_hash 47fbf8f0…, eval manifest giữ nguyên 733ef3a0…, challenge domain giữ nguyên aabec53d…, gate Tier D vẫn là +5 pp với paired CI lower > 0. Chỉ thủ tục training đổi: expert search được hướng vào trục synergy, và PPO rollout đi qua challenge domain. Candidate version lên v2; artifact attempt 1 giữ nguyên làm lịch sử NO-GO."
---

# REV-20260902-002 — Attempt 2 cho MVP release

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260902-006`
- Bản ghi hoàn tất của phiên hiện tại:
  `docs/sessions/SESSION-20260902-006-mvp-release-attempt-two.md`
- Phiên bị sửa: `SESSION-20260902-005` (verdict `NO-GO` của attempt 1)
- Artifact bị sửa: không artifact nào của attempt 1 bị sửa. Chúng được giữ
  nguyên tại `evidence/mvp/release-v1/mr04-no-go/` làm lịch sử bất biến.
- Revision/checksum trước sửa: training report attempt 1
  `0180d0da24173ad0f8e85db9f694e0a612b343512308f9c6bfe94477586ff45b`
- Revision/checksum sau sửa: attempt 2 sinh artifact mới dưới candidate version
  `v2`; không ghi đè lên attempt 1.
- Issue/finding/yêu cầu nguồn: `ROADMAP-MVP-RELEASE-001` §5 MR-04 stop rule,
  §7 decision matrix, §8.8

## 2. Lý do chỉnh sửa

Attempt 1 dừng đúng luật: BC và PPO đều đo được `-1.0` pp so với controller
prior trên 300 challenge-development seed, trong khi Tier D đòi `+5.0` pp. §8.8
nói rõ rằng một `NO-GO` phải giữ evidence để quyết định nguyên nhân, và cấm lặp
lại cùng một run mà không đổi giả thuyết.

Nguyên nhân đã được đo, không phải suy đoán. `scripts/probe_mvp_residual.py`
chạy trên development seed hỏi một câu duy nhất: action space đã khóa có chứa
residual nào cứu được các failure của prior trên challenge domain không? Câu
trả lời là có, và với biên rộng:

| Residual hằng số (chỉ segment carry) | Challenge domain | Base dev | Safety |
| --- | ---: | ---: | ---: |
| synergy `(0.00, 0.75)` | `+7.00 pp` (+24/−3) | `+1.50 pp` (+3/−0) | 0 |
| synergy `(0.25, 0.75)` | `+7.00 pp` (+26/−5) | `+0.00 pp` (+2/−2) | 0 |
| synergy `(0.50, 0.50)` | `+6.00 pp` (+25/−7) | `+0.00 pp` (+3/−3) | 0 |
| synergy `(1.00, 0.00)` | `+5.00 pp` (+25/−10) | `+1.50 pp` (+4/−1) | 0 |

Lực tiếp xúc lớn nhất quan sát được là `10.96 N` trên ngân sách `20.0 N`, và
không có safety violation hay invalid state nào.

Nghĩa là bài toán học được, và attempt 1 thất bại vì hai lỗi thi công cụ thể:

1. **Expert search không thể tìm ra lời giải.** Nó bốc mẫu nhiễu đẳng hướng
   `N(0, 0.35)` trên không gian 16 chiều (2 segment × 8 dim), 11 mẫu mỗi
   episode. Xác suất trúng “đóng chặt hơn ở segment carry mà không phá pha
   approach” là rất nhỏ. Hệ quả đo được: 644/705 demonstration mang nhãn không
   làm gì, `mean_expert_action_magnitude` `0.0078`.
2. **PPO không bao giờ đi qua miền mà đóng góp của nó được đo.** Rollout lấy từ
   split `train` ở miền base, nơi prior đã gần trần. Một chính sách không thấy
   miền challenge thì không có đường học hành vi trên miền đó.

Bản ghi `NO-GO` của attempt 1 vẫn đúng với những gì nó tuyên bố. Nó không còn
đủ để kết luận “learned residual không có giá trị trên MVP này”, vì phép đo ở
trên cho thấy điều ngược lại.

## 3. Mức độ cần thiết

- Mức đã chọn: `N3`
- Lập luận: quyết định trực tiếp về việc phát hành công khai. Nếu không mở
  attempt 2, kết luận duy nhất còn lại là `do_not_release` hoặc chuyển sang
  controller-only — cả hai đều là tuyên bố sai về khả năng của hệ, vì bằng
  chứng đo được nói rằng residual học được có đóng góp thật.
- Thời hạn/độ khẩn: trước bất kỳ locked evaluation nào. Attempt 1 chưa tiêu
  locked evaluation, nên ngân sách một-lần của MR-05 vẫn còn nguyên.
- Biện pháp giảm thiểu nếu trì hoãn: không cần; không có artifact công khai nào
  đang mang tuyên bố sai.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | expert bốc mẫu nhiễu thuần | zero, rồi 6 grip candidate có hướng, rồi 5 mẫu ngẫu nhiên | vừa | budget `candidates` vẫn được tôn trọng đúng |
| API/schema/config | PPO rollout chỉ ở miền base | thêm `challenge_path`/`challenge_fraction` | thấp | mặc định `0.0`, không đổi hành vi cũ |
| Dữ liệu/checkpoint | demonstration/checkpoint attempt 1 | attempt 2 sinh bộ mới | không | attempt 1 giữ nguyên bytes |
| Ngưỡng và tier | Tier D `+5 pp`, CI lower `> 0` | **không đổi** | không | scope/manifest/domain giữ nguyên hash |
| Số liệu/kết luận đã công bố | `NO-GO` attempt 1 | vẫn đúng trong phạm vi của nó | thấp | không đính chính số liệu nào |

Tóm tắt tác động: điều duy nhất thay đổi là *thủ tục* tạo candidate. Mọi thứ
quyết định việc candidate đó có được phát hành hay không — scope, eval manifest,
challenge domain, gate Tier D, ablation contract, safety budget — giữ nguyên
từng hash. Đây là ràng buộc quan trọng nhất của revision này: đổi cách train thì
được, đổi thước đo thì không.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Viết probe đo khả năng cứu failure của action space đã khóa | `scripts/probe_mvp_residual.py` | `V-001` |
| `CH-002` | Lưu kết quả probe làm bằng chứng cho quyết định này | `evidence/mvp/release-v1/residual-probe.json` `2520b725…` | `V-001` |
| `CH-003` | Hướng expert search vào trục synergy, giữ nguyên ngân sách `candidates` và điều kiện chấp nhận | `qdgrasp/mvp/expert.py` | `V-002` |
| `CH-004` | Cho PPO rollout đi qua challenge domain với seed offset riêng | `qdgrasp/mvp/ppo.py`, `qdgrasp/mvp/env.py` | `V-002` |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `.venv/bin/python scripts/probe_mvp_residual.py --out evidence/mvp/release-v1/residual-probe.json` | trả lời được câu hỏi action space có cứu được failure không | `+7.00 pp`, zero safety, `locked_action_space_can_recover_failures: true` | `pass` | `2520b725…` |
| `V-002` | `.venv/bin/python -m pytest tests/mvp -q` | không regression | 127 passed, 12 subtests | `pass` | log phiên |
| `V-003` | `.venv/bin/python scripts/check_static_core.py` | Ruff + Mypy pass | PASS (137 file, 32 contract) | `pass` | log phiên |
| `V-004` | `scripts/lock_mvp_scope.py --scope …v1.yaml --check` | scope và manifest không đổi | OK `47fbf8f0…` | `pass` | log phiên |

- Regression đã chạy lại: toàn bộ `tests/mvp`, static core, scope lock.
- Kiểm tra chưa chạy: chưa có locked evaluation nào. Ngân sách MR-05 còn nguyên
  vì attempt 1 dừng trước khi chạm tới nó.
- Khả năng rollback: revert commit; artifact attempt 1 không bị chạm nên không
  cần khôi phục dữ liệu.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có.
- Verdict cũ còn hiệu lực cho revision mới: `NO-GO` của attempt 1 vẫn đúng cho
  candidate của attempt 1. Nó không còn được đọc như một kết luận về việc
  learned residual có giá trị hay không.
- Loại kiểm tra lại cần thiết: `full_review` cho artifact attempt 2 khi tới
  MR-06.
- Đính chính số liệu/tài liệu cần phát hành: không có số liệu nào sai.
- Người chấp nhận rủi ro/ngoại lệ: maintainer đã giao toàn quyền thi công cho
  agent với chỉ thị làm tới khi release; quyết định N3 này được ghi ở đây để nó
  kiểm chứng được, không phải để xin phê duyệt lại.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-agent, 2026-09-02, UTC+07:00
- Người kiểm tra: chưa có; independent review là MR-06.
- Kết luận: đã hoàn tất. Attempt 2 được mở với một giả thuyết đã đo, và với mọi
  thước đo giữ nguyên hash từ MR-02.
- Liên kết bản ghi hoàn tất phiên hiện tại:
  `docs/sessions/SESSION-20260902-006-mvp-release-attempt-two.md`
