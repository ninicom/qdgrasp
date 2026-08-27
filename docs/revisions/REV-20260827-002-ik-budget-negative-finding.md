---
document_id: REV-20260827-002
document_type: revision_record
revision_schema: 2
title: Đo và bác bỏ giả thuyết IK budget cho canonical yield
status: complete
date: 2026-08-27
record_id: REV-20260827-002
session_id: SESSION-20260827-P3.1-14
created_at: 2026-08-27T05:40:00+07:00
author: claude-primary-agent
revises:
  - session_id: SESSION-20260827-001
    artifact: docs/revisions/REV-20260827-001-p3-1-14-regeneration.md
    revision: 7dd519bb9320339cd97e6e7e2ffac064fc5838f65f7f3501856fb5ec9a0fba05
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: b5d0933cc64d1b6ad3e521d0d30cec6f6b4471b4557db2f23218777d72999a98
reason: "REV-20260827-001 ghi max_iter=40 là nút thắt yield toàn cục và là hướng trực tiếp tấn công canonical 0/12. Đo terminal residual cho thấy kết luận đó sai, nên phải đính chính trước khi có ai bỏ công chạy lại."
necessity: N2
impact: "Bác bỏ hướng nâng IK budget bằng số đo; canonical 0/12 được xác định lại là bài toán proposal feasibility. Không thay đổi dataset, code hay trạng thái phase nào."
---

# REV-20260827-002 — IK budget là hướng chết

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260827-P3.1-14`.
- Bản ghi hoàn tất: tài liệu này.
- Artifacts bị sửa và checksum trước sửa: hai mục `revises` ở metadata.
- Revision code chạy evidence: `be38587e778961c176aab80a519a41407a9817c6`.
- Evidence: `evidence/phase3_1/p16-ik-budget/`; sổ thí nghiệm tổng hợp tại
  `evidence/phase3_1/EXPERIMENTS.md`.

## 2. Lý do chỉnh sửa

`REV-20260827-001` §7 ghi rằng `max_iter=40` hardcode trong orchestrator "là nút
thắt yield toàn cục" và "có thể tấn công trực tiếp canonical 0/12". Suy luận đó
dựa trên việc IK là stage chết nhiều nhất trong lưới Allegro, chứ không dựa trên
số đo về mức độ hội tụ.

Đo terminal residual bác bỏ nó. Solver dừng cách tolerance hai bậc độ lớn, tức
không nằm trong basin nào, chứ không phải sắp hội tụ thì hết vòng lặp.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: đính chính một kết luận kỹ thuật đã ghi vào record và roadmap, có
  khả năng dẫn tới công việc lãng phí quy mô lớn.
- Thời hạn/độ khẩn: trước khi bất kỳ ai bắt đầu sửa `max_iter`.
- Biện pháp nếu trì hoãn: một phiên sau sẽ nâng budget, làm mất hiệu lực bằng
  chứng P3.1-13 và P3.2.1, phải chạy lại toàn bộ, và không thu được yield nào.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | không đổi | không đổi | không | `max_iter` giữ nguyên 40+40 |
| API/schema/config | không đổi | không đổi | không | không |
| Dữ liệu/checkpoint | không đổi | không đổi | không | không |
| Tương thích CPU/GPU/export | không đổi | không đổi | không | không |
| Số liệu/kết luận đã công bố | IK budget là hướng hứa hẹn | hướng chết, đã đo | cao | chuyển trọng tâm sang proposal feasibility |

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

- `CH-001` `evidence/phase3_1/p16-ik-budget/`: harness đo terminal residual và
  kết quả 12 ô hand × object trên bốn object procedural canonical.
- `CH-002` `evidence/phase3_1/EXPERIMENTS.md`: sổ ghi mọi thí nghiệm P3.1 đã
  chạy, gồm cả hướng chết, kèm mục "rejected without testing" và "open".
- `CH-003` Cập nhật `PROJECT_PHASES.md` thay câu về `max_iter` bằng kết luận đã
  đo.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | đo terminal position residual, 3 hand × 4 object | so với tolerance `0.001 m` | tốt nhất `0.0276 m` (28×), trung vị `0.095`–`0.179 m` | pass | `ik-terminal-residuals.json` |
| `V-002` | đo terminal normal residual | so với `dot >= 0.866` (30°) | tốt nhất `64°`, trung vị `86°`–`151°` | pass | `ik-terminal-residuals.json` |
| `V-003` | đếm proposal rejection trước IK | định vị stage chặn thật | 1–6 trên mỗi 8 candidate chết ở proposal | pass | `ik-terminal-residuals.json` |

- Không chạy regression: phiên này không đổi code production, chỉ đo.
- Khả năng rollback: evidence và tài liệu; không có thay đổi hành vi để revert.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo bị ảnh hưởng: `REV-20260827-001` §7 bullet "Finding mở cho phase sau"
  bị **thay thế** bởi record này. Mọi kết luận khác của `REV-20260827-001` giữ
  nguyên hiệu lực, gồm cả việc đóng P3.1.
- Đính chính: `max_iter` **không** phải nút thắt yield. Nó là stage mà candidate
  chết nhiều nhất, nhưng chết vì mục tiêu không khả thi động học chứ không vì
  hết budget. Hai điều đó khác nhau và lưới chẩn đoán ban đầu không phân biệt
  được.
- Bài học phương pháp ghi lại để không lặp: "stage nào chết nhiều nhất" không
  chỉ ra nguyên nhân; phải đo mức độ hội tụ tại điểm dừng mới kết luận được.
- Người chấp nhận rủi ro/ngoại lệ: không có ngoại lệ.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-primary-agent, 2026-08-27 Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer.
- Kết luận: hướng IK budget đóng lại bằng số đo. Canonical `0/12` được xác định
  lại là bài toán proposal feasibility và chưa có thí nghiệm nào chạm tới.
- Liên kết bản ghi hoàn tất: `REV-20260827-002`.
