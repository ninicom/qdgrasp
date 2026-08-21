---
document_id: GOV-SESSION-001
document_type: policy
title: Quy tắc ghi việc đã làm trong mỗi phiên
status: active
date: 2026-08-21
owner: project-maintainer
revises: none
---

# Quy tắc ghi việc đã làm trong mỗi phiên

## 1. Định danh và thời điểm

- Mỗi phiên có ID `SESSION-YYYYMMDD-NNN` theo múi giờ `Asia/Bangkok`.
- Báo cáo được tạo trong chính phiên đó, không viết hồi tưởng nhiều phiên thành
  một báo cáo.
- Tên file: `SESSION-YYYYMMDD-NNN-<slug>.md`.

## 2. Nội dung bắt buộc

Mỗi session report phải tách rõ:

1. **Mục tiêu phiên**: phạm vi được phép và tiêu chí dừng.
2. **Việc đã hoàn tất**: chỉ kết quả có bằng chứng; dùng động từ ở trạng thái đã
   xong và nêu output cụ thể.
3. **Bằng chứng**: file, commit, artifact, hash hoặc dòng log.
4. **Kiểm tra đã chạy**: lệnh, exit code, kết quả; không ghi “tests pass” nếu chỉ
   chạy một phần.
5. **Việc chưa hoàn tất**: work-in-progress, blocker, giả định chưa xác nhận.
6. **Sửa đổi phiên trước**: `Không` hoặc liên kết revision record.
7. **Bàn giao**: trạng thái để phiên sau bắt đầu mà không suy đoán.

## 3. Từ vựng trạng thái

- `Hoàn tất`: đã triển khai và vượt kiểm tra được nêu.
- `Đã triển khai, chưa xác minh`: code/tài liệu tồn tại nhưng chưa qua cổng test.
- `Đang làm`: chưa tạo đủ output.
- `Bị chặn`: không thể tiếp tục vì phụ thuộc/authority bên ngoài; ghi rõ điều kiện
  gỡ chặn.
- `Không thực hiện`: nằm ngoài phạm vi hoặc bị thay quyết định.

Không dùng “xong”, “ổn”, “đã fix” hoặc “đạt” mà thiếu chủ thể, test và evidence.

## 4. Khi sửa việc của phiên trước

Front matter phải đổi `revises` từ `none` sang ID bị sửa và thêm:

- `revision_reason`: nguyên nhân cụ thể, không dùng “cập nhật cho đúng”.
- `necessity`: một trong `N0`, `N1`, `N2`, `N3`.
- `impact`: phạm vi tài liệu, số liệu, code và quyết định bị ảnh hưởng.

Phần **Sửa đổi phiên trước** phải liên kết `docs/revisions/REV-...md`, nêu claim
cũ, claim mới và lý do mức cần thiết đã chọn. Với `N2/N3`, mọi “pass” phụ thuộc
phải chuyển về chờ xác minh cho tới khi chạy lại.

## 5. Checklist đóng phiên

- [ ] Không có claim “hoàn tất” thiếu evidence.
- [ ] Lệnh kiểm tra và exit code được ghi.
- [ ] File mới/sửa được liệt kê.
- [ ] Số liệu mới dùng metric ID và run ID.
- [ ] Sửa lịch sử có revision record và necessity.
- [ ] Work chưa xong không bị trộn vào danh sách hoàn tất.
- [ ] `python3 scripts/check_docs.py --root .` trả exit code 0.
