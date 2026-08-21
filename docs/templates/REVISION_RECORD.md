---
document_id: REV-YYYYMMDD-NNN
document_type: revision_record
revision_schema: 2
title: Hồ sơ sửa kết quả của phiên trước
status: draft
date: YYYY-MM-DD
record_id: "REV-YYYYMMDD-NNN"
session_id: "CURRENT-SESSION-ID"
created_at: "YYYY-MM-DDTHH:MM:SS+07:00"
author: "Họ tên hoặc agent ID"
revises:
  - session_id: "PREVIOUS-SESSION-ID"
    artifact: "path/to/artifact"
    revision: "COMMIT-OR-CHECKSUM-BEFORE"
reason: "Mô tả nguyên nhân cụ thể khiến phải sửa kết quả phiên trước"
necessity: N2
impact: "Tóm tắt phần bị ảnh hưởng và mức thay đổi hành vi/kết quả"
---

# Hồ sơ sửa kết quả của phiên trước

> Hồ sơ này bắt buộc khi một phiên hiện tại sửa artifact đã được ghi “hoàn tất” ở phiên trước. Không sửa hoặc xóa bản ghi của phiên cũ. Xóa hướng dẫn trong dấu ngoặc nhọn và đổi `status` thành `complete` chỉ sau khi đã xác minh.

## 1. Liên kết truy vết

- Phiên thực hiện sửa: {khớp `session_id`}
- Bản ghi hoàn tất của phiên hiện tại: {đường dẫn/ID}
- Phiên bị sửa: {khớp từng `revises[].session_id`}
- Artifact bị sửa: {khớp từng `revises[].artifact`}
- Revision/checksum trước sửa: {khớp từng `revises[].revision`}
- Revision/checksum sau sửa: {commit hoặc SHA-256}
- Issue/finding/yêu cầu nguồn: {ID hoặc đường dẫn bền vững}

Nếu sửa nhiều artifact hoặc nhiều phiên cũ, thêm từng mục vào `revises`; không gộp bằng mô tả mơ hồ hoặc glob.

## 2. Lý do chỉnh sửa

{Nêu sự kiện hoặc bằng chứng mới làm phát sinh chỉnh sửa: lỗi được phát hiện, tiêu chí thay đổi, kết quả không tái lập, yêu cầu pháp lý, hoặc giả định trước đây sai. Chỉ rõ vì sao bản ghi “đã hoàn tất” của phiên trước không còn đủ. Không dùng riêng các từ “cải tiến”, “dọn dẹp”, “cập nhật” hay “tối ưu”. Nội dung phải nhất quán với trường `reason` trong metadata.}

## 3. Mức độ cần thiết

- Mức đã chọn: {`N0`, `N1`, `N2` hoặc `N3`; khớp `necessity`}
- Lập luận: {vì sao thay đổi thuộc mức này và hậu quả cụ thể nếu không sửa}
- Thời hạn/độ khẩn: {ngày hoặc mốc phải hoàn tất}
- Biện pháp giảm thiểu nếu trì hoãn: {mô tả hoặc “không được phép trì hoãn”}

Quy tắc phân loại:

- `N0` — Biên tập: chính tả/format, không đổi nghĩa hay số liệu.
- `N1` — Làm rõ: tránh hiểu nhầm nhưng không đổi kết luận.
- `N2` — Bắt buộc: sửa sai protocol, số liệu, artifact, hành vi hoặc kết luận kỹ thuật.
- `N3` — Khẩn cấp: ảnh hưởng an toàn, phát hành, pháp lý hoặc quyết định lớn.

Độ khó hoặc thời gian sửa không quyết định mức cần thiết. Không dùng `N0` hoặc
`N1` nếu output, API, dữ liệu, benchmark hoặc kết luận thay đổi.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | {mô tả} | {mô tả} | {không/thấp/vừa/cao} | {test/review} |
| API/schema/config | {mô tả} | {mô tả} | {mức} | {migration/versioning} |
| Dữ liệu/checkpoint | {mô tả} | {mô tả} | {mức} | {convert/regenerate} |
| Tương thích CPU/GPU/export | {mô tả} | {mô tả} | {mức} | {matrix cần chạy lại} |
| Số liệu/kết luận đã công bố | {mô tả} | {mô tả} | {mức} | {đính chính/re-review} |

Tóm tắt tác động: {khớp trường `impact` trong metadata; nêu rõ thành phần không bị ảnh hưởng chỉ khi đã có cơ sở xác minh.}

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

Chỉ ghi việc đã thực hiện và có thể kiểm chứng trong phiên hiện tại:

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | {mô tả ở thì đã hoàn thành} | {đường dẫn + commit/checksum} | {evidence/test ID} |

Không ghi việc đang làm, dự định hoặc chưa chạy kiểm tra vào bảng này. Các việc đó phải nằm trong backlog/follow-up riêng.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | {lệnh đầy đủ} | {tiêu chí} | {kết quả} | `pass/fail/not_run` | {log + checksum} |

- Regression đã chạy lại: {phạm vi}
- Kiểm tra chưa chạy: {lý do và rủi ro; ghi “không có” nếu đầy đủ}
- Khả năng rollback: {cách quay lại revision trước và dữ liệu liên quan}

Không được đặt `status: complete` nếu kiểm tra bắt buộc có trạng thái `fail` hoặc
`not_run` mà không có quyết định chấp nhận rủi ro được liên kết.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: {report ID/revision hoặc “không có”}
- Verdict cũ còn hiệu lực cho revision mới: {có/không; giải thích}
- Loại kiểm tra lại cần thiết: {`none`, `delta_review` hoặc `full_review`}
- Đính chính số liệu/tài liệu cần phát hành: {danh sách hoặc “không có”}
- Người chấp nhận rủi ro/ngoại lệ: {tên, quyết định và ngày; nếu có}

## 8. Xác nhận đóng hồ sơ

- Tác giả: {tên/ID, ngày và múi giờ}
- Người kiểm tra: {tên/ID, phạm vi kiểm}
- Kết luận: {đã hoàn tất / chưa hoàn tất và lý do}
- Liên kết bản ghi hoàn tất phiên hiện tại: {đường dẫn/ID}
