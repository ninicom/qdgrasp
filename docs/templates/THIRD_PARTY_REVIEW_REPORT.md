---
document_id: TPR-YYYYMMDD-NNN
document_type: third_party_review
title: Báo cáo kiểm tra độc lập
status: draft
date: YYYY-MM-DD
revises: none
report_id: "TPR-YYYYMMDD-NNN"
reviewed_session: "SESSION-ID"
reviewed_revision: "COMMIT-OR-CHECKSUM"
reviewer: "Họ tên"
reviewer_organization: "Tổ chức/nhóm"
review_date: "YYYY-MM-DD"
independence: external
verdict: inconclusive
max_severity: NONE
---

# Báo cáo kiểm tra độc lập

> Xóa mọi hướng dẫn trong dấu ngoặc nhọn trước khi chuyển `status` thành `complete`
> (hoặc `accepted` khi đã được người có thẩm quyền chấp nhận). Giá trị hợp lệ:
> `independence = external | internal_independent`; `verdict = pass |
> conditional_pass | fail | inconclusive`; `max_severity = S0 | S1 | S2 | S3 |
> INFO | NONE`.

## 1. Đối tượng và phạm vi

- Mục tiêu kiểm tra: {tuyên bố hoặc mốc cần xác nhận}
- Artifact/repository: {đường dẫn hoặc URL bền vững}
- Revision/commit/checksum: {đúng với `reviewed_revision`}
- Tiêu chí nghiệm thu tham chiếu: {tài liệu và revision}
- Trong phạm vi: {danh sách hữu hạn}
- Ngoài phạm vi: {nội dung không được kiểm}
- Thời gian thực hiện: {bắt đầu — kết thúc, kèm múi giờ}

## 2. Tuyên bố độc lập và xung đột lợi ích

- Mức độc lập: {`external` hoặc `internal_independent`, giải thích ngắn}
- Quan hệ với tác giả/nhóm triển khai: {mô tả hoặc “không có”}
- Đóng góp trước đây cho artifact: {mô tả hoặc “không có”}
- Lợi ích tài chính/phụ thuộc quản lý: {mô tả hoặc “không có”}
- Hạn chế ảnh hưởng tới tính độc lập: {mô tả hoặc “không có”}

Tôi xác nhận không viết hoặc sửa revision đang được kiểm tra và báo cáo phản ánh kết quả tôi quan sát được từ bằng chứng liệt kê bên dưới.

## 3. Môi trường và phương pháp

| Hạng mục | Giá trị |
| --- | --- |
| Hệ điều hành/runtime | {phiên bản đầy đủ} |
| CPU/RAM | {model/dung lượng} |
| GPU/driver/runtime | {model và phiên bản, hoặc không dùng} |
| Dependency lock | {đường dẫn + checksum} |
| Dataset/model manifest | {đường dẫn + checksum} |
| Seed/protocol | {giá trị hoặc tài liệu tham chiếu} |

Phương pháp: {cách tiếp nhận artifact, kiểm checksum, dựng môi trường, tái chạy và đối chiếu. Nêu rõ mọi sai khác so với protocol chuẩn.}

## 4. Bằng chứng

| Evidence ID | Cấp | Mô tả | Nguồn bền vững | Checksum | Tạo bởi |
| --- | --- | --- | --- | --- | --- |
| `E-001` | `E1` | {log/kết quả thô} | {đường dẫn} | {SHA-256} | {lệnh hoặc tác nhân} |
| `E-002` | `E2` | {kết quả tái lập} | {đường dẫn} | {SHA-256} | {người kiểm tra} |

Không dùng ảnh chụp màn hình hoặc nội dung chép tay làm bằng chứng duy nhất cho số liệu. Mỗi số trong phần kết quả phải liên kết tới một evidence ID.

## 5. Kết quả đối chiếu

| Check ID | Tuyên bố/tiêu chí | Cách kiểm | Kết quả quan sát | Sai số/ngưỡng | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `C-001` | {tiêu chí} | {lệnh/phương pháp} | {giá trị} | {ngưỡng} | `đạt / không đạt / không kiểm` | `E-001` |

Các kiểm tra không thực hiện được: {liệt kê lý do và ảnh hưởng tới kết luận; ghi “không có” nếu đầy đủ.}

## 6. Phát hiện

### F-001 — {Tiêu đề ngắn}

- Severity: {`S0`, `S1`, `S2`, `S3` hoặc `INFO`}
- Trạng thái: {`open`, `resolved`, `accepted_risk` hoặc `not_reproducible`}
- Thành phần/revision bị ảnh hưởng: {phạm vi chính xác}
- Quan sát: {dữ kiện, không trộn suy đoán}
- Tác động: {người dùng/hệ thống/tuyên bố bị ảnh hưởng}
- Bước tái hiện: {lệnh và đầu vào tối thiểu}
- Evidence: {một hoặc nhiều evidence ID}
- Điều kiện đóng: {kiểm tra hoặc bằng chứng cần có}
- Phản hồi của nhóm triển khai: {giữ nguyên văn hoặc liên kết}
- Xác minh sau sửa: {revision/evidence mới, hoặc “chưa có”}

> Lặp lại mục này cho từng phát hiện. Nếu không có phát hiện, ghi rõ “Không có phát hiện” và xóa mục mẫu.

## 7. Điều kiện còn lại

| Condition ID | Điều kiện | Chủ sở hữu | Hạn | Cách xác minh | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| `K-001` | {chỉ dùng khi verdict là `conditional_pass`} | {người/nhóm} | {ngày} | {evidence/check} | `open` |

Nếu verdict không phải `conditional_pass`, ghi “Không có”.

## 8. Kết luận

- Verdict: {phải khớp metadata}
- Severity cao nhất còn mở: {phải khớp `max_severity`}
- Cơ sở kết luận: {tóm tắt các check/evidence quyết định verdict}
- Giới hạn của kết luận: {phạm vi, môi trường hoặc dữ liệu chưa bao phủ}
- Yêu cầu delta/full re-review: {điều kiện khiến kết luận hết hiệu lực}

Verdict này chỉ áp dụng cho `reviewed_revision` và phạm vi đã nêu. Mọi thay đổi sau đó phải có hồ sơ revision và được đánh giá lại theo tác động.

## 9. Chữ ký

- Người kiểm tra: {họ tên}
- Tổ chức/nhóm: {tổ chức}
- Ngày ký: {YYYY-MM-DD, múi giờ}
- Chữ ký hoặc định danh xác thực: {chữ ký số/ID hồ sơ}
