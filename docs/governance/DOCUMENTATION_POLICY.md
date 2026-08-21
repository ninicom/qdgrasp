---
document_id: GOV-DOC-001
document_type: policy
title: Chính sách tài liệu và truy xuất bằng chứng
status: active
date: 2026-08-21
owner: project-maintainer
revises: none
---

# Chính sách tài liệu và truy xuất bằng chứng

## 1. Nguyên tắc bắt buộc

1. `PLAN.md` là nguồn sự thật về phạm vi; báo cáo không được âm thầm đổi tiêu chí
   nghiệm thu sau khi đã có kết quả.
2. Chỉ ghi “hoàn tất” khi có artifact hoặc lệnh kiểm tra có thể lặp lại. Công việc
   mới viết code nhưng chưa kiểm tra phải ghi “đã triển khai, chưa xác minh”.
3. Số liệu không có protocol, sample size, môi trường và artifact hash chỉ là số
   thăm dò, không được dùng làm kết luận.
4. Raw artifacts là bất biến. Nếu chạy lại, tạo run ID và artifact mới.
5. Không sửa im lặng tài liệu của phiên trước. Mọi sửa đổi phải có revision
   record nêu lý do, mức cần thiết, phạm vi ảnh hưởng và bằng chứng.
6. Reviewer độc lập không được vừa tạo artifact vừa là người duy nhất phê duyệt
   artifact đó.

## 2. Metadata chung

Mỗi tài liệu quản trị dùng YAML front matter và tối thiểu có:

- `document_id`: định danh duy nhất, không tái sử dụng.
- `document_type`: `plan`, `session_report`, `metrics_report`,
  `third_party_review`, `revision_record`, `decision`, `policy`, `registry` hoặc
  `schema`, `index`.
- `title`, `status`, `date`.
- `revises`: `none`, document/session ID bị sửa, hoặc danh sách mục đích danh
  khi một revision record sửa nhiều artifact. Không dùng glob hay mô tả mơ hồ.

Trạng thái hợp lệ cho hồ sơ phiên/kết quả:

- `draft`: chưa đủ bằng chứng hoặc chưa gửi review.
- `in_review`: nội dung đã khóa để người khác kiểm tra.
- `accepted`: đã qua cổng review được yêu cầu.
- `complete`: phiên đã kết thúc, có thể còn review bên ngoài.
- `superseded`: có tài liệu mới thay thế, bản cũ vẫn được giữ.
- `invalidated`: kết luận không còn hợp lệ; phải trỏ đến revision record.

Tài liệu quy phạm (`plan`, `policy`, `registry`, `schema`, `index`) được dùng
thêm `active`: phiên bản hiện đang có hiệu lực. `active` không hợp lệ cho
`session_report`, `metrics_report`, `third_party_review` hoặc
`revision_record`; các record này phải dùng trạng thái vòng đời ở trên.

File raw được lưu nguyên byte trong `docs/archive/` là ngoại lệ duy nhất được
phép không có front matter. Trạng thái, SHA-256 và revision pointer của file đó
phải nằm trong `docs/archive/README.md`; không sửa raw file chỉ để thêm metadata.

## 3. Mức cần thiết khi sửa phiên trước

| Mức | Tên | Khi dùng | Hành động |
|---|---|---|---|
| `N0` | Biên tập | Chính tả/format, không đổi nghĩa hay số liệu | Ghi revision ngắn |
| `N1` | Làm rõ | Câu chữ có thể gây hiểu nhầm, không đổi kết luận | Ghi ảnh hưởng và bản mới |
| `N2` | Bắt buộc | Sai protocol, số liệu, artifact hoặc kết luận kỹ thuật | Invalid/supersede bản cũ và review lại |
| `N3` | Khẩn cấp | Sai lệch ảnh hưởng an toàn, phát hành, pháp lý hoặc quyết định lớn | Dừng dùng kết quả cũ, thông báo và audit toàn bộ phụ thuộc |

Không được hạ mức chỉ để tránh review lại. Nếu không chắc giữa hai mức, dùng mức
cao hơn cho tới khi reviewer xác nhận.

## 4. Quy trình sửa lịch sử

1. Không chỉnh nội dung accepted/complete cũ trước khi tạo `REV-...`.
2. Revision record chụp lại claim cũ, claim mới, nguyên nhân gốc, `necessity`,
   ảnh hưởng trực tiếp/lan truyền và artifact chứng minh.
3. Đổi tài liệu cũ sang `superseded` hoặc `invalidated`, thêm liên kết đến
   revision. Nếu là raw archive cần giữ nguyên byte, ghi cùng thông tin trong
   sidecar `docs/archive/README.md` thay vì sửa file.
4. Tạo phiên bản mới; tuyệt đối không tái sử dụng document ID.
5. Chạy lại các metric phụ thuộc nếu sửa thuộc `N2` hoặc `N3`.
6. Third-party reviewer xác nhận phạm vi sửa và verdict mới.

## 5. Lưu artifact và hash

- Report chỉ liên kết artifact; không nhúng hoặc sửa raw output.
- Mỗi artifact ghi đường dẫn/URI, kích thước byte, SHA-256, thời điểm tạo và lệnh
  tạo. Với thư mục, tạo manifest sắp xếp ổn định rồi hash manifest.
- Secret, token và dữ liệu có điều khoản hạn chế không được ghi vào report.
- Artifact bên ngoài phải có immutable version hoặc content hash; tên Kaggle run
  hay URL mutable một mình là chưa đủ.

## 6. Quy tắc so sánh

Hai run chỉ được so trực tiếp khi metric registry cho phép và các trường khóa
giống nhau: dataset manifest, split/views, depth source, robot/profile hash,
frame, simulator/protocol, sample count, seed policy và code/config version.
Khác một trường phải ghi `comparability: limited|invalid` cùng giải thích.
