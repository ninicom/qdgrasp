---
document_id: GOV-REVIEW-001
document_type: policy
title: Quy chế kiểm tra độc lập bởi người thứ ba
status: active
date: 2026-08-21
owner: project-maintainer
revises: none
---

# Quy chế kiểm tra độc lập bởi người thứ ba

Tài liệu này là quy định bắt buộc cho các báo cáo kiểm tra độc lập của dự án. Mục tiêu là để mọi tuyên bố về tính đúng đắn, hiệu năng, khả năng tái lập và mức sẵn sàng phát hành đều truy vết được tới bằng chứng, thay vì chỉ dựa trên nhận xét của người triển khai.

## 1. Phạm vi áp dụng

Kiểm tra độc lập phải được thực hiện trước khi:

- chấp nhận một mốc triển khai hoặc phát hành;
- công bố số liệu so sánh, độ chính xác, tốc độ, bộ nhớ hoặc độ ổn định;
- tuyên bố tương thích CPU, GPU, định dạng export hoặc simulator;
- đóng một lỗi có mức nghiêm trọng `S0` hoặc `S1`;
- thay đổi tiêu chí nghiệm thu, protocol benchmark hoặc dữ liệu chuẩn.

Kiểm tra có thể giới hạn ở một delta cụ thể, nhưng phạm vi và phần không kiểm tra phải được nêu rõ. Không được dùng kết luận của một phạm vi hẹp để đại diện cho toàn bộ hệ thống.

## 2. Vai trò và tính độc lập

- **Tác giả/nhóm triển khai** cung cấp artifact, hướng dẫn tái lập và phản hồi phát hiện; không tự phát hành báo cáo kiểm tra độc lập cho thay đổi của mình.
- **Người kiểm tra** đánh giá bằng chứng, tái chạy kiểm tra cần thiết và đưa ra verdict. Người này không được là tác giả hoặc người phê duyệt trực tiếp của thay đổi đang xét.
- **Người chấp nhận** quyết định mốc phát hành dựa trên báo cáo; không được tự ý nâng verdict của người kiểm tra.

Mỗi báo cáo phải khai báo một trong hai mức độc lập:

- `external`: người kiểm tra thuộc tổ chức hoặc nhóm pháp lý khác, không tham gia thiết kế, triển khai hay quyết định phát hành;
- `internal_independent`: người kiểm tra cùng tổ chức nhưng không viết, sửa hoặc trực tiếp phê duyệt thay đổi đang xét.

Chỉ báo cáo có `independence: external` mới được gọi là **kiểm tra người thứ ba** trong tài liệu công khai. Báo cáo nội bộ độc lập vẫn hợp lệ cho kiểm soát chất lượng, nhưng phải dùng đúng nhãn.

Người kiểm tra phải công khai quan hệ công việc, lợi ích tài chính, phụ thuộc quản lý và mọi đóng góp trước đó có thể ảnh hưởng tới đánh giá. Khi có xung đột lợi ích đáng kể, báo cáo phải mang verdict `inconclusive` và chuyển cho người kiểm tra khác. Người kiểm tra không được âm thầm sửa artifact nhằm làm cho kiểm tra đạt; mọi sửa đổi phải tạo revision mới.

## 3. Artifact và bằng chứng bắt buộc

Trước khi kiểm tra, nhóm triển khai phải đóng băng đối tượng đánh giá bằng commit, checksum hoặc định danh revision bất biến. Bộ bàn giao tối thiểu gồm:

- mã nguồn và cấu hình chính xác;
- manifest dữ liệu, model/checkpoint và nguồn gốc của chúng;
- phiên bản hệ điều hành, runtime, thư viện, phần cứng và device;
- lệnh chạy từ môi trường sạch, seed và các biến cấu hình ảnh hưởng kết quả;
- tiêu chí nghiệm thu đã được chốt trước khi xem kết quả;
- log thô, kết quả máy đọc được và checksum của artifact đầu vào/đầu ra.

Bằng chứng được xếp theo độ mạnh:

| Cấp | Loại bằng chứng | Cách dùng |
| --- | --- | --- |
| `E1` | Artifact/log thô có checksum và provenance | Nguồn chính cho số liệu và sự kiện thực thi |
| `E2` | Kết quả được người kiểm tra tái lập độc lập | Nguồn chính cho tuyên bố khả năng tái lập |
| `E3` | Bảng/biểu đồ dẫn xuất có liên kết tới `E1` hoặc `E2` | Dùng để diễn giải, không thay thế dữ liệu gốc |
| `E4` | Nhận xét, ảnh chụp màn hình hoặc mô tả thủ công | Chỉ bổ trợ; không đủ để xác nhận một tuyên bố định lượng |

Mỗi số liệu trong báo cáo phải chỉ tới evidence ID. Không được chép số thủ công mà không có nguồn. Không dùng log đã cắt bỏ lỗi, kết quả chỉ từ một lần chạy khi protocol yêu cầu nhiều seed, hoặc bằng chứng được tạo từ revision khác mà không nêu rõ.

Dữ liệu nhạy cảm có thể được che, nhưng phải giữ checksum, provenance và mô tả đủ để audit. Việc che dữ liệu không được làm thay đổi ý nghĩa của bằng chứng.

## 4. Quy trình kiểm tra

1. **Tiếp nhận:** xác nhận câu hỏi cần kiểm tra, phạm vi, revision, tiêu chí nghiệm thu và artifact được cung cấp.
2. **Xác nhận độc lập:** khai báo mức độc lập và xung đột lợi ích trước khi xem kết quả chi tiết.
3. **Kiểm tra nguồn:** đối chiếu checksum, provenance, dữ liệu, cấu hình và môi trường với manifest.
4. **Tái lập:** chạy lại các kiểm tra trọng yếu từ hướng dẫn đã bàn giao; ghi nguyên văn lệnh, môi trường, thời gian và kết quả.
5. **Đối chiếu:** so sánh kết quả với tiêu chí nghiệm thu, baseline và sai số cho phép; không thay đổi ngưỡng sau khi đã xem kết quả.
6. **Ghi phát hiện:** mỗi phát hiện có ID, severity, bằng chứng, bước tái hiện, tác động và điều kiện đóng.
7. **Phản hồi:** nhóm triển khai có thể giải trình hoặc tạo revision sửa lỗi. Giải trình không được xóa phát hiện ban đầu.
8. **Kết luận:** người kiểm tra phát hành báo cáo theo `docs/templates/THIRD_PARTY_REVIEW_REPORT.md` và ký tên/ngày.

Nếu artifact thay đổi sau khi bắt đầu kiểm tra, revision cũ vẫn giữ nguyên báo cáo. Revision mới phải có hồ sơ theo `docs/templates/REVISION_RECORD.md`; người kiểm tra quyết định thực hiện delta review hay kiểm tra lại toàn bộ dựa trên tác động đã ghi.

## 5. Mức nghiêm trọng của phát hiện

| Mức | Định nghĩa | Ví dụ điển hình |
| --- | --- | --- |
| `S0` — Tới hạn | Làm sai bản chất kết quả, gây mất/an toàn dữ liệu, vi phạm pháp lý hoặc khiến artifact không thể tin cậy | Kết quả giả, rò rỉ dữ liệu, benchmark dùng nhầm tập test |
| `S1` — Cao | Tiêu chí nghiệm thu cốt lõi không đạt, lỗi phổ biến chặn sử dụng hoặc tuyên bố chính không tái lập được | CPU không chạy dù đã tuyên bố hỗ trợ, resume làm sai model |
| `S2` — Trung bình | Hành vi sai có phạm vi giới hạn hoặc thiếu bằng chứng quan trọng nhưng có biện pháp tránh rõ ràng | Một cấu hình phụ lỗi, log thiếu trường truy vết |
| `S3` — Thấp | Không ảnh hưởng kết luận chính; lỗi tài liệu, khả dụng hoặc bảo trì có tác động nhỏ | Thông báo lỗi khó hiểu, tài liệu thiếu ví dụ phụ |
| `INFO` | Quan sát hoặc đề xuất, không phải lỗi nghiệm thu | Cơ hội tối ưu hoặc làm rõ trong tương lai |

Severity được xác định theo tác động thực tế và khả năng xảy ra, không theo độ khó sửa. Chỉ người kiểm tra mới thay đổi severity trong báo cáo; mọi thay đổi phải có lý do và lưu lịch sử.

## 6. Verdict

| Verdict | Điều kiện |
| --- | --- |
| `pass` | Tất cả tiêu chí trong phạm vi đã được kiểm bằng bằng chứng đủ mạnh; không còn phát hiện `S0`, `S1` hoặc điều kiện bắt buộc chưa hoàn tất |
| `conditional_pass` | Không còn `S0`/`S1`; chỉ còn điều kiện cụ thể, có người chịu trách nhiệm và hạn xử lý. Chưa được gọi là `pass` trước khi điều kiện được xác minh |
| `fail` | Có `S0`/`S1`, tiêu chí nghiệm thu cốt lõi không đạt hoặc bằng chứng chứng minh tuyên bố chính là sai |
| `inconclusive` | Không đủ bằng chứng, môi trường không tái lập được, phạm vi bị cản trở hoặc tính độc lập không bảo đảm |

`inconclusive` không đồng nghĩa với đạt. Báo cáo `conditional_pass` phải liệt kê từng điều kiện và trạng thái; nếu một điều kiện quá hạn hoặc thất bại, verdict tự động trở thành `fail` cho tới khi có báo cáo cập nhật. Một báo cáo chỉ được nâng verdict bằng evidence mới và chữ ký của người kiểm tra.

## 7. Quy tắc báo cáo và lưu trữ

- Dùng template chuẩn và giữ nguyên các metadata bắt buộc.
- Báo cáo có `status: complete` hoặc `status: accepted` là hồ sơ bất biến. Điều
  chỉnh nội dung phải tạo phiên bản mới, liên kết về phiên bản cũ và nêu lý do.
- Evidence ID phải duy nhất trong báo cáo; finding phải liên kết tới ít nhất một evidence ID.
- Kết quả không kiểm được phải ghi `không kiểm`, kèm lý do; không để trống hoặc suy đoán là đạt.
- Báo cáo phải tách rõ dữ kiện, suy luận và khuyến nghị.
- Raw evidence phải được giữ ở vị trí bền vững; đường dẫn tạm, cache cục bộ hoặc URL có hạn không được xem là hồ sơ cuối.
- Mọi ngoại lệ đối với protocol phải được ghi trước verdict và đánh giá ảnh hưởng tới kết luận.

## 8. Quy tắc ghi nhận công việc theo phiên và sửa phiên trước

Mỗi phiên làm việc phải kết thúc bằng bản ghi **việc thực sự đã hoàn tất trong chính phiên đó**, kèm artifact/evidence; không ghi kế hoạch, dự định hoặc việc chưa xác minh dưới nhãn “đã xong”. Bản ghi của phiên đã đóng là bất biến.

Khi một phiên sau sửa artifact đã được ghi hoàn tất ở phiên trước, bắt buộc:

1. giữ nguyên bản ghi cũ và tạo `REVISION_RECORD` mới;
2. khai báo `revises` với session ID, artifact và revision cũ;
3. mô tả nguyên nhân cụ thể trong `reason`, không dùng câu chung như “cải tiến” hoặc “cập nhật”;
4. chọn đúng mức cần thiết `N0`–`N3` và giải thích vì sao;
5. ghi tác động hành vi, interface, dữ liệu, tương thích và số liệu đã công bố trong `impact`;
6. liên kết bằng chứng xác minh sau sửa và mọi báo cáo người thứ ba bị ảnh hưởng.

Mức cần thiết của chỉnh sửa:

| Mức | Ý nghĩa | Điều kiện sử dụng |
| --- | --- | --- |
| `N0` — Biên tập | Chỉ chính tả/format; không đổi nghĩa, hành vi hay số liệu | Ghi revision ngắn và xác nhận không đổi kết luận |
| `N1` — Làm rõ | Câu chữ có thể gây hiểu nhầm nhưng không đổi kết luận | Ghi claim trước/sau và phạm vi ảnh hưởng |
| `N2` — Bắt buộc | Sai protocol, số liệu, artifact, hành vi hoặc kết luận kỹ thuật | Supersede/invalid bản cũ và kiểm tra lại phần phụ thuộc |
| `N3` — Khẩn cấp | Ảnh hưởng an toàn, phát hành, pháp lý hoặc quyết định lớn | Dừng dùng kết quả cũ và audit toàn bộ phụ thuộc |

Nếu một chỉnh sửa làm thay đổi kết quả hoặc kết luận đã được người thứ ba xác nhận, báo cáo cũ phải được đánh dấu là chỉ áp dụng cho revision cũ; revision mới cần delta review hoặc full review trước khi tái sử dụng verdict.
