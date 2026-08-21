---
document_id: METRICS-REGISTRY-001
document_type: registry
title: Sổ đăng ký chỉ số
status: active
date: 2026-08-21
revises: none
---

# Sổ đăng ký chỉ số (Metrics Registry)

Tài liệu này là nguồn chuẩn duy nhất về cách định nghĩa, đo, kiểm tra và so sánh số liệu của dự án. Một con số chỉ được công bố là **kết quả chính thức** khi metric, protocol, môi trường chạy và artifact thô của nó đáp ứng đầy đủ các yêu cầu dưới đây.

Các từ **PHẢI**, **KHÔNG ĐƯỢC**, **NÊN** và **CÓ THỂ** được dùng theo nghĩa quy phạm.

## 1. Định danh và vòng đời metric

Metric ID có dạng:

```text
DGN-{DOMAIN}-{NAME}-v{MAJOR}
```

- `DOMAIN`: `DATA`, `MODEL`, `SIM`, `SYSTEM` hoặc `EXPORT`.
- `NAME`: tên ASCII viết hoa, các từ nối bằng dấu gạch ngang.
- `MAJOR`: số nguyên dương. Tăng phiên bản khi thay đổi định nghĩa, đơn vị, sample unit, tập dữ liệu, protocol mặc định hoặc cách tổng hợp có thể làm thay đổi ý nghĩa kết quả.
- Sửa lỗi diễn đạt không làm đổi phép đo thì giữ nguyên ID và ghi vào lịch sử thay đổi của tài liệu.
- Metric ID đã được dùng trong báo cáo PHẢI bất biến; không tái sử dụng ID cũ cho định nghĩa mới.

Trạng thái metric: `DRAFT` → `ACTIVE` → `DEPRECATED`. Chỉ metric `ACTIVE` được dùng làm release gate. Kết quả cũ không bị xóa khi metric bị deprecate; báo cáo phải trỏ tới metric thay thế.

## 2. Trường bắt buộc của một metric

Mỗi metric đăng ký PHẢI có đủ các trường sau:

| Trường | Yêu cầu |
|---|---|
| Metric ID, tên, phiên bản, trạng thái | Định danh duy nhất và vòng đời hiện tại. |
| Mục đích | Câu hỏi kỹ thuật mà metric trả lời; không mô tả chung chung. |
| Giá trị tốt hơn | `higher`, `lower`, `target` hoặc `informational`. |
| Định nghĩa và đơn vị | Công thức, miền giá trị, quy tắc làm tròn và xử lý giá trị lỗi/thiếu. |
| Sample unit | Đơn vị độc lập dùng để tính `n` và CI, ví dụ scene, object hoặc inference request; không lấy số grasp có tương quan làm `n` nếu protocol lấy mẫu theo scene. |
| Protocol ID và phiên bản | Quy trình cố định để sinh mẫu và tính metric. Protocol phải chỉ rõ dataset/split/manifest, preprocessing, seed, warm-up, số lần lặp và cách tổng hợp. |
| Môi trường bắt buộc | Phần cứng, OS, runtime, dependency, dtype, device, thread count và simulator/export backend cần ghi lại. |
| Cỡ mẫu | `n` thực tế, `n` dự kiến và lý do nếu thiếu mẫu; với phép đo phân tầng phải báo `n` từng tầng. |
| Độ bất định | Loại CI, mức tin cậy, thuật toán/resampling unit, số bootstrap và seed; metric đếm chính xác ghi `CI=N/A (exact)`. |
| Artifact thô | Đường dẫn tương đối hoặc URI bền vững, định dạng/schema, kích thước byte và SHA-256 chữ thường 64 ký tự. |
| Điều kiện so sánh | Các trường bắt buộc phải giống nhau và các sai khác được phép khi so với baseline. |
| Điều kiện vô hiệu | Thay đổi nào khiến kết quả không còn hợp lệ và cách đánh dấu kết quả bị thay thế. |
| Người sở hữu và người kiểm tra | Người/nhóm chịu trách nhiệm metric và người kiểm tra độc lập. |

Không được nhập thủ công số tổng hợp nếu không thể tái tính nó từ artifact thô bằng lệnh hoặc script đã ghi trong báo cáo.

## 3. Quy ước protocol, môi trường và artifact

### 3.1 Protocol

Protocol ID có dạng `PROTO-{NAME}-v{MAJOR}`. Mỗi lần chạy phải lưu snapshot protocol hoặc SHA-256 của file protocol. Protocol phải khóa tối thiểu:

- commit code, model YAML, checkpoint và robot profile;
- dataset manifest/split và thứ tự hoặc seed lấy mẫu;
- chuỗi preprocessing/postprocessing và tham số của chúng;
- số warm-up, số lần lặp, batch size, top-K và dtype;
- công thức aggregate, chính sách mẫu thiếu/lỗi và phương pháp CI;
- simulator/backend cùng phiên bản nếu có.

### 3.2 Môi trường

Mỗi run có một `environment_id` và một manifest máy đọc được. Manifest tối thiểu gồm:

```yaml
environment_id: ENV-YYYYMMDD-<short-sha256>
os: <name/version/kernel>
cpu: <model, physical/logical cores>
ram_bytes: <integer>
gpu: <model hoặc null>
driver: <version hoặc null>
cuda: <version hoặc null>
python: <version>
torch: <version>
dependencies_lock_sha256: <64-hex>
device: cpu|cuda:<index>
dtype: fp32|amp-fp16|amp-bf16
threads: {intraop: <integer>, interop: <integer>}
```

CPU và GPU là hai cohort kết quả khác nhau. Không gộp chúng thành một số nếu metric không quy định cách tổng hợp từ trước.

### 3.3 Artifact thô và SHA-256

- Artifact thô PHẢI chứa quan sát theo sample unit, không chỉ chứa mean/median cuối cùng.
- Artifact ưu tiên định dạng mở (`jsonl`, `csv`, `parquet`, `npz`) và phải có schema/version.
- Mỗi artifact có entry: `path_or_uri`, `media_type`, `schema_version`, `bytes`, `sha256`, `created_utc`, `producer_command`.
- SHA-256 tính trên đúng byte được lưu: `sha256sum <artifact>` trên Linux. Nếu artifact nén, hash file nén và ghi thêm hash nội dung khi cần kiểm tra lâu dài.
- Artifact không được sửa tại chỗ sau khi công bố. Kết quả tái chạy dùng `run_id` và artifact mới.
- Log, stdout hoặc ảnh chụp chỉ là bằng chứng phụ; không thay thế dữ liệu thô có cấu trúc.

## 4. Quy tắc độ bất định

Mức tin cậy mặc định là 95%. Báo cả ước lượng điểm và hai đầu CI, không dùng ký hiệu `±` nếu đó không thực sự là nửa độ rộng CI đối xứng.

| Loại metric | Cách báo mặc định |
|---|---|
| Tỷ lệ thành công nhị phân | Tỷ lệ + Wilson 95% CI. Khi so sánh cùng sample giữa model mới và baseline: paired bootstrap theo sample unit, tối thiểu 10.000 lần, báo CI của chênh lệch. |
| Metric liên tục theo scene/object | Mean hoặc median đã đăng ký + percentile bootstrap 95% theo sample unit, tối thiểu 10.000 lần. |
| Latency | Median, p90, p95; bootstrap 95% CI cho median. Tách warm-up, đo đồng bộ device, lưu từng lần lặp. |
| Peak memory | Giá trị cực đại trên từng lần chạy độc lập; báo median và range/CI theo số lần chạy. Ghi rõ RSS, allocator hay device peak. |
| Số tham số/kích thước file | Giá trị exact; `CI=N/A (exact)`. |

Nếu dữ liệu có cấu trúc lồng nhau, bootstrap phải resample ở đơn vị độc lập cao nhất đã đăng ký (thường là scene), không resample từng grasp để làm CI hẹp giả tạo. Số mẫu loại bỏ và lý do phải xuất hiện trong báo cáo.

## 5. Khả năng so sánh và vô hiệu hóa

Mỗi cặp kết quả phải được gán một trong ba mức:

- `COMPARABLE`: cùng metric major version, protocol, dataset/split manifest, sample unit, checkpoint selection rule và cohort thiết bị; mọi khác biệt được metric cho phép.
- `QUALIFIED`: có sai khác đã biết nhưng phép so sánh vẫn hữu ích; báo cáo phải định lượng/giải thích ảnh hưởng và không dùng làm release gate.
- `NOT_COMPARABLE`: khác định nghĩa metric, tập mẫu, source/frame/robot, simulator protocol, precision hoặc điều kiện thiết bị có ảnh hưởng trực tiếp đến kết luận.

Kết quả phải chuyển thành `INVALID` khi xảy ra ít nhất một điều sau:

- artifact hoặc SHA-256 không khớp, thiếu quan sát thô, hoặc script không tái tính được kết quả;
- phát hiện data leakage, sample trùng, sai frame/unit/joint order hoặc lỗi gắn nhãn;
- code/config/checkpoint/dataset/protocol thực tế khác provenance đã báo;
- lỗi implementation làm thay đổi giá trị metric;
- cỡ mẫu thấp hơn mức tối thiểu mà không có waiver được ghi và phê duyệt.

Không xóa lịch sử. Entry vô hiệu phải giữ `run_id`, ghi `invalidated_at_utc`, người xác nhận, nguyên nhân, mức ảnh hưởng (`LOCAL`, `METRIC`, `REPORT`, `RELEASE`) và `superseded_by` nếu có. Sửa đổi từ báo cáo phiên trước phải có mục “Lịch sử chỉnh sửa”, nêu rõ nguyên nhân và mức độ cần thiết theo thang `N0`–`N3` trong `docs/governance/DOCUMENTATION_POLICY.md`.

## 6. Registry khởi tạo

Các metric dưới đây là bộ tối thiểu để xây pipeline; chi tiết protocol phải được đăng ký trước khi sinh kết quả chính thức.

| Metric ID | Tên | Tốt hơn | Đơn vị / aggregate | Sample unit | CI mặc định | Trạng thái |
|---|---|---:|---|---|---|---|
| `DGN-MODEL-PARAMS-v1` | Số tham số trainable | informational | count, exact | model build | N/A | ACTIVE |
| `DGN-MODEL-SIM-SUCCESS-v1` | Tỷ lệ grasp thành công | higher | proportion | scene-grasp trial đã khóa | Wilson; paired bootstrap khi so baseline | DRAFT |
| `DGN-MODEL-TRANSLATION-ERROR-v1` | Sai số tịnh tiến | lower | m, median | scene | bootstrap | DRAFT |
| `DGN-MODEL-ROTATION-ERROR-v1` | Sai số geodesic rotation | lower | degree, median | scene | bootstrap | DRAFT |
| `DGN-MODEL-JOINT-ERROR-v1` | Sai số joint có mask | lower | degree/mm theo loại joint | scene | bootstrap | DRAFT |
| `DGN-SYSTEM-LATENCY-v1` | Độ trễ inference end-to-end | lower | ms; median/p90/p95 | request | bootstrap median | DRAFT |
| `DGN-SYSTEM-PEAK-MEMORY-v1` | Bộ nhớ đỉnh | lower | MiB | run độc lập | median + range/CI | DRAFT |
| `DGN-EXPORT-PARITY-v1` | Sai lệch backend xuất | lower | max/quantile error theo output | sample | bootstrap hoặc exact theo protocol | DRAFT |

Metric `DRAFT` không được dùng làm tuyên bố release cho tới khi protocol tương ứng được duyệt và trạng thái đổi sang `ACTIVE` trong một thay đổi có review.

## 7. Bảng kết quả bắt buộc

Mọi báo cáo phải có ít nhất bảng sau; một hàng tương ứng một metric trong một cohort môi trường:

| Run ID | Metric ID | Protocol ID | Model/checkpoint | Dataset manifest | Environment ID | Device/dtype | n | Estimate | 95% CI | Raw artifact | SHA-256 | So với baseline | Trạng thái |
|---|---|---|---|---|---|---|---:|---:|---|---|---|---|---|
| `<run>` | `<metric-vN>` | `<proto-vN>` | `<id/hash>` | `<hash>` | `<env>` | `<device/dtype>` | `<n>` | `<value unit>` | `<low, high/N/A>` | `<path/URI>` | `<64-hex>` | `COMPARABLE/QUALIFIED/NOT_COMPARABLE` | `VALID/INVALID` |

Không được để trống ô. Dùng `N/A` kèm lý do trong phần ghi chú nếu trường không áp dụng.

## 8. Kiểm tra độc lập

Trước khi gắn nhãn `VERIFIED`, một người không tạo run và không sửa artifact phải:

1. kiểm tra provenance, manifest môi trường và SHA-256;
2. tái tính toàn bộ metric từ artifact thô bằng command đã ghi;
3. đối chiếu cỡ mẫu, missing/exclusion và CI;
4. xác nhận mức comparability và mọi điều kiện invalidation;
5. chạy lại ít nhất một mẫu đại diện khi môi trường cho phép;
6. ghi danh tính, thời điểm UTC, commit review và kết luận `PASS`, `PASS_WITH_NOTES` hoặc `FAIL`.

Kết quả chưa kiểm tra phải ghi `UNVERIFIED`; không được diễn đạt như đã được bên thứ ba xác nhận.
