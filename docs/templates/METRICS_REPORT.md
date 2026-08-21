---
document_id: METRICS-YYYYMMDD-RUNID
document_type: metrics_report
title: Báo cáo số liệu - tên phiên hoặc chủ đề
status: draft
date: YYYY-MM-DD
run_id: RUN-YYYYMMDD-ID
author: AUTHOR-ID
reviewer: unassigned
revises: none
---

# Báo cáo số liệu: `<tên phiên/chủ đề>`

> Mẫu này phải được điền cùng [Metrics Registry](../metrics/METRICS_REGISTRY.md). Không xóa các mục bắt buộc; dùng `N/A` và giải thích nếu không áp dụng.

## 1. Thông tin báo cáo

| Trường | Giá trị |
|---|---|
| Report ID | `REPORT-YYYYMMDD-<slug>-vN` |
| Trạng thái | `DRAFT / REVIEWED / VERIFIED / INVALID` |
| Phiên làm việc | `<session ID hoặc ngày UTC>` |
| Tác giả | `<tên/định danh>` |
| Người kiểm tra độc lập | `<tên/định danh hoặc UNASSIGNED>` |
| Commit code | `<full Git SHA>` |
| Nhánh/tag | `<branch/tag>` |
| Thời gian bắt đầu/kết thúc | `<ISO-8601 UTC>` |
| Mục tiêu | `<câu hỏi mà báo cáo cần trả lời>` |
| Kết luận một câu | `<kết luận có giới hạn, không vượt quá bằng chứng>` |

## 2. Việc hoàn thành trong phiên

Chỉ ghi việc đã hoàn tất và có bằng chứng kiểm tra được. Việc chưa xong chuyển sang mục “Tồn đọng”, không dùng các cụm từ mơ hồ như “gần xong”.

| Work ID | Việc đã hoàn thành | Bằng chứng (commit/test/artifact) | Kết quả kiểm tra | Người thực hiện |
|---|---|---|---|---|
| `<W-001>` | `<mô tả ở thì hoàn thành>` | `<SHA/path/URI>` | `PASS / FAIL / N/A` | `<id>` |

### Tồn đọng và giới hạn

| Item | Trạng thái | Ảnh hưởng tới kết luận | Hành động tiếp theo |
|---|---|---|---|
| `<item>` | `OPEN / BLOCKED / DEFERRED` | `<ảnh hưởng>` | `<hành động có thể kiểm chứng>` |

## 3. Chỉnh sửa công việc từ phiên trước

Nếu không có, ghi “Không có”. Mọi thay đổi vào công việc từng được báo hoàn tất phải nêu nguyên nhân và mức độ cần thiết.

| Change ID | Phiên/work ID cũ | Nội dung đã sửa | Nguyên nhân có bằng chứng | Mức cần thiết | Phạm vi ảnh hưởng | Kết quả tái kiểm tra |
|---|---|---|---|---|---|---|
| `<C-001>` | `<session/W-ID>` | `<before → after>` | `<bug/requirement/evidence>` | `N0 / N1 / N2 / N3` | `LOCAL / METRIC / REPORT / RELEASE` | `<test/artifact + PASS/FAIL>` |

Định nghĩa mức cần thiết (nguồn chuẩn:
`docs/governance/DOCUMENTATION_POLICY.md`):

- `N0` — biên tập: chỉ chính tả/format, không đổi nghĩa hoặc số liệu.
- `N1` — làm rõ: tránh hiểu nhầm nhưng không đổi kết luận.
- `N2` — bắt buộc: sửa sai protocol, số liệu, artifact hoặc kết luận kỹ thuật.
- `N3` — khẩn cấp: ảnh hưởng an toàn, phát hành, pháp lý hoặc quyết định lớn.

## 4. Protocol và phạm vi mẫu

| Trường | Giá trị |
|---|---|
| Protocol ID | `PROTO-<NAME>-vN` |
| Protocol path + SHA-256 | `<path> / <64-hex>` |
| Metric ID áp dụng | `<DGN-...-vN>` |
| Dataset/split | `<name/version/split>` |
| Dataset manifest + SHA-256 | `<path/URI> / <64-hex>` |
| Sample unit | `<scene/object/request/run/...>` |
| Cỡ mẫu dự kiến | `<n>` |
| Cỡ mẫu thực tế | `<n>` |
| Missing/excluded | `<count và lý do từng nhóm>` |
| Seed | `<integer/list>` |
| Pre/postprocessing | `<config path + SHA-256>` |
| Warm-up/lặp/batch/top-K | `<giá trị hoặc N/A>` |
| Aggregate | `<mean/median/rate/p90/...>` |
| Phương pháp CI | `<method, 95%, resampling unit, số lần, seed>` |

Mô tả mọi sai lệch khỏi protocol đã đăng ký và đánh giá liệu sai lệch đó làm kết quả `QUALIFIED` hay `INVALID`:

```text
<Không có / mô tả sai lệch, nguyên nhân và ảnh hưởng>
```

## 5. Môi trường thực thi

| Trường | Giá trị |
|---|---|
| Environment ID | `ENV-YYYYMMDD-<short-sha256>` |
| Environment manifest | `<path/URI + SHA-256>` |
| OS/kernel | `<name/version>` |
| CPU/RAM | `<model, core/thread, bytes>` |
| GPU/driver/CUDA | `<model/version hoặc N/A>` |
| Python/PyTorch | `<versions>` |
| Dependency lock SHA-256 | `<64-hex>` |
| Device/dtype | `<cpu|cuda:N> / <fp32|amp-fp16|amp-bf16>` |
| Thread settings | `<intraop/inter-op/OMP/MKL>` |
| Simulator/export backend | `<name/version hoặc N/A>` |
| Model YAML SHA-256 | `<64-hex>` |
| Checkpoint SHA-256 | `<64-hex>` |
| Robot profile SHA-256 | `<64-hex hoặc N/A>` |

## 6. Kết quả metric

Một hàng chỉ chứa một metric và một cohort môi trường. Không gộp CPU/GPU, precision hoặc protocol khác nhau.

| Run ID | Metric ID | Protocol ID | Model/checkpoint | Dataset manifest | Environment ID | Device/dtype | n | Estimate | 95% CI | Raw artifact | SHA-256 | So với baseline | Trạng thái |
|---|---|---|---|---|---|---|---:|---:|---|---|---|---|---|
| `<run>` | `<metric-vN>` | `<proto-vN>` | `<id/hash>` | `<hash>` | `<env>` | `<device/dtype>` | `<n>` | `<value unit>` | `<low, high/N/A>` | `<path/URI>` | `<64-hex>` | `COMPARABLE/QUALIFIED/NOT_COMPARABLE` | `VALID/INVALID` |

### So sánh với baseline

| Metric ID | Baseline run | Candidate run | Chênh lệch (candidate − baseline) | CI của chênh lệch | Mức so sánh | Lý do |
|---|---|---|---:|---|---|---|
| `<metric>` | `<run>` | `<run>` | `<value unit>` | `<low, high>` | `COMPARABLE / QUALIFIED / NOT_COMPARABLE` | `<đối chiếu protocol/data/env>` |

### Diễn giải

- `<Kết luận nào được dữ liệu hỗ trợ>`
- `<Kết luận nào chưa được dữ liệu hỗ trợ>`
- `<Nguồn bất định hoặc sai lệch còn lại>`

## 7. Manifest artifact

Artifact thô phải chứa quan sát theo sample unit. Mỗi file là bất biến sau khi báo cáo.

| Artifact ID | Vai trò | Path hoặc URI | Media type/schema | Bytes | SHA-256 | Created UTC | Producer command |
|---|---|---|---|---:|---|---|---|
| `<A-001>` | `raw / aggregate / log / config / plot` | `<path/URI>` | `<type/vN>` | `<bytes>` | `<64-hex>` | `<ISO-8601>` | `<lệnh tái tạo>` |

Lệnh kiểm tra hash:

```bash
sha256sum <artifact-paths>
```

Lệnh tái tính bảng metric từ artifact thô:

```bash
<exact command, arguments và working directory>
```

Kết quả tái tính: `<PASS/FAIL>`; sai khác tối đa: `<value/unit>`.

## 8. Vô hiệu hóa và thay thế

| Trường | Giá trị |
|---|---|
| Có trigger invalidation không? | `NO / YES` |
| Trigger | `<hash mismatch/data leakage/frame/unit/config/code/... hoặc N/A>` |
| Mức ảnh hưởng | `LOCAL / METRIC / REPORT / RELEASE / N/A` |
| Invalidated at UTC/by | `<time/id hoặc N/A>` |
| Run/report thay thế | `<ID hoặc N/A>` |
| Biện pháp ngăn tái diễn | `<test/validation/process hoặc N/A>` |

## 9. Kiểm tra độc lập

Người kiểm tra phải độc lập với người tạo run và không sửa artifact đang kiểm tra.

| Hạng mục | Kết quả | Bằng chứng/ghi chú |
|---|---|---|
| Commit, config, checkpoint và provenance khớp báo cáo | `PASS / FAIL` | `<chi tiết>` |
| SHA-256 và kích thước mọi artifact khớp | `PASS / FAIL` | `<chi tiết>` |
| Cỡ mẫu, missing và exclusion khớp dữ liệu thô | `PASS / FAIL` | `<chi tiết>` |
| Tái tính estimate và CI từ artifact thô | `PASS / FAIL` | `<command/output>` |
| Metric/protocol đúng phiên bản | `PASS / FAIL` | `<chi tiết>` |
| Mức comparability hợp lệ | `PASS / FAIL` | `<chi tiết>` |
| Không có trigger invalidation chưa xử lý | `PASS / FAIL` | `<chi tiết>` |
| Chạy lại mẫu đại diện | `PASS / FAIL / NOT_RUN` | `<sample/env/result/lý do>` |

| Trường xác nhận | Giá trị |
|---|---|
| Reviewer | `<tên/định danh>` |
| Thời điểm UTC | `<ISO-8601>` |
| Review commit/signature | `<SHA/chữ ký>` |
| Kết luận | `PASS / PASS_WITH_NOTES / FAIL` |
| Ghi chú bắt buộc | `<nội dung hoặc N/A>` |

Chỉ đổi trạng thái báo cáo thành `VERIFIED` khi kết luận là `PASS` hoặc `PASS_WITH_NOTES` và mọi note không làm vô hiệu kết quả.

## 10. Lịch sử chỉnh sửa báo cáo

| Phiên bản | Thời điểm UTC | Người sửa | Nội dung | Nguyên nhân | Mức cần thiết | Có đổi kết luận? |
|---|---|---|---|---|---|---|
| `v1` | `<time>` | `<id>` | `Tạo báo cáo` | `Báo cáo phiên` | `N/A (tạo mới)` | `N/A` |
