---
document_id: ADR-0007
document_type: decision
title: Phát hành QDGrasp như thư viện cộng đồng AGPL-3.0
status: accepted
date: 2026-08-22
revises: none
supersedes: ADR-0002
---

# ADR-0007 — AGPL community library

## Bối cảnh

ADR-0002 từng yêu cầu bỏ toàn bộ cây dẫn xuất và xây một repository Apache-2.0
clean-room. Maintainer sau đó chốt mục tiêu khác: giữ mô hình cộng đồng/copyleft
giống YOLO, phát triển model grasp mới hoàn toàn, không dùng checkpoint, code,
data hoặc asset của DexGraspNet2, và cài QDGrasp như thư viện ngay từ Phase 0.

## Quyết định

1. Đổi tên public từ DexGrasp thành **QDGrasp** để không bị hiểu là dự án của
   tác giả khác. Distribution/import/CLI/repository chuẩn đều dùng `qdgrasp`.
2. Code chính thức phát hành dưới `AGPL-3.0-only`. Root `LICENSE`, package
   metadata, source/sdist/wheel và public repository phải nhất quán.
3. Tiếp tục repository và Git history hiện tại. File dẫn xuất Ultralytics giữ
   nguyên copyright/header; reuse MIT/BSD/Apache giữ license và notice tương ứng.
4. Public API mới là `qdgrasp`; notebook, benchmark và downstream project cài
   package từ exact release/commit, không copy source vào notebook.
5. DGN2 chỉ là literature context cho bài toán và nguyên lý grasping. Cấm import
   code/config, dùng dataset/checkpoint/label/distillation target, hoặc đưa URDF,
   mesh và metadata DGN2 vào sản phẩm.
6. Kiến trúc, schema, model, generator và checkpoint QDGrasp được phát triển và
   train mới. Reference source chỉ được dùng theo exact license/provenance.
7. Apache-2.0/MIT/BSD source có thể được kết hợp vào AGPL project khi giữ đủ
   notice và kiểm hướng tương thích. License của dataset, checkpoint và asset
   vẫn độc lập với license code.
8. RH56E2 bị loại khỏi mọi scope hoạt động. Những nhắc tới trong raw archive chỉ
   là lịch sử bất biến, không phải manifest hay quyền tái nhập.
9. Phase 0 chỉ đóng khi wheel/sdist cài ngoài source tree, CPU gate pass và
   notebook repo riêng chạy CUDA/cu128 trên GPU NVIDIA thật bằng public package.

## Bằng chứng tương thích license

- GNU mô tả Apache License 2.0 tương thích với GPLv3 theo một hướng và các GNU
  license phiên bản 3 có thể bao hàm điều khoản Apache 2.0:
  `https://www.gnu.org/licenses/license-compatibility.en.html`.
- Apache Software Foundation cũng ghi Apache-2.0 có thể được đưa vào dự án GPLv3,
  nhưng chiều ngược lại không đúng:
  `https://www.apache.org/licenses/GPL-compatibility`.
- GNU GPL FAQ làm rõ các phát biểu tương thích GPLv3 nhìn chung cũng áp dụng cho
  AGPLv3: `https://www.gnu.org/licenses/gpl-faq.en.html`.

Các nguồn trên không thay thế audit theo file hoặc tư vấn pháp lý. Không suy từ
license repository sang weights, dataset, sample assets hay submodule.

## Hệ quả

- Kế hoạch Apache clean-room, fresh-history repository và zero-copyleft gate bị
  hủy. ADR-0002 còn lại như hồ sơ quyết định đã bị thay thế.
- AGPL buộc người phân phối/sửa đổi và người vận hành bản sửa đổi qua mạng tuân
  thủ nghĩa vụ cung cấp source tương ứng; đây là chủ ý cộng đồng của dự án.
- P1 trở đi vẫn ưu tiên API/schema QDGrasp gọn và model mới, không biến YAML
  thành DSL tùy ý và không dùng DGN2 checkpoint làm lối tắt.
