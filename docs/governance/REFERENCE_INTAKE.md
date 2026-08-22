---
document_id: REFERENCE-INTAKE-001
document_type: policy
title: Quy trình bổ sung source tham chiếu khi gặp ca khó
version: 1.0.0
status: active
date: 2026-08-22
revises: none
necessity: N2
impact: Cho phép mở rộng reference có kiểm soát mà không biến chúng thành dependency hoặc làm trôi kiến trúc đã khóa.
revision_reason: Maintainer cho phép tiếp tục tham khảo code public khi implementation gặp vấn đề khó.
---

# Quy trình bổ sung reference

Baseline hiện tại đã đủ để triển khai. Không tiếp tục sưu tầm repository nếu
chưa có một câu hỏi kỹ thuật cụ thể.

## Khi nào được bổ sung

Một reference mới phải trả lời ít nhất một vấn đề đã tái hiện được:

- lỗi toán học/numerical, FK, rotation, collision hoặc diffusion/flow;
- contract cross-hand chưa giải được bằng baseline hiện tại;
- chênh lệch CPU/CUDA, memory hoặc distributed behavior;
- ablation cho paper cần một phương pháp đối chứng độc lập;
- thiếu test oracle tối giản cho một công thức hoặc interface.

Issue phải ghi input tối thiểu, expected behavior, actual behavior, stack trace
hoặc metric và các reference hiện tại đã kiểm tra.

## Intake gate

1. Tìm paper và repository chính chủ; ưu tiên MIT/BSD/Apache-2.0.
2. Kiểm exact commit, root license, file-level notice, submodule, checkpoint,
   dataset, asset và dependency riêng biệt.
3. Clone detached dưới `.references/architectures/<name>`; không thêm vào import
   path, package data hoặc release tests.
4. Thêm entry vào `architecture_references.lock.yaml`, gồm câu hỏi cần trả lời,
   phần được xem, phần bị cấm và license review còn thiếu.
5. Trích ra contract/toán học/test case tối thiểu. Mặc định viết implementation
   mới; nếu copy code thì giữ attribution/NOTICE và lập file-level provenance.
6. Nếu reference làm đổi schema, model topology, dataset hoặc release license,
   mở ADR/revision tương ứng trước khi merge.
7. Sau khi giải quyết issue, lưu kết luận và negative findings; không xóa pin để
   tránh lặp lại cùng sai lầm.

## Quy tắc quyết định

Không chấp nhận một thiết kế chỉ vì nhiều repository cùng dùng nó nếu chúng có
chung nguồn tổ tiên. Cần ít nhất hai trong ba loại bằng chứng: derivation/toán
học, test tổng hợp độc lập, hoặc kết quả simulator/ablation. Reference code chỉ
là bằng chứng hỗ trợ.

Không tải checkpoint/dataset để “thử nhanh” nếu license/provenance chưa qua gate.
DGN2 tiếp tục là paper/reference-only theo ADR-0004.
