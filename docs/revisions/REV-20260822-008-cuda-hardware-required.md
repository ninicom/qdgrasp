---
document_id: REV-20260822-008
document_type: revision_record
revision_schema: 2
title: Bắt buộc CUDA workload chạy trên GPU NVIDIA
status: complete
date: 2026-08-22
record_id: REV-20260822-008
session_id: SESSION-20260822-017
created_at: 2026-08-22T18:40:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-016
    artifact: PLAN.md
    revision: b1b5e28b031ac2f0f5a820fcf90265aff9c4cbb784b36db14d295f7bce12dbd4
reason: Environment lock resolve chưa đủ rõ; maintainer chốt mọi phần CUDA phải chạy trên CUDA thật.
necessity: N2
impact: PLAN 3.3 cấm CPU fallback cho CUDA và yêu cầu hardware evidence trước khi đóng các model/training gates.
supersedes: none
---

# REV-20260822-008 — CUDA hardware gate

## Liên kết truy vết

- Phiên thực hiện: `SESSION-20260822-017`.
- Phiên bị sửa: `SESSION-20260822-016`.
- Artifact: `PLAN.md`.
- Checksum trước: `b1b5e28b031ac2f0f5a820fcf90265aff9c4cbb784b36db14d295f7bce12dbd4`.
- Checksum sau: `b9c74d6b47adc63813912bdf734a2589dce54fb481c2ab9b119cfaeffd20ec18`.
- Nguồn: yêu cầu maintainer ngày 2026-08-22.

## Lý do chỉnh sửa

Phiên 016 đã khóa cu128 nhưng mới xác minh resolution. Nếu không fail-closed,
CPU fallback có thể làm một CUDA command chạy “thành công” mà không kiểm kernel,
driver, VRAM, AMP hoặc GPU-specific behavior.

## Mức độ cần thiết

- Mức `N2`: thay execution/test protocol và điều kiện chấp nhận benchmark.
- Có hiệu lực trước P0 CUDA gate và mọi model training.
- Không được trì hoãn bằng CPU evidence.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| CUDA runtime | Lock resolve | GPU NVIDIA thật, cu128, fail-closed | Cao | Hardware smoke |
| CPU | CI/correctness | Giữ nguyên, không đại diện CUDA | Thấp | Tiếp tục CPU CI |
| Training | CUDA dự kiến | CUDA bắt buộc | Cao | GPU runner |
| Benchmark/paper | Chờ hardware evidence | Không claim trước evidence | Cao | Lưu raw GPU metadata |
| License/schema | Không đổi | Không đổi | Không | Giữ gates cũ |

Tóm tắt tác động: PLAN 3.3 cấm CPU fallback cho CUDA và yêu cầu hardware
evidence trước khi đóng các model/training gates.

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| CH-001 | Ghi CUDA fail-closed decision | ADR-0006 / `25525b4f…` | session E-02 |
| CH-002 | Cập nhật PLAN 3.3 | PLAN / `b9c74d6b…` | session E-01 |
| CH-003 | Cập nhật environment/roadmap | protocol, manifest, roadmap | docs gate |

## Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| V-001 | Documentation validator | Hợp lệ | Pass | pass | session T-01 |
| V-002 | Whitespace check | Không lỗi | Pass | pass | session T-02 |
| V-003 | Policy consistency scan | Không CPU fallback trong CUDA scope | PLAN/ADR/protocol/roadmap nhất quán | pass | repository text |

- Regression: documentation contract và existing CPU environment không đổi.
- Downstream gate chưa chạy: CUDA hardware test; gate này không cần để xác minh
  thay đổi tài liệu nhưng bắt buộc trước khi đóng P0/P1 CUDA hoặc P4–P7.
- Rollback: khôi phục PLAN theo checksum trước, nhưng rollback sẽ trái yêu cầu
  fail-closed mới của maintainer.

## Ảnh hưởng tới báo cáo và quyết định cũ

- CPU verdict phiên 016 còn hiệu lực cho CPU.
- CUDA `resolved_not_hardware_verified` vẫn là trạng thái chính xác.
- Cần delta review sau khi có GPU evidence.
- Không có public metric cần đính chính.
- Không có ngoại lệ CPU fallback.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22, Asia/Bangkok.
- Người kiểm tra: automated documentation gate; GPU reviewer chưa áp dụng.
- Kết luận: policy/docs đã chốt; CUDA execution gate chưa pass và được ghi rõ.
- Session: `docs/sessions/SESSION-20260822-017-cuda-hardware-required.md`.
