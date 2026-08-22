---
document_id: REV-20260822-007
document_type: revision_record
revision_schema: 2
title: Khóa môi trường và reference-on-demand trong PLAN
status: complete
date: 2026-08-22
record_id: REV-20260822-007
session_id: SESSION-20260822-016
created_at: 2026-08-22T18:20:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-015
    artifact: PLAN.md
    revision: 8bf07953d390a47b5826cceab74724f54d50345804daca01f7f3ea8c76d15bd5
reason: Baseline kiến trúc đã đủ; cần khóa môi trường cài đặt và chỉ mở rộng reference khi có vấn đề kỹ thuật tái hiện được.
necessity: N2
impact: PLAN 3.2 khóa Python/PyTorch/MuJoCo, CPU/cu128 profiles và reference intake; không đổi license, DGN2 boundary hay kiến trúc HandGraph-flow.
supersedes: none
---

# REV-20260822-007 — Environment freeze

## Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260822-016`.
- Bản ghi hoàn tất: `docs/sessions/SESSION-20260822-016-environment-freeze.md`.
- Phiên bị sửa: `SESSION-20260822-015`.
- Artifact: `PLAN.md`.
- Checksum trước: `8bf07953d390a47b5826cceab74724f54d50345804daca01f7f3ea8c76d15bd5`.
- Checksum sau: `b1b5e28b031ac2f0f5a820fcf90265aff9c4cbb784b36db14d295f7bce12dbd4`.
- Yêu cầu nguồn: maintainer ngày 2026-08-22.

## Lý do chỉnh sửa

PLAN 3.1 đã chọn stack nhưng chưa có phiên bản cài đặt, backend lock hoặc quy
trình giới hạn việc thêm code tham chiếu. Điều đó làm implementation và kết quả
paper khó tái lập, đồng thời có thể khiến dependency/reference surface tiếp tục
mở rộng không có câu hỏi kỹ thuật cụ thể.

## Mức độ cần thiết

- Mức: `N2` vì thay protocol phát triển, dependency resolution và evidence bắt
  buộc cho CPU/CUDA.
- Hiệu lực: trước implementation model mới.
- Nếu trì hoãn: chỉ được làm documentation/provenance, không train benchmark.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Runtime | Stack chưa pin | Python 3.11, Torch 2.11, Fabric 2.6.5, MuJoCo 3.12 | Vừa | Implement trên lock |
| Config | Chưa có install contract | Pydantic + PyYAML pin | Vừa | Viết schema riêng |
| CPU/GPU | Chỉ nêu parity | CPU và cu128 lock tách biệt | Cao | GPU smoke trên NVIDIA |
| Reference | Có bốn baseline | Đóng baseline, intake on-demand | Vừa | Mở issue trước clone mới |
| License/data | Boundary đã khóa | Không đổi | Không | Giữ gates cũ |

Tóm tắt tác động: PLAN 3.2 khóa Python/PyTorch/MuJoCo, CPU/cu128 profiles và
reference intake; không đổi license, DGN2 boundary hay kiến trúc HandGraph-flow.

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| CH-001 | Khóa environment contract và transitive dependencies | `environments/` | session E-02–E-04 |
| CH-002 | Tạo validator và cài CPU profile | `scripts/check_environment.py` | session T-01/T-02 |
| CH-003 | Quy định reference-on-demand | `docs/governance/REFERENCE_INTAKE.md` | ADR-0005 |
| CH-004 | Cập nhật PLAN 3.2 | `PLAN.md` / `b1b5e28b…` | session E-01 |

## Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| V-001 | CPU environment validator | Torch/Fabric/MuJoCo pass | Pass | pass | session T-01 |
| V-002 | uv dependency check | Không conflict | 48 packages compatible | pass | session T-02 |
| V-003 | cu128 dry-run | Lock resolve | 67 packages resolve | pass | session T-03 |
| V-004 | docs/unit/diff gates | Không regression | Pass | pass | session T-04–T-06 |

- Regression: documentation, reference/train registries và 47 unit tests.
- Chưa chạy: CUDA hardware smoke vì host không có NVIDIA GPU.
- Rollback: khôi phục PLAN theo checksum trước; lock/docs mới có thể giữ làm
  evidence mà không ảnh hưởng runtime hiện tại.

## Ảnh hưởng tới báo cáo và quyết định cũ

- Third-party review: không có review cũ bị vô hiệu.
- Verdict phiên 015: còn hiệu lực.
- Review cần thiết: delta review khi bắt đầu GPU benchmark.
- Đính chính public: không có.
- Ngoại lệ/rủi ro: CUDA giữ `resolved_not_hardware_verified`, không được báo pass.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22, Asia/Bangkok.
- Người kiểm tra: automated gates; GPU reviewer chưa áp dụng.
- Kết luận: environment CPU và documentation đã hoàn tất; CUDA hardware gate
  được ghi rõ là follow-up trước benchmark.
- Session: `docs/sessions/SESSION-20260822-016-environment-freeze.md`.
