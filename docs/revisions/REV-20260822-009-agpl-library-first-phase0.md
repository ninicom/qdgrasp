---
document_id: REV-20260822-009
document_type: revision_record
revision_schema: 2
title: Chuyển Phase 0 sang AGPL library-first
status: draft
date: 2026-08-22
record_id: REV-20260822-009
session_id: SESSION-20260822-018
created_at: 2026-08-22T21:00:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-017
    artifact: PLAN.md
    revision: b9c74d6b47adc63813912bdf734a2589dce54fb481c2ab9b119cfaeffd20ec18
reason: Maintainer chốt tên QDGrasp, AGPL-3.0, cài như thư viện từ đầu, public repository hiện tại và loại RH56E2.
necessity: N3
impact: Supersede Apache clean-room, giữ history/notice AGPL và thêm package plus external Kaggle CUDA gate cho Phase 0.
supersedes: REV-20260822-006
---

# REV-20260822-009 — AGPL library-first Phase 0

## Liên kết truy vết

- Phiên thực hiện: `SESSION-20260822-018`.
- Phiên bị sửa: `SESSION-20260822-017` và quyết định Apache ở phiên 015.
- Artifact chính: `PLAN.md`.
- Checksum trước: `b9c74d6b47adc63813912bdf734a2589dce54fb481c2ab9b119cfaeffd20ec18`.
- Checksum sau: `dc6a71e38de16641e665cbf5f9de2b821bb3c78fefbcea0602adffea90de783f`.

## Lý do chỉnh sửa

Apache clean-room không còn phản ánh lựa chọn maintainer. AGPL cho phép tiếp tục
cây dẫn xuất hợp lệ và hỗ trợ mục tiêu community/copyleft, trong khi DGN2 vẫn bị
giữ ở biên paper-only và model/checkpoint được phát triển mới hoàn toàn.

## Mức độ cần thiết

`N3`: đổi license/repository strategy và release gate. Quyết định cũ không được
dùng để xóa history, notice hoặc trì hoãn public package.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Tác động | Hành động |
| --- | --- | --- | --- | --- |
| Code license | Apache clean-room dự kiến | AGPL-3.0-only | Rất cao | ADR-0007, metadata, NOTICE |
| Tên public | DexGrasp | QDGrasp / `qdgrasp` | Cao | package, CLI và repo đồng nhất |
| Repository | Fresh history | Giữ history hiện tại | Cao | Public `ninicom/qdgrasp` |
| Packaging | P1 | Library từ P0 | Cao | wheel/sdist/import/CLI gate |
| GPU evidence | Máy NVIDIA chưa chốt | Kaggle repo riêng | Cao | exact-commit notebook |
| Robot | LEAP/Allegro/Shadow + lịch sử RH | RH56E2 bị loại | Trung bình | manifest + sidecar scan |

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| CH-001 | Chốt AGPL và DGN2 boundary | ADR-0007, PLAN 4.0 | docs gate |
| CH-002 | Tạo public package/CLI/runtime | `pyproject.toml`, `qdgrasp/runtime.py` | wheel test |
| CH-003 | Loại RH56E2 khỏi active scope | robot lock, archive sidecar | Phase 0 checker |
| CH-004 | Tách Kaggle harness | external repository | pending GPU run |

## Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| V-001 | wheel install ngoài source tree | import/CLI pass | pass | pass | session T-01 |
| V-002 | CPU environment + runtime tests | pass | pass | pass | session T-02 |
| V-003 | docs/legacy tests | pass | pass | pass | session T-03 |
| V-004 | Kaggle cu128 hardware run | pass | pending | pending | external run |

## Ảnh hưởng tới báo cáo và quyết định cũ

- ADR-0002 và REV-006 là lịch sử đã bị supersede, không còn quy phạm.
- ADR-0004 DGN2 paper-only tiếp tục hiệu lực; chỉ phần nhắc Apache clean-room bị
  ADR-0007 thay thế.
- CUDA fail-closed ADR-0006 tiếp tục hiệu lực và phải có Kaggle hardware evidence.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22.
- Reviewer: automated gates; external GPU runtime pending.
- Kết luận: draft cho tới khi Kaggle run và public repositories pass.
- Session: `docs/sessions/SESSION-20260822-018-phase0-agpl-library.md`.
