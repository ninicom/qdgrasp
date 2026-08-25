---
document_id: REV-20260825-006
document_type: revision_record
revision_schema: 2
title: Đóng băng và bàn giao release candidate Phase 3.3
status: complete
date: 2026-08-25
record_id: REV-20260825-006
session_id: SESSION-20260825-P3.3-RELEASE-CANDIDATE
created_at: 2026-08-25T18:30:00+07:00
author: codex-primary-agent
revises:
  - session_id: ROADMAP-P3.3-001
    artifact: docs/roadmap/PHASE3_3_SCENE_GRASP_DATA_PLAN.md
    revision: 2edc8dc95cd8d82539a568e7d066b7c654c87e36cf7a9c2e5ed25366db36115c
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 05f1a23a0e60f53dc9119960c5f5a182373a333c57dc27537ed32dfcc9613238
reason: "P3.3-14 đã tạo đủ bounded three-hand scene release và P3.3-15 đã có deterministic, integrity, resource và license gates; roadmap phải thay checkpoint runtime bằng trạng thái release candidate chính xác và giữ independent acceptance gate mở."
necessity: N2
impact: "Bộ triển khai và artifact P3.3 được đóng băng để bàn giao; điều kiện hoàn tất 1–7 đạt, còn điều kiện 8 chờ reviewer độc lập theo GOV-REVIEW-001 nên chưa nâng trạng thái plan thành complete."
---

# REV-20260825-006 — Phase 3.3 release candidate

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260825-P3.3-RELEASE-CANDIDATE`.
- Bản ghi hoàn tất của phiên hiện tại: tài liệu này.
- Phiên bị sửa: `ROADMAP-P3.3-001` và `ROADMAP-PROJECT-001`.
- Artifact bị sửa: hai roadmap trong metadata.
- Revision/checksum trước sửa: hai SHA-256 trong metadata.
- Revision/checksum sau sửa: commit tài liệu chứa record này.
- Yêu cầu nguồn: P3.3-14, P3.3-15 và GOV-REVIEW-001.

## 2. Lý do chỉnh sửa

Checkpoint trước chỉ có renderer, target crop và một LEAP rollout. Candidate mới
đã bổ sung explicit approach/acquisition cho ba hand, generator có checkpoint/
resume, exact-stage rendered evidence, immutable scene dataset và checker tự tái
tính identity từ asset. Vì artifact và kết luận gate thay đổi, roadmap cũ không
còn mô tả đúng trạng thái bàn giao.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: dataset release, runtime acquisition và completion matrix thay đổi.
- Thời hạn/độ khẩn: trước khi giao candidate cho independent reviewer.
- Biện pháp nếu trì hoãn: không được gọi checkpoint cũ là release candidate.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | một LEAP scene fixture | genuine acquisition cho ba hand | cao | independent review |
| API/schema/config | scene contracts chưa có release CLI | bounded generator/checker/resume | vừa | giữ compatibility |
| Dữ liệu/checkpoint | chưa có release artifact | 12-scene immutable tiny dataset | cao | pin exact commit |
| Tương thích CPU/GPU/export | CPU scene fixtures | bounded CPU release gate | thấp | P3.3 không claim GPU |
| Số liệu/kết luận đã công bố | P3.3-14/15 pending | 1–7 đạt, review còn mở | cao | không overclaim complete |

Tóm tắt tác động: bộ triển khai và artifact P3.3 được đóng băng để bàn giao;
điều kiện hoàn tất 1–7 đạt, còn điều kiện 8 chờ reviewer độc lập theo
GOV-REVIEW-001 nên chưa nâng trạng thái plan thành complete.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Thêm explicit pregrasp/approach và measured scene acquisition ba hand | `b6143a5` | 38 focused tests |
| `CH-002` | Thêm bounded deterministic release generator và exact-stage QA render | `ec71cfc`, `476dbe2` | generator/resume tests và visual QA |
| `CH-003` | Thêm micro/release acceptance checker, parity, resource và license policy | `f18b4a1`, `a28f210` | 63-test micro gate + release audit |
| `CH-004` | Phát hành `QDGrasp-Scene-Tiny` | `e9c87b8` | 12 scenes, 108 artifact hashes |
| `CH-005` | Cập nhật trạng thái roadmap không overclaim independent verdict | P3.3 plan v1.6.0 | docs checker |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `scripts/check_phase3_3.py --profile micro` | tests, two-run byte parity và coverage pass | 63 tests; parity pass; đủ positive/negative | pass | checker JSON output |
| `V-002` | `scripts/check_phase3_3.py --profile release` | release integrity và coverage pass | 108 artifacts; 12/24/33/17 records; 3 positive | pass | checker JSON output |
| `V-003` | checker recompute identities | manifest không tự chứng thực | object/camera/environment/robot/split hashes khớp | pass | `a28f210` |
| `V-004` | kiểm trực quan exact-stage images | target/contact/path/scene đọc được | ba hand, bin và shelf đạt | pass | `datasets/qdgrasp-scene-tiny/qa/` |
| `V-005` | external real-source smoke | chỉ chạy khi root được cấu hình | ba root `not_configured`; không claim | pass | opt-in boundary |

- Regression đã chạy lại: acquisition, generation/resume, micro checker, full
  release audit và repository documentation hooks, tuần tự một thread.
- Kiểm tra chưa chạy: independent review; real full-dataset smoke không thuộc CI
  và chưa có user-provided roots.
- Khả năng rollback: revert từng commit runtime/checker; dataset artifact được pin
  riêng tại `e9c87b8`.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có report P3.3 trước đó.
- Verdict cũ còn hiệu lực cho revision mới: không áp dụng; chưa có verdict.
- Loại kiểm tra lại cần thiết: `full_review` trên exact candidate.
- Đính chính số liệu/tài liệu cần phát hành: roadmap được sửa trong revision này.
- Người chấp nhận rủi ro/ngoại lệ: không có; independent gate không được waive.

## 8. Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-25 Asia/Bangkok.
- Người kiểm tra: self-verification chỉ cho implementation/gates; không phải
  `internal_independent` hoặc `external` review.
- Kết luận: release candidate và bộ bàn giao đã hoàn tất; phase acceptance chưa
  hoàn tất cho tới khi có independent verdict.
- Liên kết bản ghi hoàn tất phiên hiện tại: `REV-20260825-006`.
