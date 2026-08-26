---
document_id: REV-20260827-001
document_type: revision_record
revision_schema: 2
title: Regenerate DGN-Open-Tiny bằng measured positive và giữ release blocked cho Allegro
status: complete
date: 2026-08-27
record_id: REV-20260827-001
session_id: SESSION-20260827-P3.1-14
created_at: 2026-08-27T02:10:00+07:00
author: claude-primary-agent
revises:
  - session_id: ROADMAP-P3.1-001
    artifact: docs/roadmap/PHASE3_1_DATA_CORRECTNESS_REMEDIATION_PLAN.md
    revision: 1508cae60d3d692aca23a246d7d81c965a83535154ad614d9bb6ed01bd55b6ee
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 8377a7e4635405b20fbd58add8c87af4c2409a92696beedda87a86641bd3ccfd
reason: "Release DGN-Open-Tiny cũ bị REV-20260823-009 vô hiệu vì positive là fixture bịa; regeneration đo thật cho thấy mười hai object procedural không sinh positive nào nên release phải đổi thành phần object thay vì đổi ngưỡng."
necessity: N2
impact: "P3.1-14 hoàn tất một phần: dataset mới có measured positive ở 5/6 shard và invalidated=false, nhưng val shard của wonik_allegro vẫn 0 positive nên release_blocked=true. P3.1-15 và P3 tiếp tục pending."
---

# REV-20260827-001 — P3.1-14 regeneration

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260827-P3.1-14`.
- Bản ghi hoàn tất: tài liệu này.
- Artifacts bị sửa và checksum trước sửa: hai mục `revises` ở metadata.
- Revision code chạy release evidence: `790f5c65b1ad1d54fc2b704f7a7ba879d8b69d5d`.
- Evidence: `evidence/phase3_1/p14-regeneration/`.
- Finding nguồn: P3.1-14 trong `ROADMAP-P3.1-001`; invalidation trong
  `REV-20260823-009`; recipe selection trong `REV-20260825-001`.

## 2. Lý do chỉnh sửa

`datasets/dgn-open-tiny` phát hành trước đó mang đúng một positive mỗi shard, và
`REV-20260823-009` đã xác định đó là fixture dựng tay bị thay vào `outcomes[0]`,
không có provenance liên tục. Manifest vì vậy mang `invalidated=true`.

Regeneration đầu tiên bằng recipe đã chốt `region_opposition_v1` trên đúng mười
hai object procedural cũ cho **0 positive ở cả sáu shard**. Kết quả này loại bỏ
giả thiết rằng chỉ cần chạy lại generator là đủ: thành phần object của release,
chứ không phải ngưỡng hay recipe, là thứ phải thay đổi.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: thay thành phần object của một release artifact và thay kết luận về
  khả năng đóng P3.1, nhưng không đổi schema, recipe, threshold hay gate rule.
- Thời hạn/độ khẩn: trước khi bất kỳ công việc P4 nào dùng tiny dataset.
- Biện pháp nếu trì hoãn: release tree tiếp tục chứa positive bịa đã bị vô hiệu.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | generator chỉ sinh object procedural | thêm positive-control object gắn theo hand | trung bình | giữ frozen stream và budget |
| API/schema/config | không đổi | `build_grasp_bar` mới; `build_generated_reachable_object` giữ nguyên geometry | thấp | không |
| Dữ liệu/checkpoint | 144 sample, 6 positive bịa, invalidated | 186 sample, 5 measured positive, invalidated=false | cao | pin exact revision |
| Tương thích CPU/GPU/export | CPU lock | không đổi | không | không |
| Số liệu/kết luận đã công bố | P3.1-14 `ready` | hoàn tất một phần, còn blocker Allegro | cao | không claim P3.1 đóng |

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

- `CH-001` Refactor `qdgrasp/dataset/pipeline/generated_reachable.py`: tách
  `build_grasp_bar(profile, upper_height, upper_center_z)` và giữ
  `build_generated_reachable_object` trả về geometry **byte-identical** (mesh
  sha256 ba hand không đổi), nên fixture pin của P3.2.1 không bị ảnh hưởng.
- `CH-002` `scripts/generate_dgn_open_tiny.py` nhận năm positive-control object.
  Mỗi object đi qua đúng `create_object_asset` như object procedural, chỉ ghép
  với hand đã calibrate, dùng frozen proposal stream `generated_reachable_rng`
  và candidate budget đã validated (LEAP 4, Allegro 14, Shadow 10), có guard
  chặn budget vượt `candidate_budget`.
- `CH-003` Thêm `generated_reachable.py` vào `generator_source_hashes`.
- `CH-004` Thay nội dung `datasets/dgn-open-tiny/`; xóa `dataset_stats.json`
  (artifact của generator cũ, không còn code hay tài liệu nào tham chiếu).
- `CH-005` Lưu evidence P3.1-14 và cập nhật roadmap P3.1 cùng `PROJECT_PHASES`.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | regeneration canonical, recipe đã chốt | đo yield thật | 0 positive ở 6/6 shard | pass | `canonical-baseline.log` |
| `V-002` | probe variant pre-registered, 1 tham số × 2 giá trị | tìm variant thứ hai mỗi hand | LEAP có, Shadow có, Allegro 0/4 | pass | `variant-probe.json` |
| `V-003` | probe Allegro ở ceiling budget 16 | trong envelope đã validated | vẫn 0/4 variant | pass | `variant-probe.json` |
| `V-004` | hai clean regeneration | byte-identical | 6/6 shard, 34/34 object, manifest trùng khớp | pass | `release-run1.log`, `release-run2.log` |
| `V-005` | mesh hash `build_generated_reachable_object` trước/sau refactor | không đổi | ba hand trùng khớp | pass | preflight commit `790f5c6` |
| `V-006` | `check_dataset_manifest.py --root datasets/dgn-open-tiny` | fail-closed đúng lý do | FAIL `release_blocked=True` | pass | terminal log |

- Regression đã chạy: `tests/test_generated_reachable.py`,
  `test_dataset_manifest_audit.py`, `test_objects_manifest.py`,
  `test_dataset_schema.py`, `test_no_positive_substitution.py` — 21 passed.
- Khả năng rollback: revert commit dataset và `790f5c6`; nhánh
  `feature/phase3-1-closure` tách riêng khỏi `feature/phase3-data-layer`.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo bị ảnh hưởng: `REV-20260823-009` giữ nguyên hiệu lực; nguyên nhân
  invalidation đã được gỡ bỏ khỏi release tree bằng dữ liệu đo thật.
- Verdict cũ còn hiệu lực: `REV-20260825-001` được củng cố — ba positive-control
  cell tái lập được **ngoài** harness ablation, 3/3 hand.
- Đính chính bắt buộc: canonical procedural yield là `0`, không phải "thấp".
  Positive của release đến từ positive-control object, và điều đó phải được ghi
  ở mọi nơi trích dẫn tiny dataset; đây không phải bằng chứng generalization.
- Người chấp nhận rủi ro/ngoại lệ: không có ngoại lệ. Không hạ threshold, không
  sửa quy tắc `release_blocked`, không nhét positive thủ công.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-primary-agent, 2026-08-27 Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer; hồ sơ này không tự phát hành
  verdict cho P3.1.
- Kết luận: P3.1-14 hoàn tất một phần. `wonik_allegro` không có positive ở split
  `val` nên `release_blocked` vẫn `true`; P3.1-15 và P3 tiếp tục pending.
- Liên kết bản ghi hoàn tất: `REV-20260827-001`.
