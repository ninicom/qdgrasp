---
document_id: REV-20260822-005
document_type: revision_record
revision_schema: 2
title: Sửa lại lý do bỏ license-gate — public AGPL-3.0, không phải nội bộ
status: in_review
date: 2026-08-22
record_id: REV-20260822-005
session_id: SESSION-20260822-005
created_at: 2026-08-22T04:50:06+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-004
    artifact: PLAN.md
    revision: edc98c5eba522eda5d67e9b957ebf54676213a14538b99c78601c3a15efe1ca2
reason: Maintainer đổi ý ngay sau REV-20260822-004 — "internal-only, không phát hành" không còn đúng. Maintainer xác nhận dexgrasp sẽ được public toàn bộ source cho cộng đồng (không dùng nội bộ trực tiếp), và sẽ public dưới AGPL-3.0. Lý do "internal-only" trong REV-004 sai với ý định thật, phải sửa trước khi dùng để mở khóa fork.
necessity: N3
impact: Thay lý do pháp lý cho phép fork/sửa code Ultralytics từ "nội bộ, không phân phối" sang "public dưới AGPL-3.0 — copyleft compliance hợp pháp, không cần văn bản cấp phép thương mại RIÊNG miễn toàn bộ dexgrasp giữ AGPL-3.0". Không đổi kết luận "được phép bắt đầu fork" nhưng đổi CĂN CỨ pháp lý hoàn toàn — căn cứ cũ (REV-004) không còn áp dụng cho hành vi thật sắp diễn ra (public release).
supersedes: REV-20260822-004
---

# REV-20260822-005 — Căn cứ pháp lý đúng: public AGPL-3.0

## Liên kết truy vết

- Phiên sửa: `SESSION-20260822-005`.
- Bản ghi hoàn tất của phiên hiện tại: `SESSION-20260822-005`.
- Phiên bị sửa: `SESSION-20260822-004` (đã bump `PLAN.md` lên `2.2.0` tại
  commit `3bd51503...` trên `feature/train-argument-registry`, sau đó merge
  vào `develop`).
- Artifact bị sửa: `PLAN.md`.
- Revision/checksum trước sửa:
  `edc98c5eba522eda5d67e9b957ebf54676213a14538b99c78601c3a15efe1ca2`.
- Revision/checksum sau sửa:
  `7ef0353f2233f8ef407381b0260a5d65bc6264edf09c19703074b6b9f254b922`.
- Issue/finding/yêu cầu nguồn: maintainer tự sửa ý định trong cùng phiên chat
  2026-08-22, ngay sau khi `REV-20260822-004` được ghi và bắt đầu fork.

## Lý do chỉnh sửa

`REV-20260822-004` ghi nhận lý do bỏ license-gate là "dự án chỉ dùng nội bộ,
không phát hành/phân phối". Ngay sau đó, maintainer nói ngược lại: "dự án sau
này sẽ được public full source" rồi "tôi muốn làm để public cho cộng đồng chứ
không trực tiếp sử dụng". Đây không phải làm rõ (N1) mà là **đảo ngược tiền đề
pháp lý** — "internal-only" và "public" dẫn tới nghĩa vụ AGPL-3.0 khác nhau
hoàn toàn (internal use không kích hoạt nghĩa vụ phân phối; public release
kích hoạt trực tiếp). Việc dùng REV-004 làm căn cứ cho một hành vi thực tế
khác hẳn (public, không phải internal) là sai và phải sửa trước khi tiếp tục
vendor code Ultralytics.

Sau khi hỏi lại, maintainer xác nhận: `dexgrasp` sẽ public dưới `AGPL-3.0`
(copyleft, không đóng gói lại permissive/độc quyền), mục đích cộng đồng, không
phải sản phẩm nội bộ. AGPL-3.0 của Ultralytics tự cho phép phân phối lại bản
dẫn xuất dưới CÙNG license mà không cần văn bản cấp phép thương mại riêng —
đây là con đường tuân thủ chuẩn của mọi fork AGPL, không phải ngoại lệ hay lỗ
hổng. Kết luận "được phép fork" không đổi, nhưng căn cứ pháp lý đã đổi hoàn
toàn và phải được ghi đúng.

## Mức độ cần thiết

- Mức đã chọn: `N3`.
- Lập luận: thay đổi trực tiếp căn cứ pháp lý cho phép sao chép/sửa code bên
  thứ ba có license — đúng định nghĩa N3.
- Thời hạn/độ khẩn: có hiệu lực ngay; bắt buộc trước khi commit vendor import
  đã chuẩn bị sẵn (365 file từ `.references/ultralytics`, chưa commit).
- Biện pháp giảm thiểu nếu trì hoãn: không commit vendor import cho tới khi
  revision này tồn tại.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa (REV-004) | Sau sửa (REV-005) | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Căn cứ pháp lý cho fork Ultralytics | "Nội bộ, không phân phối" (evidence E4, lời nói) | "Public dưới AGPL-3.0 cho cộng đồng" (evidence E4, lời nói) — copyleft compliance chuẩn, không cần văn bản thương mại RIÊNG | Cao | Không cần văn bản NẾU dexgrasp thật sự giữ AGPL-3.0 khi public |
| `PLAN.md` §1 | Không có câu nào về license phát hành | Ghi rõ: public AGPL-3.0, cấm đóng gói lại permissive/độc quyền trừ khi có văn bản thương mại thật | Cao | Câu này chính là điều kiện ràng buộc thay license-gate cũ |
| `PLAN.md` M0 | Không có bullet license | Thêm bullet "chốt public AGPL-3.0" + bullet đánh giá pháp lý DGN2 riêng | Vừa | Theo dõi ở M0 |
| DexGraspNet2 CC-BY-NC-4.0 | Chưa giải quyết | Vẫn CHƯA giải quyết — ghi rõ là release gate riêng | Không đổi | Chưa port code/dữ liệu DGN2 |
| `references.lock.yaml` | `distribution_status` cũ, mâu thuẫn REV-004 | Vẫn mâu thuẫn, giờ mâu thuẫn theo hướng khác (public AGPL, không phải "blocked_pending_commercial_license") | Vừa | Cần maintainer quyết định giá trị mới, agent không tự sửa |
| Vendor import đã chuẩn bị | 365 file staged rồi unstage, chưa commit | Không đổi nội dung, chỉ đổi căn cứ pháp lý cho phép commit | Cao | Commit sau revision này, message trích rõ AGPL-3.0 |

Tóm tắt tác động: thay lý do pháp lý cho phép fork/sửa code Ultralytics từ
"nội bộ, không phân phối" sang "public dưới AGPL-3.0"; không đổi kết luận
được phép fork nhưng đổi hoàn toàn căn cứ; DGN2 CC-BY-NC-4.0 vẫn mở.

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| CH-001 | Ghi nhận đảo ngược tiền đề của maintainer, không tự suy đoán lý do | `docs/revisions/REV-20260822-005-agpl-public-release-basis.md` | mục này |
| CH-002 | Sửa `PLAN.md` §1: thêm câu AGPL-3.0 public + câu DGN2 CC-BY-NC-4.0 còn mở | `PLAN.md` | `git diff PLAN.md` |
| CH-003 | Sửa `PLAN.md` M0: thêm bullet chốt license public + đánh giá DGN2 riêng | `PLAN.md` | `git diff PLAN.md` |
| CH-004 | Bump front matter `PLAN.md`: `version → 2.3.0`, `latest_revision_record → REV-005` | `PLAN.md` | hash `7ef0353f…` |
| CH-005 | Unstage vendor import đã chuẩn bị trước đó (365 file), chờ revision này trước khi commit | `vendor/ultralytics-329682a/` (untracked, chưa commit) | `git status` |

## Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| V-001 | `sha256sum PLAN.md` trước/sau khớp `revises[0].revision` và hash sau sửa | Khớp | Khớp | pass | console |
| V-002 | `python3 scripts/check_docs.py --root .` | Pass (PLAN.md chỉ kiểm COMMON_FIELDS) | `44 file đã kiểm tra`, exit 0 | pass | console |
| V-003 | Vendor import (365 file) chưa commit cho tới khi revision này tồn tại | Đúng | `git status` cho thấy `vendor/` untracked, chưa add | pass | console |
| V-004 | Xác nhận maintainer về license public trước khi ghi vào PLAN.md | Có, qua `AskUserQuestion` | "AGPL-3.0 (copyleft, giống Ultralytics)" | pass | chat |
| V-005 | Đánh giá pháp lý bằng văn bản (khuyến nghị, không bắt buộc) | Văn bản luật sư hoặc tư vấn chính thức | Chưa có | not_run | — |
| V-006 | Đánh giá riêng DGN2 CC-BY-NC-4.0 trước khi port code/dữ liệu | Đánh giá cụ thể | Chưa làm | not_run | — |

- Regression đã chạy lại: `check_docs.py` (44 file); không hồi quy khác vì chỉ
  sửa tài liệu.
- Kiểm tra chưa chạy: đánh giá pháp lý chính thức bằng văn bản; đánh giá DGN2.
- Khả năng rollback: khôi phục `PLAN.md` về nội dung REV-004 (hash trước sửa ở
  trên); revision này không xóa REV-004, chỉ supersede.

## Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có.
- Verdict cũ còn hiệu lực cho revision mới: KHÔNG — `REV-20260822-004` bị
  supersede vì lý do gốc ("internal-only") sai với ý định thật của maintainer;
  `REV-004` được giữ nguyên làm hồ sơ lịch sử, không xóa.
- Loại kiểm tra lại cần thiết: `none` cho revision record; khuyến nghị
  `delta_review` trước khi merge bất kỳ commit nào thật sự public source.
- Đính chính số liệu/tài liệu cần phát hành: không có.
- Người chấp nhận rủi ro/ngoại lệ: maintainer (qua chat, 2026-08-22,
  Asia/Bangkok); chấp nhận dựa trên evidence E4 (lời nói), chưa có văn bản.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22 Asia/Bangkok.
- Người kiểm tra: chưa có; cần review trước khi coi căn cứ AGPL-3.0 là cuối
  cùng cho release thật.
- Kết luận: `in_review`; căn cứ pháp lý đã sửa đúng ý định maintainer, front
  matter đã bump, gate đã pass; còn thiếu đánh giá pháp lý bằng văn bản và
  đánh giá riêng cho DGN2.
- Liên kết bản ghi hoàn tất phiên hiện tại: `SESSION-20260822-005`.
