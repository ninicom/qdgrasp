---
document_id: REV-20260822-004
document_type: revision_record
revision_schema: 2
title: Bỏ điều kiện văn bản cấp phép thương mại vì phạm vi chỉ nội bộ
status: in_review
date: 2026-08-22
record_id: REV-20260822-004
session_id: SESSION-20260822-004
created_at: 2026-08-22T04:36:41+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260821-007
    artifact: PLAN.md
    revision: 3bb353e6db014bc8742f912c5e33de8ea4167762d917f43338a2b1e4df24b084
reason: Maintainer quyết định 2026-08-22 rằng dự án chỉ dùng nội bộ, không phát hành/phân phối, nên điều kiện release gate "văn bản cấp phép thương mại trước khi sao chép/sửa/phân phối code Ultralytics và DGN2" không còn áp dụng cho phạm vi hiện tại.
necessity: N3
impact: Mở khóa việc bắt đầu fork/sửa code Ultralytics vào dexgrasp/ trong nội bộ mà không cần văn bản cấp phép thương mại; KHÔNG xóa nghĩa vụ giữ attribution/license file gốc, KHÔNG tự động cho phép phát hành/phân phối ra ngoài, và KHÔNG giải quyết điều khoản non-commercial CC-BY-NC-4.0 của DexGraspNet2.
supersedes: none
---

# REV-20260822-004 — Bỏ gate license cho phạm vi nội bộ

## Liên kết truy vết

- Phiên sửa: `SESSION-20260822-004`.
- Bản ghi hoàn tất của phiên hiện tại: `SESSION-20260822-004`.
- Phiên bị sửa: `SESSION-20260821-007` (đã bump PLAN.md lên `2.1.0` tại commit
  `5d86a426ff7f250bc7a77e80e241610d10a9958a`).
- Artifact bị sửa: `PLAN.md`.
- Revision/checksum trước sửa:
  `3bb353e6db014bc8742f912c5e33de8ea4167762d917f43338a2b1e4df24b084`.
- Revision/checksum sau sửa:
  `f3a172a7963f0c53204322e6357ae4ae984f6ec291c25497ff2aae69933de7cd`.
- Issue/finding/yêu cầu nguồn: yêu cầu trực tiếp từ maintainer trong phiên chat
  2026-08-22 ("đã cập nhật plan, không cần license"); không có finding TPR.

## Lý do chỉnh sửa

`PLAN.md` §1 (bản `2.1.0`) có câu: "Việc sao chép, sửa hoặc phân phối code
Ultralytics và DGN2 chỉ bắt đầu sau khi có văn bản cấp phép thương mại bao phủ
đúng hành vi đó. Đây là release gate." Mục M0 (`## 4. Lộ trình`) cũng có gạch
đầu dòng "Nhận license thương mại, clone/pin bốn upstream, lập provenance và
baseline manifest." Maintainer xác nhận trực tiếp (không có văn bản, chỉ lời
nói/chat) rằng dự án hiện tại chỉ dùng nội bộ, không phát hành hay phân phối ra
ngoài tổ chức, và đã tự xóa CẢ HAI câu trên khỏi `PLAN.md` trước khi revision
record này được tạo. Vì AGPL-3.0 (Ultralytics) và CC-BY-NC-4.0 (DGN2) đặt nghĩa
vụ chủ yếu lên hành vi phân phối/thương mại hóa, maintainer quyết định hai câu
điều kiện trên không còn cần thiết cho phạm vi đã thu hẹp. Đây không phải sửa
sai sót kỹ thuật của phiên trước, mà là thay đổi quyết định phạm vi/chính sách
của maintainer — được ghi lại đây, không phải do agent chủ động đề xuất.

## Mức độ cần thiết

- Mức đã chọn: `N3`.
- Lập luận: thay đổi ảnh hưởng trực tiếp tới điều kiện phát hành và nghĩa vụ
  pháp lý (bản quyền/giấy phép mã nguồn bên thứ ba) — đúng định nghĩa N3 trong
  `THIRD_PARTY_REVIEW.md` §8 ("ảnh hưởng an toàn, phát hành, pháp lý hoặc quyết
  định lớn").
- Thời hạn/độ khẩn: có hiệu lực ngay khi revision này ở trạng thái hợp lệ; bắt
  buộc trước khi bất kỳ commit nào sao chép/sửa code Ultralytics vào `dexgrasp/`.
- Biện pháp giảm thiểu nếu trì hoãn: không sao chép/sửa code Ultralytics/DGN2
  cho tới khi revision này tồn tại và được maintainer xác nhận.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Release gate M1 fork | Chặn tới khi có văn bản cấp phép thương mại | Không chặn cho phạm vi nội bộ, không phân phối | Cao | Bắt đầu fork chỉ khi maintainer xác nhận rõ ràng bằng lệnh riêng |
| Nghĩa vụ AGPL-3.0 (Ultralytics) | Chưa đánh giá | Vẫn áp dụng nếu có network interaction (§13 Affero) dù không "phân phối" truyền thống | Cao | Không triển khai dexgrasp như dịch vụ mạng cho người ngoài mà không đánh giá lại |
| Nghĩa vụ CC-BY-NC-4.0 (DGN2) | Chưa đánh giá | KHÔNG được giải quyết bởi quyết định "nội bộ, không phân phối" — trục non-commercial độc lập với trục distribution | Cao | Cần đánh giá riêng nếu tổ chức dùng DGN2 cho mục đích thương mại nội bộ |
| `references.lock.yaml` | `distribution_status: blocked_pending_commercial_license` / `blocked_noncommercial_terms_and_pending_legal_review` | Không đổi trong revision này | Vừa | Cần maintainer quyết định có sửa field này hay không; chưa sửa để tránh tự ý đổi ý nghĩa evidence pháp lý |
| Train-argument registry/checker | Không đổi | Không đổi | Không | Không cần chạy lại |
| Runtime/model | Chưa có | Chưa có | Không | Không claim |

Tóm tắt tác động: mở khóa bắt đầu fork/sửa code Ultralytics vào `dexgrasp/`
trong nội bộ mà không cần văn bản cấp phép thương mại; không xóa nghĩa vụ giữ
attribution/license gốc; không tự động cho phép phát hành/phân phối; không
giải quyết điều khoản non-commercial của DGN2.

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| CH-001 | Ghi nhận quyết định maintainer và lý do vào revision record | `docs/revisions/REV-20260822-004-plan-internal-only-scope.md` | mục này |
| CH-002 | Xác nhận `PLAN.md` §1 và mục M0 đã bỏ hai câu license-gate | `PLAN.md` (working tree, hash `f3a172a7…`) | `git diff PLAN.md` |
| CH-003 | Bump front matter `PLAN.md`: `version → 2.2.0`, `latest_revision_record → REV-20260822-004`; giữ nguyên `revises`/`revision_record`/`revision_reason`/`necessity`/`impact` gốc theo tiền lệ REV-006 | `PLAN.md` | `git diff PLAN.md` |

## Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| V-001 | `git show HEAD:PLAN.md \| sha256sum` so với `sha256sum PLAN.md` | Hash trước/sau khác nhau, khớp với `revises[0].revision` và mô tả trên | Khớp | pass | console |
| V-002 | `python3 scripts/check_docs.py --root .` | Vẫn pass (PLAN.md chỉ kiểm COMMON_FIELDS) | `44 file đã kiểm tra`, exit 0 | pass | console |
| V-003 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | Không hồi quy do thay đổi tài liệu | `38/38 pass`, exit 0 | pass | console |
| V-004 | Xác nhận rõ ràng của maintainer trước khi bắt đầu fork thật | Lệnh riêng, không suy đoán từ revision này | chưa có | not_run | — |
| V-005 | Đánh giá pháp lý bằng văn bản (không bắt buộc theo quyết định này, nhưng khuyến nghị) | Văn bản cấp phép hoặc đánh giá rủi ro chính thức | chưa có | not_run | — |

- Regression đã chạy lại: `check_docs.py` (44 file) và unit suite (38/38); cả
  hai đều không liên quan trực tiếp tới nội dung PLAN.md nên chỉ xác nhận
  không có hồi quy phụ.
- Kiểm tra chưa chạy: đánh giá pháp lý chính thức bằng văn bản (rủi ro: quyết
  định hiện dựa trên evidence E4 — lời nói/chat, không phải văn bản).
- Khả năng rollback: khôi phục câu license-gate vào `PLAN.md` từ commit
  `5d86a426ff7f250bc7a77e80e241610d10a9958a` (hash trước sửa ở trên); revision
  này không xóa hay sửa các revision record cũ.

## Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có (chưa từng có TPR nào đánh giá
  câu license-gate này).
- Verdict cũ còn hiệu lực cho revision mới: không áp dụng — đây là thay đổi
  chính sách/phạm vi, không phải sửa sai của review trước.
- Loại kiểm tra lại cần thiết: `none` cho revision record này; khuyến nghị
  `delta_review` trước khi merge bất kỳ commit nào thật sự sao chép code
  Ultralytics/DGN2 vào `dexgrasp/`.
- Đính chính số liệu/tài liệu cần phát hành: không có.
- Người chấp nhận rủi ro/ngoại lệ: maintainer (qua chat, 2026-08-22, Asia/Bangkok);
  chấp nhận rủi ro dựa trên evidence E4 (lời nói), chưa có văn bản chính thức.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22 Asia/Bangkok.
- Người kiểm tra: chưa có; cần review trước khi dùng revision này để bắt đầu
  fork code Ultralytics thật sự.
- Kết luận: `in_review`; quyết định đã ghi nhận, front matter đã bump và gate
  đã replay sạch; còn thiếu xác nhận rõ ràng riêng của maintainer trước khi
  chạm vào code Ultralytics, và đánh giá pháp lý bằng văn bản.
- Liên kết bản ghi hoàn tất phiên hiện tại: `SESSION-20260822-004`.
