---
document_id: REV-20260822-006
document_type: revision_record
revision_schema: 2
title: Sửa evidence sai của SESSION-006 — .gitignore âm thầm loại 51 file vendor
status: in_review
date: 2026-08-22
record_id: REV-20260822-006
session_id: SESSION-20260822-007
created_at: 2026-08-22T05:04:51+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-006
    artifact: docs/sessions/SESSION-20260822-006-vendor-ultralytics-import.md
    revision: 2ff13f5eb267d8f04882619acfd44927356708bd7f14d9d2b096b660307d3c93
reason: SESSION-20260822-006 (commit 2bc248c) tuyên bố "365 file, byte-for-byte identical, nothing stripped" nhưng thực tế chỉ 314 file được git track — .gitignore có pattern không anchor "datasets/" khớp nhầm dexgrasp/cfg/datasets/ (51 file YOLO dataset config), bị `git add vendor/` âm thầm bỏ qua mà không có cảnh báo. Phát hiện khi chạy `git rm -r ... dexgrasp/cfg/datasets` cho bước strip kế tiếp và gặp lỗi "pathspec did not match any files".
necessity: N2
impact: Sửa số liệu/evidence sai trong SESSION-006 (E-03 manifest 365 file không khớp nội dung thật đã commit); KHÔNG đổi kết luận "vendor import hợp lệ" vì phần bị thiếu (YOLO dataset config) đã nằm trong danh sách dự kiến strip ngay sau đó. Sửa .gitignore để ngăn tái diễn cho các vendor import sau (GraspGenX, rectified-flow).
supersedes: none
---

# REV-20260822-006 — Sửa evidence vendor manifest, sửa .gitignore

## Liên kết truy vết

- Phiên sửa: `SESSION-20260822-007`.
- Bản ghi hoàn tất của phiên hiện tại: `SESSION-20260822-007`.
- Phiên bị sửa: `SESSION-20260822-006`, commit `2bc248c` (vendor import).
- Artifact bị sửa: `docs/sessions/SESSION-20260822-006-vendor-ultralytics-import.md`
  (nội dung file KHÔNG bị sửa/xóa — giữ nguyên làm hồ sơ lịch sử; revision này
  chỉ ghi đè phần evidence sai bằng bản ghi mới, đúng quy tắc "không sửa bản ghi
  phiên đã đóng").
- Revision/checksum trước sửa (SESSION-006 hiện tại):
  `2ff13f5eb267d8f04882619acfd44927356708bd7f14d9d2b096b660307d3c93`.
- Revision/checksum sau sửa: không áp dụng — SESSION-006 giữ nguyên, sự thật
  đúng nằm ở revision record này.
- Issue/finding/yêu cầu nguồn: tự phát hiện trong lúc thực hiện bước strip kế
  tiếp (`git rm -r dexgrasp/cfg/datasets` báo lỗi pathspec).

## Lý do chỉnh sửa

`SESSION-20260822-006` (E-03) tuyên bố vendor manifest có 365 file và commit
`2bc248c` "byte-for-byte identical... nothing stripped". Khi chạy lệnh strip
tiếp theo, `git rm -r dexgrasp/cfg/datasets` báo `pathspec did not match any
files` — kiểm tra `git ls-files dexgrasp | wc -l` cho kết quả **314**, không
phải 365.

Nguyên nhân: `.gitignore` (dòng gốc) có pattern `datasets/` không anchor (không
có `/` ở đầu), nên khớp với BẤT KỲ thư mục nào tên `datasets` ở BẤT KỲ độ sâu
nào trong repo — bao gồm `dexgrasp/cfg/datasets/` (51 file YAML cấu hình
dataset của YOLO, vốn không liên quan gì tới ý định gốc của rule này, vốn chỉ
nhằm loại thư mục dataset runtime ở gốc dự án). `git add vendor/` ở phiên
trước đã âm thầm bỏ qua 51 file này mà không có output cảnh báo, và không ai
đối chiếu `git ls-files` với manifest disk-based trước khi ghi "365 file,
byte-for-byte" vào SESSION-006.

Đối chiếu cụ thể (`comm` giữa danh sách file trên đĩa và `git ls-files`) xác
nhận: đúng 51 file thiếu, TẤT CẢ đều nằm trong `cfg/datasets/`, không có file
nào khác bị ảnh hưởng ở bất kỳ nơi nào trong repo (đã quét toàn bộ working
tree bằng `git status --ignored`).

## Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: sai artifact/evidence cụ thể (số file, tuyên bố "unmodified") —
  đúng định nghĩa N2. Không phải N3 vì không ảnh hưởng pháp lý/an toàn/phát
  hành trực tiếp (phần thiếu vốn đã nằm trong danh sách dự kiến xóa).
- Thời hạn/độ khẩn: sửa trước khi tiếp tục bất kỳ vendor import nào khác
  (GraspGenX, rectified-flow) để tránh lặp lại lỗi tương tự.
- Biện pháp giảm thiểu nếu trì hoãn: không vendor thêm nguồn nào cho tới khi
  `.gitignore` được anchor lại.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Evidence SESSION-006 | "365 file, byte-for-byte, nothing stripped" (sai) | Số liệu đúng ghi trong REV-006: 314 file thật sự committed; 51 file `cfg/datasets/` bị `.gitignore` loại âm thầm | Vừa | Không sửa SESSION-006, chỉ supersede evidence qua revision này |
| `.gitignore` | `runs/`, `weights/`, `checkpoints/`, `datasets/` không anchor — khớp mọi độ sâu | Anchor về root: `/runs/`, `/weights/`, `/checkpoints/`, `/datasets/` | Cao | Ngăn tái diễn cho vendor import sau |
| Nội dung `dexgrasp/` | Có `cfg/datasets/` (do lỗi, không chủ đích) | Không có `cfg/datasets/` (giờ là quyết định chủ đích, gộp vào commit strip cùng phiên) | Thấp | Không cần khôi phục — phần này vốn nằm trong danh sách strip dự kiến (YOLO dataset config) |
| Vendor manifest evidence file | `docs/reports/evidence/VENDOR-ULTRALYTICS-20260822-329682a-manifest.sha256.txt` (365 dòng, đúng cho ĐĨA, không đúng cho COMMIT) | Không sửa file — vẫn là bằng chứng đúng cho trạng thái đĩa tại thời điểm đó; REV-006 ghi rõ chênh lệch | Thấp | Giữ nguyên làm bằng chứng lịch sử, không sửa |
| Runtime/model | Chưa có | Chưa có | Không | Không claim |

Tóm tắt tác động: sửa số liệu/evidence sai trong SESSION-006, không đổi kết
luận "vendor import hợp lệ"; sửa `.gitignore` để vendor import sau không lặp
lỗi tương tự.

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| CH-001 | Phát hiện và định lượng chênh lệch (314 tracked vs 365 disk, đúng 51 file `cfg/datasets/`) | — | `comm`/`git ls-files`/`git status --ignored` |
| CH-002 | Anchor lại 4 pattern trong `.gitignore` (`runs/`, `weights/`, `checkpoints/`, `datasets/` → có `/` đầu) | `.gitignore` | `git check-ignore` xác nhận hết khớp nhầm |
| CH-003 | Ghi revision record này, không sửa/xóa `SESSION-20260822-006` | `docs/revisions/REV-20260822-006-vendor-manifest-gitignore-gap.md` | mục này |

## Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| V-001 | `git ls-files dexgrasp \| wc -l` trước sửa | 314 | 314 | pass | console |
| V-002 | `comm` giữa disk và tracked, lọc khác `cfg/datasets/` | Rỗng (không có file nào khác bị ảnh hưởng) | Rỗng | pass | console |
| V-003 | `git status --ignored --short .` toàn repo | Chỉ `dexgrasp/cfg/datasets/` và `__pycache__/` bị ignore | Đúng | pass | console |
| V-004 | `git check-ignore -v dexgrasp/cfg/datasets/coco.yaml` sau khi anchor `.gitignore` | Không còn bị ignore | "not ignored now" | pass | console |
| V-005 | Full project gate sau khi sửa `.gitignore` + strip | Zero failure | refs/registry/docs/38 test/shell/diff pass | pass | console; xem `SESSION-20260822-007` |

- Regression đã chạy lại: full gate (xem T-table của `SESSION-20260822-007`).
- Kiểm tra chưa chạy: không có.
- Khả năng rollback: revert `.gitignore` về 4 dòng không anchor nếu cần (không
  khuyến nghị — đây là bug fix thật); `SESSION-006` không bị động tới nên
  không cần rollback riêng.

## Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có.
- Verdict cũ còn hiệu lực cho revision mới: không áp dụng — đây là sửa evidence
  không phải sửa verdict review.
- Loại kiểm tra lại cần thiết: `none`.
- Đính chính số liệu/tài liệu cần phát hành: `SESSION-20260822-006` E-03 (số
  365 file) nay được biết là số file TRÊN ĐĨA tại thời điểm đó, không phải số
  file COMMIT; số đúng đã commit là 314. Không claim nào khác trong SESSION-006
  bị ảnh hưởng (LICENSE hash, `.references` cleanliness vẫn đúng).
- Người chấp nhận rủi ro/ngoại lệ: không cần — đây là bug fix kỹ thuật thuần
  túy, không phải risk acceptance.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22 Asia/Bangkok.
- Người kiểm tra: chưa có.
- Kết luận: `in_review`; bug đã sửa, evidence đã đính chính, gate đã pass;
  chưa có independent review.
- Liên kết bản ghi hoàn tất phiên hiện tại: `SESSION-20260822-007`.
