---
document_id: REV-20260822-006
document_type: revision_record
revision_schema: 2
title: Chuyển kế hoạch từ fork AGPL sang Apache-2.0 clean-room
status: in_review
date: 2026-08-22
record_id: REV-20260822-006
session_id: SESSION-20260822-015
created_at: 2026-08-22T16:49:00+07:00
author: codex-primary-agent
revises:
  - session_id: SESSION-20260822-005
    artifact: PLAN.md
    revision: 7ef0353f2233f8ef407381b0260a5d65bc6264edf09c19703074b6b9f254b922
reason: Maintainer yêu cầu bỏ ràng buộc AGPL/non-commercial, nạp corpus robot public gồm Shadow Hand và dùng các paper mới để xây DexGrasp cross-embodiment; audit chứng minh cây hiện tại là dẫn xuất Ultralytics gần như toàn phần và DGN2 không thể là nguồn train permissive.
necessity: N3
impact: Supersede căn cứ AGPL của REV-005, dừng nhánh fork như product, yêu cầu repository/history clean-room mới, thay engine, kiến trúc model và data strategy; code AGPL hiện tại không bị rellicense hay xóa notice.
supersedes: REV-20260822-005
---

# REV-20260822-006 — Chuyển sang Apache-2.0 clean-room

## Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260822-015`.
- Bản ghi hoàn tất của phiên hiện tại:
  `docs/sessions/SESSION-20260822-015-apache-clean-room-plan.md`.
- Phiên bị sửa: `SESSION-20260822-005`.
- Artifact bị sửa: `PLAN.md`.
- Revision/checksum trước sửa:
  `7ef0353f2233f8ef407381b0260a5d65bc6264edf09c19703074b6b9f254b922`.
- Revision/checksum sau sửa:
  `8bf07953d390a47b5826cceab74724f54d50345804daca01f7f3ea8c76d15bd5`.
- Issue/finding/yêu cầu nguồn: yêu cầu maintainer trong chat ngày 2026-08-22;
  audit repository và `docs/decisions/0002-apache-clean-room.md`.

## Lý do chỉnh sửa

REV-005 chọn public toàn bộ DexGrasp dưới AGPL-3.0 vì sản phẩm được fork trực tiếp
từ Ultralytics. Maintainer nay yêu cầu một nền permissive để cộng đồng có thể dùng
cả cá nhân, public và thương mại mà không phải mở toàn bộ downstream source.

Audit xác nhận 93/93 file tracked trong `dexgrasp/` map tới upstream; 87 file
trùng byte, 44.475/44.500 dòng text khớp và 90/91 file text có header AGPL (file
còn lại là marker rỗng). Đây là bản dẫn xuất thực chất, nên chỉ đổi root license
là không hợp lệ. DGN2 chưa được port, nhưng tuyên bố cả work/dataset dưới
CC-BY-NC-4.0; nếu dùng làm training source thì official weights không đạt mục
tiêu permissive. Hai tiền đề đó buộc phải thay cả source strategy và data
strategy trước khi tiếp tục product implementation. Yêu cầu cross-hand/Shadow
mới đồng thời buộc robot assets và research architecture phải có provenance độc
lập ngay từ đầu.

## Mức độ cần thiết

- Mức đã chọn: `N3`.
- Lập luận: thay license đích, quyền phát hành, source provenance, Git history,
  framework base, dataset và official weights; đây là thay đổi pháp lý/phát hành
  và quyết định kiến trúc lớn.
- Thời hạn/độ khẩn: có hiệu lực ngay trước mọi commit product tiếp theo hoặc public
  release.
- Biện pháp giảm thiểu nếu trì hoãn: freeze nhánh dẫn xuất; không đổi `LICENSE`,
  không xóa AGPL notice và không quảng bá cây hiện tại là Apache-2.0.

## Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | Trim engine Ultralytics | Façade/engine viết mới trên PyTorch + Lightning Fabric | Cao | Behavior spec và contract tests |
| API/schema/config | Tương thích lifecycle, 115 keys và YAML grammar Ultralytics | UX một lệnh nhưng schema/CLI độc lập | Cao | Version schema `dexgrasp/*/v1` |
| Dữ liệu/checkpoint | DGN2 converter/corpus dự kiến | DGN2 paper/reference-only; `DGN-Open` tự sinh | Cao | Xây generator và data/weight manifests |
| Model/robot | LEAP-specific wrist+joint pipeline dự kiến | Variable HandGraph, direct executable flow, FK-keypoint/contact/force phụ; LEAP/Allegro/Shadow | Cao | Implementation và ablation clean-room |
| Assets | Robot assets DGN2 chưa rõ quyền | Exact-pin local corpus; per-model license; Barrett blocked | Cao | Independent asset audit trước ship |
| Tương thích CPU/GPU/export | Dựa trên Ultralytics backends | PyTorch/Fabric + TorchScript/ONNX viết mới | Cao | Chạy lại toàn matrix |
| Số liệu/kết luận đã công bố | Baseline/gates chưa có model grasp | Mọi kết quả AGPL branch không là release evidence cho code Apache | Vừa | Full rerun trong repository mới |
| License/phát hành | AGPL-3.0 toàn sản phẩm | Apache-2.0 cho code dự án mới; per-domain manifests | Rất cao | Fresh repository/history + independent review |

Tóm tắt tác động: supersede căn cứ AGPL của REV-005, dừng nhánh fork như product,
yêu cầu repository/history clean-room mới, thay engine, kiến trúc model và data
strategy; code AGPL hiện tại không bị rellicense hay xóa notice.

## Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Audit provenance cây `dexgrasp/` và biên DGN2 | Working tree + Git history | `E-01` trong session report |
| `CH-002` | Ghi quyết định Apache-2.0/clean-room, nguồn thay thế và license domains | `docs/decisions/0002-apache-clean-room.md` / `7172e42f…` | `E-02` |
| `CH-003` | Viết lại PLAN 3.1 với stack, graph-flow architecture, data generator, roadmap và release gates mới | `PLAN.md` / `1678cd87…` | `E-03` |
| `CH-004` | Giữ nguyên root AGPL `LICENSE` và code đang sửa dở | `LICENSE` / `0d96a4ff…` | `E-04` |
| `CH-005` | Xác minh/correct 10 paper và khóa kiến trúc cross-embodiment clean-room | `docs/decisions/0003-cross-embodiment-flow-architecture.md` / `9c7f8ccf…` | `E-05` |
| `CH-006` | Sparse-clone exact-pin URDF/MJCF/mesh/meta fixtures, gồm Shadow; ghi license/tree/hash và test tương thích | `robot_assets.lock.yaml` / `8774c36d…`; evidence / `812e94a8…` | `E-06` |
| `CH-007` | Chuẩn hóa boundary: DGN2 chỉ là paper/reference-only; không phải oracle/runtime/data/checkpoint source | `docs/decisions/0004-dgn2-paper-reference-boundary.md` / `8fe124be…`; cập nhật `PLAN.md`, session report | `E-07` |

## Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `git diff --check` | Không whitespace error | Exit 0 | pass | console |
| `V-002` | `python3 scripts/check_docs.py --root .` | Mọi normative/record contract hợp lệ | 61 file, exit 0 | pass | console |
| `V-003` | `python3 scripts/check_references.py --lock-only` | Historical lock không bị sửa | Exit 0 | pass | console |
| `V-004` | Independent license/provenance review | Reviewer xác nhận N3 scope và stack pins | Chưa có reviewer độc lập | not_run | release gate |
| `V-005` | Parse URDF/MJCF, resolve mesh, MuJoCo 3.11 load/forward | Ba hand chính có đường parser + simulation hợp lệ | LEAP/Shadow pass raw; Allegro URDF cần normalization, paired MJCF pass | pass_with_limit | `ROBOT-ASSETS-20260822-compatibility.txt` |
| `V-006` | Đối chiếu arXiv/CVF/NeurIPS/project/repository primary sources | Title/date/venue/claim/license boundary rõ | 10/10 paper xác minh; hai correction ghi trong ADR-0003 | pass | `docs/decisions/0003-cross-embodiment-flow-architecture.md` |

- Regression đã chạy lại: documentation/reference/config gates, 47 unit tests,
  diff whitespace và exact-pin/clean-status cho bốn asset checkouts đều pass.
- Kiểm tra chưa chạy: functional model benchmark không thuộc phiên vì không sửa
  runtime/model. Independent review vẫn bắt buộc trước public beta.
- Khả năng rollback: Git giữ blob PLAN trước sửa và toàn bộ code/license hiện tại;
  rollback chỉ hợp lệ nếu maintainer chủ động quay lại AGPL bằng revision N3 mới.

## Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: các review hiện có không audit license,
  model, dataset hoặc weights nên không thể dùng làm approval cho hướng Apache.
- Verdict cũ còn hiệu lực cho revision mới: các kết quả kỹ thuật/lịch sử của
  sessions 006–014 vẫn đúng cho cây AGPL, nhưng không chứng minh source eligibility
  của sản phẩm Apache.
- Loại kiểm tra lại cần thiết: `full_review` cho repository mới.
- Đính chính số liệu/tài liệu cần phát hành: không có public release trước đó;
  PLAN 2.x và REV-005 được supersede, không bị xóa.
- Người chấp nhận rủi ro/ngoại lệ: không có; license allowlist không có ngoại lệ ở
  quyết định này.

## Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-22, Asia/Bangkok.
- Người kiểm tra: chưa có; cần reviewer độc lập trước public beta.
- Kết luận: nội dung pivot, research architecture, asset corpus và documentation
  gates đã hoàn tất; record giữ `in_review` cho tới khi independent review được lập.
- Liên kết bản ghi hoàn tất phiên hiện tại:
  `docs/sessions/SESSION-20260822-015-apache-clean-room-plan.md`.
