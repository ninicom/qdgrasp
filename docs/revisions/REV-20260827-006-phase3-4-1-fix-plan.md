---
document_id: REV-20260827-006
document_type: revision_record
revision_schema: 2
title: Lập corrective plan P3.4.1 cho ba measured/structural blocker
status: complete
date: 2026-08-27
record_id: REV-20260827-006
session_id: SESSION-20260827-P3.4.1-PLAN
created_at: 2026-08-27T23:30:00+07:00
author: codex-primary-agent
revises:
  - session_id: ROADMAP-P3.4-001
    artifact: docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md
    revision: 4a6132a5be065cdbe25b9ea7c86896558068ec80fb7f80f1e5297a415e7152b7
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: 7b67f6c4485ea5f1ae38b8d8ff9fba9df30ba15e1923fda10df4eab4a0b24494
reason: "P3.4 v9 đạt speed 4.444x nhưng full GPU gate fail vì 29/1024 world non-finite và VRAM đo sai allocator; Shadow failure nằm trước closure và chưa có independent reviewer."
necessity: N2
impact: "Thêm ROADMAP-P3.4.1-001, giữ P3.4 pending, ưu tiên GPU stability/VRAM telemetry, Shadow corrective delta và immutable review packet; không sửa code/runtime/evidence trong revision này."
---

# REV-20260827-006 — Corrective plan Phase 3.4.1

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260827-P3.4.1-PLAN`.
- Bản ghi hoàn tất: `REV-20260827-006` (tài liệu này).
- Artifact active bị sửa: `ROADMAP-P3.4-001` và `ROADMAP-001`; checksum trước
  sửa nằm trong metadata `revises`.
- Artifact mới: `docs/roadmap/PHASE3_4_1_FIX_PLAN.md`, SHA-256
  `c03c11981fcfbc279fe5d8f5ab7d3b96aa5c6067e53cb34159e87f007c481c09`.
- Checksum sau sửa của parent plan:
  `d2c197b88ff06c9d633c061bd861a66ee5070b69e2d1add9620eb3aeb6ef0931`.
- Checksum sau sửa của project roadmap:
  `b2a9eeb0eb7a65e5e348ca7fa964e5566e2df41697b47ca4d0c54459dd17827e`.
- Findings nguồn: `evidence/phase3_4/p15-throughput/`,
  `evidence/phase3_4/p16-contactrich-tiny/` và điều kiện §10/§16 của parent plan.

## 2. Lý do chỉnh sửa

P3.4 ban đầu đo `0.764x` tại 64 worlds. v9 sau đó đạt `4.444x` tại operating
point 1024 worlds, nhưng 29/1024 identical worlds non-finite nên full CUDA gate
vẫn fail. Shadow cũng đã chứng minh failure tồn tại ở no-closure và không được
giải bằng 40-candidate closure search. Cuối cùng, governance không cho tác giả
tự phát hành independent verdict. Parent plan chưa có decision tree hoặc blast
radius để sửa các blocker này mà không đổi threshold hay rewrite evidence.

Research 2026 bổ sung ba phát hiện ảnh hưởng protocol: `Data.overflow` của MJWarp
phải được kiểm vì overflow tạo undefined behavior; Warp có debug/sanitizer path
cho non-finite; và bộ đếm VRAM PyTorch không bao phủ Warp allocator. CUDA Graph
là option có điều kiện nếu speed sau stability fix tụt dưới `2x`.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: revision sửa protocol stability/VRAM, đường kiến trúc performance có
  điều kiện và phạm vi correction của shared Shadow recipe; đây không phải biên tập.
- Thời hạn: trước lần tuyên bố P3.4 release candidate hoặc đưa ContactRich data
  vào P4.
- Nếu trì hoãn: giữ P3.4 `pending`, giữ dataset `release_blocked=true`, không
  phát hành verdict.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | v9 speed pass nhưng 29 world non-finite | overflow/first-bad-tick/sanitizer triage trước; CUDA Graph chỉ nếu speed tụt | cao, chưa triển khai | P3.4.1-01–05 |
| API/schema/config | không có corrective schema | versioned Shadow delta và evidence root mới được quy định | vừa, chưa triển khai | P3.4.1-06–08 |
| Dữ liệu/checkpoint | ContactRich hiện release-blocked | cấm dùng làm P4 release input cho tới corrective pass | cao | regenerate v2 sau fix |
| Tương thích CPU/GPU | `4.444x`, 29 NaN; VRAM đo sai allocator | zero invalid worlds, workload/output parity và device-level VRAM bắt buộc | cao | T4 evidence mới |
| Số liệu/kết luận | `0.764x` cũ; v9 `4.444x` + 29 NaN; Shadow 40/40; chưa reviewer | speed subcriterion pass, full phase vẫn pending | làm rõ verdict | reviewer khác ký RC sau cùng |

Tóm tắt tác động khớp metadata: revision thêm plan và quy tắc, không sửa
code/runtime/evidence trong phiên này.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Tạo research-backed corrective plan với ba nhánh blocker | `ROADMAP-P3.4.1-001@1.0.0` | source links + local evidence trong plan |
| `CH-002` | Liên kết P3.4 parent tới corrective phase, tăng version | `ROADMAP-P3.4-001@1.1.0` | SHA-256 mục 1 |
| `CH-003` | Cập nhật project roadmap, giữ P3.4 pending và khóa P4 contact-rich input | `ROADMAP-001@1.22.0` | SHA-256 mục 1 |
| `CH-004` | Ghi nhận VRAM allocator mismatch và cấm suy diễn GPU idle từ `~0 GiB` | `ROADMAP-P3.4.1-001` §2.3 | PyTorch/Warp/NVML primary docs |
| `CH-005` | Ingest v9 truthfully: speed pass, stability fail; thêm overflow/sanitizer decision tree | `ROADMAP-P3.4.1-001` §2.4/§3 | v9 P15 evidence + MJWarp/Warp/NVIDIA docs |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `git diff --check` | không whitespace error | không có output, exit 0 | pass | terminal phiên |
| `V-002` | manual logic review baseline → option → gate | không đường tự đổi fail thành pass | threshold/safety/evidence/reviewer đều giữ fail-closed | pass | plan §1, §3–§5, §8 |
| `V-003` | đối chiếu code backend và raw evidence | finding có căn cứ local | v9 `4.444x`, 29/1024 non-finite; backend chưa report `Data.overflow` | pass | `mjwarp_cuda.py`, evidence P15/P16 |
| `V-004` | research primary sources, cutoff 2026-08-27 | kiến trúc/source hiện hành | MJWarp, MuJoCo, NVIDIA, Warp, PyTorch, Menagerie, Isaac Lab | pass | plan §2 |
| `V-005` | workload/test mô phỏng | không thuộc revision plan | không chạy theo yêu cầu kiểm logic | not_run | không thay code/runtime |

- Regression đã chạy lại: không có; revision không thay runtime/data.
- Kiểm tra chưa chạy: simulation/test suite; rủi ro runtime được giữ nguyên là
  blocker, không bị ghi thành pass.
- Rollback: revert ba living-plan changes và xóa plan mới trong một revision
  successor; không sửa record/evidence cũ.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `REV-20260827-005` còn hiệu lực cho exact revision cũ. Evidence P15 hiện có
  successor v9: speed subcriterion pass, stability fail; baseline 64-world không
  bị xóa.
- `TPR-20260825-001` vẫn có hiệu lực cho exact P3.3 candidate của nó nhưng không
  review corrective plan hoặc Shadow recipe tương lai.
- Verdict cũ không tự mở rộng sang revision mới.
- Loại kiểm lại: delta review cho thay đổi living plans; full independent review
  cho P3.4 release candidate sau P3.4.1-05/08.
- Không sửa raw số liệu; roadmap được cập nhật từ historical `0.764x` sang trạng
  thái v9 đầy đủ `4.444x + 29 NaN`. Diễn giải peak PyTorch `~0 GiB` không phải
  peak Warp/device VRAM.
- Không có ngoại lệ/rủi ro nào được chấp nhận thay reviewer.

## 8. Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-27, Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer; tác giả chỉ hoàn tất hồ sơ plan.
- Kết luận: revision plan đã hoàn tất; P3.4.1 execution `active`, P3.4 vẫn
  `pending`, P4 contact-rich input vẫn blocked.
- Liên kết bản ghi hoàn tất: `REV-20260827-006`.
