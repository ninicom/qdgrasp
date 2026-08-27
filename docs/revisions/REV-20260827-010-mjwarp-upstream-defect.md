---
document_id: REV-20260827-010
document_type: revision_record
revision_schema: 2
title: Xác định defect uninitialized read của MuJoCo Warp 1.16.0 và hệ quả backend
status: complete
date: 2026-08-27
record_id: REV-20260827-010
session_id: SESSION-20260827-002
created_at: 2026-08-27T23:55:00+07:00
author: claude-primary-agent
revises:
  - session_id: ROADMAP-P3.4.1-001
    artifact: docs/roadmap/PHASE3_4_1_FIX_PLAN.md
    revision: 863809fc81e381283b043493db6ecbaf7cfbf0ad039c63f5af2e81230535f020
reason: "P3.4.1-01 kết thúc với nguyên nhân gốc nằm ngoài QDGrasp; cây quyết định §3.4 và bảng backend §3.7 cần được cập nhật bằng kết quả đo thay vì để hàng upstream ở trạng thái giả định."
necessity: N2
impact: "Chọn hàng `upstream MJWarp 1.16.0 bug` trong §3.4 bằng bằng chứng; P3.4.1-02 không còn là fix trong repo mà là compatibility spike theo §3.7. Không đổi code, threshold hay trạng thái phase."
---

# REV-20260827-010 — Defect upstream của MuJoCo Warp 1.16.0

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260827-002`.
- Bản ghi hoàn tất: tài liệu này.
- Artifact bị sửa và checksum trước sửa: mục `revises` ở metadata.
- Revision code chạy evidence: `1f0dffb54f3d97ab584162457ae86b3eb6e9119f`.
- Evidence: `evidence/phase3_4_1/perf-v9-1024/sanitizer/`.

## 2. Lý do chỉnh sửa

`§3.4` liệt kê năm nguyên nhân khả dĩ cho world phân kỳ và `§3.7` liệt kê năm
phương án backend. Cả hai được viết khi chưa có số đo. Nay có, và chúng chọn ra
đúng một hàng ở mỗi bảng.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: chuyển một hàng giả định thành kết luận đã đo, và đổi bản chất của
  P3.4.1-02 từ sửa code trong repo sang compatibility spike.
- Thời hạn/độ khẩn: trước khi ai đó bắt đầu audit `put_data`/`reset` của QDGrasp.
- Biện pháp nếu trì hoãn: công sức đổ vào wrapper vốn đã được loại trừ.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | không đổi | không đổi | không | không |
| API/schema/config | không đổi | không đổi | không | không |
| Dữ liệu/checkpoint | không đổi | không đổi | không | không |
| Tương thích CPU/GPU | GPU search chưa tái lập được | nguyên nhân xác định, vẫn chưa tái lập | trung bình | compatibility spike |
| Số liệu/kết luận đã công bố | nguyên nhân là giả thuyết | đã đo và loại trừ hết các giả thuyết khác | cao | cập nhật §3.4/§3.7 |

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

- `CH-001` Chạy `compute-sanitizer` `initcheck` và `racecheck` trên reproducer
  thu nhỏ; in nguyên văn bản ghi thay vì đếm.
- `CH-002` Chạy `§3.3` bước 6: `put_model`/`put_data`/`step` trực tiếp, không có
  backend QDGrasp trong call path.
- `CH-003` Lưu ba log kernel và README ghi cả kết luận lẫn giới hạn.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `racecheck` trên reproducer | phát hiện race nếu có | `0 hazards displayed (0 errors, 0 warnings)`, hai lần | pass | `kernel-v11`, `kernel-v14` |
| `V-002` | `initcheck` in nguyên văn | chỉ đích danh kernel | `_linesearch_iterative_kernel`, `solver.py:1359/1562/3533/3726` | pass | `kernel-v14-verbatim.log` |
| `V-003` | reproduction không có wrapper | loại trừ hoặc buộc tội QDGrasp | tái hiện y hệt: `67531 errors`, cùng offset `+0x7be0`, cùng thread `(25,0,0)` | pass | `kernel-v16-upstream-only.log` |

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- **`§3.4` chốt hàng `upstream MJWarp 1.16.0 bug`.** Bốn hàng còn lại đã bị loại
  trừ bằng số đo: không phải overflow đọc sai, không phải index cố định, không
  phải race, không phải solver instability thông thường.
- **`§3.7` loại hàng `Native MJWarp 1.16 fix`** khỏi vị trí mặc định: defect nằm
  trong chính 1.16.0. Hàng `MJWarp newer pinned version` có điều kiện kích hoạt
  đã thoả — *"upstream 1.16 bug đã cô lập"* — nên compatibility spike là bước
  tiếp theo đúng.
- **Đính chính nội bộ:** tác giả từng viết rằng bước kế tiếp là audit
  `mujoco_warp.put_data(..., nworld=N)` và `MjWarpCudaBackend.reset`. Sai; `V-003`
  loại trừ wrapper. Bản ghi này thay thế phát biểu đó.
- Gate `check_phase3_4` giữ nguyên hành vi thoát nonzero khi có NaN. Speed đạt
  `4.444x` không làm kết quả GPU search tái lập được, nên tiêu chí không đổi.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-primary-agent, 2026-08-27 Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer.
- Kết luận: nguyên nhân gốc nằm ngoài QDGrasp. P3.4.1-02 chuyển từ sửa code
  trong repo sang compatibility spike theo `§3.7`. Không đổi trạng thái phase;
  P3.4 vẫn paused theo `ADR-0008`.
- Giới hạn còn lại: `67531` lỗi, in `6`, các bản ghi bắt được đều cùng kernel —
  phù hợp với một site duy nhất nhưng chưa chứng minh. Bisect version hoặc báo
  cáo upstream sẽ chốt.
- Liên kết bản ghi hoàn tất: `REV-20260827-010`.
