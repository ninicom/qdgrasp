---
document_id: REV-20260822-011
document_type: revision_record
revision_schema: 2
title: Sửa FK, dữ liệu inertial, mesh gate và portability của Phase 2
status: in_review
date: 2026-08-22
record_id: "REV-20260822-011"
session_id: "SESSION-20260822-021"
created_at: "2026-08-22T21:54:25+07:00"
author: "claude-primary-agent"
revises:
  - session_id: "SESSION-20260822-020"
    artifact: "docs/sessions/SESSION-20260822-020-phase2-robot-layer.md"
    revision: "3e7d34d4e8f0f2b2ed0742ea5807cd0e314e1c3da8fd584618e018de55ab1047"
reason: "Gate FK của Phase 2 chỉ so forward_kinematics với chính nó nên không thể phát hiện FK sai; đối chiếu với mj_forward cho thấy Shadow đặt sai rh_wrist 0.216 m và rh_forearm 0.247 m, cùng bốn defect khác mà các gate liên quan không bắt được."
necessity: N2
impact: "Sửa kết luận kỹ thuật của W-04, W-05, W-06, W-07, W-08 và W-12 trong SESSION-20260822-020; thay đổi hành vi FK, node feature của HandGraph, dữ liệu inertial, tên manifest và artifact dẫn xuất đã commit. Không ảnh hưởng Phase 0, Phase 1 hay license/provenance."
---

# Hồ sơ sửa kết quả của phiên trước

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260822-021`
- Bản ghi hoàn tất của phiên hiện tại:
  `docs/sessions/SESSION-20260822-021-phase2-defect-remediation.md`
- Phiên bị sửa: `SESSION-20260822-020`
- Artifact bị sửa: `docs/sessions/SESSION-20260822-020-phase2-robot-layer.md`
  (giữ nguyên byte; claim được sửa bằng bản ghi kế tiếp)
- Revision/checksum trước sửa:
  `3e7d34d4e8f0f2b2ed0742ea5807cd0e314e1c3da8fd584618e018de55ab1047`
  tại commit `7bd4a96`
- Revision/checksum sau sửa: commit `ffd5dec` (code), bản ghi kế tiếp
  `SESSION-20260822-021`
- Issue/finding/yêu cầu nguồn: review cấu trúc và logic do maintainer yêu cầu

## 2. Lý do chỉnh sửa

`SESSION-20260822-020` ghi W-08 (FK) là `Hoàn tất`, dựa trên
`check_forward_kinematics_and_batch_parity` trong `scripts/check_phase2.py`. Kiểm
tra đó so `forward_kinematics(batch)` với `forward_kinematics(single)`; hai vế
chạy cùng một hàm nên một FK sai hoàn toàn vẫn pass. Bằng chứng không đủ để đỡ
claim.

Khi đối chiếu với `mj_forward` làm oracle độc lập, sáu defect lộ ra:

1. `forward_kinematics` fallback `T_parent = T_palm` khi chưa có transform của
   parent, nên mọi link nằm trên palm trong cây bị gắn nhầm vào palm. Shadow sai
   `rh_wrist` 0.216 m và `rh_forearm` 0.247 m; `rh_wrist` chính là link mà profile
   khai báo ở `wrist_link`.
2. Đường MJCF chuyển quaternion sang RPY rồi dựng lại ma trận. Phép này suy biến
   ở `|pitch| = 90 deg`, đúng vị trí của Allegro `palm` (quat `0, .707, 0, .707`)
   và Shadow `rh_forearm`; ma trận thu lại sai tới 2.0.
3. Đường MJCF gán cứng `mass=0.05` và `inertia=(1e-4, ...)` cho mọi link, và tạo
   `surface_anchors` toàn số 0 nhưng khai báo có 16 anchor mỗi link.
4. `check_mesh_resolution` lặp trên `link.mesh_paths` mà đường MJCF luôn để rỗng;
   cả ba profile phát hành đều là MJCF nên tiêu chí zero-missing-mesh được thoả
   mãn bằng cách không có mesh nào để kiểm. Đường URDF thì nuốt lỗi mesh bằng
   `except Exception: pass`.
5. `normalize_urdf` ghi manifest vào tên cố định `normalization_manifest.json`
   theo thư mục, nên asset thứ hai sẽ xoá provenance của asset thứ nhất. URDF dẫn
   xuất đã commit còn nhúng 42 đường dẫn tuyệt đối của máy phát triển, trái với
   quy tắc relative path của `PLAN.md`.
6. `validate_semantic_bodies` của đường MJCF không kiểm `wrist_link` và
   `contact_links`, tức hai vai trò này chỉ được kiểm trên đường URDF mà không
   profile phát hành nào dùng.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`
- Lập luận: sai hành vi kỹ thuật và sai kết luận nghiệm thu, đúng định nghĩa `N2`
  trong `DOCUMENTATION_POLICY.md` §3. Nếu không sửa, mọi consumer của frame wrist
  và mọi node feature dựa trên mass/inertia/anchor đều nhận giá trị sai, và
  Phase 4 sẽ condition trên hằng số.
- Thời hạn/độ khẩn: trước khi bất kỳ công việc Phase 3 hoặc Phase 4 nào dùng
  `RobotSpec`.
- Biện pháp giảm thiểu nếu trì hoãn: không được phép trì hoãn; gate cũ không phát
  hiện được lớp lỗi này.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | FK sai với link trên palm; ma trận sai ở điểm suy biến Euler | Khớp `mj_forward` `3.75e-08` trên mọi body | cao | gate ground-truth mới |
| API/schema/config | `LinkSpec.origin_rpy`; node feature 14 chiều | `LinkSpec.origin_rotation`; node feature 17 chiều với rot6d | vừa | consumer nội bộ, chưa có API công khai |
| Dữ liệu/checkpoint | mass/inertia hằng số, anchor giả | giá trị thật từ MuJoCo, anchor rỗng tường minh | vừa | anchor thật để lại phase sau |
| Tương thích CPU/GPU/export | không đổi | không đổi | không | `scripts/phase2_cuda_fk_parity.py` chưa chạy |
| Số liệu/kết luận đã công bố | W-04/05/06/07/08/12 ghi `Hoàn tất` | các claim đó được thay bằng `SESSION-20260822-021` | cao | independent review record này |

Tóm tắt tác động: chỉ Phase 2 bị ảnh hưởng. `check_phase0.py` vẫn PASS với PLAN
hash `f1d4b9eb…f245c`, `check_phase1.py` vẫn PASS, `dummy-hand.yaml` vẫn ở
`qdgrasp/robot/v1`, không thêm dependency nào.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Thêm oracle `mj_forward` vào gate và test, commit ở trạng thái fail để ghi nhận gate có hiệu lực | commit `bebb653` | `V-001` |
| `CH-002` | Dựng FK theo cây thật, rebase bằng nghịch đảo rigid, raise thay vì fallback; bỏ Euler khỏi đường MJCF; node feature dùng rot6d | commit `d2bc6d3` | `V-002` |
| `CH-003` | Đọc `body_mass`/`body_inertia` thật; từ chối body nhiều joint và joint anchor khác 0; validate wrist/contact trên đường MJCF | commit `93c80e9` | `V-003` |
| `CH-004` | Lấy mesh path thật cho MJCF; bỏ `except Exception: pass`; gate đếm mesh và fail khi bằng 0; anchor rỗng thay vì zeros giả | commit `c70ce25` | `V-004` |
| `CH-005` | Manifest đặt tên theo artifact; regenerate URDF dẫn xuất bằng đường dẫn tương đối; kiểm bất đẳng thức tam giác; gate kiểm portability | commit `ab84edb` | `V-005` |
| `CH-006` | Registry schema document thay cho import ngược; sửa kiểu trả về; từ chối preset trùng basename; gate in summary | commit `ffd5dec` | `V-006` |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `scripts/check_phase2.py` tại commit `bebb653` | fail, chứng minh gate có hiệu lực | exit 1; `leap 0.100000 m`, `shadow 0.246727 m` | `pass` | commit `bebb653` |
| `V-002` | So FK với `mj_forward` cho mọi body của ba hand | sai số `<= 1e-4` | worst `3.75e-08` m, gồm cả ancestor của palm | `pass` | `scripts/check_phase2.py`, `tests/test_robot_kinematics.py` |
| `V-003` | Đếm mass phân biệt trên ba profile | không còn hằng số | 12/12/9 giá trị phân biệt; tổng 3.794/0.644/0.746 kg | `pass` | summary của gate |
| `V-004` | Đếm mesh đã kiểm trên ba profile | lớn hơn 0, không thiếu file | 25/21/21 mesh, missing 0 | `pass` | summary của gate |
| `V-005` | `grep` đường dẫn tuyệt đối trong `qdgrasp/assets/derived/` | không còn | 0; MuJoCo vẫn load `nq=16` | `pass` | `check_derived_assets_are_portable` |
| `V-006` | `pytest tests/ -q` | pass | 148 passed, 1 skipped | `pass` | log phiên |
| `V-007` | `check_phase0.py`, `check_phase1.py`, `check_docs.py`, `unittest scripts/tests` | pass, PLAN hash không đổi | PASS; PLAN `f1d4b9eb…f245c`; 76 file; 50 tests | `pass` | log phiên |

- Regression đã chạy lại: toàn bộ gate Phase 0, Phase 1, Phase 2 và hai bộ test.
- Kiểm tra chưa chạy: `scripts/phase2_cuda_fk_parity.py` (cần GPU thật, ngoài gate
  chuyển phase theo `PHASE2_EXECUTION_PLAN.md` §10).
- Khả năng rollback: `git revert` sáu commit `bebb653..ffd5dec`; trạng thái trước
  sửa nằm ở commit `7bd4a96`.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có
- Verdict cũ còn hiệu lực cho revision mới: không; claim W-08 của
  `SESSION-20260822-020` dựa trên một gate không thể fail
- Loại kiểm tra lại cần thiết: `delta_review` trên Phase 2
- Đính chính số liệu/tài liệu cần phát hành: `SESSION-20260822-021` thay các claim
  W-04, W-05, W-06, W-07, W-08 và W-12
- Người chấp nhận rủi ro/ngoại lệ: maintainer yêu cầu review và sửa, 2026-08-22

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-primary-agent, 2026-08-22 `Asia/Bangkok`
- Người kiểm tra: chưa chỉ định; cần reviewer không phải tác giả
- Kết luận: chưa hoàn tất — thay đổi đã áp dụng và mọi kiểm tra CPU pass, nhưng
  `N2` cần independent review trước khi chuyển sang `accepted`
- Liên kết bản ghi hoàn tất phiên hiện tại:
  `docs/sessions/SESSION-20260822-021-phase2-defect-remediation.md`
