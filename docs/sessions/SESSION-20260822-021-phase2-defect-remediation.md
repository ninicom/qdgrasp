---
document_id: SESSION-20260822-021
document_type: session_report
title: Sửa defect Phase 2 sau review logic và cấu trúc
status: complete
date: 2026-08-22
session_id: SESSION-20260822-021
author: claude-primary-agent
revises: SESSION-20260822-020
related_plan: PLAN-V2
revision_reason: Gate FK của Phase 2 so forward_kinematics với chính nó nên không thể phát hiện FK sai; oracle mj_forward cho thấy sáu defect trong đó có link đặt sai tới 0.247 m.
necessity: N2
impact: Thay kết luận nghiệm thu của W-04, W-05, W-06, W-07, W-08 và W-12 trong SESSION-20260822-020; đổi hành vi FK, node feature HandGraph, dữ liệu inertial, tên manifest và artifact dẫn xuất đã commit.
---

# SESSION-20260822-021 — Sửa defect Phase 2

## Mục tiêu phiên

- Review logic và cấu trúc của Phase 2 theo yêu cầu maintainer.
- Sửa mọi defect tìm được, không mở rộng phạm vi sang Phase 3.
- Tiêu chí dừng: gate Phase 2 có oracle độc lập và pass; Phase 0/Phase 1 không
  regression.

## Việc đã hoàn tất

| ID | Kết quả | Output | Trạng thái |
| --- | --- | --- | --- |
| W-01 | Gate FK dùng oracle `mj_forward` thay vì tự so với chính nó; commit ở trạng thái fail để ghi nhận | `scripts/check_phase2.py`, `tests/test_robot_kinematics.py` | Hoàn tất |
| W-02 | FK dựng theo cây thật, rebase bằng nghịch đảo rigid, raise thay vì fallback im lặng | `qdgrasp/robot/spec.py`, `kinematics.py` | Hoàn tất |
| W-03 | Bỏ vòng quaternion→RPY→ma trận ở đường MJCF; node feature dùng rot6d | `qdgrasp/robot/spec.py` | Hoàn tất |
| W-04 | `body_mass`/`body_inertia` thật thay hằng số; từ chối body nhiều joint và joint anchor khác 0 | `qdgrasp/robot/mjcf.py`, `spec.py` | Hoàn tất |
| W-05 | `validate_semantic_bodies` phủ wrist và contact như đường URDF | `qdgrasp/robot/mjcf.py` | Hoàn tất |
| W-06 | Mesh path thật cho MJCF; bỏ `except Exception: pass`; gate đếm mesh | `qdgrasp/robot/mjcf.py`, `spec.py`, `scripts/check_phase2.py` | Hoàn tất |
| W-07 | Manifest đặt tên theo artifact; URDF dẫn xuất dùng đường dẫn tương đối; gate kiểm portability | `qdgrasp/robot/normalize.py`, `qdgrasp/assets/derived/` | Hoàn tất |
| W-08 | Registry schema document thay import ngược; kiểu trả về thật; preset trùng basename bị từ chối | `qdgrasp/config/` | Hoàn tất |
| W-09 | Gate Phase 2 in summary máy đọc được | `scripts/check_phase2.py` | Hoàn tất |
| W-10 | Scene fixture dựng bằng `MjSpec` nên thật sự chứa vật thể; contact chỉ đếm cặp tay–vật | `qdgrasp/sim/fixtures.py` | Hoàn tất |
| W-11 | Mimic multiplier được đối chiếu với fixed tendon của MJCF | `qdgrasp/robot/spec.py`, `mjcf.py` | Hoàn tất |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- | --- |
| E-01 | Commit gate có hiệu lực, cố ý fail | `git` | `bebb653` |
| E-02 | Commit sửa FK | `git` | `d2bc6d3` |
| E-03 | Commit dữ liệu inertial và từ chối ca chưa hỗ trợ | `git` | `93c80e9` |
| E-04 | Commit mesh | `git` | `c70ce25` |
| E-05 | Commit manifest và portability | `git` | `ab84edb` |
| E-06 | Commit cấu trúc config | `git` | `ffd5dec` |
| E-07 | Revision record | `docs/revisions/REV-20260822-011-phase2-fk-and-mesh-defects.md` | `N2`, `in_review` |
| E-08 | Commit sửa fixture và nối tendon vào validation | `git` | `366c76e` |
| E-09 | PLAN không đổi | `PLAN.md` | `f1d4b9eb1692f229704593502afe088b73ae7f769367f7d9e6a515cc0cfe245c` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| T-01 | `scripts/check_phase2.py` tại `bebb653` | 1 | FAIL có chủ ý: `leap 0.100000 m`, `shadow 0.246727 m` |
| T-02 | `scripts/check_phase2.py` tại `ffd5dec` | 0 | PASS; FK worst `3.75e-08` m trên mọi body của ba hand |
| T-03 | `python3 -m pytest tests/ -q` | 0 | 150 passed, 1 skipped |
| T-04 | `python3 -m unittest discover -s scripts/tests` | 0 | 50 tests OK |
| T-05 | `scripts/check_phase0.py` | 0 | PASS, PLAN hash không đổi |
| T-06 | `scripts/check_phase1.py` | 0 | PASS |
| T-07 | `scripts/check_docs.py --root .` | 0 | 78 file |
| T-08 | `scripts/check_robot_assets.py` | 0 | 4 checkouts sạch |
| T-09 | `evaluate_grasp_fixture` trên ba hand sau khi sửa scene | 0 | scene compile được và chứa `target_object`; contact tay–vật `0/0/0`; lặp lại cùng seed cho metric giống hệt |
| T-10 | Sửa multiplier `rh_FFJ1` thành `0.5` rồi dựng `RobotSpec` | — | bị từ chối: mâu thuẫn tendon `rh_FFJ0` |

Số liệu gate sau sửa: LEAP 18 link / 16 joint / 21 mesh / 0.746 kg; Allegro 22 /
16 / 21 / 0.644 kg; Shadow 26 link / 20 joint / 4 mimic / 25 mesh / 3.794 kg.

## Việc chưa hoàn tất

- **Fixture grasp/squeeze/lift chưa nắm được vật ở cấu hình mặc định.** Trước phiên
  này `build_evaluation_scene_xml` ghép hand bằng `<include>` rồi compile từ
  chuỗi; model dựng từ chuỗi không có model directory nên `meshdir` của hand
  không được áp và mọi mesh reference fail. Một `except Exception` nuốt lỗi rồi
  load hand trơ **không có vật thể**, nên fixture báo cáo kết quả cho một scene
  không có gì để nắm. Scene nay dựng bằng `mujoco.MjSpec.from_file` và fail
  closed. Khi có scene thật, cách đếm contact cũng lộ ra là vô nghĩa: `data.ncon`
  tính cả vật nằm trên sàn nên `squeeze_contacts` luôn dương dù tay không chạm gì.
  Sau khi lọc chỉ đếm cặp tay–vật, kết quả trung thực là **0 contact tay–vật trên
  cả ba hand**, `success=False` và `stable_lift=False`.

  Đây là giới hạn cần khai báo, không phải regression: gate P2 theo
  `PROJECT_PHASES.md` chỉ đòi fixture repeatability, và điều đó vẫn pass — nay
  pass một cách có nghĩa thay vì pass rỗng. Muốn `stable_lift` đúng thật phải
  tune hình học vật thể và hướng đóng của từng ngón theo từng hand; đó là protocol
  grasp/squeeze/lift thuộc M5 trong `PLAN.md`, không phải P2. Không được nới tiêu
  chí `success` để làm số liệu đẹp lên.
- **Giới hạn đã khai báo, không phải defect.** Body MJCF mang nhiều joint và
  joint anchor offset khác 0 nay bị từ chối tường minh thay vì xử lý sai im lặng.
  Không hand nào trong corpus đã pin rơi vào hai ca này (đo được 0/24, 0/16, 0/16
  anchor khác 0 và không body nào nhiều joint), nên hỗ trợ thật để lại cho khi có
  hand cần.
- **Surface anchor để rỗng.** Anchor thật cần pose offset ở mức geom, và
  `trimesh.sample_surface` không tái lập được giữa các version — dev chạy 5.0.0
  còn `cu128.lock` pin 4.12.2. Nhét giá trị phụ thuộc môi trường vào node feature
  còn tệ hơn để rỗng. Thuộc phase sau.
- **`scripts/phase2_cuda_fk_parity.py` chưa chạy.** Cần GPU thật; theo
  `PHASE2_EXECUTION_PLAN.md` §10 đây là hạng mục ngoài gate chuyển phase.
- `scripts/check_qdgrasp_imports.py` vẫn báo 7 lỗi resolve trong cây legacy, có
  sẵn từ trước Phase 1, ngoài phạm vi.

## Sửa đổi phiên trước

`docs/revisions/REV-20260822-011-phase2-fk-and-mesh-defects.md` (`N2`). Claim cũ:
`SESSION-20260822-020` ghi W-04, W-05, W-06, W-07, W-08 và W-12 là `Hoàn tất`.
Claim mới: các hạng mục đó chỉ đạt sau sáu commit sửa lỗi trong phiên này; riêng
W-08 (FK) trước đó được nghiệm thu bằng một gate không thể fail. Bản ghi
`SESSION-20260822-020` giữ nguyên byte.

## Bàn giao

- `feature/phase2-robot-layer` chứa commit gốc của Phase 2 cộng sáu commit sửa.
  Chưa merge vào `develop`, chưa push.
- Trước khi mở Phase 3: hoàn tất independent review cho `REV-20260822-010` và
  `REV-20260822-011`, rồi chạy `scripts/git/finish_feature.sh`.
- Phiên sau phải kiểm trước: khi thêm hand mới, hai giới hạn đã khai báo ở trên
  sẽ raise chứ không im lặng; đó là hành vi đúng và cần xử lý bằng cách bổ sung
  hỗ trợ, không bằng cách nới điều kiện.
