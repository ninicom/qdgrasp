---
document_id: SESSION-20260822-015
document_type: session_report
title: Pivot Apache clean-room, graph-flow architecture và robot compatibility corpus
status: in_review
date: 2026-08-22
session_id: SESSION-20260822-015
author: codex-primary-agent
revises: SESSION-20260822-005
related_plan: PLAN-V2
revision_reason: Maintainer thay mục tiêu từ public AGPL fork sang permissive clean-room và bổ sung yêu cầu cross-embodiment, Shadow Hand cùng research-paper basis.
necessity: N3
impact: Supersede product-license basis cũ; thay source history, engine, model, data, assets và release gates, nhưng giữ nguyên code/license AGPL hiện tại.
---

# SESSION-20260822-015 — Apache clean-room, paper architecture và robot assets

## Mục tiêu phiên

Đánh giá yêu cầu maintainer về việc bỏ ràng buộc bản quyền mạnh, đổi hướng sang
một dự án permissive có trải nghiệm cộng đồng như YOLO, tìm stack thay thế hợp
lệ, tải một corpus URDF/mesh/meta public gồm Shadow Hand để thử compatibility,
và kiểm chứng mười paper mới trước khi đưa ý tưởng vào kiến trúc/paper plan.

Phiên này là planning/provenance/compatibility work. Nó không cấp phép lại source
hiện tại, không tạo repository Apache mới và không port model code hoặc data từ
DGN2/paper repositories. DGN2 chỉ được dùng như paper/reference-only để hiểu
bài toán và nguyên lý grasping, không phải oracle hay runtime dependency.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
| --- | --- | --- | --- |
| W-01 | Audit cây hiện tại với exact upstream/Git history: 93/93 tracked file `dexgrasp/` map Ultralytics, 87 byte-identical, 94,01% bytes và 44.475/44.500 text lines khớp; DGN2 chưa được port và chỉ là paper/reference, còn work/data upstream là CC-BY-NC-4.0 | ADR-0002 + REV-006 | Git/source audit |
| W-02 | Chọn Apache-2.0 cho code sở hữu mới, freeze AGPL branch và bắt buộc repository/history mới; không đổi root `LICENSE` | `PLAN.md`, ADR-0002 | hash + review boundary |
| W-03 | Thay stack kế hoạch bằng PyTorch/Lightning Fabric, GraspGenX/Pointcept/rectified-flow candidates và MuJoCo/MJX; thêm `DGN-Open` procedural/CC0 | `PLAN.md` 3.1.0 | documentation gate |
| W-04 | Xác minh 10/10 paper bằng nguồn primary, sửa title MachaGrasp và phân loại DexGraspNet 3.0 là dataset của DexVLG | ADR-0003 | primary-source review |
| W-05 | Khóa kiến trúc: URDF/MJCF HandGraph biến độ dài, direct executable palm+joint flow, differentiable FK, topology-keypoint/contact/force phụ; paper-only boundary | ADR-0003 | decision record |
| W-06 | Sparse-clone/detach exact pins cho Menagerie Shadow/Allegro/LEAP, LEAP official, Wonik Allegro official và DexSuite parser fixtures | `.references/robot-assets/` | clean detached HEAD |
| W-07 | Ghi machine-readable source/commit/tree/license/hash/meta/status cho robot corpus; chặn Barrett khỏi redistribution | `robot_assets.lock.yaml` | YAML parse + source audit |
| W-08 | Parse XML, resolve mesh, đếm links/joints/limits và chạy MuJoCo load/forward | compatibility evidence | pass với giới hạn Allegro raw inertia |
| W-09 | Bảo toàn ba file exporter user đang sửa dở và session 014 chưa tracked; không chạm runtime source | Git worktree | status review |

## Bằng chứng

| Evidence ID | Loại | Artifact/URI | SHA-256 hoặc kết luận |
| --- | --- | --- | --- |
| E-01 | Plan mới | `PLAN.md` | `8bf07953d390a47b5826cceab74724f54d50345804daca01f7f3ea8c76d15bd5` |
| E-02 | License/clean-room decision | `docs/decisions/0002-apache-clean-room.md` | `7172e42f5ae8f2e0322994722d1b90081eb43516318103edb933fca801dc8fe2` |
| E-03 | Research architecture decision | `docs/decisions/0003-cross-embodiment-flow-architecture.md` | `9c7f8ccfb2bc5ccebdff5c1f674c5a0af78bef4dbee4cc7350fe4cf362c562ef` |
| E-04 | Robot source/license lock | `robot_assets.lock.yaml` | `8774c36dab2cda1e51a1ff30f774dbe15a5c5e3854f562acc22f9444dfb9a2be` |
| E-05 | Raw compatibility result | `docs/reports/evidence/ROBOT-ASSETS-20260822-compatibility.txt` | `812e94a85fca68aa713a00a0e849a0c4fef0542a5d359526369b76a7c4910d4f` |
| E-06 | Root license không đổi | `LICENSE` | `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` (AGPL-3.0) |
| E-07 | Asset pins | four detached local checkouts | Menagerie `da76818…`; LEAP `150bc3d…`; Wonik `80bd4a8…`; DexSuite `f5e7132…` |
| E-08 | Paper primary sources | arXiv/CVF/NeurIPS links in ADR-0003 | 10/10 title/date/venue/technical claims checked |
| E-09 | DGN2 boundary clarification | `docs/decisions/0004-dgn2-paper-reference-boundary.md` | `8fe124bef0b8c5fe6583860c6bac2846c1d36a3310cface8a2163f5f79aee125` |

Compatibility result chính:

- Official LEAP URDF: 17 links, 16 movable joints, zero missing mesh, MuJoCo
  forward pass.
- Menagerie Shadow: `nq=24`, `nu=20`; Allegro/LEAP: `nq=nu=16`; cả ba MJCF
  forward pass.
- Wonik official Allegro URDF parse/mesh resolution pass nhưng raw MuJoCo load
  fail vì `fullinertia` không positive-definite. Asset gốc không bị sửa;
  simulator dùng paired Menagerie MJCF hoặc transform normalization có manifest.
- DexSuite Barrett parse/sim pass cục bộ nhưng license text thiếu disclaimer và
  upstream không có full LICENSE; kết quả kỹ thuật không cấp quyền redistribution.

## Kiểm tra đã chạy

| Test ID | Lệnh/phương pháp | Exit code | Kết quả |
| --- | --- | ---: | --- |
| T-01 | `git diff --check -- PLAN.md docs/decisions/0002-apache-clean-room.md docs/decisions/0003-cross-embodiment-flow-architecture.md robot_assets.lock.yaml docs/reports/evidence/ROBOT-ASSETS-20260822-compatibility.txt` | 0 | pass |
| T-02 | `yaml.safe_load(robot_assets.lock.yaml)` + schema/source assertions | 0 | pass; schema 1, 4 sources |
| T-03 | stdlib XML parse + relative/ROS-package mesh resolution | 0 | pass; zero missing refs ở fixtures được báo cáo |
| T-04 | MuJoCo 3.11 `MjModel` load + `mj_forward` + finite qpos | 0 | pass cho LEAP/Shadow và paired Allegro MJCF; raw Allegro limitation được ghi, không bị che |
| T-05 | `python3 scripts/check_references.py --lock-only` | 0 | pass; historical immutable lock không đổi |
| T-06 | `python3 scripts/check_docs.py --root .` | 0 | pass; 60 file |
| T-07 | `python3 scripts/check_train_args.py --registry-only` | 0 | pass; 127 names, zero mismatch |
| T-08 | `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | 0 | pass; 47/47 |
| T-09 | Kiểm exact HEAD/origin/clean status của bốn asset checkout + `git check-ignore` | 0 | pass; detached exact pins, clean, ignored |
| T-10 | `git diff --check` | 0 | pass |

## Việc chưa hoàn tất

- Chưa tạo repository/Git history Apache-2.0 mới vì maintainer chưa cung cấp
  destination remote/name; tuyệt đối không đổi root `LICENSE` của repo hiện tại.
- Chưa có independent reviewer cho N3 license pivot, paper/source pins và robot
  asset redistribution. `eligible_pending_independent_audit` không đồng nghĩa
  “đã được ship”.
- Chưa implement framework, HandGraph, flow, FK, data generator hoặc model; các
  kết quả phiên này là plan/spec/fixture validation, không phải model benchmark.
- Chưa tạo DGN-Open và chưa train checkpoint. Không có claim paper về accuracy,
  latency, generalization hoặc novelty.
- Barrett bị chặn. Shadow URDF DexSuite chỉ là local parser fixture; public
  baseline an toàn hiện tại là Menagerie Shadow MJCF Apache-2.0.
- Raw official Allegro URDF cần normalization có provenance hoặc paired MJCF;
  không chỉnh source asset tại chỗ.

## Sửa đổi phiên trước

`REV-20260822-006` supersede quyết định AGPL product basis của
`SESSION-20260822-005` ở mức N3. Kết quả sessions 006–014 vẫn là bằng chứng đúng
cho nhánh dẫn xuất AGPL, nhưng không được dùng làm release evidence cho source
Apache mới. PLAN trước/sau được khóa bằng SHA-256; code/license cũ không bị xóa
hoặc relabel.

## Bàn giao

1. Freeze nhánh hiện tại sau khi hoàn tất/commit riêng công việc exporter của
   maintainer; không tiếp tục dùng nó làm product base.
2. Tạo repository mới, history mới, Apache-2.0 `LICENSE`/`NOTICE`, DCO,
   `THIRD_PARTY.yml`, SBOM và license CI.
3. Re-fetch các asset `eligible_pending_independent_audit` từ exact pins trong
   `robot_assets.lock.yaml`; chỉ copy release selection sau independent review.
4. Viết behavior/schema spec rồi implement P0: `RobotSpec`/HandGraph, FK,
   object encoder và direct executable flow. Không cho implementer đọc/copy cây
   Ultralytics/DGN2.
5. Tạo `DGN-Open-Tiny`, chạy ba baseline direct/keypoint+IK/dual-consistency;
   khóa paper protocol trước full-scale training.
