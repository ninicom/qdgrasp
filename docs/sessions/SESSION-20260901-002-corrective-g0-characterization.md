---
document_id: SESSION-20260901-002
document_type: session_report
title: R1/G0 — characterization suite và hard-stop cho public training
status: complete
date: 2026-09-01
session_id: SESSION-20260901-002
author: claude-agent
related_plan: PLAN-V2@4.6.0
revises: none
---

# SESSION-20260901-002 — R1/G0: characterization và hard-stop

## Mục tiêu phiên

- Thi công `R1` theo `PLAN.md` §9.11: characterization tests cho toàn bộ chuỗi
  lỗi trong §9.2, cộng hard-stop trên public training path.
- Giữ đúng ranh giới của `R1`: không sửa semantics, không regenerate dữ liệu,
  không auto-fix lint, không tạo release evidence mới.
- Làm cho trạng thái "finding còn mở" trở thành dữ liệu máy đọc được, để khi
  `G1`–`G7` đóng thì chính test đó chuyển thành regression test mà không phải
  viết lại.

## Việc đã hoàn tất

| ID | Kết quả | Output | Trạng thái |
|---|---|---|---|
| `R1-01` | Registry `COR-00`…`COR-12` dạng dữ liệu, kèm gate/PR đóng từng finding | `qdgrasp/corrective/registry.py` | complete |
| `R1-02` | Khai báo trước các schema sẽ bump và finding gây ra bump | `PLANNED_SCHEMA_BUMPS` trong cùng file | complete |
| `R1-03` | Hard-stop: canonical audit và positive gate chạy trước khi public train/val mở dataset | `qdgrasp/corrective/gate.py`, `qdgrasp/api/facade.py` | complete |
| `R1-04` | 39 characterization test phủ đủ 13 finding, mỗi test viết theo trạng thái đích | `tests/corrective/test_cor00_*.py` … `test_cor12_*.py` | complete |
| `R1-05` | Test đồng bộ registry ↔ `PLAN.md` §9.2 ↔ suite (id, severity, độ phủ) | `tests/corrective/test_registry.py` | complete |
| `R1-06` | Regression test cho chính hard-stop, gồm việc gate không có đường vòng | `tests/corrective/test_g0_hard_stop.py` | complete |
| `R1-07` | Thông báo schema mismatch nêu rõ schema mong đợi thay vì chỉ schema đọc được | `qdgrasp/engine/checkpoint.py`, `qdgrasp/mvp/policy.py` | complete |

Cơ chế: mỗi characterization test mang decorator `@characterization("COR-xx")`.
Khi finding còn `open`, test là `xfail(strict=True)`; khi gate tương ứng đóng và
registry chuyển sang `closed`, cùng test đó trở thành regression test. `strict`
là điều kiện bắt buộc: nếu ai đó sửa xong mà quên đóng registry, suite fail vì
`XPASS` chứ không im lặng.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/kết quả | Tham chiếu |
|---|---|---|---|
| `E-01` | hard-stop trên corpus hiện tại | `QDGrasp().train("configs/data/dgn_open_tiny.yaml")` raise `CorrectiveGateError` nêu cả hai check fail | `tests/corrective/test_g0_hard_stop.py` |
| `E-02` | tái hiện thực thi pickle | shard và MVP checkpoint chứa reducer tạo được marker khi load | `test_cor00_artifact_io.py` |
| `E-03` | tái hiện path escape | absolute path, `..` traversal và sibling-prefix đều được chấp nhận | `test_cor00_artifact_io.py` |
| `E-04` | tái hiện protocol leakage | `comp_dumbbell_01`, `comp_l_shape_01`, `comp_t_shape_02` — 12 sample mỗi active hand trong train view | `test_cor02_protocol_view.py` |
| `E-05` | tái hiện held-out embodiment | `wonik_allegro` có 54 sample trong split train | `test_cor02_protocol_view.py` |
| `E-06` | tái hiện joint parameterization | `decode(encode(q))` lệch `1.100919` rad (LEAP) và `0.786500` rad (Allegro) | `test_cor05_joint_parameterization.py` |
| `E-07` | tái hiện quality tie | hai candidate khác nhau trên cùng object cho cùng logit | `test_cor06_candidate_quality.py` |
| `E-08` | tái hiện metric phụ thuộc batch | cùng model, 5 sample: `0.9077` ở `batch_size=2`, `0.9297` ở `batch_size=5` | `test_cor07_validation_determinism.py` |
| `E-09` | tái hiện cross-robot resume | resume LEAP được nhận vào run Allegro | `test_cor08_resume_identity.py` |
| `E-10` | tái hiện fingerprint bypass | `_worker_init` dựng policy từ checkpoint mang fingerprint lạ | `test_cor11_mvp_fingerprint.py` |
| `E-11` | kiểm kê code-execution surface | 8 vị trí `exec`/`eval`/`weights_only=False`/`allow_pickle=True` trong wheel | `test_cor12_package_boundary.py` |

Payload dùng cho probe pickle chỉ tạo một thư mục trong `tmp_path` của test và
không ghi gì ngoài đó; test khẳng định thư mục đó không được tạo, nên nội dung
payload không phải thứ ai đó phải tin.

## Kiểm tra đã chạy

| Test ID | Lệnh/phương pháp | Kết quả | Exit/đánh giá |
|---|---|---|---|
| `T-01` | `.venv/bin/pytest -q tests/corrective` | 17 passed, 39 xfailed | pass |
| `T-02` | `.venv/bin/pytest -q tests/corrective --runxfail` | 39 fail, từng cái nêu đúng chuỗi lỗi đã đăng ký | expected |
| `T-03` | `.venv/bin/pytest -q` trên worktree sạch sau commit | 1323 passed, 1 skipped, 39 xfailed, 93 subtests | pass |
| `T-04` | `.venv/bin/ruff check qdgrasp/corrective tests/corrective` | All checks passed | pass |
| `T-05` | `.venv/bin/mypy qdgrasp/corrective --ignore-missing-imports` | 0 finding trong package mới | pass |
| `T-06` | `.venv/bin/python scripts/check_docs.py --root .` | hợp lệ | pass |

`T-02` là kiểm tra quan trọng nhất của phiên: nó xác nhận từng test fail vì đúng
lỗi mà §9.2 mô tả, chứ không fail vì import sai hay fixture hỏng. Một xfail
không đọc nguyên nhân là một xfail vô nghĩa.

`T-03` chỉ được tính sau khi commit. Trên worktree bẩn,
`tests/model_flow/test_phase4_gate.py::test_verify_accepts_the_committed_packet_on_a_clean_tree`
fail vì đúng lý do nó tồn tại — packet chỉ ký được trên cây sạch — nên một lần
chạy trên cây bẩn không nói được gì về suite.

## Việc chưa hoàn tất

- `COR-00`…`COR-12` chưa có finding nào được sửa. `R1` cố ý không chứa semantic
  fix; toàn bộ vẫn `open` trong registry.
- `R2`–`R9` chưa bắt đầu. Thứ tự và ràng buộc giữ nguyên theo `PLAN.md` §9.11.
- Chưa regenerate DGN-Open-Tiny, chưa retrain Flow/MVP, chưa tạo evidence thay
  thế, chưa chạy CUDA/AMP gate, chưa có independent review.
- Interaction test matrix ở §9.12 mới được phủ ở dạng characterization từng
  chuỗi; các chuỗi end-to-end đầy đủ thuộc về `R2`–`R8`.
- Hard-stop hiện định vị canonical audit và positive gate theo đường dẫn
  `scripts/`. Đây là ràng buộc tạm: `G1` phải đưa audit vào package qua
  `DatasetArtifact.open_verified()`. Cho tới lúc đó, một bản cài đặt không kèm
  `scripts/` sẽ bị từ chối train trên corpus có manifest, và đó là hành vi cố ý.

## Sửa đổi phiên trước

Không. Phiên này thi công `SESSION-20260901-001` và `PLAN-V2@4.6.0` theo đúng
những gì hai tài liệu đó bàn giao; không có claim, verdict hay evidence nào của
phiên trước bị sửa hoặc đảo ngược.

## Bàn giao

1. `R2` là bước kế tiếp: safe artifact I/O và một sample/manifest schema duy
   nhất. Các test `COR-00`/`COR-01` đã mô tả sẵn trạng thái đích, gồm cả việc
   audit phải nằm trong package.
2. Khi một gate đóng, sửa `status` của finding tương ứng trong
   `qdgrasp/corrective/registry.py` trong cùng PR với fix. Suite sẽ tự nói nếu
   đóng sớm (test vẫn fail) hoặc đóng muộn (`XPASS` dưới `strict`).
3. Khi bump schema thật, cập nhật `PLANNED_SCHEMA_BUMPS` cùng lúc;
   `test_registry.py` ghim giá trị hiện tại nên hai bên không trôi khỏi nhau.
4. Không nới hard-stop bằng cờ hoặc biến môi trường. Nếu một corpus hợp lệ bị
   chặn, đó là audit của corpus đó đang fail và phải sửa ở corpus.
5. `datasets/dgn-open-tiny` giữ nguyên trạng thái non-release input; phiên này
   không đổi manifest, không đổi byte và không tạo evidence mới.
