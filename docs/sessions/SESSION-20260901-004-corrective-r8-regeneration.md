---
document_id: SESSION-20260901-004
document_type: session_report
title: R8 — tái sinh DGN-Open-Tiny và chạy lại bằng chứng Phase 4
status: complete
date: 2026-09-01
session_id: SESSION-20260901-004
author: claude-agent
related_plan: PLAN-V2@4.6.0
revises: none
---

# SESSION-20260901-004 — R8: tái sinh dữ liệu và bằng chứng

## Mục tiêu phiên

- Tái sinh `DGN-Open-Tiny` dưới sample contract `v3`, vì manifest schema đã bump
  khi target-validity thành field bắt buộc và corpus cũ không còn đọc được.
- Chạy lại bằng chứng Phase 4 CPU, thứ mà `PLAN.md` §9.1 đã tuyên bố superseded
  sau khi joint parameterization và quality objective đổi.
- Ghi lại ngữ nghĩa kiến trúc vào chính artifact, để một checkpoint không thể
  được load như thể nó cùng nghĩa.

## Việc đã hoàn tất

| ID | Kết quả | Output | Trạng thái |
|---|---|---|---|
| `S4-01` | Tái sinh corpus từ commit sạch `4941be8`, recipe `region_opposition_v1`, seed 42 | `datasets/dgn-open-tiny` | verified |
| `S4-02` | Canonical audit pass lần đầu kể từ audit 2026-09-01 | `scripts/check_dataset_manifest.py` | verified |
| `S4-03` | Shadow rời khỏi release: 4 shard cho 2 active hand, đúng ADR-0008 | cùng artifact | complete |
| `S4-04` | Sửa test G0 theo sự thật mới: audit pass, floor fail | `tests/corrective/test_g0_hard_stop.py` | complete |
| `S4-05` | Thêm test cho corpus có source drift: không đếm, không báo số bên cạnh audit fail | cùng file | complete |
| `S4-06` | Ghi `joint_parameterization` và `quality_conditioning` vào model, bundle và gate load theo chúng | `qdgrasp/models/flow.py`, `engine/checkpoint.py` | complete |
| `S4-07` | Bằng chứng Phase 4 lên schema `v2`, mang identity đầy đủ | `scripts/overfit_qdgrasp_flow.py` | complete |
| `S4-08` | Chạy lại overfit CPU cho cả hai active hand từ worktree sạch | `evidence/phase4/overfit-*.json` | verified |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/kết quả | Tham chiếu |
|---|---|---|---|
| `E-01` | canonical audit | `PASS`; 4 shard, 132 sample, 5 positive, 16 object | `scripts/check_dataset_manifest.py --root datasets/dgn-open-tiny` |
| `E-02` | provenance corpus | schema `v3`, commit `4941be8`, worktree sạch, recipe `region_opposition_v1` | `dataset_manifest.json` |
| `E-03` | validity flag | mọi sample mang `kinematics_valid`, `pose_target_valid`, `joint_target_valid`, `fk_target_valid` | cùng artifact |
| `E-04` | recipe là biến quyết định positive | `wrench_guided_v1` cho 0 positive; `region_opposition_v1` cho 5, khớp yield của bản bị thay | log tái sinh |
| `E-05` | positive floor | 5 positive toàn corpus nhưng `0` trong train split mà protocol thừa nhận; sàn là 25 mỗi training hand | `scripts/check_phase5_inputs.py` |
| `E-06` | Phase 4 LEAP | palm `0.0413` m, rot `0.0228` rad, joint `0.0516` rad, tip `0.0418` m; coverage 186/186 | `evidence/phase4/overfit-leap-cpu.json` |
| `E-07` | Phase 4 Allegro | palm `0.0559` m, rot `0.0158` rad, joint `0.0487` rad, tip `0.0585` m; coverage 186/186 | `evidence/phase4/overfit-allegro-cpu.json` |
| `E-08` | identity của bằng chứng mới | schema `v2`, commit `f45ca65`, worktree sạch, `atanh-normalized-limits/v1` | cùng hai file |

Hai lần chạy đầu tiên bị loại vì bản Allegro ghi `worktree_dirty=True`: bản LEAP
đã ghi file bằng chứng của nó trước, và điều đó làm bẩn cây cho lần sau. Cả hai
được chạy lại vào thư mục tạm rồi mới đưa vào `evidence/`, nên cả hai khai báo
cùng một commit sạch.

## Kiểm tra đã chạy

| Test ID | Lệnh/phương pháp | Kết quả | Exit/đánh giá |
|---|---|---|---|
| `T-01` | `.venv/bin/pytest -q tests/corrective` | 67 pass, không còn expected-fail | pass |
| `T-02` | `.venv/bin/pytest -q` | 1402 pass, 1 skip, 93 subtest | pass |
| `T-03` | `.venv/bin/python scripts/check_dataset_manifest.py --root datasets/dgn-open-tiny` | `PASS` | pass |
| `T-04` | `.venv/bin/python scripts/check_phase5_inputs.py` | exit 1 vì positive floor | expected blocker |
| `T-05` | `.venv/bin/python scripts/check_phase4.py --profile contract` | 10/12 delivered; `P4-11b`, `P4-12` blocked | expected blocker |
| `T-06` | `.venv/bin/ruff check` trên file đã sửa | không thêm finding | pass |
| `T-07` | `.venv/bin/python scripts/check_docs.py --root .` | hợp lệ | pass |

## Việc chưa hoàn tất

- `COR-02` còn mở và `blocked_on` đã được viết lại: phần plumbing xong, gate
  `G2` còn chờ positive yield. Đọc kỹ hơn thì tình hình xấu hơn con số tổng:
  cả 5 positive đều nằm ngoài train matrix của protocol — chúng thuộc các
  positive-control bar hiệu chỉnh theo từng hand, thứ mà `protocol-v2` cố ý loại
  khỏi so sánh cross-hand, hoặc thuộc val. Train split được protocol thừa nhận
  có `0` positive. Đó là câu hỏi về pipeline sinh dữ liệu, không phải về đường
  dẫn, và không có cách dán nhãn nào đóng được nó.
- `COR-12` còn mở, thuộc `R9`: `qdgrasp/models/data.py` vẫn là contract thứ hai,
  và static gate cho active core chưa sạch.
- `P4-11b` CUDA gate cần chạy trên Kaggle T4; code đã đổi nên notebook phải ghim
  lại commit mới. Chưa push branch nào.
- `P4-12` independent review vẫn treo; tác giả một artifact không tự ký nó.
- Review packet Phase 4 **chưa** được ký lại. `PLAN.md` §9.10 đặt việc đó sau
  khi mọi corrective gate pass, nên packet hiện tại vẫn đúng khi bị từ chối.
- Chưa chạy lại bằng chứng MVP; fingerprint hiện vẫn khớp world hiện tại.

## Sửa đổi phiên trước

Không. Phiên này thi công `R8` theo dependency của `PLAN.md` §9.11 sau khi
`R2`–`R7` đã merge, và không sửa claim hay verdict nào của phiên trước. Bằng
chứng Phase 4 cũ không bị viết lại mà bị thay thế: nó được sinh dưới ngữ nghĩa
khác và §9.1 đã tuyên bố điều đó.

## Bàn giao

1. Hai việc còn lại của `R8` cần thứ mà máy này không có: CUDA gate chạy ở
   Kaggle, và một người review độc lập.
2. Trước khi bàn giao notebook CUDA, ghim lại commit đã push và diễn tập
   notebook trên một checkout sạch của đúng commit đó.
3. Positive yield là điều kiện còn lại của `G2`. Hai đường: đổi recipe sinh dữ
   liệu, hoặc thay bằng corpus lớn hơn. Cả hai đều là quyết định phạm vi, không
   phải sửa lỗi.
4. `R9` là phần cuối: bỏ contract trùng, làm sạch static gate, rồi mới ký lại
   packet và các card theo §9.10.
5. Không nới hard-stop. Corpus mới pass audit nhưng vẫn bị chặn ở floor, và đó
   là câu trả lời đúng.
