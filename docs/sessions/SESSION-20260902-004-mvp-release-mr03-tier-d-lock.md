---
document_id: SESSION-20260902-004
document_type: session_report
title: MVP release MR-03 — hiệu chuẩn và khóa Tier D
status: complete
date: 2026-09-02
session_id: SESSION-20260902-004
author: claude-agent
revises: none
related_plan: ROADMAP-MVP-RELEASE-001
scope: mujoco_cpu_leap_state_cuboid_only
candidate_branch: feature/mvp-grasp-policy
candidate_commit: 07e8b85148ae1e8a0ecb805e20c6b7b588034479
---

# SESSION-20260902-004 — MVP release MR-03: hiệu chuẩn và khóa Tier D

## Mục tiêu phiên

- Tìm một challenge domain mà controller prior chưa bão hòa, bằng
  development-only seed và trong ngân sách tối đa ba cấu hình.
- Giải thích từng failure bucket và loại domain khó vì lý do không hợp lệ.
- Khóa domain, manifest và hash trước khi train candidate cuối.
- Chứng minh train/dev/challenge-development/locked seed sets rời nhau.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
| --- | --- | --- | --- |
| `MR03-01` | Cài challenge domain thành artifact có schema, hash và ràng buộc narrow-only | `qdgrasp/mvp/challenge.py` | verified |
| `MR03-02` | Nối domain vào env và evaluator; chỉ challenge split bị ảnh hưởng | `qdgrasp/mvp/env.py`, `qdgrasp/mvp/evaluate.py` | verified |
| `MR03-03` | Script khảo sát một cấu hình bằng development seed root | `scripts/calibrate_mvp_challenge.py` | verified |
| `MR03-04` | Cấu hình C1 đo được: prior 67.0% (201/300), 99 failure, zero safety/invalid | `evidence/mvp/release-v1/challenge-development/c1.json` | verified |
| `MR03-05` | Khóa domain vào tài liệu bất biến; hash khớp đúng domain đã đo | `configs/mvp/dexacquire-mvp-v1.challenge.json` | verified |
| `MR03-06` | Test seed disjointness và narrow-only; release gate parse domain qua model | `tests/mvp/test_mvp_release_contract.py`, `scripts/check_mvp.py` | verified |

### Cách chọn domain, và tại sao chỉ tốn một cấu hình

Ngân sách là ba cấu hình. Trước khi tiêu cấu hình nào, các ledger đã publish của
round-3 được đọc để biết prior hỏng ở đâu — đó là artifact v0 đã tồn tại, không
phải một lần chạy mới:

| Trục | Ở episode fail | Ở episode success |
| --- | --- | --- |
| `friction_slide` | trung bình 0.227 | trung bình 0.732 |
| `mass` | trung bình 0.101 kg | trung bình 0.051 kg |
| `density` | trung bình 1742 | trung bình 1357 |
| variant | w240 hỏng 9/47, w210 hỏng 4/42 | w090/w120/w180 gần như không hỏng |
| `yaw`, `drop_height` | không có tín hiệu | không có tín hiệu |

Domain C1 đi theo đúng ba trục có tín hiệu — `friction_slide` `[0.10, 0.55]`,
`density` từ `1000`, `half_width` `[0.018, 0.024]` (ba variant train rộng
nhất) — và để nguyên `yaw`/pose vì chúng không phân biệt được fail với success.
Ước lượng từ ledger cũ là ~63%; đo thật được 67.0%. Không cần cấu hình thứ hai.

Việc dùng ledger v0 để *thiết kế* domain không phải đọc locked result của v1:
chúng thuộc scope v0, seed root khác, và Tier D của v1 được test là rời hoàn
toàn khỏi mọi seed v0 cũng như khỏi train/dev/challenge-development của v1.

### Giải thích failure bucket

99 failure gồm `contact_loss` 66 và `drop` 33.

- `contact_loss`: kết thúc episode với ít hơn số nhóm ngón tiếp xúc mà success
  predicate đòi. Ma sát thấp làm vật trượt khỏi kẹp — đây là thất bại grasp thật.
- `drop`: vật đã được nâng rồi rơi, hoặc rời khỏi bàn. Khối lượng cao làm
  moment vượt khả năng giữ — cũng là thất bại grasp thật.
- `simulator_error` (0), `penetration` (0), `excess_force` (0): đúng bằng
  không. Một domain khó vì simulator invalid hoặc vì vi phạm safety budget sẽ bị
  loại, và điều kiện đó được kiểm chứ không được giả định.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- | --- |
| `MR03-E01` | commit khóa Tier D | branch `feature/mvp-grasp-policy` | `07e8b85148ae1e8a0ecb805e20c6b7b588034479` |
| `MR03-E02` | locked challenge domain | `configs/mvp/dexacquire-mvp-v1.challenge.json` | file `0ac2beae8f33180c8b6b0b738debc9c376951890dbd831fb62dfbb6903fd1c18`, content `aabec53df8291d3e434cf526da57ced54cd8b6439083f329343488027ed9a971` |
| `MR03-E03` | calibration report C1 | `evidence/mvp/release-v1/challenge-development/c1.json` | `512f9b79c8b4136209739f31a888951495bdd054efbb9d9f090183685333eeaf` |
| `MR03-E04` | Tier D seeds | `configs/mvp/dexacquire-mvp-v1.eval-manifest.json` | 300 seed, manifest hash `733ef3a04073337b82e586b60e442cdc7bdf386e38578a4d389decfa78f8dc12` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| `MR03-T01` | `.venv/bin/python scripts/calibrate_mvp_challenge.py --domain … --episodes 300` | 0 | `admissible`; 67.0%, 99 failure, zero safety/invalid, 33.0 s |
| `MR03-T02` | `.venv/bin/python -m pytest -q` | 0 | 1509 passed, 1 skipped, 105 subtests |
| `MR03-T03` | `.venv/bin/python -m pytest tests/mvp/test_mvp_release_contract.py -q` | 0 | 36 passed |
| `MR03-T04` | `.venv/bin/python -m pytest tests/mvp/test_check_mvp_release_gate.py -q` | 0 | 27 passed |
| `MR03-T05` | `.venv/bin/python scripts/check_static_core.py` | 0 | Ruff 137 file, Mypy 32 contract |
| `MR03-T06` | `.venv/bin/python scripts/check_docs.py --root .` | 0 | 159 tài liệu |
| `MR03-T07` | `.venv/bin/python scripts/check_mvp.py --root . --runs runs/mvp --release` | 1 | expected; `challenge_domain_present` và `challenge_domain_contract` nay PASS |

## Việc chưa hoàn tất

- Chưa sinh demonstration v1, chưa train BC/PPO: đó là MR-04.
- Chưa có locked evaluation nào; Tier D chưa từng được chạy trên locked seed.
- `.venv/bin/ruff check .` toàn repository vẫn còn 357 finding có sẵn, không
  finding nào thuộc file của MR-01–MR-03; khoảng cách với ma trận MR-07 vẫn mở.

## Sửa đổi phiên trước

Không. Không artifact v0 nào bị chạm. Một defect fail-closed trong release gate
do chính MR-02 viết đã được sửa: gate khớp key của challenge domain như dict nên
sẽ nhận một domain khai báo đúng tên trục nhưng vươn ra ngoài dải đã khóa của
scope. Gate nay parse tài liệu qua model biết rằng challenge domain chỉ được thu
hẹp scope. Đây là sửa bug làm gate hở, không phải đổi semantics của contract, nên
không cần bump artifact version theo §8.2.

## Bàn giao

MR-03 đạt exit gate: Tier D manifest bất biến, prior report development có 99
failure đo được, zero safety/invalid, và không có seed leakage.

Work package tiếp theo là MR-04: sinh demonstration train/dev v1 dưới scope v1,
train BC, train tối đa một PPO từ đúng BC parent, và chọn candidate chỉ bằng
train/dev/challenge-development. Ngân sách là một BC run và một PPO run. Nếu
không candidate learned nào có tín hiệu uplift trên challenge development,
verdict là `NO-GO`.
