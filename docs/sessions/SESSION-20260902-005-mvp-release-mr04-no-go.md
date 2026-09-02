---
document_id: SESSION-20260902-005
document_type: session_report
title: MVP release MR-04 — candidate không có tín hiệu uplift, verdict NO-GO
status: complete
date: 2026-09-02
session_id: SESSION-20260902-005
author: claude-agent
revises: none
related_plan: ROADMAP-MVP-RELEASE-001
scope: mujoco_cpu_leap_state_cuboid_only
candidate_branch: feature/mvp-grasp-policy
verdict: NO-GO
---

# SESSION-20260902-005 — MVP release MR-04: verdict `NO-GO`

## Mục tiêu phiên

- Sinh demonstration v1 dưới scope v1, có phủ challenge domain.
- Train một BC run và tối đa một PPO run từ đúng BC parent.
- Chọn candidate chỉ bằng train/dev/challenge-development.
- Áp dụng stop rule của MR-04 lên kết quả đo được.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
| --- | --- | --- | --- |
| `MR04-01` | Prior dựng lại từ scope v1 ra đúng bytes đã commit; không artifact prior nào bị đổi | `configs/mvp/leap-pinch-prior-v0.json` `fcaeb88a…` | verified |
| `MR04-02` | Generator hỗ trợ demonstration trộn base + challenge domain, ghi cờ `challenged` vào từng ledger row | `scripts/generate_mvp_demos.py`, `qdgrasp/mvp/env.py` | verified |
| `MR04-03` | Sinh 800 train + 240 dev episode; 705 + 210 accepted | `evidence/mvp/release-v1/mr04-no-go/demonstrations/` | verified |
| `MR04-04` | Train BC, rồi một PPO từ đúng BC parent | `policy/bc.pt`, `policy/ppo.pt` | verified |
| `MR04-05` | Đo prior/BC/PPO trên challenge-development seeds và tính paired CI | `policy/training-report.json` | verified |
| `MR04-06` | Áp stop rule: không candidate nào có tín hiệu uplift → `NO-GO` | phần §Verdict | verified |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- | --- |
| `MR04-E01` | demonstration index | `evidence/mvp/release-v1/mr04-no-go/demonstrations/index.json` | `b6026a172f865ccc101a490711b443ef1663be49c5054f5cd492226160150ee6` |
| `MR04-E02` | train ledger | `.../demonstrations/train/ledger.jsonl` | `01589865b41e698f3200f0687c7cb16218000095e5eaaeaf4e1b28a525e659e9` |
| `MR04-E03` | dev ledger | `.../demonstrations/dev/ledger.jsonl` | `98730fe9d4aac6dd696b88c0250c7580c2b320a72dadcefbb7063f138f5dacdb` |
| `MR04-E04` | BC checkpoint | `.../policy/bc.pt` | `dc866271442054b46e8ef71b492af88e99dd03fd4c1932341c11aa4c1de95e9f` |
| `MR04-E05` | PPO checkpoint | `.../policy/ppo.pt` | `14a4d23bc78bbe7eb55418a9f0344eab98e39bf0dfdcc39ba8c2a4c6aa2cdfc6` |
| `MR04-E06` | training report | `.../policy/training-report.json` | `0180d0da24173ad0f8e85db9f694e0a612b343512308f9c6bfe94477586ff45b` |
| `MR04-E07` | challenge domain đã dùng | `configs/mvp/dexacquire-mvp-v1.challenge.json` | content `aabec53df8291d3e434cf526da57ced54cd8b6439083f329343488027ed9a971` |

`MR04-E04` và `MR04-E05` nằm trên đĩa nhưng không được git theo dõi: `.gitignore:27`
loại `*.pt` trên toàn repository, và checkpoint của `round-1`…`round-3` cũng được
giữ đúng như vậy. SHA-256 ghi ở bảng trên là bản ghi provenance của chúng. Nếu một
attempt sau đạt tới MR-06, review packet sẽ cần một đường dẫn artifact được hash
và phân phối tách khỏi git, chứ không dựa vào việc file còn nằm trên máy này.

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| `MR04-T01` | `scripts/lock_mvp_scope.py --scope configs/mvp/dexacquire-mvp-v1.yaml --check` | 0 | manifest v1 khớp scope |
| `MR04-T02` | `scripts/build_mvp_prior.py --scope …v1.yaml --out <tmp>` | 0 | bytes trùng prior đã commit |
| `MR04-T03` | `scripts/generate_mvp_demos.py … --challenge-train-episodes 400 --challenge-dev-episodes 120` | 0 | 496.2 s; train 705/800, dev 210/240 |
| `MR04-T04` | `scripts/train_mvp_policy.py … --challenge-dev-episodes 300` | 0 | 591.7 s; BC + PPO |
| `MR04-T05` | `.venv/bin/python -m pytest tests/mvp -q` | 0 | 127 passed, 12 subtests |
| `MR04-T06` | `scripts/check_static_core.py` | 0 | Ruff 137 file, Mypy 32 contract |

## Số liệu quyết định

Demonstration, tách theo miền (train split, 800 episode):

| Nửa | Episode | Prior tự thành công | Accepted | Search rescued |
| --- | ---: | ---: | ---: | ---: |
| base domain | 400 | 377 | 388 | 11 |
| challenge domain | 400 | 267 | 317 | 50 |

Nửa challenge cho prior đúng 66.75%, khớp với 67.0% mà MR-03 đo — miền được xây
đúng như đã khóa. Tổng số episode mà expert search cứu được là 61, so với 7 của
v0: gấp gần chín lần, và vẫn chỉ là 8.7% của tập demonstration.

Đo trên 300 challenge-development seed, cùng seed cho cả ba arm:

| Arm | Success | Rate | Paired vs prior | 95% CI |
| --- | ---: | ---: | ---: | --- |
| controller prior | 201/300 | 0.670 | — | — |
| BC | 198/300 | 0.660 | `-1.00 pp` | `[-3.33, +1.00]` |
| PPO | 198/300 | 0.660 | `-1.00 pp` | `[-3.33, +1.00]` |

Trên base dev: BC 0.907, PPO 0.927.

## Verdict

`NO-GO`.

Gate Tier D đòi candidate hơn prior ít nhất `5.0` pp với paired CI lower `> 0`.
Đo được là `-1.0` pp với CI `[-3.33, +1.00]` — không những không đạt ngưỡng, mà
điểm ước lượng còn âm và khoảng tin cậy nằm vắt qua không. Cả BC lẫn PPO đều
như vậy, và cả hai có đúng cùng cấu trúc paired (4 episode chỉ candidate thắng,
7 episode chỉ prior thắng), nghĩa là PPO không làm dịch chuyển hành vi trên
miền challenge chút nào.

Theo `ROADMAP-MVP-RELEASE-001` §5 MR-04 stop rule và §7 decision matrix
(“BC và PPO đều không đạt D → `NO-GO`; không fallback prior âm thầm”), phiên
dừng tại đây. Không chạy locked evaluation MR-05: tiêu một lần locked evaluation
duy nhất cho một candidate không có tín hiệu development chính là điều stop rule
tồn tại để ngăn.

## Phân tích nguyên nhân

Giữ evidence để quyết định nguyên nhân là yêu cầu §8.8. Ba quan sát đo được:

1. **Expert là minimum-intervention.** Search dừng ngay khi candidate 0 (chính
   là prior) thành công, nên 644/705 demonstration accepted mang nhãn “không
   làm gì”. `mean_expert_action_magnitude` là `0.0078` trên thang unit action.
   BC học từ tập đó tất yếu hội tụ về residual gần không: giá trị dự đoán đo
   được là `0.0105`. Đây đúng là trạng thái suy biến mà ablation contract ở
   §2.3.3 được viết ra để bắt.
2. **PPO không hề đi qua miền challenge.** `train_residual_ppo` lấy rollout từ
   split `train` với `seed_offset=10_000`, tức miền base, nơi prior đã gần trần.
   Một chính sách không bao giờ thấy miền mà đóng góp của nó được đo thì không
   có đường nào để học đóng góp đó. Đây là quan sát về thiết kế hiện có, không
   phải bug làm run invalid: generator đã được làm challenge-aware ở phiên này,
   PPO rollout thì chưa.
3. **Cải thiện duy nhất nằm sai chỗ.** PPO hơn BC 2.0 pp trên base dev
   (0.927 so với 0.907) — đúng miền mà prior đã đạt trần và §2.3 nói rằng thắng
   ở đó không chứng minh được gì.

Không quan sát nào trong ba điều trên được dùng để mở tuning loop trong phiên
này. §8.8 cấm lặp lại cùng một run mà không đổi giả thuyết, và việc đổi giả
thuyết là quyết định của maintainer chứ không phải của phiên thi công.

## Việc chưa hoàn tất

- MR-05 đến MR-09 không được thực hiện. Public release vẫn bị chặn.
- Chưa có locked evaluation nào; Tier D chưa từng chạy trên locked seed, và
  candidate chưa từng thấy locked seed.
- `.venv/bin/ruff check .` toàn repository vẫn còn 357 finding có sẵn.

## Sửa đổi phiên trước

Không. Không artifact v0 hay v1 nào đã đóng bị sửa. Scope v1, eval manifest v1
và challenge domain giữ nguyên hash; chúng là contract mà kết quả này được đo
dưới, và việc đo ra `NO-GO` không phải lý do để sửa chúng.

## Bàn giao

MVP release attempt thứ nhất dừng ở `NO-GO` tại MR-04, với evidence đầy đủ và
mọi contract còn nguyên. Theo §5 MR-05 và §8.8, một attempt mới cần một giả
thuyết khác, một scope/candidate version mới và một revision record — đó là
quyết định N3 của maintainer, không phải việc mà phiên này được phép tự mở.

Ba hướng mà evidence ở trên trực tiếp gợi ra, để maintainer cân nhắc:

1. Cho PPO rollout đi qua challenge domain, vì đó là miền mà đóng góp được đo.
2. Đổi expert từ minimum-intervention sang một expert thực sự tối ưu trên miền
   khó, để demonstration mang tín hiệu chứ không mang nhãn “không làm gì”.
3. Kết luận rằng residual learned trên state-based LEAP MVP không đủ giá trị để
   phát hành, và chuyển sản phẩm thành controller-only — theo §0.6 việc này cần
   quyết định N3 và một release claim mới.
