---
document_id: RPT-MVP-GRASP-POLICY-001
document_type: release_report
title: Model card — QDGrasp-Leap-Grasp-MVP (experimental, non-release)
status: complete
date: 2026-08-31
author: claude-agent
revises: none
related_plan: ROADMAP-MVP-001
release_class: experimental_non_release
artifact_id: QDGrasp-Leap-Grasp-MVP
environment_id: QDGrasp-DexAcquire-MVP-v0
---

# Model card — QDGrasp-Leap-Grasp-MVP

**Đây không phải release.** Artifact mang `release_class: experimental_non_release`.
Nó không đóng P3.4.3, P3.5, P4 hay P5, không gỡ bất kỳ blocker nào, và không
được trích dẫn như bằng chứng release hay bằng chứng GPU physics.

## Tóm tắt

Checkpoint `QDGrasp-Leap-Grasp-MVP` là một actor-critic MLP nhỏ sinh residual có
biên quanh một controller prior pinch cho LEAP Hand, trong môi trường
`QDGrasp-DexAcquire-MVP-v0`: một bàn cố định, một cuboid, quan sát state đặc
quyền, MuJoCo CPU.

Nó **đạt cả ba tier** của `ROADMAP-MVP-001` §7 sau khi reload checkpoint, với
`invalid_state=0`, `safety_violation=0` và `checkpoint_reload_mismatch=0`.

Nó **không cải thiện được controller prior**. Đây là kết luận quan trọng nhất
của tài liệu này và nó nằm ngay ở đây thay vì ở cuối:

| Tier | Miền | Controller prior | Checkpoint | Gate |
| --- | --- | ---: | ---: | ---: |
| A | canonical, 6 train size, pose trung tâm | 100/100 | **100/100 = 100.0%** | `>=95%` |
| B | randomized, 6 train size, 300 locked seed | 284/300 | **284/300 = 94.7%** | `>=85%`, Wilson `>=80%` |
| C | held-out size, 200 locked seed | 188/200 | **187/200 = 93.5%** | `>=70%` |

Tier A và Tier B trùng khớp đến từng episode. Tier C lệch một episode theo
hướng xấu hơn. Residual trung bình mà policy phát ra là `0.0039` trong không
gian action đơn vị, tức khoảng **32 micromet** dịch chuyển palm target: policy
đã học sao chép prior chứ không học vượt prior. Gate MVP đạt được là nhờ
controller, không nhờ phần học.

Wilson 95% lower bound: A `0.963`, B `0.915`, C `0.892`.

## Giới hạn và phạm vi

- Một tay (LEAP), một preset đã pin. Không có claim nào cho Allegro; Shadow vẫn
  `paused_by_ADR-0008`.
- Một họ cuboid: sáu `half_width` train `0.009–0.024 m`, bốn held-out
  `0.0075/0.0135/0.0195/0.0255 m`, `half_height` cố định `0.020 m`. Không có
  raw mesh, không CoACD, không clutter, không scene resolver.
- Quan sát là **privileged simulator state**. Không có RGB/depth/point cloud/
  tactile. Vì prior đọc pose thật của target, randomization hình học (vị trí,
  yaw) bị prior khử gần như hoàn toàn theo thiết kế; phần ngẫu nhiên thực sự khó
  là ma sát `0.10–1.30` và mật độ `300–2400 kg/m^3`.
- Tất cả số liệu là **simulation-only** trên MuJoCo CPU. Không có claim nào về
  một bàn tay vật lý, và không có bằng chứng GPU physics nào ở đây.
- `QDGrasp-DexAcquire-MVP-v0` không phải Gymnasium contract của P3.5; nó chỉ có
  `reset/step` theo §1 của plan tạm thời.
- Không có independent reviewer. Tác giả tạo artifact cũng là người đo, nên theo
  `docs/governance/THIRD_PARTY_REVIEW.md` kết quả này không đủ tư cách cho bất kỳ
  cổng release nào.

## Đầu vào, đầu ra và an toàn

- Observation `qdgrasp/mvp-observation/v0`, 99 chiều, mỗi trường khai báo frame
  và đơn vị trong `qdgrasp/mvp/env.py`.
- Action là residual 8 chiều đã chuẩn hóa: `delta_xyz` (`<=8 mm`), `delta_rot`
  (`<=0.10 rad`) và hai finger-synergy (`<=0.20 rad`), đi qua low-pass
  `alpha=0.20` rồi bị kẹp bởi workspace, joint limit và safety budget trước khi
  tới `ctrl`. Policy không có đường nào ghi vào pose của target, vào trạng thái
  solver, vào cờ termination hay vào verdict.
- Safety budget: penetration `<=2 mm`, lực tiếp xúc `<=20 N`, xung lượng
  `<=0.5 N·s` trong một chu kỳ điều khiển. Mọi vi phạm kết thúc episode và tính
  là failure.

## Bằng chứng

Artifact bất biến nằm ở `evidence/mvp/round-3/`, manifest có SHA-256 đầy đủ tại
`evidence/mvp/round-3/MANIFEST.json`.

| Hạng mục | Giá trị |
| --- | --- |
| `scope_hash` | `a897e36084c9ab11cbf8046e446ef318c48b7463d8b2db4af4d4ca0593109b8b` |
| `eval_manifest_hash` | `ac57bf61fb1e4294f840d29fea4f26865f542164eb08ee6b6f39915875840e3a` |
| `prior_hash` | `ab96a87a275ec574568458f14a28c929828fcfada01ca8d6cf06f8c6a647be43` |
| `observation_schema` | `qdgrasp/mvp-observation/v0` |
| `policy/ppo.pt` | `498d03ae6f062ed2…`, 763 504 byte |
| `policy/bc.pt` (rollback) | `2e470a384239c6d5…`, 763 805 byte |
| Ledger từng episode | `evidence/mvp/round-3/evaluation/ppo/tier-{a,b,c}.jsonl` |
| Ledger generator | `evidence/mvp/round-3/demonstrations/{train,dev}/ledger.jsonl` |

Dữ liệu huấn luyện: 390/400 episode train được nhận, 7 trong số đó cần residual
để cứu, 915/1170 rollout nhiễu được nhận, tổng 137 025 transition. Tập `dev`
không có episode nào cần cứu (117/117 prior tự đạt).

Behaviour cloning đạt `144/150 = 96.0%` trên `dev`; residual PPO đạt
`143/150 = 95.3%`, nằm trong dung sai 2 điểm phần trăm nên được promote, và
checkpoint BC được giữ làm rollback.

### Failure bucket của candidate

| Tier | contact_loss | drop | timeout | penetration | excess_force | simulator_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0 | 0 | 0 | 0 | 0 | 0 |
| B | 9 | 7 | 0 | 0 | 0 | 0 |
| C | 9 | 4 | 0 | 0 | 0 | 0 |

Toàn bộ failure còn lại là mất tiếp xúc ở bước cuối hoặc rơi vật sau khi đã
nâng; không còn timeout, không vi phạm an toàn, không lỗi simulator.

## Ba vòng tune và cái gì đã sai

`ROADMAP-MVP-001` §8 cho phép tối đa ba vòng. Cả ba đã dùng, và hai vòng đầu
thất bại theo cách đo được; artifact của chúng được giữ nguyên tại
`evidence/mvp/round-1/` và `evidence/mvp/round-2/`.

| Vòng | Thay đổi | Tier A | Tier B | Tier C |
| --- | --- | ---: | ---: | ---: |
| 1 | Expert = ứng viên residual tốt nhất theo margin | 37.0% | 34.7% | 23.5% |
| 2 | Expert minimum-intervention + previous-action dropout | 16.0% | 15.7% | 10.0% |
| 3 | Low-pass residual, predicate đúng §4, demo có nhiễu | **100.0%** | **94.7%** | **93.5%** |

Vòng 1 sụp đổ vì causal confusion: nhãn expert là hằng số trong mỗi đoạn và
previous action nằm trong observation, nên `a_t = a_{t-1}` khớp gần như mọi mẫu
(train loss `0.0014`, validation `0.0021`). Khi rollout, đó là một random walk
không neo: tương quan giữa hai action liên tiếp đo được `0.98–0.997` và action
trôi tới `±0.6`.

Vòng 2 sửa nhãn nhưng tệ hơn, vì nguyên nhân còn lại không phải nhãn mà là
covariate shift: mọi state được demo đều nằm đúng trên quỹ đạo expert, nên
milimet sai lệch đầu tiên đã đưa policy ra ngoài dữ liệu. Đo được: action đi từ
`0.005` ở bước đầu lên `0.9` ở bước 40.

Vòng 3 sửa ba thứ, và chỉ một trong ba là thay đổi thuật toán:

1. **Predicate sai so với plan.** §4 yêu cầu *độ cao* giữ liên tục `0.50 s`, và
   tiếp xúc `>=2` finger group ở **bước cuối**. Bản cài đặt đầu gộp cả tiếp xúc
   vào bộ đếm liên tục, nên một bước chattering tiếp xúc xóa sạch nửa giây nâng
   hợp lệ. Sửa lại đúng §4 làm baseline của prior tăng từ `B 89.7% / C 88.5%`
   lên `B 94.7% / C 94.0%` mà không đổi một dòng controller nào — toàn bộ 15
   timeout của Tier B là artifact của predicate.
2. **Interface action không chịu được nhiễu.** Bơm `N(0, 0.15)` vào action làm
   mất vật ở **42/42** rollout, trong khi một residual hằng số lớn gấp ba lần
   thì vô hại. Vấn đề là biến thiên tần số cao, nên low-pass `alpha=0.20` được
   đặt vào interface thay vì yêu cầu policy tự mượt.
3. **Dữ liệu không phủ sai số của chính policy.** Demo được thu thêm dưới nhiễu
   action, nhãn là residual không nhiễu.

Hai thay đổi đầu làm `scope_hash` và `eval_manifest_hash` đổi; tier, kích thước
mẫu, miền randomization và dẫn xuất seed **không** đổi, nên số của cả ba vòng vẫn
so sánh được trực tiếp. Chi tiết tại `docs/revisions/REV-20260831-001-grasp-policy-mvp-closure.md`.

## Cách tái lập

```bash
python scripts/lock_mvp_scope.py --check
python scripts/build_mvp_prior.py
python scripts/evaluate_mvp.py --label controller_prior
python scripts/generate_mvp_demos.py
python scripts/train_mvp_policy.py
python scripts/evaluate_mvp.py --checkpoint runs/mvp/policy/ppo.pt
python scripts/check_mvp.py --runs evidence/mvp/round-3
```

`notebooks/mvp_grasp_policy.ipynb` chạy cùng chuỗi đó trên Kaggle/Colab từ một
commit public đã pin, có resume sau khi runtime bị thu hồi. Notebook đó dùng GPU
để train MLP; physics vẫn là MuJoCo CPU và không cell nào trong đó là bằng chứng
GPU physics.

## Kết luận

MVP đạt điều kiện đóng của `ROADMAP-MVP-001` §10: MVP-00–07 đều có artifact thật
và §7 pass sau checkpoint reload. Kết quả kỹ thuật đi kèm là một kết quả âm cần
được ghi đúng: trong phạm vi hẹp này, một controller prior pinch có bù lực nắm
đã đủ, và phần học không đo được đóng góp nào trên nó.

Điều đó định hướng việc quay lại roadmap chính theo §10: mở rộng miền cho tới khi
prior thực sự hỏng — object ngoài họ cuboid, quan sát không đặc quyền, tay thứ
hai — trước khi bỏ thêm công vào kiến trúc policy. Một policy chỉ có thể chứng
minh giá trị trên một bài toán mà prior không giải được.
