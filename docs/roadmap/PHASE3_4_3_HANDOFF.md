---
document_id: ROADMAP-P3.4.3-003
document_type: plan
title: Điều kiện bàn giao contact-rich input cho Phase 4
version: 1.0.0
status: active
date: 2026-08-28
parent_plan: ROADMAP-P3.4.3-001
related_decision: ADR-0008
active_hands:
  - leap_hand
  - wonik_allegro
paused_hands:
  - shadow_hand
---

# P3.4.3 → P4 handoff (G11)

Tài liệu này nói rõ P4 được nhận gì, khi nào, và cái gì vẫn đóng.

## 1. Interface freeze

Ba contract dưới đây là bề mặt bàn giao. Đổi bất kỳ cái nào cần version mới và
một revision record; không sửa tại chỗ.

| Interface | Schema | Ý nghĩa |
| --- | --- | --- |
| Trajectory | `qdgrasp/dynamic-trajectory/v2` | fixed-rate state samples + sparse contact stream, timebase khai báo, palm pose thật |
| ReplayCapsule | `qdgrasp/replay-capsule/v1` | đủ để CPU oracle replay đúng candidate mà không cần process của search |
| Manifest | `qdgrasp/contactrich-manifest/v2` | counts tính từ shard, coverage tách khỏi coverage_status |

Loader public là `qdgrasp.dataset.contactrich_active.load`. Mặc định nó từ chối
v1, dirty worktree, hash lệch, path traversal, split rò và artifact
`release_blocked`.

## 2. Cái gì đã mở

P4 **static/offline** không bị chặn. Nó chưa bao giờ phụ thuộc contact-rich
input, và không có gì trong P3.4.3 làm nó xấu đi.

## 3. Cái gì vẫn đóng

P4 **contact-rich training input** vẫn đóng cho tới khi đủ cả ba:

1. `scripts/check_phase3_4_3.py --scope active --profile release` trả `0` trên
   commit sạch, tức completeness manifest không còn required item mở;
2. CUDA gate chạy thật trên Kaggle T4 với verdict `PASS` trên đúng commit ứng
   viên — `kaggle-phase3-4-3/` là harness, `scripts/check_phase3_4_3_cuda.py`
   là gate;
3. reviewer độc lập ký `PASS` trên exact packet hash, zero S0/S1 mở.

Thiếu bất kỳ mục nào thì artifact giữ `release_blocked=true` và loader từ chối
nó theo mặc định. Đó là hành vi đúng, không phải lỗi cấu hình.

## 4. Cái gì không bao giờ được suy ra từ artifact này

- `QDGrasp-ContactRich-Active-Tiny` là **2/2 active hand**. Nó không phải
  three-hand coverage và không đổi P3.4 lịch sử thành pass; P3.4 giữ
  `paused_by_ADR-0008`.
- Mọi số đo tiếp xúc ở đây là simulation-only. Không có claim nào về việc một
  bàn tay vật lý chịu được các giới hạn đó.
- MPPI (`P3.4-10`) là `deferred_not_claimed`. Không coverage claim nào được bao
  gồm nó.
- Ablation static-vs-dynamic hiện cho `no_measured_difference` trên scene của
  release recipe. Không threshold nào bị đổi để tránh kết quả đó.

## 5. Lệnh kiểm tra

```bash
python scripts/check_phase3_4_3.py --scope active --profile cpu
python scripts/check_contactrich_active.py datasets/contactrich-active-tiny
python -m pytest tests/dynamic_grasp tests/contactrich_active -q
python scripts/check_docs.py --root .
git diff --check
```

Gate release đầy đủ thêm `--profile release`, `--cuda-evidence` và
`--review-packet`; nó chỉ trả `0` khi cả ba đều có và đều pass.
