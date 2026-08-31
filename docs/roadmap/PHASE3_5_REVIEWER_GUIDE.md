---
document_id: ROADMAP-P3.5-002
document_type: plan
title: Hướng dẫn kiểm tra độc lập Phase 3.5
version: 1.0.0
status: active
date: 2026-08-31
revises: none
related_plan: ROADMAP-P3.5-001
latest_revision_record: docs/revisions/REV-20260831-002-phase3-5-execution.md
---

# Hướng dẫn reviewer Phase 3.5 (P3.5-18)

Tài liệu này dành cho người **không phải tác giả** của P3.5. Tác giả chỉ chuẩn bị
packet bất biến; verdict do reviewer ghi.

## 0. Trạng thái trước khi bắt đầu

`scripts/check_phase3_5.py --profile micro` báo **17/19 mục delivered** và trả
`1`. Hai mục còn mở:

| Mục | Trạng thái | Vì sao chưa đóng |
| --- | --- | --- |
| `P3.5-15` backend decision | `blocked` | Chưa có CUDA run thật. `ADR-0006` cấm CPU fallback làm bằng chứng CUDA; §7 cấm chọn backend khi chưa có parity hai tay đo được. |
| `P3.5-18` independent review | `blocked` | Chính là việc của tài liệu này. |

**Exit code `1` là kết quả đúng.** Nếu cổng trả `0` mà chưa có GPU evidence và
chưa có verdict, nghĩa là cổng đã bị nới, không phải phase đã xong.

## 1. Lấy packet

```bash
python scripts/phase3_5_review_packet.py --out evidence/phase3_5/review
```

Packet ghi `commit`, `worktree_clean`, hash của 23 artifact bắt buộc, phạm vi
kiểm theo §13.9, và `known_absent` — những bằng chứng plan yêu cầu nhưng máy tạo
packet không sinh được. Verdict luôn `null`.

Chỉ ký trên `packet_digest` của một packet có `worktree_clean: true`.

## 2. Lệnh tái lập

```bash
python -m pytest tests/assets_ingest tests/rl_env -q
python scripts/check_phase3_5.py --profile micro
python scripts/phase3_5_gpu_rl_readiness.py --backend mujoco-cpu --device cpu
python scripts/generate_rl_env_tiny.py --out /tmp/rl-env-tiny-replay
python scripts/check_docs.py --root .
git diff --check
```

Lệnh thứ tư sinh lại artifact ở thư mục khác; so `summary` với
`datasets/qdgrasp-rl-env-tiny/dataset_manifest.json`. Hash từng file **không**
bắt buộc trùng: asset procedural phụ thuộc seed của generator, còn lớp outcome
của từng case thì phải trùng.

## 3. Điều cần kiểm, theo §13.9

| Lĩnh vực | Ở đâu | Câu hỏi |
| --- | --- | --- |
| Asset transform và unit | `qdgrasp/objects/ingest.py` | Scale áp đúng một lần? Hash byte gốc trước mọi transform? Thiếu unit thì từ chối hay đoán? |
| CoACD | `qdgrasp/objects/coacd.py` | Không network, không tự ghi dataset? Mọi tham số official có type/range và nằm trong config hash? Tham số upstream không hỗ trợ bị từ chối hay bị bỏ im lặng? |
| Settle | `qdgrasp/scenes/settle.py` | `settled` có yêu cầu **mọi** vật đồng thời yên trong đủ số bước liên tiếp? Chín lớp fail có precedence cố định? |
| Scene resolution | `qdgrasp/scenes/resolver.py` | Scene hỏng có fail thay vì âm thầm thành scene ảo? |
| Gym API | `qdgrasp/rl/contracts.py`, `qdgrasp/rl/envs/` | `terminated` và `truncated` tách bạch? Mỗi bước terminated có đúng một lý do? |
| Reward | `qdgrasp/rl/contracts.py` | Total bằng tổng term được log? Term dương có thể trả giá cho barrier an toàn không? |
| Backend parity | — | **ABSENT.** Xem §4. |
| Cloud evidence | `notebooks/phase3_5_rl_readiness.ipynb` | Có pin commit bất biến? Có từ chối gắn nhãn GPU cho một CPU run? |
| Artifact | `datasets/qdgrasp-rl-env-tiny/` | Positive/negative/random có đúng lớp của nó? Có hash? |

## 4. Những gì reviewer **không** được kết luận từ packet này

- Không có backend GPU nào được chọn. Không có parity hai tay nào được đo.
  `scripts/phase3_5_gpu_rl_readiness.py` từ chối chạy `--device cuda:*` khi không
  có CUDA, và packet ghi `P3.5-15` là `known_absent`.
- Parity output-class của CoACD với artifact Stage 0 **chưa** được kiểm: CoACD và
  ManifoldPlus không được cài. Tham số của profile đã pin và có test; các convex
  part nó sinh ra thì chưa.
- `QDGrasp-RL-Env-Tiny` là `experimental_non_release`. Fixture positive dùng grasp
  prior đọc pose đặc quyền của target — đó là fixture chứng minh environment
  *giải được*, không phải bằng chứng một policy học được.
- Shadow Hand giữ `paused_by_ADR-0008` và bị `qdgrasp/rl/envs/hand_scene.py` từ
  chối load. Không kết quả nào ở đây là three-hand coverage.

## 5. Ghi verdict

Dùng `docs/templates/THIRD_PARTY_REVIEW_REPORT.md`, ghi `packet_digest` đã kiểm,
và phân loại từng finding theo S0–S3. P3.5 chỉ được ghi `complete` khi **cả**
`P3.5-15` có GPU evidence đo được **và** review này `pass` với zero S0/S1.
