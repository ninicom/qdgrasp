---
document_id: ROADMAP-P4-002
document_type: plan
title: Hướng dẫn kiểm tra độc lập Phase 4
version: 1.0.0
status: active
date: 2026-08-31
revises: none
related_plan: ROADMAP-P4-001
latest_revision_record: docs/revisions/REV-20260831-003-phase4-architecture.md
---

# Hướng dẫn reviewer Phase 4 (P4-12)

Tài liệu này dành cho người **không phải tác giả** của P4. Tác giả chỉ chuẩn bị
packet bất biến; verdict do reviewer ghi.

## 0. Trạng thái trước khi bắt đầu

`scripts/check_phase4.py --profile micro` báo **11/13 mục delivered** và trả `1`.
Hai mục còn mở:

| Mục | Trạng thái | Vì sao chưa đóng |
| --- | --- | --- |
| `P4-11b` CUDA gate evidence | `blocked` | Chưa có CUDA run thật. `ADR-0006` cấm CPU fallback làm bằng chứng CUDA; §7.4/§7.5 đòi overfit trên CUDA và parity FP32 CPU/CUDA. |
| `P4-12` independent review | `blocked` | Chính là việc của tài liệu này. |

**Exit code `1` là kết quả đúng.** Nếu cổng trả `0` mà chưa có CUDA evidence và
chưa có verdict, nghĩa là cổng đã bị nới, không phải phase đã xong.

## 1. Điều duy nhất P4 tuyên bố

**Kiến trúc học được.** Không hơn.

Bằng chứng là một lần overfit tám sample: palm 1.871 → 0.040 m, rotation
2.617 → 0.021 rad, joint 0.864 → 0.053 rad, fingertip 1.838 → 0.041 m, gradient
phủ 186/186 tham số. Overfit tám sample **không** nói gì về grasping, và §7 của
plan cấm trích nó như thể có. Một trong những việc của reviewer là kiểm rằng
không tài liệu nào trong repo vi phạm điều đó.

## 2. Lấy packet

```bash
python scripts/phase4_review_packet.py --out evidence/phase4/review
```

Packet ghi `commit`, `worktree_clean`, hash của 22 artifact bắt buộc, phạm vi
kiểm theo §7, và `known_absent` — bằng chứng plan yêu cầu nhưng máy tạo packet
không sinh được. Verdict luôn `null`.

Chỉ ký trên `packet_digest` của một packet có `worktree_clean: true`.

## 3. Lệnh tái lập

```bash
python -m pytest tests/model_flow -q
python scripts/check_phase4.py --profile micro
python scripts/overfit_qdgrasp_flow.py --device cpu --steps 1200
python scripts/phase4_cuda_gate.py --device cpu
python scripts/check_docs.py --root .
git diff --check
```

Lệnh thứ tư **phải fail** với `verdict: refused`. Đó là kiểm tra rằng harness
CUDA không chịu gắn nhãn CUDA cho một lần chạy CPU. Nếu nó trả `measured` trên
máy không có GPU thì đó là finding S0.

Lệnh thứ ba mất khoảng 5 phút CPU và phải hội tụ dưới bốn ngưỡng pose đã pin.

## 4. Điều cần kiểm, theo §7

| Lĩnh vực | Ở đâu | Câu hỏi |
| --- | --- | --- |
| Tokenizer | `qdgrasp/models/tokenizer.py` | Key có phải theo vị trí và đơn ánh, không hash? Grid mịn quá khả năng pack bị từ chối hay bị alias im lặng? |
| Không `N×N` | `qdgrasp/models/encoder.py`, `flow.py` | Có attention call nào hai chiều đều dài? Probe khẳng định "không" có negative control chạy được không? |
| Cross-embodiment | `qdgrasp/models/hand_graph.py` | LEAP 18 node và Allegro 22 node qua **một** encoder, không hằng số nào bị compile vào? Đổi thứ tự node có đổi kết quả không? |
| Output hợp lệ | `qdgrasp/models/flow.py` | Rotation trong SO(3) dung sai 1e-4 và joint trong limit **do cấu trúc**, hay do clip sau khi sinh? |
| Loss accounting | `qdgrasp/models/losses.py` | Total bằng tổng term được log? Term lạ bị từ chối? Rotation error là geodesic? |
| Gradient coverage | `evidence/phase4/overfit-leap-cpu.json` | Mọi tham số trainable nhận gradient hữu hạn sau một backward? |
| Verdict của overfit | `scripts/overfit_qdgrasp_flow.py` | Verdict đọc trên pose error chứ không trên tổng loss? Lý do có được ghi ngay chỗ đặt ngưỡng? |
| Config/registry | `qdgrasp/models/config.py`, `qdgrasp/presets/qdgrasp-flow-*.yaml` | Tham số preset lạ bị từ chối hay bị bỏ qua? Preset có đổi được shape mà scale table sở hữu không? |
| CUDA gate | `scripts/phase4_cuda_gate.py` | **ABSENT.** Xem §5. Harness có từ chối CPU run thay vì gắn nhãn cho nó? |
| Phạm vi tuyên bố | `docs/sessions/SESSION-20260831-003-phase4-qdgrasp-flow.md` | Có câu nào đọc được thành kết quả grasping không? |

## 5. Những gì reviewer **không** được kết luận từ packet này

- Không có số CUDA nào. Forward/backward, parity FP32 CPU/CUDA và memory scaling
  chưa chạy trên NVIDIA thật; `scripts/phase4_cuda_gate.py` từ chối `--device cpu`
  và packet ghi `P4-11` là `known_absent`.
- Overfit có trên cả hai active hand nhưng vẫn chỉ là tám sample mỗi lần, với
  label sinh bằng FK của chính profile. Hai lần hội tụ không cộng lại thành một
  bằng chứng về dữ liệu thật.
- Không có chất lượng grasp nào được đo. Không benchmark, không held-out object,
  không held-out embodiment, không model card. Đó là P5.
- Shadow Hand giữ `paused_by_ADR-0008`. Không kết quả nào ở đây là three-hand
  coverage.
- Model chưa được train trên `DGN-Open-Tiny`. Fixture của overfit là label sinh
  bằng FK của chính profile, nên nó chắc chắn với tới được — đó là điều kiện để
  một lần không hội tụ có nghĩa là "kiến trúc sai", chứ không phải bằng chứng
  model đọc được dữ liệu thật.

## 6. Ghi verdict

Dùng `docs/templates/THIRD_PARTY_REVIEW_REPORT.md`, ghi `packet_digest` đã kiểm,
và phân loại từng finding theo S0–S3. P4 chỉ được ghi `complete` khi **cả**
`P4-11b` có CUDA evidence đo được cho hai active hand **và** review này `pass`
với zero S0/S1.
