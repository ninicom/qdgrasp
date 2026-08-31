---
document_id: SESSION-20260831-002
document_type: session_report
title: Thi công Phase 3.5 — ingest, scene resolver, settle và RL environment
status: complete
date: 2026-08-31
session_id: SESSION-20260831-002
author: claude-agent
revises: none
related_plan: ROADMAP-P3.5-001
---

# SESSION-20260831-002 — Phase 3.5 (P3.5-01 … P3.5-16)

## Mục tiêu phiên

- Thực thi phase tiếp theo sau khi `ROADMAP-MVP-001` đóng. Theo
  `docs/roadmap/PROJECT_PHASES.md`, đó là `ROADMAP-P3.5-001`, contract đã bị MVP
  hạ ưu tiên và chưa có work package nào được thi công.
- Tiêu chí dừng: mỗi work package trong §9 hoặc có artifact chạy được kèm test,
  hoặc được ghi rõ là `open`/`blocked` cùng lý do. Không ghi `complete` cho
  package chưa đo.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Raw mesh ingest: một source duy nhất, hash byte gốc trước mọi transform, unit bắt buộc, scale áp đúng một lần | `qdgrasp/objects/ingest.py` | verified |
| W-02 | Public CoACD API đủ tham số official, typed error, cache content-addressed, từ chối tham số upstream không hỗ trợ | `qdgrasp/objects/coacd.py` | verified |
| W-03 | `ObjectAssetManifestV2` bất biến, từ chối ghi đè im lặng | `qdgrasp/objects/manifest_v2.py` | verified |
| W-04 | Scene resolver phân biệt load / fail / generate; scene hỏng không được thay bằng scene ảo | `qdgrasp/scenes/resolver.py`, `qdgrasp/scenes/serialize.py` | verified |
| W-05 | Virtual drop scene có support hữu hạn, spawn region, đặt vật không overlap, seed stream tách riêng | `qdgrasp/scenes/virtual_drop.py` | verified |
| W-06 | Settle certifier với chín lớp outcome, precedence cố định và `SceneSnapshot` bất biến replay được | `qdgrasp/scenes/settle.py` | verified |
| W-07 | RL contract độc lập simulator: observation schema, action spec, reward accounting, terminated/truncated tách bạch | `qdgrasp/rl/contracts.py` | verified |
| W-08 | Ba environment `ObjectSettle` / `DexAcquire` / `DexAcquireScene` chạy `reset`/`step` cho hai active hand | `qdgrasp/rl/envs/` | verified |
| W-09 | Fixture scripted và random-policy probe | `qdgrasp/rl/tasks/scripted.py` | verified |
| W-10 | Seed stream, domain randomization, scene signature bucketing, evaluation split theo hash | `qdgrasp/rl/randomization.py` | verified |
| W-11 | Cổng CPU của phase, báo cáo từng package là delivered/open/blocked | `scripts/check_phase3_5.py` | verified |
| W-12 | Harness GPU spike từ chối gắn nhãn CUDA khi không có CUDA | `scripts/phase3_5_gpu_rl_readiness.py` | verified |
| W-13 | Runner Kaggle/Colab pin commit, có resume | `notebooks/phase3_5_rl_readiness.ipynb` | verified |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | code | `qdgrasp/objects/{ingest,coacd,manifest_v2}.py` | commit của phiên |
| E-02 | code | `qdgrasp/scenes/{resolver,serialize,virtual_drop,settle}.py` | commit của phiên |
| E-03 | code | `qdgrasp/rl/` | commit của phiên |
| E-04 | test | `tests/assets_ingest/` (47 test) | chạy trong phiên |
| E-05 | test | `tests/rl_env/` (51 test) | chạy trong phiên |
| E-06 | gate | `scripts/check_phase3_5.py --profile micro` | 16/19 delivered, exit 1 |
| E-07 | harness | `scripts/phase3_5_gpu_rl_readiness.py`, `notebooks/phase3_5_rl_readiness.ipynb` | commit của phiên |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `python -m pytest tests/assets_ingest -q` | 0 | 47 passed |
| T-02 | `python -m pytest tests/rl_env -q` | 0 | 51 passed |
| T-03 | `python -m pytest tests scripts/tests -q` | 0 | 1168 passed, 1 skipped (trước khi thêm test GPU harness) |
| T-04 | `python scripts/check_phase3_5.py --profile micro` | 1 | 16/19 delivered; ba package còn mở |
| T-05 | `python scripts/phase3_5_gpu_rl_readiness.py --backend mujoco-cpu --device cpu` | 0 | CPU oracle pass cho hai hand, 771 step/s |
| T-06 | `python scripts/phase3_5_gpu_rl_readiness.py --backend mjx-warp --device cuda:0` | 1 | từ chối: `verdict=refused`, không có CUDA |

Đường sống asset → scene → drop → settle → reset/step chạy đủ cho **cả hai
active hand**: scene generated settle sau 147 bước, scripted fixture kết thúc ở
`horizon` không vi phạm an toàn, random policy giữ observation hữu hạn và không
ghi được success nào.

## Năm hiệu chỉnh dựa trên đo đạc

1. **Actuator khớp theo transmission target, không theo tên.** LEAP đặt tên
   `<joint>_act`, Allegro đặt `ffa0` cho khớp `ffj0`. Quy ước tên đúng cho một
   tay thì loại tay kia một cách im lặng.
2. **Hướng ban đầu của bàn tay lấy từ profile.** Identity vô tình đúng cho LEAP
   và sai cho Allegro. Trục `approach` của fingertip là đại lượng sai và đã thử
   trước: nó chỉ vào vật bị kẹp nên trên tay đối ngón thì triệt tiêu.
3. **Episode bắt đầu ở 0.20 m, không phải 0.14 m** — ở 0.14 m cả hai tay đã
   xuyên vật ngay lúc reset. Environment nay từ chối bắt đầu khi đang chạm.
4. **Xuyên thấu cần hai ngân sách.** Vật rơi nén tiếp xúc 7.7 mm lúc chạm đất và
   ổn định ở 0.07 mm; ngón chạm vật ở 0.08 m/s nén 5.75 mm ở bước đầu. Đó là
   xung lực, không phải lỗi. Ngân sách trạng thái ổn định giữ 2 mm; trần tức
   thời riêng bắt tunnelling.
5. **Descent của fixture kết thúc khi chạm, không theo số bước.** Số bước cố
   định hiệu chỉnh theo hình học một tay: open-loop, LEAP nâng được 28 cm còn
   Allegro nằm 5 mm trong hộp suốt sáu bước điều khiển và chạm ngưỡng an toàn.

## Việc chưa hoàn tất

- **P3.5-15 — backend decision record:** `blocked`. Không có GPU NVIDIA trên máy
  phát triển; `ADR-0006` cấm CPU fallback làm bằng chứng CUDA và §7 cấm chọn
  backend khi chưa có parity hai tay đo được. Harness và notebook đã sẵn sàng để
  chạy trên Kaggle/Colab; kết quả của run đó là đầu vào của P3.5-15.
- **P3.5-17 — `QDGrasp-RL-Env-Tiny`:** `open`. Artifact cần positive scripted
  fixture, mà fixture hiện tại chạy hết horizon không gắp được. Đây là kết luận
  cấu trúc, không phải thiếu tinh chỉnh: descend-and-close không bao được một
  hộp, và một fixture positive cần grasp prior có pose giải IK và aperture đã
  fit — đúng thứ MVP tạm thời phải xây cho một tay. Một pha "seat" (tiếp tục hạ
  sau khi chạm) đã được đo và làm xấu đi trên cả hai tay, nên không được giữ lại.
- **P3.5-18 — independent review:** `blocked`. Tác giả artifact không được tự ký
  verdict.
- CoACD và ManifoldPlus không được cài và không được thêm làm dependency. Vì vậy
  parity output-class với artifact Stage 0 chưa được kiểm; tham số của profile đã
  pin, các part nó sinh ra thì chưa.

## Sửa đổi phiên trước

Không. Phiên này chỉ thi công `ROADMAP-P3.5-001`; không kết luận nào của P3.3,
P3.4, P3.4.3, ADR-0008 hay `ROADMAP-MVP-001` bị sửa.

## Bàn giao

- Trạng thái: P3.5 `in_progress`, 16/19 package delivered. P3 tổng vẫn `pending`.
- Bước hợp lệ tiếp theo, theo thứ tự phụ thuộc:
  1. chạy `notebooks/phase3_5_rl_readiness.ipynb` trên Kaggle/Colab GPU và lưu
     evidence dưới `evidence/phase3_5/`; đó là đầu vào duy nhất hợp lệ cho
     P3.5-15;
  2. xây grasp prior hai tay để có positive fixture, rồi mới sinh
     `QDGrasp-RL-Env-Tiny` (P3.5-17);
  3. reviewer độc lập cho P3.5-18.
- Điều kiện phiên sau phải kiểm trước: `python scripts/check_phase3_5.py
  --profile micro` phải còn trả `1` với đúng ba package mở; nếu nó trả `0` mà
  chưa có GPU evidence và review thì cổng đã bị nới, không phải phase đã xong.
