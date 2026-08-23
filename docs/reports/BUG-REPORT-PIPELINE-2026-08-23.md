---
document_id: BUG-REPORT-PIPELINE-2026-08-23
document_type: bug_report
title: Báo cáo phân tích lỗi hệ thống Pipeline sinh dữ liệu & Mô phỏng vật lý
status: draft
date: 2026-08-23
author: antigravity-agent
revises: none
severity: critical
affected_modules:
  - qdgrasp/dataset/pipeline/orchestrator.py
  - qdgrasp/dataset/pipeline/proposals/wrench_guided.py
  - qdgrasp/dataset/pipeline/proposals/region_opposition.py
  - qdgrasp/dataset/pipeline/solvers/fixed_contact_dls.py
  - qdgrasp/dataset/pipeline/validators/mujoco_rollout.py
  - qdgrasp/dataset/pipeline/observers/contact_load.py
  - scripts/render_4view_rollout.py
  - scripts/build_kaggle_notebook.py
---

# Báo cáo phân tích lỗi hệ thống — Pipeline sinh dữ liệu & Mô phỏng vật lý

## Tóm tắt

Báo cáo này ghi nhận 5 lỗi nền tảng trong pipeline sinh dữ liệu kẹp vật
(Dataset Generation Pipeline), bao gồm thuật toán khởi tạo tư thế bàn tay,
bộ giải động học ngược, cấu hình actuator MuJoCo, bộ kiểm định và hệ thống
dán nhãn. Tất cả đều ảnh hưởng trực tiếp đến chất lượng mẫu trong
`datasets/dgn-open-tiny/` và kết quả video rollout trên Kaggle.

---

## BUG-01: Hướng cổ tay / Bàn tay đâm xuyên vật thể khi khởi tạo

**Mức độ:** Critical
**Hiện tượng:** Bàn tay khi được sinh ra luôn có cổ tay hướng về phía bàn
hoặc đâm trực diện vào bên trong vật thể. Đặc biệt đối với LEAP Hand, khi
bật mô phỏng tại `t=0`, bàn tay đã nằm lồng vào vật thể, MuJoCo kích hoạt
lực đẩy va chạm cực lớn (penetration rejection force), bắn văng vật thể đi.

**Module ảnh hưởng:**
- `qdgrasp/dataset/pipeline/orchestrator.py` — hàm `_fit_palm_pose` (dòng 27–49)

**Nguyên nhân gốc rễ:**

Hàm `_fit_palm_pose` sử dụng thuật toán Kabsch (Point-set registration) để
khớp 4 điểm đầu ngón tay nguồn (`source_tips`) với 4 điểm tiếp xúc trên vật
thể (`target_tips`):

```
R_palm = V @ U^T    với    U Σ V^T = SVD( Σ (p_tip - p̄) (c - c̄)^T )
p_palm = c̄ - R_palm @ p̄_tip
```

Khuyết điểm:

1. Thuật toán Kabsch chỉ tối ưu vị trí 4 đầu ngón mà hoàn toàn không có
   ràng buộc về hướng không gian của thân bàn tay (Palm Body), cánh tay hay
   mặt phẳng sàn (`z=0`).
2. Do các đầu ngón tay trong hệ quy chiếu gốc đều nằm phía trước palm, khi
   xoay để khớp vào vật thể trên bàn, ma trận xoay `R_palm` có thể lật ngược
   cổ tay cắm xuống đất.
3. Không có bước kiểm tra va chạm thể tích (SDF / GJK) giữa thân bàn tay và
   vật thể trước khi chấp nhận vị trí khởi tạo.

**Bằng chứng:** Video Kaggle Version 12 trở về trước — tất cả 3 bàn tay đều
có hiện tượng này.

---

## BUG-02: Lực kẹp yếu / Không đạt Force Closure

**Mức độ:** Critical
**Hiện tượng:** Wonik Allegro và Shadow Hand tiếp cận vật thể nhưng chỉ chạm
hờ bề mặt, không tạo đủ áp lực tiếp xúc hai phía (bilateral normal force).
Khi nhấc bàn tay lên, vật thể bị trượt và rơi ngay lập tức.

**Module ảnh hưởng:**
- `qdgrasp/dataset/pipeline/proposals/wrench_guided.py` — hàm
  `compute_preliminary_wrench_score` (dòng 6–38)
- `qdgrasp/dataset/pipeline/solvers/fixed_contact_dls.py` (dòng 130–220)

**Nguyên nhân gốc rễ:**

1. **Tiêu chuẩn Wrench Heuristic sai bản chất:**
   `wrench_guided.py` tính điểm kẹp dựa trên tích các giá trị kỳ dị (singular
   values) của ma trận `G ∈ R^{6×K}`:

   ```
   Score = Π σ_i(G) × rank(G) / 6
   ```

   Chỉ số này chỉ là phép xấp xỉ hướng pháp tuyến của `K=4` điểm, hoàn toàn
   không kiểm tra xem nón ma sát (Friction Cone) có bao bọc gốc tọa độ hay
   không — đây là điều kiện tiên quyết của Force Closure theo Ferrari-Canny.

2. **DLS Solver chỉ giải kinematics:**
   Thuật toán DLS IK chỉ cố đưa đầu ngón tay đến đúng tọa độ bề mặt
   (`e_pos → 0`). Nó không có thành phần bù độ sâu tiếp xúc (contact
   penetration offset / virtual spring displacement) để khi servo ở chế độ vị
   trí, ngón tay tạo ra áp lực nén `F_N = K_p × (q_target − q_contact) > mg`.

---

## BUG-03: Rung giật ngón tay cường độ cao (Finger Jitter)

**Mức độ:** High
**Hiện tượng:** Trong quá trình mô phỏng, các khớp ngón tay rung bần bật liên
tục với tần số cao, nhìn rõ trên video rollout.

**Module ảnh hưởng:**
- `qdgrasp/dataset/pipeline/validators/mujoco_rollout.py` — hàm
  `build_rollout_scene_model`
- Mô hình Menagerie XML của Wonik Allegro, LEAP Hand

**Nguyên nhân gốc rễ:**

Actuator của MuJoCo Menagerie sử dụng mô hình servo PD affine:

```
τ = gainprm[0] × (u − q) + biasprm[1] × q + biasprm[2] × q̇
```

Trong cấu hình hiện tại, khi tăng độ cứng lò xo `K_p = 8.0` để kẹp chặt vật
thể nhưng bỏ trống hệ số cản `K_d = 0.0` tại `biasprm[:, 2]`, hệ cơ khí trở
thành dao động điều hòa không tắt dần (undamped harmonic oscillator) kết hợp
với xung lực va chạm rời rạc từ bộ giải LCP, sinh ra rung giật liên tục
(contact chatter).

**Cách xử lý đã kiểm chứng:** Đặt `biasprm[:, 2] = −0.15` (critical damping)
triệt tiêu hoàn toàn rung giật. Đã xác nhận trên Kaggle Kernel Version 14.

---

## BUG-04: Dán nhãn sai PASS dù video và vật lý thất bại hoàn toàn

**Mức độ:** High
**Hiện tượng:** Giao diện HTML và báo cáo hiển thị PASS (badge xanh), nhưng
trong video vật thể trượt khỏi tay và rơi xuống sàn.

**Module ảnh hưởng:**
- `scripts/build_kaggle_notebook.py` — hàm render HTML card
- `qdgrasp/dataset/pipeline/observers/contact_load.py` — dòng 131

**Nguyên nhân gốc rễ:**

1. Mã tạo HTML kiểm tra theo tên kịch bản (`category == 'pass'`) thay vì đọc
   thuộc tính `actual_outcome` hoặc đo độ dịch chuyển trọng lực thực tế
   `Δz = z_final − z_init`.
2. Bộ quan sát `extract_contact_loads` đặt ngưỡng lực tác động `1e-3 N`
   (1 mN), khiến những va quẹt thoáng qua cũng được tính là "Active Finger
   Contact".

**Cách xử lý đã kiểm chứng:** Thay đổi tiêu chí PASS/FAIL dựa trên
`lift_achieved > 0.04m` sau khi hạ sàn xuống `z = −10m` (zero-support gravity
lift). Đã xác nhận trên Kaggle Kernel Version 14.

---

## BUG-05: Nhầm lẫn giữa AI inference loss và thuật toán sinh dữ liệu

**Mức độ:** Medium (gây hiểu lầm, không ảnh hưởng logic)
**Hiện tượng:** Notebook Kaggle in ra `Loss = 1.225`, `Train Step` khiến
người dùng tưởng quy trình đang huấn luyện mô hình AI trong khi chưa yêu cầu.

**Module ảnh hưởng:**
- `kaggle-phase1/qdgrasp-phase-1-cuda-framework-gate.ipynb`

**Nguyên nhân:**

Kernel Kaggle chạy một mạng nơ-ron benchmark mẫu (Dummy Regression Network)
song song để kiểm tra cổng phần cứng CUDA FP32/AMP Gate. Điều này gây hiểu
lầm rằng pipeline tạo dữ liệu phụ thuộc vào AI chưa được huấn luyện, trong khi
bản chất DexGraspNet/QD-Grasp là thuật toán tối ưu hình học và mô phỏng vật lý
giải tích (Analytical Optimization & Physics Simulation).

---

## Kế hoạch khắc phục

| STT | Bug | Giải pháp | Module cần sửa | Trạng thái |
|---|---|---|---|---|
| 1 | BUG-01 Hướng bàn tay | Spherical Standoff Sampling + SDF collision check | `orchestrator.py`, `proposals/` | Chưa sửa |
| 2 | BUG-02 Lực kẹp yếu | Antipodal Opposition + contact penetration offset | `solvers/fixed_contact_dls.py`, `proposals/region_opposition.py` | Chưa sửa |
| 3 | BUG-03 Rung giật | Critical damping `K_d = −0.15` | `validators/mujoco_rollout.py` | **Đã sửa (v14)** |
| 4 | BUG-04 Nhãn sai | Dùng `lift_achieved > 0.04m` + zero-support | `validators/mujoco_rollout.py`, `build_kaggle_notebook.py` | **Đã sửa (v14)** |
| 5 | BUG-05 Hiểu lầm loss | Tách rõ CUDA gate benchmark khỏi pipeline output | `kaggle-phase1/` notebook | Chưa sửa |

## Bằng chứng

| Evidence ID | Mô tả | Nguồn |
|---|---|---|
| E-01 | Video rollout Kaggle v12 — bàn tay đâm xuyên vật thể, vật bị bắn đi | Kaggle Kernel v12 output |
| E-02 | Video rollout Kaggle v12 — ngón tay rung giật liên tục | Kaggle Kernel v12 output |
| E-03 | Video manifest v12 — nhãn PASS nhưng lift = 0.0m | `video_manifest.json` Kernel v12 |
| E-04 | Video rollout Kaggle v14 — BUG-03 và BUG-04 đã sửa | Kaggle Kernel v14 output |
| E-05 | `video_manifest.json` v14 — 3/4 pass có lift > 10cm, fail có lift < 0 | Kaggle Kernel v14 output |

## Kết luận

Đã xác nhận 5 lỗi nền tảng. BUG-03 (rung giật) và BUG-04 (nhãn sai) đã được
khắc phục và kiểm chứng trên Kaggle Kernel Version 14. BUG-01 (hướng bàn tay),
BUG-02 (lực kẹp yếu) và BUG-05 (hiểu lầm loss) cần được xử lý trong các
phiên tiếp theo bằng cách tái cấu trúc lõi pipeline.
