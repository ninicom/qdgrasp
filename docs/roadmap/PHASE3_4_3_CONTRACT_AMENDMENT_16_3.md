---
document_id: ROADMAP-P3.4.3-AMEND-16.3
document_type: amendment_proposal
title: Đề xuất sửa §16.3 — tiêu chí paired static-fail/dynamic-pass không thoả được với đặc tả predicate hiện tại
version: 1.1.0
status: accepted
date: 2026-08-28
parent_plan: ROADMAP-P3.4.3-001
blocks: [R-DOD-03, C05]
decision_owner: contract-owner
---

# Đề xuất sửa §16.3

## 1. Vấn đề

§16.3 của `ROADMAP-P3.4-001` yêu cầu bằng chứng ghép đôi: một grasp mà phép thử
**tĩnh** từ chối còn phép thử **động** chấp nhận. Ý đồ rõ ràng và đúng — chứng
minh rằng mô phỏng động mở ra được vùng grasp mà giả định vật đóng băng bỏ sót.

Với đặc tả hiện tại của hai predicate, cặp đó **không thể tồn tại**.

## 2. Chứng minh

Hai predicate dùng chung một sàn:

| | ràng buộc | nguồn |
|---|---|---|
| tĩnh | `certify_force_closure` cần **≥2** contact | `qdgrasp/dataset/pipeline/certifiers/contact_force.py` |
| động | `active_contact_sustained` cần `min_active_fingers=2` duy trì qua perturbation window | `qdgrasp/dataset/pipeline/validators/mujoco_rollout.py:914` |

Grasp nào quá thưa để qua phép thử tĩnh thì cũng quá thưa để qua phép thử động,
theo **đúng cùng một phép đếm**. Nên hai predicate không thể bất đồng theo chiều
§16.3 giả định — chỉ có thể bất đồng theo chiều ngược lại, và thực tế đo được
đúng như vậy: 4 case static-pass/dynamic-fail, 0 case ngược lại.

Predicate động còn một ràng buộc thứ hai làm chặt thêm:
`floor_support_after_lift` phải `False` — vật phải rời giá đỡ. Điều này loại trừ
**đúng** lớp grasp được môi trường đỡ, tức lớp duy nhất khiến một phép thử force
closure đóng băng thất bại một cách có ý nghĩa vật lý.

## 3. Bằng chứng đo

Câu hỏi đã được đặt bốn cách độc lập, tất cả trong
`evidence/phase3_4_3/s11/static-vs-dynamic.json`:

1. contact theo recipe dự kiến — 0 cặp
2. quét khối lượng mười điểm — 0 cặp
3. force closure trên contact **đo được** ở enclosure cuối — 0 cặp
4. scene môi trường-đỡ dựng có chủ đích, tường cạnh target, **8 hình học khai
   báo × 2 tay**, báo cáo toàn bộ điểm — 0 cặp

Nhánh 4 là nhánh quyết định: nó dựng đúng trường hợp sách giáo khoa cho cặp này,
và mọi điểm đều fail **cả hai** nhánh, không bao giờ chỉ một.

## 4. Ba phương án

| | Sửa gì | Đánh đổi |
|---|---|---|
| **A** | Phép thử tĩnh fail vì lý do khác số contact — ví dụ **wrench-space margin** dưới ngưỡng, thay vì đếm contact | Giữ nguyên predicate động; cần định nghĩa và hiệu chuẩn margin mới |
| **B** | Predicate động chấp nhận thành công **có môi trường đỡ** (bỏ `floor_support_after_lift`, hoặc tách thành lớp outcome riêng) | Mở đúng lớp grasp §16.3 muốn; nhưng nới định nghĩa "nhấc được", cần xem lại mọi kết quả dựa trên nó |
| **C** | Rút §16.3, ghi nhận tiêu chí không thoả được và không claim | Trung thực và rẻ nhất; mất một tiêu chí đã có ý đồ đúng |

Khuyến nghị: **A**. Nó giữ nguyên ngữ nghĩa "nhấc được" của predicate động — thứ
mọi bằng chứng khác đang dựa vào — và nhắm đúng chỗ hỏng: phép thử tĩnh hiện fail
vì *đếm*, chứ không vì *cơ học*. Một wrench-space margin làm phép thử tĩnh bảo
thủ đúng nghĩa, và khi đó bất đồng theo chiều §16.3 trở nên khả dĩ.

## 5. Việc này không tự làm được

Cả ba phương án đều sửa contract, không sửa implementation.
`ROADMAP-P3.4.3-001` không trao quyền đó: plan được phép đo tiêu chí, không
được phép viết lại tiêu chí mình không đạt. Nếu tự sửa thì mất đúng cái tính
chất khiến ledger có giá trị.

Vì vậy `R-DOD-03` và `C05` giữ `failed` cho tới khi chủ contract chọn một phương
án. Đây là một quyết định, không phải một hạng mục tồn đọng.


## 6. Quyết định và kết quả đo (2026-08-28)

Chủ contract chọn **phương án A**. Đã cài đặt và đo.

`certify_force_closure` nhận thêm `quality_margin_threshold`, mặc định `0.0` —
tức hành vi lịch sử **y nguyên**, nên 65 requirement đã passed và toàn bộ bằng
chứng thu trước sửa đổi vẫn còn hiệu lực (969 test xanh sau thay đổi).

Ngưỡng **không được chọn** mà **suy ra**: τ = chuẩn của perturbation wrench mà
protocol động thực sự áp lên chính bàn tay đó. Phân tích đóng băng vì thế bị hỏi
đúng câu hỏi có nghĩa — grasp này có chịu nổi nhiễu loạn nó sắp gặp không —
thay vì câu hỏi đếm ngón.

| tay | τ | GWS margin | tĩnh | động | cặp |
|---|---|---|---|---|---|
| `leap_hand` | 0.1388 | 0.0530 | **fail** | pass | ✅ |
| `wonik_allegro` | 0.2128 | 0.1294 | **fail** | pass | ✅ |

Ablation đổi verdict từ `no_measured_difference` sang **`dynamic_admits_more`**,
với **8 cặp** static-fail/dynamic-pass trên **2/2 active hand**. Tiêu chí §16.3
vừa có nghĩa vừa **đạt**. `R-DOD-03` và `C05` chuyển sang `passed`.

## 7. Một khẳng định sai đã được sửa

Bản 1.1.0 của tài liệu này viết rằng `leap_hand` không sinh ra cặp vì recipe của
nó không khai báo `perturbation_wrench`, và rằng do đó tiêu chí
`disturbance_survival_pass` của LEAP là "rỗng". **Cả hai đều sai.**

`validators/mujoco_rollout.py` **tự suy ra** một perturbation wrench khi recipe
không khai báo — lực bằng `0.5 × trọng lượng vật`, mô-men bằng
`0.25 × trọng lượng × kích thước đặc trưng`. LEAP luôn bị nhiễu loạn; tiêu chí
của nó chưa bao giờ rỗng.

Lỗi nằm ở hàm `declared_disturbance` của ablation: nó chỉ đọc `rollout_kwargs`
nên chấm LEAP thành τ = 0 và báo rằng phép thử tĩnh đã pass, trong khi protocol
vẫn đang nhiễu loạn nó suốt. Đây là bug của phép đo, không phải lỗ hổng của
recipe, và **không** cần quyết định nào về contract.

Hàm đã sửa để phản chiếu đúng công thức của validator, và
`test_every_active_hand_faces_a_real_disturbance` giữ cho sai lầm này chỉ xảy ra
một lần.
