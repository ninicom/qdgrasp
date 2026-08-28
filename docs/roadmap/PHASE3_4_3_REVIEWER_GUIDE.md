---
document_id: ROADMAP-P3.4.3-REVIEW-GUIDE
document_type: reviewer_guide
title: Hướng dẫn cho reviewer độc lập Phase 3.4.3
version: 1.0.0
status: active
date: 2026-08-28
parent_plan: ROADMAP-P3.4.3-001
unblocks: [G10, G11, C08, P3.4-17, R-DOD-08]
---

# Hướng dẫn cho reviewer độc lập

Bạn là điều kiện duy nhất còn lại của năm requirement. Tài liệu này để bạn ra
verdict nhanh nhất mà không phải tin lời ai.

## 0. Điều kiện tiên quyết

Bạn **không được** là người viết phần lớn patch hoặc tạo evidence quyết định.
Tác giả của toàn bộ thay đổi này là `claude-implementation-agent`. Nếu bạn là
tác nhân đó, bạn không đủ tư cách — đây không phải thủ tục hình thức, nó là lý
do tồn tại của gate này.

## 1. Lấy packet

```bash
python scripts/phase3_4_3_review_packet.py \
  --cuda-evidence evidence/phase3_4_3/s10/kaggle-run-v8/cuda-gate.json
```

Packet **phải** dựng trên cây sạch. Nếu worktree bẩn, script tự từ chối và nói
rõ nó không phải release candidate. Ghi lại `packet_sha256` — verdict của bạn
phải trỏ đúng hash đó.

Packet không mang verdict, và cố ý như vậy.

## 2. Bảy mục checklist

Packet liệt kê bảy mục. Với mỗi mục, thứ cần kiểm và chỗ tự kiểm được:

| # | Kiểm gì | Tự chạy |
|---|---|---|
| 1 | Số học lực và xung lực, gồm hai đầu cửa sổ trượt | `pytest tests/contactrich_active -k safety` |
| 2 | Không giới hạn an toàn nào thiếu sensor | `qdgrasp/dynamic/safety_budget.py`, `require_full_coverage` |
| 3 | Lineage replay GPU→CPU, từng capsule | `evidence/phase3_4_3/s10/kaggle-run-v8/` |
| 4 | Manifest counts/splits/shard hash so với đĩa | `python scripts/check_contactrich_active.py` |
| 5 | Công bố phạm vi ADR-0008 ở mọi nơi xuất hiện | `qdgrasp/roadmap/scope_audit.py` |
| 6 | Exit code của checker dưới mutation | `pytest tests/contactrich_active/test_completeness_ledger.py` |
| 7 | Hạn chế đã khai báo có khớp thực tế đo | mục 3 dưới đây |

## 3. Bốn chỗ đáng nghi nhất — hãy tấn công vào đây trước

Đây là những chỗ tôi cho là dễ sai nhất trong toàn bộ công việc. Nếu bạn chỉ có
thời gian cho một phần, hãy dùng cho phần này.

**a. Verdict CUDA là FAIL, và tốc độ thì pass.** Kiểm rằng không có đường nào
cho phép tốc độ mua lại đúng đắn. Xem `scripts/check_phase3_4_3_cuda.py`, đặc
biệt `read_sanitizer_report` và chỗ sanitizer trở thành tiêu chí gate — trước
v7 nó chạy ở cell riêng và **gate có thể trả PASS trong khi initcheck báo 68k
lỗi**. Xác nhận lỗ đó đã đóng thật.

**b. Quy tắc bằng chứng dương.** Hai lần một probe chết in ra `ERROR SUMMARY: 0
errors` — giống hệt một run sạch. Kiểm rằng `probe_did_not_run` và
`inconclusive_no_error_summary` không thể bị đọc nhầm thành `clean` ở bất kỳ
đâu. Đây là chỗ suýt tạo ra hai kết luận sai ngược hẳn sự thật.

**c. Ngưỡng margin của §16.3 có bị chỉnh cho ra kết quả không.** Đây là chỗ dễ
gian nhất trong toàn bộ hồ sơ. τ phải là chuẩn của `perturbation_wrench` mà
recipe khai báo, không phải một số chọn tay. Kiểm `declared_disturbance` trong
`scripts/phase3_4_3_ablation.py` và đối chiếu với
`qdgrasp/scenes/release_recipes.py`. Nếu τ đến từ đâu khác, verdict phải là fail.

**d. Mặc định `quality_margin_threshold=0.0`.** Toàn bộ 65 requirement đã passed
dựa vào hành vi cũ của certifier. Kiểm rằng mặc định thật sự giữ nguyên hành vi
đó, chứ không phải chỉ trên giấy.

## 4. Những gì hồ sơ **không** claim

Đừng mất thời gian xác minh những thứ này — chúng đã được khai là không đạt:

- GPU parity, sanitizer, non-finite worlds: **failed, đã đo** trên T4
- Cặp §16.3: **1/2 tay**; LEAP không có vì recipe không khai báo nhiễu loạn
- MPPI (`P3.4-10`): deferred, không claim coverage
- Three-hand: paused theo ADR-0008
- MJX fallback: chưa kết luận được, probe timeout ở 1800 s

## 5. Ghi verdict

Một file riêng, với `reviewer`, `reviewer_verdict`, `open_findings`,
`packet_sha256`. `scripts/check_phase3_4_3.py` đọc file đó.

`PASS` chỉ khi zero finding mức S0–S3 còn mở. Verdict của bạn không biến 14
requirement `failed` thành passed — nó chỉ đóng phần review. Phase vẫn `blocked`
cho tới khi mục 11 và 14 được giải quyết.
