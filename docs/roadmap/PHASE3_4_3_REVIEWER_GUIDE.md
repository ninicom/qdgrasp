---
document_id: ROADMAP-P3.4.3-REVIEW-GUIDE
document_type: reviewer_guide
title: Hướng dẫn cho reviewer độc lập Phase 3.4.3
version: 2.0.0
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

Một lượt static review đã chạy trước bạn và tìm ra sáu finding (`REV-20260828-016`),
trong đó hai cái là lỗi suy luận của tác giả chứ không phải lỗi gõ. Bốn chỗ dưới
đây là nơi tôi cho là còn dễ sai nhất **sau** khi remediation xong.

**a. Verdict CUDA giờ được tính lại, không phải đọc.** `recompute_cuda_verdict`
trong `scripts/check_phase3_4_3.py` dựng verdict từ metric; trường `verdict` khai
báo chỉ dùng để **đối chiếu**. Hãy kiểm rằng không còn đường nào để một bundle tự
khẳng định pass — đây từng là RRV-01, và trước v7 sanitizer chạy ở cell riêng nên
gate có thể trả PASS trong khi initcheck báo 68k lỗi.

**b. Quy tắc bằng chứng dương.** Hai lần một probe chết in ra
`ERROR SUMMARY: 0 errors` — giống hệt một run sạch. Kiểm rằng
`probe_did_not_run` và `inconclusive_no_error_summary` không thể bị đọc thành
`clean` ở bất kỳ đâu.

**c. `alpha` có bị chỉnh cho ra kết quả không.** Đây vẫn là chỗ dễ gian nhất.
`alpha` phải **không thứ nguyên**: nghiệm của bài toán resistance trong
`qdgrasp/dataset/pipeline/certifiers/static_resistance.py`, dưới trần lực lấy từ
safety budget, với nhiễu loạn lấy từ `resolve_perturbation_wrench` dùng chung với
validator. Kiểm ba tính chất: bất biến khi đổi đơn vị chiều dài, đơn điệu theo
force limit, tỉ lệ nghịch với nhiễu loạn. Nếu thiếu trần lực mà vẫn trả số thì
verdict phải là fail — LP không chặn sẽ siết tuỳ thích và chứng nhận bất cứ thứ gì.

**d. Hai nhánh có thật sự cùng một snapshot không.** `CandidateSnapshot.fork`
phải cho `one_factor_diff == ('physics_mode',)` ở **mọi** điểm sweep, kể cả khi
mass thay đổi. RRV-04 chính là chỗ này: sweep đổi mass động mà giữ ngưỡng tĩnh ở
mass gốc, nên từ điểm thứ hai hai nhánh mô tả hai thí nghiệm khác nhau.

## 4. Những gì hồ sơ **không** claim

Đừng mất thời gian xác minh — đã khai là không đạt:

- GPU parity, sanitizer, non-finite worlds: **failed, đã đo** trên T4
- Cặp §16.3: **0 cặp trên 2/2 tay** dưới tiêu chí resistance; `C05` và
  `R-DOD-03` giữ `failed`. Claim "8 cặp" của bản trước **đã bị thu hồi** vì so
  sánh hai đại lượng không cùng thứ nguyên (RRV-03).
- MPPI (`P3.4-10`): deferred, không claim coverage
- Three-hand: paused theo ADR-0008
- MJX fallback: chưa kết luận, probe timeout ở 1800 s

Ledger hiện tại: 65 passed, 14 failed, 5 blocked, 1 deferred, 0 pending —
`manifest b56deb1efa39`.

## 5. Ghi verdict

Một file riêng, với `reviewer`, `reviewer_verdict`, `open_findings`,
`packet_sha256`. `scripts/check_phase3_4_3.py` đọc file đó.

`PASS` chỉ khi zero finding mức S0–S3 còn mở. Verdict của bạn không biến 14
requirement `failed` thành passed — nó chỉ đóng phần review. Phase vẫn `blocked`
cho tới khi mục 11 và 14 được giải quyết.
