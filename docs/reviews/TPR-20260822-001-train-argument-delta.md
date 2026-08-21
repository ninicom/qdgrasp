---
document_id: TPR-20260822-001
document_type: third_party_review
title: Delta audit lần một cho remediation train-argument registry
status: complete
date: 2026-08-22
revises: none
report_id: TPR-20260822-001
reviewed_session: SESSION-20260821-007
reviewed_revision: fdb60eb51e4793ee5764b272acc3a40960608846
reviewer: codex-agent-third-party-train-args
reviewer_organization: codex-multi-agent-internal-review
review_date: 2026-08-22
independence: internal_independent
verdict: fail
max_severity: S1
---

# TPR-20260822-001 — Delta audit remediation train arguments

## 1. Đối tượng và phạm vi

Reviewer kiểm read-only commit
`fdb60eb51e4793ee5764b272acc3a40960608846`, tree
`2b26aed81b7f8647e85d2bc64b2a6ecf2da979f5`, đối chiếu từng F-001–F-009
của `TPR-20260821-003`. Phạm vi gồm parser/fingerprint registry, Markdown
contract, source/reference gates, governance và replay evidence. Model/runtime,
metric grasp và CPU/CUDA smoke nằm ngoài phạm vi vì chưa được triển khai.

## 2. Tuyên bố độc lập

Mức độc lập là `internal_independent`. Reviewer không viết/sửa artifact, không
commit và không tham gia implementation của revision được kiểm. Không có xung
đột tài chính hoặc quản lý được khai báo. Một sub-audit độc lập trong cùng nhóm
xác nhận lại verdict và mutation probes.

## 3. Môi trường và phương pháp

Đây là static/source audit trên checkout local có hai clone đã pin. Reviewer chạy
baseline checker/test, sau đó tạo bản registry/Markdown đột biến trong bộ nhớ
hoặc vùng tạm, so kết quả checker với semantic YAML, và thử full-source trên
checkout giữ nguyên HEAD nhưng có tracked-file drift. Không dùng dataset, model,
GPU, seed hoặc benchmark.

## 4. Bằng chứng

| Evidence | Cấp | Artifact | Kích thước | SHA-256 | Tạo bởi |
|---|---|---|---:|---|---|
| E-001 | E2 | `docs/reviews/evidence/TPR-20260822-001-delta-checks.txt` | 3.088 byte | `739837d1dd3f559594cb68a99ae8111311d1b53e4755907f6ba5dd606693a29c` | independent reviewer observations |
| E-002 | E1 | `docs/reports/evidence/TRAIN-ARGS-20260821-audit-remediation.txt` | 4.178 byte | `fd7888018bfa8e691dd8a6e236ffbca9fbfb6e864f483ed97522abec0ced037f` | implementation gate |

## 5. Kết quả đối chiếu

| Finding gốc | Trạng thái | Kết quả delta | Evidence |
|---|---|---|---|
| F-001 | `open` S1 | Quote, tab và thiếu space vẫn bị parser chuẩn hóa/nhận sai | E-001 |
| F-002 | `open` S1 | Typed scalar và nhiều nội dung Markdown quy phạm vẫn drift mà pass | E-001 |
| F-003 | `resolved` | Hai nhánh merge `cfg=`/không `cfg=` đã được ghi | E-001 |
| F-004 | `partial/open` S2 | Nullable/range tốt hơn nhưng thiếu structured quantize alias và prose pin | E-001 |
| F-005 | `partial/open` S2 | Test tăng nhưng registry/full guarantee vẫn cao hơn checker thực | E-001 |
| F-006 | `resolved` | Living-contract và revision schema 2 đã nhất quán | E-001 |
| F-007 | `resolved` | Exact remediation commit, replay log, reference/license evidence đã pin | E-001, E-002 |
| F-008 | `resolved` | Tổng 115 canonical và 127 public đã tách rõ | E-001 |
| F-009 | `resolved` | Combined hooks/Git gates chạy full khi clone hiện diện | E-001 |

## 6. Phát hiện

### F-010 — Standalone full-source không bắt checkout sạch

- Severity: `S2`.
- Trạng thái: `open`.
- Quan sát: thay tracked `ultralytics/data/augment.py` trong checkout vẫn giữ
  pinned HEAD làm `check_train_args.py --source ...` pass khi chạy độc lập.
- Tác động: standalone claim không bảo đảm toàn bộ source tree đúng revision;
  combined project gate hiện được `check_references.py` giảm thiểu.
- Điều kiện đóng: full-source tự kiểm clean worktree/origin/HEAD và có regression.

Các reproduction quyết định verdict gồm `100 → "100"`, `True → "True"`,
`null → "null"`, quoted schema version, `default:100`, `default:<TAB>100`, thay
type `epochs`/`quantize`, đổi behavior legacy `half`, làm sai prose `scale` và
đảo merge-order `cfg`. Tất cả đều pass ngoài dự kiến tại reviewed revision.

## 7. Điều kiện còn lại

Không có conditional acceptance. Phải đóng toàn bộ F-001/F-002/F-004/F-005 và
F-010, thêm regression tương ứng, rồi thực hiện delta review mới trên exact clean
commit. Không được merge/release snapshot này.

## 8. Kết luận

- Verdict: `fail`.
- Severity cao nhất còn mở: `S1`.
- Cơ sở: checker vẫn có false-negative làm thay đổi kiểu/cấu trúc registry và
  nội dung quy phạm, dù baseline 24/24 test và mọi gate hiện có đều xanh.
- Giới hạn: không kết luận model/runtime/metric; inventory 127 tên hiện tại
  không bị chứng minh sai.
- Re-review: bắt buộc trên revision mới; verdict chỉ áp dụng cho `fdb60eb`.

## 9. Chữ ký

- Người kiểm tra: `codex-agent-third-party-train-args`.
- Tổ chức: `codex-multi-agent-internal-review`.
- Ngày hoàn tất: `2026-08-22T00:14:10+07:00`, Asia/Bangkok.
- Định danh: `TPR-20260822-001`; reviewer xác nhận không sửa file.
