---
document_id: TPR-20260822-002
document_type: third_party_review
title: Delta audit lần hai cho train-argument registry schema v2
status: complete
date: 2026-08-22
revises: none
report_id: TPR-20260822-002
reviewed_session: SESSION-20260822-001
reviewed_revision: 363a671734ea4d2c89d399cec000fd97d4750870
reviewer: codex-agent-third-party-train-args
reviewer_organization: codex-multi-agent-internal-review
review_date: 2026-08-22
independence: internal_independent
verdict: fail
max_severity: S2
---

# TPR-20260822-002 — Delta audit schema v2

## 1. Đối tượng và phạm vi

Reviewer read-only kiểm commit
`363a671734ea4d2c89d399cec000fd97d4750870`, tree
`dfb454f1d0fb5af3252a423ff3a563d57131a2a6`; implementation parent
`8dc0e2e0a122e46df95f2f4f7a3cfbfee9dec58c`. Phạm vi là F-001–F-010,
typed fingerprint, Markdown contract, source cleanliness và evidence. Không kiểm
model/runtime/metric/CPU-CUDA vì các phần đó chưa tồn tại.

## 2. Tuyên bố độc lập

Mức độc lập `internal_independent`. Reviewer không viết implementation, không
sửa/commit project hoặc clone nguồn. Tất cả mutation dùng bộ nhớ hoặc clone tạm;
project, Ultralytics và DGN2 worktree vẫn sạch. Không có xung đột lợi ích được
khai báo.

## 3. Môi trường và phương pháp

Reviewer replay baseline, liệt kê 968 scalar leaves, mutation từng contract và
thử source drift trong isolated clone exact origin/HEAD. Ngoài dirty tracked/
untracked thông thường, audit đặt Git index bits `assume-unchanged` và
`skip-worktree`. Front matter được đột biến rồi chạy cả train checker và generic
documentation checker. Không dùng dataset, model, GPU hoặc seed.

## 4. Bằng chứng

| Evidence | Cấp | Artifact | Kích thước | SHA-256 | Tạo bởi |
|---|---|---|---:|---|---|
| E-001 | E2 | `docs/reviews/evidence/TPR-20260822-002-delta-checks.txt` | 4.172 byte | `adc4a5ac12f605be121a1f0e30cbb3ce7ff3ac0b397bbd318d39f0622fc2bcb3` | independent reviewer |
| E-002 | E1 | `docs/reports/evidence/TRAIN-ARGS-20260822-dialect-v2.txt` | 4.725 byte | `49b29aa755890422863cc2821a293abed612a4981e93d384ac112edc480cdda1` | implementation replay |

## 5. Kết quả đối chiếu

| Finding | Trạng thái | Kết quả | Evidence |
|---|---|---|---|
| F-001 | `resolved` | Quote/spacing/tab/trailing/flow corruption đều reject | E-001 |
| F-002 | `resolved` phạm vi gốc | 967/968 leaves ảnh hưởng typed fingerprint; self-reference được guard riêng | E-001 |
| F-003 | `resolved` | Hai nhánh merge và authority đúng; prose mutation reject | E-001 |
| F-004 | `resolved` | Nullable/range đúng; 11/11 quantize aliases khớp AST source | E-001 |
| F-005 | `partial/open` S2 | Coverage 30 tests nhưng thiếu hai index-flag bypass | E-001 |
| F-006 | `resolved` cho artifact hiện tại | Chain/version/hash nhất quán; enforcement gap tách thành F-011 | E-001 |
| F-007 | `resolved` | Commit/tree/evidence/reference hashes replay đúng | E-001, E-002 |
| F-008 | `resolved` | 115 và 127 disposition totals đúng | E-001 |
| F-009 | `resolved` wiring | Hooks/gates gọi full checks; implementation cleanliness còn F-010 | E-001 |
| F-010 | `partial/open` S2 | Dirt thường reject; index flags che mutation và checker vẫn pass | E-001 |

## 6. Phát hiện

### F-011 — Front matter quản trị không được khóa

- Severity: `S2`; trạng thái: `open`.
- Quan sát: bỏ/đổi `latest_revision_record`, đặt `date: never`, đổi title hoặc
  thêm `runtime_implemented: True` đều làm cả hai validator trả zero problem.
- Tác động: pointer giả vi phạm living-contract policy và unknown claim có thể
  mâu thuẫn giới hạn “chưa có runtime”.
- Điều kiện đóng: exact allowed schema/values, ISO date, pointer bắt buộc tồn tại
  và nhận diện current revision, unknown metadata bị reject, có combined tests.

F-010 còn mở vì `git status` tôn trọng `assume-unchanged` (`h`) và
`skip-worktree` (`S`). Trong isolated pinned clone, sửa
`ultralytics/data/augment.py` với mỗi flag giữ status rỗng và full checker exit 0.

## 7. Điều kiện còn lại

Không conditional acceptance. Phải đóng F-010/F-011, bổ sung regression cho cả
hai Git flags và front-matter probes, rồi delta review exact commit mới. Candidate
này không được merge/release.

## 8. Kết luận

- Verdict: `fail`.
- Severity cao nhất còn mở: `S2`; không còn S0/S1.
- Baseline reference/registry/docs/30 tests/shell/diff đều pass nhưng không phủ
  hai bypass nên không đủ acceptance.
- Inventory 127 tên, default/group/type sets và quantize map được xác nhận đúng.
- Verdict chỉ áp dụng cho `363a671`; model/runtime nằm ngoài kết luận.

## 9. Chữ ký

- Reviewer: `codex-agent-third-party-train-args`.
- Tổ chức: `codex-multi-agent-internal-review`.
- Hoàn tất: `2026-08-22T00:33:47+07:00`, Asia/Bangkok.
- Định danh: `TPR-20260822-002`; không có file nào được reviewer sửa/commit.
