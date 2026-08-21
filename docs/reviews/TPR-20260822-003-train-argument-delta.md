---
document_id: TPR-20260822-003
document_type: third_party_review
title: Delta audit lần ba cho source và metadata hardening
status: complete
date: 2026-08-22
revises: none
report_id: TPR-20260822-003
reviewed_session: SESSION-20260822-002
reviewed_revision: 42914f550929789fe5d0df45bafedaa0b9ae469b
reviewer: codex-agent-third-party-train-args/delta-checker
reviewer_organization: codex-multi-agent-internal-review
review_date: 2026-08-22
independence: internal_independent
verdict: fail
max_severity: S2
---

# TPR-20260822-003 — Delta audit source/metadata hardening

## 1. Đối tượng và phạm vi

Reviewer read-only kiểm commit
`42914f550929789fe5d0df45bafedaa0b9ae469b`, tree
`e6186cdcbfbe4bcf71571bbafbf33982118c4784`; implementation parent
`62111ed4b0c31e85e47da02c33831f2cf5a32894`, tree
`a52d4782654d8b5c5b2228392965beff1b4da817`. Phạm vi là toàn bộ
F-001–F-011, đặc biệt source cleanliness và exact front-matter/revision
identity. Model/runtime/CPU-CUDA, dataset, export, simulator và benchmark nằm
ngoài phạm vi vì chưa tồn tại.

## 2. Tuyên bố độc lập

Mức độc lập `internal_independent`. Reviewer không viết implementation, không
sửa/commit project hoặc hai clone nguồn. Mutation được thực hiện trong clone
tạm hoặc artifact tạm. Đây không phải review external hoặc human.

## 3. Môi trường và phương pháp

Reviewer xác minh commit/tree/hash trước, replay positive gates, rồi thử mutation
registry/Markdown và source checkout trong bản sao disposable. Source probes bao
gồm tracked, untracked, `assume-unchanged`, `skip-worktree` và
`fsmonitor-valid`; metadata probes bao gồm mọi field quản trị và identity của
revision pointer. Không dùng model, dataset, GPU hoặc seed.

## 4. Bằng chứng

| Evidence | Cấp | Artifact | Kích thước | SHA-256 | Tạo bởi |
|---|---|---|---:|---|---|
| E-001 | E2 | `docs/reviews/evidence/TPR-20260822-003-delta-checks.txt` | 4.703 byte | `05c992a54e1bec6b6f9ee0022ebab5d9ce5b7e7507df91e5567511fbcffe3095` | independent reviewer; primary materialized verbatim results |
| E-002 | E1 | `docs/reports/evidence/TRAIN-ARGS-20260822-source-metadata-hardening.txt` | 5.540 byte | `15b540f7dbf4a1be3b173882d02488e93b7638915ceba82cf4e58f67ecd1de17` | implementation replay |

## 5. Kết quả đối chiếu

| Finding | Trạng thái | Kết quả | Evidence |
|---|---|---|---|
| F-001–F-004 | `resolved` | Dialect, fingerprint, body và quantize map đều đóng | E-001 |
| F-005 | `partial/open` S2 | 35 tests pass nhưng chưa có hai nhóm regression mới | E-001 |
| F-006 | `resolved` cho artifact hiện tại | Enforcement residual nằm ở F-011 | E-001 |
| F-007–F-009 | `resolved` | Provenance/count/gate wiring đúng | E-001, E-002 |
| F-010 | `open` S2 | h/S và dirt thường reject; `fsmonitor-valid` vẫn pass | E-001 |
| F-011 | `open` S2 | Schema đúng nhưng toàn bộ exact scalar vẫn bị normalize | E-001 |

Positive baseline xác nhận 115 canonical + 2 extra + 9 legacy + 1 API = 127
tên; 39 tài liệu và 35/35 tests pass; 968 fingerprint leaves không có leaf
ngoài contract, trừ self-reference có chủ đích.

## 6. Phát hiện

### F-010 — `fsmonitor-valid` che source drift

- Severity: `S2`; trạng thái: `open`.
- Quan sát: với file tracked đã đổi, porcelain rỗng; `ls-files -v` vẫn là `H`,
  trong khi `ls-files -f` là `h`; full checker trả `0` thay vì `1`.
- Tác động: standalone source-clean claim có thể sai dù origin/HEAD đúng.
- Điều kiện đóng: kiểm cả hai view `-v`/`-f`, chỉ cho phép `H`, và thêm
  regression `fsmonitor-valid`.

### F-011 — Exact scalar bị thay bằng giá trị đã chuẩn hóa

- Severity: `S2`; trạng thái: `open`.
- Quan sát: đổi từng field trong 11 field từ `value` thành `" value "` vẫn làm
  train checker và generic checker trả `0`; cả năm identity/status field của
  revision target có cùng hành vi.
- Tác động: YAML semantic value và đường dẫn thực tế có thể khác với giá trị mà
  checker đem so, phá tuyên bố exact contract.
- Điều kiện đóng: so raw lexeme, reject quote/padding/separator không canonical
  cho toàn bộ 11 + 5 field, có regression exhaustive.

## 7. Điều kiện còn lại

Không conditional acceptance. Phải đóng F-010/F-011 và phần còn lại của F-005,
khóa evidence trên commit mới rồi review lại. Snapshot này không được
merge/release.

## 8. Kết luận

- Verdict: `fail`.
- Severity cao nhất còn mở: `S2`; không có S0/S1.
- Inventory và phần contract hiện hữu đúng, nhưng hai bypass khiến acceptance
  guarantee chưa đúng với tuyên bố.
- Verdict chỉ áp dụng cho `42914f5`; không đưa ra kết luận model/runtime.

## 9. Chữ ký

- Reviewer: `codex-agent-third-party-train-args/delta-checker`.
- Tổ chức: `codex-multi-agent-internal-review`.
- Ngày: 2026-08-22, Asia/Bangkok.
- Định danh: `TPR-20260822-003`; reviewer không sửa/commit file nào.
