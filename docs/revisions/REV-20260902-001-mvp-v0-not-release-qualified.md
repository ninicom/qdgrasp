---
document_id: REV-20260902-001
document_type: revision_record
revision_schema: 2
title: MVP v0 không đủ điều kiện phát hành và bị tách khỏi contract release v1
status: complete
date: 2026-09-02
record_id: REV-20260902-001
session_id: SESSION-20260902-002
created_at: 2026-09-02T14:00:00+07:00
author: claude-agent
revises:
  - session_id: SESSION-20260831-001
    artifact: docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md
    revision: 90b3accb76ae1417199167661c58cd937b00d9b9
  - session_id: ROADMAP-MVP-001
    artifact: configs/mvp/dexacquire-mvp-v0.yaml
    revision: a897e36084c9ab11cbf8046e446ef318c48b7463d8b2db4af4d4ca0593109b8b
  - session_id: SESSION-20260901-004
    artifact: datasets/dgn-open-tiny/dataset_manifest.json
    revision: 2cb68ef5985a7a24014ada5549791e8b1ae86af8
reason: "MVP v0 đạt gate A/B/C nhưng không có bằng chứng nào cho thấy learned policy đóng góp gì so với controller prior: không có challenge tier chưa bão hòa, không có paired comparison, không có ablation tắt residual, và release_class chỉ là một chuỗi trong tài liệu chứ không phải giá trị mà checker đọc được. Một artifact như vậy không thể mở release, và gate experimental đang pass không được phép bị trích dẫn như một release verdict."
necessity: N3
impact: "Không sửa artifact, số liệu hay kết luận nào của v0: round-1..round-3 và model card v0 giữ nguyên là experimental_non_release. Thay đổi là bổ sung: scope v1 với release_class release_candidate, Tier D uplift gate, paired bootstrap có seed, ablation contract, safety budget dạng bound, release-mode checker riêng, và version identity 0.1.0a2 / v0.1.0-alpha.2. Việc bump version buộc phải re-pin generator source closure của DGN-Open-Tiny: bốn entry hash đổi, không sample/shard/object manifest nào bị chạm. Public release bị chặn cho tới khi artifact v1 tồn tại và qua independent review."
---

# REV-20260902-001 — MVP v0 không đủ điều kiện phát hành

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260902-002`
- Bản ghi hoàn tất của phiên hiện tại:
  `docs/sessions/SESSION-20260902-003-mvp-release-mr02-contract-lock.md`
- Phiên bị sửa: `SESSION-20260831-001` và `ROADMAP-MVP-001`
- Artifact bị sửa: `docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md` (kết luận
  “đủ để phát hành” chưa từng được nêu, nhưng phạm vi hiệu lực của nó nay được
  ghi rõ), `configs/mvp/dexacquire-mvp-v0.yaml` (giữ nguyên bytes; bị tách khỏi
  release path bằng một scope mới)
- Revision/checksum trước sửa: commit `90b3acc`; scope v0 `scope_hash`
  `a897e36084c9ab11cbf8046e446ef318c48b7463d8b2db4af4d4ca0593109b8b`
- Revision/checksum sau sửa: scope v0 `scope_hash` **không đổi**
  (`a897e360…`), `eval_manifest_hash` **không đổi** (`ac57bf61…`); scope v1
  `scope_hash` `47fbf8f0bb786473993f9196d442a8e0fe08f8e7bdcdec4117a12cfd70aff11d`,
  `eval_manifest_hash`
  `733ef3a04073337b82e586b60e442cdc7bdf386e38578a4d389decfa78f8dc12`
- Issue/finding/yêu cầu nguồn: `ROADMAP-MVP-RELEASE-001` §0.5, §2.3, §5 MR-02

## 2. Lý do chỉnh sửa

Bản ghi hoàn tất của `SESSION-20260831-001` là đúng với những gì nó tuyên bố:
MVP v0 đạt Tier A/B/C và artifact của nó là `experimental_non_release`. Nó
không còn đủ cho câu hỏi đang được hỏi bây giờ, là câu hỏi phát hành.

Sự kiện làm phát sinh chỉnh sửa là việc đọc lại chính số liệu v0 dưới yêu cầu
release: trên A/B/C, controller prior đã bão hòa, nên một candidate đạt gate
không chứng minh được rằng phần learned đóng góp gì. Chạy release-mode checker
lên `runs/mvp` cho thấy điều đó thành một con số cụ thể — trên Tier C, candidate
thua prior một success theo paired count — trong khi gate experimental vẫn báo
tier đó pass, vì gate experimental không so sánh hai arm.

Ba thiếu sót còn lại là thiếu sót về cấu trúc, không phải về kết quả:

1. Không có tier nào mà prior chưa bão hòa, nên không có chỗ để đo uplift.
2. Không có ablation tắt residual, nên một cải thiện do wrapper hay do clamp
   không phân biệt được với cải thiện do model.
3. `release_class` là chuỗi trong tài liệu; không có checker nào từ chối dùng
   kết quả experimental làm release verdict.

Đây không phải “cải tiến” hay “dọn dẹp”: nếu không sửa, con đường duy nhất tới
release là trích dẫn một gate không được thiết kế để trả lời câu hỏi release.

## 3. Mức độ cần thiết

- Mức đã chọn: `N3`
- Lập luận: ảnh hưởng trực tiếp tới quyết định phát hành công khai. Nếu không
  sửa, alpha.2 có thể được phát hành dựa trên một verdict experimental, kèm
  một release claim (“learned grasp policy”) mà bằng chứng hiện có không đỡ
  được. Sai lệch đó chạm tới người dùng đầu tiên và tới tuyên bố an toàn.
- Thời hạn/độ khẩn: trước `MR-03`; không được phép trì hoãn qua bất kỳ lần
  train release nào, vì mọi ngưỡng phải tồn tại trước kết quả mà nó phán xét.
- Biện pháp giảm thiểu nếu trì hoãn: không được phép trì hoãn.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | `environment_fingerprint` gắn cứng `environment_id` v0 | fingerprint lấy `environment_id` từ scope | thấp | v0 fingerprint không đổi; test pin |
| API/schema/config | scope chỉ có schema v0, tier A–C, `release_class` một giá trị | thêm schema v1, tier D, `release_candidate`, `challenge`/`release` block | vừa | `MvpScopeConfig` fail-closed theo từng schema |
| Dữ liệu/checkpoint | DGN closure pin ở commit `2cb68ef` | pin lại ở commit `5e3c164`; 3 hash đổi, thêm `qdgrasp/version.py` | thấp | không sample/shard/object manifest nào bị chạm |
| Tương thích CPU/GPU/export | MuJoCo CPU | không đổi | không | ngoài phạm vi release |
| Số liệu/kết luận đã công bố | v0 pass A/B/C, không có contribution claim | giữ nguyên; bổ sung kết luận “không đủ điều kiện phát hành” | vừa | model card v1 riêng ở MR-06 |

Tóm tắt tác động: không có số liệu v0 nào bị sửa. `scope_hash` và
`eval_manifest_hash` của v0 được pin bằng test để việc mở rộng schema không thể
làm chúng trôi. Thay đổi duy nhất có thể quan sát trên đường v0 là
`environment_id` nay đọc từ scope thay vì hằng số của class, và với v0 hai giá
trị đó bằng nhau.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Tạo scope release v1 với Tier D, challenge contract, release criteria và safety budget | `configs/mvp/dexacquire-mvp-v1.yaml`, `scope_hash` `47fbf8f0…` | `V-002` |
| `CH-002` | Sinh eval manifest v1 A–D bằng locker, không chép tay | `configs/mvp/dexacquire-mvp-v1.eval-manifest.json`, `733ef3a0…` | `V-003` |
| `CH-003` | Thêm `release_class` đọc được bằng máy và parser fail-closed theo từng schema | `qdgrasp/mvp/config.py` | `V-004` |
| `CH-004` | Thêm release-mode checker tách khỏi gate experimental | `scripts/check_mvp.py --release` | `V-005`, `V-006` |
| `CH-005` | Cài paired bootstrap deterministic có ghi seed và resample count | `qdgrasp/mvp/evaluate.py::paired_uplift` | `V-004` |
| `CH-006` | Sửa version identity về một nguồn khai báo duy nhất và một mapping | `VERSION`, `pyproject.toml`, `qdgrasp/version.py`, `scripts/check_version_identity.py` | `V-007` |
| `CH-007` | Pin `scope_hash`/`eval_manifest_hash` của v0 bằng test | `tests/mvp/test_mvp_release_contract.py` | `V-004` |
| `CH-008` | Re-pin generator source closure của DGN-Open-Tiny sau khi version literal đổi | `datasets/dgn-open-tiny/dataset_manifest.json`, commit `e68d70b` | `V-009` |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `.venv/bin/python -m pytest -q` | full suite pass | 1496 passed, 1 skipped, 105 subtests | `pass` | log phiên |
| `V-002` | `.venv/bin/python scripts/lock_mvp_scope.py --check` | manifest v0 khớp scope v0 | OK `a897e360…` | `pass` | log phiên |
| `V-003` | `.venv/bin/python scripts/lock_mvp_scope.py --scope configs/mvp/dexacquire-mvp-v1.yaml --check` | manifest v1 khớp scope v1 | OK `47fbf8f0…` | `pass` | log phiên |
| `V-004` | `.venv/bin/python -m pytest tests/mvp/test_mvp_release_contract.py -q` | schema từ chối mọi biến thể sai | 26 passed | `pass` | log phiên |
| `V-005` | `.venv/bin/python -m pytest tests/mvp/test_check_mvp_release_gate.py -q` | mỗi tiêu chí release có một test làm hỏng đúng một thứ | 24 passed | `pass` | log phiên |
| `V-006` | `.venv/bin/python scripts/check_mvp.py --root . --runs runs/mvp --release` | artifact v0 bị release gate chặn | 53/104, `blocked_with_evidence (release gate)` | `pass` | log phiên |
| `V-007` | `.venv/bin/python -m pytest tests/test_version_identity.py -q` | một nguồn version, mapping hai chiều | 23 passed | `pass` | log phiên |
| `V-008` | `.venv/bin/python scripts/check_static_core.py` | Ruff + Mypy pass | PASS (136 file, 32 contract) | `pass` | log phiên |
| `V-009` | `.venv/bin/python -m pytest tests/corrective tests/model_training -q` | corpus lại tự chứng minh được provenance | 35 passed | `pass` | log phiên |

- Regression đã chạy lại: full suite, static core, docs, wheel, Phase 0, cả hai
  mode của MVP checker.
- Kiểm tra chưa chạy: chưa có locked evaluation nào của v1 — đó là `MR-05` và
  cố tình chưa chạy; mọi ngưỡng phải tồn tại trước kết quả.
- Khả năng rollback: revert về commit cha; scope v0, manifest v0 và toàn bộ
  `evidence/mvp/round-*` không bị chạm nên không cần khôi phục dữ liệu. Việc
  re-pin DGN là một commit riêng (`e68d70b`) và revert được độc lập.

Ghi chú về `CH-008`: `qdgrasp/__init__.py` chứa version literal và nằm trong
import closure mà DGN-Open-Tiny ghi hash, nên mọi lần bump version đều làm corpus
mất khả năng tự chứng minh. Closure được pin lại chứ không bị thu hẹp: bảy entry
mà một generator run nạp lazy vẫn được giữ và hash lại, thay vì rút xuống tập mà
một lần import trần chạm tới. Không có sample, shard hay object manifest nào bị
sửa, và hash được lấy trên cây sạch tại đúng commit mà manifest ghi.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có; MVP v0 chưa từng có independent
  review.
- Verdict cũ còn hiệu lực cho revision mới: có, trong phạm vi của nó. MVP v0
  vẫn là `experimental_non_release` đã đạt A/B/C. Nó không còn được coi là một
  bước trên đường phát hành.
- Loại kiểm tra lại cần thiết: `none` cho artifact v0; `full_review` cho artifact
  v1 khi chúng tồn tại (`MR-06`).
- Đính chính số liệu/tài liệu cần phát hành: không có số liệu nào sai. Model
  card v1 riêng sẽ nêu prior/BC/PPO/ablation ở `MR-06`; model card v0 không bị
  sửa.
- Người chấp nhận rủi ro/ngoại lệ: không có ngoại lệ nào được cấp.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-agent, 2026-09-02, UTC+07:00
- Người kiểm tra: chưa có; independent review là `MR-06` và áp dụng cho artifact
  v1, không cho hồ sơ này.
- Kết luận: đã hoàn tất. Contract release v1 được khóa trước khi tồn tại bất kỳ
  kết quả nào mà nó phán xét.
- Liên kết bản ghi hoàn tất phiên hiện tại:
  `docs/sessions/SESSION-20260902-003-mvp-release-mr02-contract-lock.md`
