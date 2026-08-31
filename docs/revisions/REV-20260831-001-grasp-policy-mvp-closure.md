---
document_id: REV-20260831-001
document_type: revision_record
revision_schema: 2
title: Đóng Grasp Policy MVP và sửa predicate/interface của môi trường MVP
status: complete
date: 2026-08-31
record_id: REV-20260831-001
session_id: SESSION-20260831-001
created_at: 2026-08-31T09:30:00+07:00
author: claude-agent
revises:
  - session_id: ROADMAP-MVP-001
    artifact: docs/roadmap/GRASP_POLICY_MVP_TEMPORARY_PLAN.md
    revision: ROADMAP-MVP-001@1.0.0
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: ROADMAP-001@1.28.0
  - session_id: PLAN-V2
    artifact: PLAN.md
    revision: PLAN-V2@4.4.0
reason: "Thi công MVP-00–07 và đo được ba vòng tune; hai vòng đầu fail và vòng ba pass sau khi sửa success predicate cho đúng §4 và thêm low-pass vào interface residual, làm scope_hash/eval_manifest_hash đổi."
necessity: N2
impact: "ROADMAP-MVP-001 chuyển từ active sang complete với artifact experimental_non_release; baseline controller prior đo lại là B 94.7% / C 94.0% thay cho B 89.7% / C 88.5%; không gate release nào đổi và không blocker P3.4.3/P3.5/P4/P5 nào được gỡ."
---

# REV-20260831-001 — Đóng Grasp Policy MVP

## 1. Liên kết truy vết

- Phiên thực hiện sửa: `SESSION-20260831-001`.
- Bản ghi hoàn tất của phiên hiện tại:
  `docs/sessions/SESSION-20260831-001-grasp-policy-mvp.md`.
- Phiên bị sửa: `ROADMAP-MVP-001` (`docs/roadmap/GRASP_POLICY_MVP_TEMPORARY_PLAN.md`),
  `ROADMAP-PROJECT-001` (`docs/roadmap/PROJECT_PHASES.md`), `PLAN-V2` (`PLAN.md`).
- Revision/checksum trước sửa: `ROADMAP-MVP-001@1.0.0`, `ROADMAP-001@1.28.0`,
  `PLAN-V2@4.4.0`.
- Revision/checksum sau sửa: `ROADMAP-MVP-001@1.1.0`, `ROADMAP-001@1.29.0`,
  `PLAN-V2@4.5.0`.
- Artifact kết quả: `evidence/mvp/round-3/` (manifest SHA-256 tại
  `evidence/mvp/round-3/MANIFEST.json`), model card
  `docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md`.
- Artifact của hai vòng thất bại: `evidence/mvp/round-1/`, `evidence/mvp/round-2/`.

## 2. Lý do chỉnh sửa

`ROADMAP-MVP-001` được ghi `active` với MVP-00–07 chưa thi công. Phiên này thi
công đủ tám work package và đo ba vòng tune theo đúng ngân sách §8. Hai lý do
buộc phải có revision record thay vì chỉ thêm một session report:

Thứ nhất, **success predicate của môi trường bị cài chặt hơn §4**. §4 yêu cầu độ
cao giữ liên tục `0.50 s`, tiếp xúc `>=2` finger group ở **bước cuối**, và không
support-assisted trong cửa sổ retain — ba điều kiện tách rời. Bản cài đặt đầu
gộp tiếp xúc vào bộ đếm liên tục, nên một bước chattering tiếp xúc xóa sạch nửa
giây nâng hợp lệ. Đây là sai protocol, không phải sai tinh chỉnh: nó làm hỏng
mọi số đã đo trước đó, kể cả baseline của controller prior.

Thứ hai, **interface action không chịu được biến thiên tần số cao**. Bơm
`N(0, 0.15)` vào action làm mất vật ở 42/42 rollout, trong khi một residual hằng
số lớn gấp ba lần thì vô hại. Vì mọi policy học được và mọi lần lấy mẫu PPO đều
sinh ra đúng loại biến thiên đó, interface được bổ sung low-pass
`residual_low_pass = 0.20`. Đây là thay đổi của môi trường, thuộc loại mà §3.1
yêu cầu làm trước khi đổ lỗi cho phần học.

Cả hai làm `scope_hash` và `eval_manifest_hash` thay đổi, nên chúng phải được
ghi lại chứ không được sửa im lặng.

## 3. Mức độ cần thiết

- Mức đã chọn: `N2`.
- Lập luận: sai protocol đo lường và thay đổi interface điều khiển làm mọi số
  liệu MVP trước đó không còn hợp lệ. Nếu không sửa, baseline controller prior
  bị báo thấp hơn thực tế `5` điểm phần trăm ở Tier B và `5.5` điểm ở Tier C, và
  mọi so sánh policy-với-prior đứng trên một predicate không đúng plan. Không
  phải `N3` vì không có gate release, license hay quyết định an toàn nào phụ
  thuộc: artifact luôn là `experimental_non_release`.
- Thời hạn/độ khẩn: đã hoàn tất trong phiên này.
- Biện pháp giảm thiểu nếu trì hoãn: không áp dụng.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước sửa | Sau sửa | Mức tác động | Hành động tiếp theo |
| --- | --- | --- | --- | --- |
| Hành vi/runtime | `held` yêu cầu đồng thời độ cao, tiếp xúc và không support | `held` chỉ yêu cầu độ cao; tiếp xúc kiểm ở bước cuối, support kiểm trên cửa sổ retain | cao | Đo lại toàn bộ ba tier |
| API/schema/config | `ActionSpec` không có làm mượt | thêm `residual_low_pass` (mặc định `1.0`, khóa ở `0.20`) | vừa | `scope_hash` mới, `lock_mvp_scope.py --check` |
| Dữ liệu/checkpoint | chưa có | `evidence/mvp/round-3/policy/{bc,ppo}.pt` | vừa | Chỉ dùng trong phạm vi MVP |
| Tương thích CPU/GPU/export | không đổi | không đổi | không | Không có claim GPU physics |
| Số liệu/kết luận đã công bố | chưa công bố số MVP nào | model card `RPT-MVP-GRASP-POLICY-001` | vừa | Ghi rõ kết quả âm của phần học |

Tóm tắt tác động: `ROADMAP-MVP-001` chuyển sang `complete` với artifact
`experimental_non_release`; baseline prior đo lại là `A 100/100`, `B 284/300`,
`C 188/200`. Không gate của P3.4.3, P3.5, P4 hay P5 bị đổi, không dataset hay
checkpoint `release_blocked` nào được gỡ, và `QDGrasp-ContactRich-Active-Tiny`
vẫn giữ `release_blocked=true`.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/revision sau sửa | Evidence |
| --- | --- | --- | --- |
| `CH-001` | Khóa scope/eval protocol MVP-00 | `configs/mvp/dexacquire-mvp-v0.yaml`, `configs/mvp/dexacquire-mvp-v0.eval-manifest.json` | `scope_hash a897e360…`, `eval_manifest_hash ac57bf61…` |
| `CH-002` | Dựng môi trường `QDGrasp-DexAcquire-MVP-v0` và pinch prior MVP-01 | `qdgrasp/mvp/{env,scene,prior}.py`, `configs/mvp/leap-pinch-prior-v0.json` | `tests/mvp`, `prior_hash ab96a87a…` |
| `CH-003` | Sửa `held` cho đúng `ROADMAP-MVP-001` §4 | `qdgrasp/mvp/env.py` | Baseline prior B `269/300` → `284/300` |
| `CH-004` | Thêm low-pass residual vào interface action | `qdgrasp/mvp/{config,env}.py`, `configs/mvp/dexacquire-mvp-v0.yaml` | Nhiễu action `sigma=0.15`: `0/42` → `14/20` giữ được vật |
| `CH-005` | Expert minimum-intervention và demo có nhiễu MVP-02 | `qdgrasp/mvp/expert.py` | `evidence/mvp/round-3/demonstrations/` |
| `CH-006` | BC + residual PPO và checkpoint có fingerprint MVP-03/04 | `qdgrasp/mvp/{bc,ppo,policy}.py` | `evidence/mvp/round-3/policy/training-report.json` |
| `CH-007` | Sửa deadlock pool khi fork từ tiến trình đã dùng torch | `qdgrasp/mvp/{evaluate,ppo}.py` | `tests/mvp/test_mvp_learning.py::test_worker_pool_survives_a_torch_trained_parent` |
| `CH-008` | Sửa hằng số MuJoCo dẫn xuất từ khối lượng khi randomize | `qdgrasp/mvp/env.py` | Baseline prior B `91.7%` → `89.7%` khi sửa |
| `CH-009` | Locked evaluation MVP-05 và cổng đóng MVP-07 | `scripts/evaluate_mvp.py`, `scripts/check_mvp.py` | `evidence/mvp/round-3/closure.json` |
| `CH-010` | Runner Kaggle/Colab MVP-06 | `notebooks/mvp_grasp_policy.ipynb` | `scripts/build_mvp_notebook.py` |
| `CH-011` | Model card và cập nhật roadmap/plan MVP-07 | `docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md`, `docs/roadmap/*`, `PLAN.md` | Tài liệu này |

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | `python scripts/check_mvp.py --runs evidence/mvp/round-3` | tất cả check pass | 38/38 pass | `pass` | `evidence/mvp/round-3/closure.json` |
| `V-002` | `python scripts/evaluate_mvp.py --checkpoint runs/mvp/policy/ppo.pt` | ba tier pass | A `100/100`, B `284/300`, C `187/200` | `pass` | `evidence/mvp/round-3/evaluation/ppo.json` |
| `V-003` | `python scripts/evaluate_mvp.py --label controller_prior` | baseline đo được | A `100/100`, B `284/300`, C `188/200` | `pass` | `evidence/mvp/round-3/evaluation/controller_prior.json` |
| `V-004` | `python -m pytest tests/mvp -q` | toàn bộ pass | 49 passed | `pass` | log phiên |
| `V-005` | `python scripts/lock_mvp_scope.py --check` | manifest khớp scope | OK | `pass` | `configs/mvp/dexacquire-mvp-v0.eval-manifest.json` |
| `V-006` | `python scripts/check_docs.py --root .` | tài liệu hợp lệ | 141 file pass | `pass` | log phiên |
| `V-007` | `python -m pytest tests -q` | không regression | toàn bộ pass | `pass` | log phiên |

- Regression đã chạy lại: toàn bộ `tests/` và `scripts/tests/`.
- Kiểm tra chưa chạy: independent review của MVP — cố ý không chạy, vì tác giả
  artifact không được tự ký verdict và artifact là `experimental_non_release` nên
  không cần cổng đó.
- Khả năng rollback: `evidence/mvp/round-3/policy/bc.pt` là checkpoint BC được
  giữ làm rollback; hai vòng trước còn nguyên tại `evidence/mvp/round-{1,2}/`.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Báo cáo người thứ ba bị ảnh hưởng: không có.
- Verdict cũ còn hiệu lực cho revision mới: có. Không verdict nào của P3.3,
  P3.4, P3.4.3 hay ADR-0008 bị đụng tới.
- Loại kiểm tra lại cần thiết: `none` cho các phase cũ; `delta_review` cho
  chính MVP nếu sau này ai đó muốn dùng artifact ngoài phạm vi `experimental`.
- Đính chính số liệu/tài liệu cần phát hành: không có số MVP nào đã công bố
  trước phiên này.
- Người chấp nhận rủi ro/ngoại lệ: không áp dụng.

## 8. Xác nhận đóng hồ sơ

- Tác giả: claude-agent, 2026-08-31, UTC+07.
- Người kiểm tra: chưa có independent reviewer; artifact giữ
  `experimental_non_release` và không được dùng cho bất kỳ cổng release nào.
- Kết luận: đã hoàn tất. MVP-00–07 có artifact thật, §7 pass sau checkpoint
  reload, và kết quả âm của phần học được ghi rõ trong model card.
- Liên kết bản ghi hoàn tất phiên hiện tại:
  `docs/sessions/SESSION-20260831-001-grasp-policy-mvp.md`.
