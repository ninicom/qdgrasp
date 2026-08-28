---
document_id: REV-20260829-001
document_type: revision_record
revision_schema: 2
title: Chuyển ưu tiên sang Grasp Policy MVP vertical slice
status: complete
date: 2026-08-29
record_id: REV-20260829-001
session_id: SESSION-20260829-GRASP-POLICY-MVP-PLAN
created_at: 2026-08-29T00:00:00+07:00
author: codex-agent
revises:
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: ROADMAP-001@1.27.0
  - session_id: ROADMAP-P3.5-001
    artifact: docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md
    revision: ROADMAP-P3.5-001@1.1.0
  - session_id: DOCS-INDEX
    artifact: docs/README.md
    revision: DOCS-INDEX@1.7.0
reason: "Roadmap đầy đủ yêu cầu đóng asset/scene, multi-hand, GPU backend và model architecture trước khi có policy; critical path quá dài để kiểm chứng sớm khả năng gắp thành công."
necessity: N3
impact: "Thêm ROADMAP-MVP-001 làm execution lane ưu tiên cho LEAP state-based vertical slice. P3.5 được hạ ưu tiên thực thi; không gate cũ nào được đổi thành pass và artifact MVP không phải release claim."
---

# REV-20260829-001 — Chuyển ưu tiên sang Grasp Policy MVP

## 1. Liên kết truy vết

- Plan mới: `docs/roadmap/GRASP_POLICY_MVP_TEMPORARY_PLAN.md`
  (`ROADMAP-MVP-001@1.0.0`).
- Roadmap: `docs/roadmap/PROJECT_PHASES.md` (`ROADMAP-001@1.28.0`).
- Plan được hạ ưu tiên: `docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md`
  (`ROADMAP-P3.5-001@1.2.0`).
- Chỉ mục: `docs/README.md` (`DOCS-INDEX@1.8.0`).

## 2. Lý do chỉnh sửa

Roadmap hiện tại đi tuần tự qua closure contact-rich, asset/scene ingestion,
multi-hand RL readiness, model flow và full training. Chuỗi này đúng cho release
nhưng quá dài trước khi dự án trả lời câu hỏi thực dụng đầu tiên: có thể train,
reload và đánh giá một policy gắp vật có success cao hay không.

## 3. Mức độ cần thiết

`N3`. Thay đổi thứ tự thực thi ảnh hưởng trực tiếp đến critical path và cách đọc
trạng thái roadmap, nhưng không thay safety/release gate đã tồn tại.

## 4. Phạm vi và tác động

Thêm một plan vertical slice, cập nhật roadmap/chỉ mục và đánh dấu P3.5 bị hạ
ưu tiên thực thi. Không thay code, dataset, checkpoint, kết quả đo hoặc trạng
thái pass/fail của gate kỹ thuật.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

Tạo `ROADMAP-MVP-001` làm đường thực thi ưu tiên tạm thời:

- chỉ LEAP Hand;
- một target cuboid family trên bàn, MuJoCo CPU;
- privileged state observation;
- controller prior -> behavior cloning -> residual PPO;
- checkpoint và locked evaluation bắt buộc;
- Kaggle/Colab dùng để train model, không tạo claim GPU physics.

P3.5 giữ status tài liệu `active` nhưng có
`execution_priority: deferred_by_ROADMAP-MVP-001`. P3.4.3/P3.5/P4/P5 vẫn giữ
toàn bộ finding, gate và release condition hiện có.

Plan mới đã khóa thêm guardrail chống MVP giả:

- Không teleport object, không action trực tiếp lên object state.
- Timeout, safety termination, NaN/Inf và simulator error đều là failure.
- Success cần lift 5 cm, retain 0.5 s, ít nhất hai finger group và không còn
  table support trong retain window.
- Eval seed/object variants được khóa trước vòng tune cuối.
- Tier B cần ít nhất 85% trên 300 episode và Wilson 95% lower bound ít nhất 80%.
- Chỉ số phải tái lập sau khi reload checkpoint cùng config/normalizer/hash.

## 6. Xác minh

- `git diff --check`: pass.
- `python3 scripts/check_docs.py --root .`: pass, 138 file được kiểm tra.
- Không chạy mô phỏng, test model hay tạo claim performance trong revision lập
  kế hoạch này.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- Không đóng P3, P3.4.3, P3.5, P4 hoặc P5.
- Không mở lại Shadow Hand; ADR-0008 giữ nguyên.
- Không giải phóng `release_blocked` artifact.
- Không xóa raw mesh/CoACD/scene generation, GPU parity, multi-hand, vision,
  generalization hoặc independent-review backlog.

## 8. Xác nhận đóng hồ sơ

Revision này chỉ đổi **thứ tự học hỏi và thi công**: chứng minh vertical slice
trước, sau đó dùng failure evidence để ưu tiên correctness/generalization. Mọi
claim release vẫn phải quay về roadmap đầy đủ và gate tương ứng.

- Tác giả: codex-agent, 2026-08-29 Asia/Bangkok.
- Reviewer độc lập: chưa yêu cầu cho thay đổi plan; mọi verdict kỹ thuật về MVP
  vẫn chờ artifact và evidence thật.
- Kết luận: hồ sơ thay đổi plan hoàn tất; MVP implementation chưa bắt đầu và
  không có phase kỹ thuật nào được đóng bởi hồ sơ này.
