---
document_id: REV-20260827-012
document_type: revision_record
revision_schema: 2
title: Mở rộng P3.4.3 thành closed-world gate completion contract
status: complete
date: 2026-08-27
record_id: REV-20260827-012
session_id: SESSION-20260827-P3.4-PLAN-COMPLETENESS
created_at: 2026-08-27T23:59:30+07:00
author: codex-primary-agent
revises:
  - session_id: ROADMAP-P3.4.3-001
    artifact: docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md
    revision: 92a76da494208595ff4542a579d8982ff4183bce1791a8e3fdcab1b877a66746
  - session_id: PLAN-V2
    artifact: PLAN.md
    revision: a52708dacbf94002112f3c2e16251a526256b377dec9d92b6616535758e5c6dd
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: bed3894b688ebc55af6bced2d2b5b6781bd00589208716365f63a1a5e6018793
  - session_id: DOCS-INDEX
    artifact: docs/README.md
    revision: 3811d10dfe62d897778a4d09bdcda975846409980496121ab79d3bcbf36249e1
reason: "Bản P3.4.3 1.0 chỉ map các defect audit trực tiếp, chưa đối chiếu closed-world với toàn bộ §4–§16 và P3.4-00–17; thiếu backend request hydration, CPU summary, controller/search invariants, full dataset coverage, ablation, operations và public handoff gates."
necessity: N3
impact: "P3.4.3 tăng từ B-01–B-10/G00–G11 thành contract B-01–B-20 + G00–G11 + C01–C08, có original-work-package traceability và completeness checker bắt buộc; không đổi ADR-0008 hoặc P3.4 historical verdict."
---

# REV-20260827-012 — P3.4.3 completeness expansion

## 1. Liên kết truy vết

- Artifact chính: `docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md`
  (`ROADMAP-P3.4.3-001@1.1.0`).
- Manifest mới: `docs/roadmap/phase3_4_3_requirements.yaml`.
- Artifact bị sửa: `PLAN.md`, `docs/roadmap/PROJECT_PHASES.md`, `docs/README.md`.
- Bản ghi trước: `REV-20260827-011`.

## 2. Lý do chỉnh sửa

- Bản 1.0: corrective plan bao phủ mười defect nhìn thấy trực tiếp.
- Bản 1.1: closed-world contract bao phủ toàn bộ normative source, work package,
  test/resource/release criteria và bắt buộc machine-readable completeness map.

## 3. Mức độ cần thiết

`N3`. Một plan chỉ map defect nhìn thấy được sẽ đóng phase khi các defect đó
được sửa, kể cả khi phần lớn contract gốc chưa có ai kiểm. Đó là lỗi release
provenance, không phải thiếu sót trình bày.

## 4. Phạm vi và tác động

P3.4.3 tăng từ B-01–B-10/G00–G11 thành contract B-01–B-20 + G00–G11 + C01–C08,
có original-work-package traceability và completeness checker bắt buộc. Không
đổi `ADR-0008` hoặc P3.4 historical verdict.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

Khoảng trống được bổ sung:

1. typed invalid-corpus và schema migration;
2. per-request state hydration, complete scene signature, world isolation;
3. CPU oracle contact-rich summary;
4. exact contact taxonomy/self-contact policy/terminal stage semantics;
5. objective/CEM/refinement fail-closed;
6. 2×3×2×3 dataset coverage, paired evidence và controlled ablation;
7. sparse storage/provenance/render/public loader/package/card;
8. resource preflight, checkpoint/resume/dedup và prior CUDA gates;
9. original P3.4-00–17 traceability;
10. completeness manifest/checker với zero-unmapped rule.

Checksum sau sửa:

- `PLAN.md`:
  `9832e2a2f5a99ef3e90e8dbed22857f631eab64587b4bd89070ab8a6dcf9148c`;
- `docs/README.md`:
  `cd23752f1efe35e85ccf1ad975db15ba0594a81592bf77fcc9a2f7f345c2aed8`;
- `docs/roadmap/PROJECT_PHASES.md`:
  `3678a863bfca796bccfab4257ece2899812066d7f3769a79bdd0334ac3c705e6`;
- `docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md`:
  `75e4b7c5d20d9c59ea3a1bbe89dc752c1df6ee092dfc60ac1bba409d8f1c70bd`;
- `docs/roadmap/phase3_4_3_requirements.yaml`:
  `0c1f169bcf2d5aec425c9ad4a0f3ab805968591c4c5f4507f41d1d058f4ec1e9`.

## 6. Xác minh

| Verification ID | Lệnh/phương pháp | Kết quả mong đợi | Kết quả thực tế | Trạng thái | Evidence |
| --- | --- | --- | --- | --- | --- |
| `V-001` | đếm tĩnh inventory trong manifest | mọi ID duy nhất | blocker 20, direct gate 12, completeness gate 8, work package 18, requirement 85, duplicate 0 | pass | `phase3_4_3_requirements.yaml` |
| `V-002` | đối chiếu C01–C08 với §4–§16 của `ROADMAP-P3.4-001` | zero clause không có gate | mọi clause có mapping | pass | `ROADMAP-P3.4.3-001` §6–§8 |

Revision này chỉ hoàn thiện plan. Không có technical gate nào được đổi thành
pass, không chạy test/simulation/benchmark và không sửa raw evidence.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- `REV-20260827-011` không bị thu hồi; bản này mở rộng contract của nó.
- Không đổi `ADR-0008`, không đổi P3.4 historical three-hand verdict.
- Mọi requirement giữ `pending` cho tới khi có implementation, test, evidence
  và review tương ứng.

## 8. Xác nhận đóng hồ sơ

- Tác giả: codex-primary-agent, 2026-08-27 Asia/Bangkok.
- Người kiểm tra: chưa có independent reviewer.
- Kết luận: contract P3.4.3 đã đóng kín về phạm vi; thực thi chưa bắt đầu tại
  thời điểm ghi bản này.
- Liên kết bản ghi hoàn tất: `REV-20260827-012`.
