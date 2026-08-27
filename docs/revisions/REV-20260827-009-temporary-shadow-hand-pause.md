---
document_id: REV-20260827-009
document_type: revision_record
revision_schema: 2
title: Tạm dừng Shadow Hand khỏi active corpus và gate mới
status: complete
date: 2026-08-27
record_id: REV-20260827-009
session_id: SESSION-20260827-SHADOW-PAUSE
created_at: 2026-08-27T21:34:47+07:00
author: codex-primary-agent
revises:
  - session_id: PLAN-V2
    artifact: PLAN.md
    revision: f1d4b9eb1692f229704593502afe088b73ae7f769367f7d9e6a515cc0cfe245c
  - session_id: ROOT-README
    artifact: README.md
    revision: 16a013b0e8821998838724efb93be2df45fdfca3e5304d62ddced73c49c5732c
  - session_id: DOCS-INDEX
    artifact: docs/README.md
    revision: 88462f3921274cbed97eee412ce78dc024d254306a41490a93053cc0bf4182e8
  - session_id: QDGRASP-ROBOT-V2
    artifact: docs/configuration/ROBOT_PROFILE.md
    revision: 6615177bb757f1d71cf816bc06c40bb479bf3def711fe4bd3812ee080b0d3700
  - session_id: ROADMAP-P3-001
    artifact: docs/roadmap/PHASE3_EXECUTION_PLAN.md
    revision: f6f6994d77ecb7a36e43df42a52a91269c7219d5d3d73e3c82570de79b07a813
  - session_id: ROADMAP-P3.4-001
    artifact: docs/roadmap/PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md
    revision: 0257b6732cf5c4404d169c2a4599dd37c2ccde2730aad3b68a33f4800aea8e3c
  - session_id: ROADMAP-P3.4.2-001
    artifact: docs/roadmap/PHASE3_4_2_CORRECTNESS_RECOVERY_PLAN.md
    revision: 102e47c7e3b039809ff735195179852dbf5304e20b56fd31d7634e10a3a22c35
  - session_id: ROADMAP-P3.5-001
    artifact: docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md
    revision: 6b86333c9ba3956e55a22b8a7cea657d65c391a2442a747fd7a9ef1f18f0a50e
  - session_id: ROADMAP-PROJECT-001
    artifact: docs/roadmap/PROJECT_PHASES.md
    revision: f696f30727f370160c87d027b7b45802000e2b139996502387cd575a5915ebdc
reason: "Maintainer quyết định tạm dừng dùng Shadow Hand vì cấu hình underactuated/contact-control khó và đang chặn tiến độ; active corpus/gate mới chuyển sang LEAP + Wonik Allegro."
necessity: N3
impact: "Thêm ADR-0008 và cập nhật source-of-truth/living docs: giữ Shadow preset/evidence nhưng loại khỏi workload, dataset/checkpoint/release/backend/RL gate mới; P3.4 ba-hand tạm dừng và không được ghi pass, P3.5 tiếp tục với 2/2 active hand."
---

# REV-20260827-009 — Temporary Shadow Hand pause

## 1. Liên kết truy vết

- Phiên thực hiện: `SESSION-20260827-SHADOW-PAUSE`.
- Quyết định mới: `ADR-0008`,
  `docs/decisions/0008-temporary-shadow-hand-pause.md`, SHA-256
  `894cd93574f18b5dfbc987f990c6744d3e2f347b8815b60f3ce089b385743af7`.
- Checksum trước sửa của từng living artifact nằm trong metadata `revises`.
- Checksum sau sửa:
  - `PLAN.md`: `900d78227abe5f2abc8279ebe73060ef6bd441ef2a6013d9bc78cf765859f3eb`;
  - `README.md`: `c6f810aa6e4fb1d126f4f1616166b9c362c95082d5de68b5780a58ada0f90d4b`;
  - `docs/README.md`: `238f20628b09e3b43c2358a523f2b594091ce9bf9f59b405092057de42edcef8`;
  - `ROBOT_PROFILE.md`: `af7ffa8a1cb23a9e8ce3d0a37f08c82bfe1d261619ccb9d387b93ba868a17f16`;
  - `PHASE3_EXECUTION_PLAN.md`: `40ee011b95de85715b19b33fa70af83d978f2e4c3817a8cd174644513763dcaf`;
  - `PHASE3_4_CONTACT_RICH_DYNAMIC_GRASP_PLAN.md`:
    `c196cc7529a0d8514336043f6196b6b6938aa85828c1239313ee92c0a824efeb`;
  - `PHASE3_4_2_CORRECTNESS_RECOVERY_PLAN.md`:
    `9d17110495257757e20e9d82b240b3c2b6faffdac87c845a19badd0cb60fe87e`;
  - `PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md`:
    `084a59bffac115e834506879ed400656139760f55e01b2d595fa0578c14af4fd`;
  - `PROJECT_PHASES.md`: `c56dd7ad3ee71723bc04fe4d272a7d10e1adff3ef165ff88176f792f739339c0`.

## 2. Lý do chỉnh sửa

Shadow đã tạo giá trị như một underactuated/transmission fixture nhưng dynamic
grasp configuration hiện tốn nhiều công sức hơn LEAP/Allegro: tendon-coupled
target semantics, inactive-finger self-contact và controller/validator chưa có
miền positive hợp lệ. Maintainer quyết định tạm dừng thay vì tiếp tục để Shadow
chặn mọi backend, dataset và RL gate mới.

Đây là scope decision có ảnh hưởng release nên phải ghi ở source-of-truth và ADR,
không chỉ thêm một câu vào session report. Historical evidence vẫn bất biến.

## 3. Mức độ cần thiết

- Mức: `N3` vì thay active corpus, release gate và phase scheduling.
- Áp dụng ngay từ 2026-08-27 cho workload/artifact mới.
- Mitigation: preset/parser/evidence Shadow được giữ; active output phải khai báo
  rõ 2/2 hand và `paused_hands=[shadow_hand]`.
- Không được dùng quyết định này để đổi P3.4 three-hand failure thành pass.

## 4. Phạm vi và tác động

| Lĩnh vực | Trước | Sau | Tác động | Hành động |
| --- | --- | --- | --- | --- |
| Active corpus | LEAP/Allegro/Shadow | LEAP/Allegro; Shadow paused | cao | manifest/disclosure mới ghi active/paused |
| Runtime/schema | Shadow preset/transmission tồn tại | giữ nguyên, chưa thêm runtime guard | không trong phiên này | implementation task riêng nếu cần enforcement |
| P3.4 | exact three-hand gate chưa đạt | paused, vẫn chưa đạt | cao | không closure/pass |
| P3.4.2 | active closure plan | superseded/paused resumption contract | cao | chỉ mở lại bằng revision mới |
| P3.5/P4/P5 mới | ngầm ba hand | active two-hand scope | cao | gate/checkpoint/claim phải ghi 2/2 |
| Historical P2/P3 evidence | three-hand pass tại exact revision | giữ nguyên historical truth | không | không rewrite |

Tóm tắt tác động khớp metadata: phiên chỉ sửa policy/documentation. Không chỉnh
`shadow_hand.yaml`, runtime defaults, test selection, dataset hoặc raw evidence.

## 5. Nội dung thay đổi đã hoàn tất trong phiên này

| Change ID | Việc đã làm | Artifact/evidence |
| --- | --- | --- |
| `CH-001` | Tạo decision trung tâm, scope và điều kiện mở lại | `ADR-0008` |
| `CH-002` | Cập nhật `PLAN.md` và root README với active corpus LEAP/Allegro | post-hash §1 |
| `CH-003` | Ghi Shadow operational status trong robot registry | `QDGRASP-ROBOT-V2@1.1.0` |
| `CH-004` | Cập nhật P3 living plan/gate từ ba xuống hai active hand | `ROADMAP-P3-001@1.1.0` |
| `CH-005` | Giữ P3.4 three-hand unclosed và pause P3.4.2 | P3.4 §19; P3.4.2 §12 |
| `CH-006` | Chuyển P3.5 backend/RL parity sang 2/2 active hand | `ROADMAP-P3.5-001@1.1.0` |
| `CH-007` | Cập nhật project/docs index và giữ historical phase truth | roadmap/docs index |

## 6. Xác minh

| Verification ID | Phương pháp | Mong đợi | Thực tế | Trạng thái |
| --- | --- | --- | --- | --- |
| `V-001` | `python3 scripts/check_docs.py --root .` | graph/front matter pass | 112 file checked, pass | pass |
| `V-002` | `git diff --check` | zero whitespace error | không có output, exit 0 | pass |
| `V-003` | scope-consistency search active docs | active two-hand, Shadow paused; historical references labeled | source-of-truth/README/registry/P3/P3.4/P3.5/roadmap nhất quán | pass |
| `V-004` | runtime/test/data workload | ngoài note/policy revision | không chạy | not_run_nonblocking |

- Runtime regression không chạy vì runtime/config/data không đổi.
- Chưa có enforcement code tự chặn Shadow; ADR/living plans là nguồn scope cho
  công việc sau. Nếu cần CLI hard block/default-list change, tạo implementation
  revision riêng và replay tests.
- Rollback/resume phải dùng ADR/revision mới và thỏa điều kiện ADR-0008; không
  xóa record này.

## 7. Ảnh hưởng tới báo cáo và quyết định cũ

- P2/P3.2/P3.3 three-hand pass giữ hiệu lực cho exact historical revisions.
- P3.4/P3.4.1/P3.4.2 failure/evidence giữ nguyên; không được đổi thành pass.
- ADR-0008 supersede một phần active-corpus/checkpoint commitment của ADR-0003,
  không đổi variable-topology architecture.
- Mọi future release/review phải kiểm disclosure 2/2 active hand và không claim
  three-hand coverage.
- Loại review khi phát hành artifact mới: delta/full review tùy blast radius;
  resume Shadow luôn cần independent review.

## 8. Xác nhận đóng hồ sơ

- Tác giả: `codex-primary-agent`, 2026-08-27, Asia/Bangkok.
- Người quyết định scope: project maintainer (yêu cầu trực tiếp trong phiên).
- Kết luận: revision hoàn tất; Shadow paused, không bị xóa.
- Bản ghi: `REV-20260827-009`.
