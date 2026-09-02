---
document_id: SESSION-20260902-003
document_type: session_report
title: MVP release MR-02 — khóa contract release v1
status: complete
date: 2026-09-02
session_id: SESSION-20260902-003
author: claude-agent
revises: REV-20260902-001
revision_reason: "MVP v0 không đủ điều kiện phát hành và DGN-Open-Tiny phải re-pin generator source closure sau khi version literal đổi; chi tiết ở REV-20260902-001."
necessity: N3
impact: "Không sửa số liệu hay kết luận nào của MVP v0; scope v0 giữ nguyên scope_hash a897e360… và eval_manifest_hash ac57bf61…. Thêm scope v1, eval manifest v1, release-mode gate và version identity 0.1.0a2. DGN-Open-Tiny đổi 4 hash trong generator_source_hashes, không sample/shard nào bị chạm."
related_plan: ROADMAP-MVP-RELEASE-001
scope: mujoco_cpu_leap_state_cuboid_only
candidate_branch: feature/mvp-grasp-policy
candidate_commit: e68d70be7f5afc24d28bfec3537a3e23f9924fea
---

# SESSION-20260902-003 — MVP release MR-02: khóa contract release v1

## Mục tiêu phiên

- Tạo scope release v1 và eval manifest v1 mà không chạm scope v0.
- Đưa `release_class` thành giá trị máy đọc được và tách release gate khỏi
  experimental gate.
- Khóa candidate selection, paired comparison, Tier D, ablation và safety
  criteria **trước** khi tồn tại bất kỳ kết quả nào mà chúng phán xét.
- Sửa version identity về một nguồn khai báo duy nhất và một mapping.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
| --- | --- | --- | --- |
| `MR02-01` | Scope v1 với Tier D, challenge contract và release criteria; scope v0 không đổi một byte | `configs/mvp/dexacquire-mvp-v1.yaml`, `scope_hash` `47fbf8f0…` | verified |
| `MR02-02` | Eval manifest v1 A–D sinh bằng locker; locker lấy đường dẫn output từ scope thay vì mặc định vào manifest của scope khác | `configs/mvp/dexacquire-mvp-v1.eval-manifest.json`, `733ef3a0…` | verified |
| `MR02-03` | `release_class` máy đọc được (`release_candidate`) và parser fail-closed theo từng schema | `qdgrasp/mvp/config.py` | verified |
| `MR02-04` | Release-mode checker `--release`, đọc scope v1, tái tính từ raw ledger; experimental verdict tự khai không phải release verdict | `scripts/check_mvp.py` | verified |
| `MR02-05` | Khóa selection/paired/Tier D/ablation/safety; paired bootstrap deterministic có ghi seed và resample count | scope v1 `release` block, `qdgrasp/mvp/evaluate.py::paired_uplift` | verified |
| `MR02-06` | Revision record N3 cho việc v0 không đủ điều kiện phát hành | `REV-20260902-001` | verified |
| `MR02-07` | Version identity: `VERSION` là nguồn duy nhất, mapping `0.1.0a2` ↔ `0.1.0-alpha.2` ↔ `v0.1.0-alpha.2` | `VERSION`, `qdgrasp/version.py`, `scripts/check_version_identity.py`, `scripts/release_gate.sh` | verified |
| `MR02-08` | Test cho release script, version source duy nhất, contract v1 và release gate | `tests/test_version_identity.py`, `tests/mvp/test_mvp_release_contract.py`, `tests/mvp/test_check_mvp_release_gate.py` | verified |

### Quyết định thiết kế cần ghi lại

1. **Miền Tier D nằm ngoài scope v1.** §6 yêu cầu scope v1 bất biến trước MR-03,
   còn miền challenge chỉ được hiệu chuẩn *trong* MR-03. Hai yêu cầu đó chỉ
   đồng thời thỏa được nếu scope khóa *luật* (trục được phép, dải success của
   prior `0.40–0.85`, tối thiểu 50 failure, tối đa 3 cấu hình development) còn
   miền cụ thể nằm ở `configs/mvp/dexacquire-mvp-v1.challenge.json` do MR-03
   viết. `evaluate_tier` từ chối chạy tier challenge khi chưa được đưa miền đó,
   thay vì im lặng đo bằng randomization nền.
2. **Tier D dùng biến thể `train`.** Tier C đã đo generalization sang kích thước
   held-out. Nếu D cũng held-out thì “miền khó hơn” và “vật chưa từng thấy” bị
   trộn, và một tier bị nhiễu như vậy không mang được contribution claim.
3. **Manifest giữ tính thuần túy.** `eval_manifest()` không đọc file ngoài;
   ràng buộc miền challenge được kiểm bằng hash trong release checker, để manifest
   vẫn tái dựng được từ riêng scope document.
4. **v0 được tuần tự hóa theo đúng schema của nó.** Các trường v1 bị loại khỏi
   document v0 thay vì ghi thành `null`, vì `null` cũng là một document khác và
   sẽ làm trôi `scope_hash` mà ba vòng evidence đang neo vào.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- | --- |
| `MR02-E01` | commit contract release | tests + config + checker | `5e3c164a3a57532acfc44cd12316fe11d09981d1` |
| `MR02-E02` | commit re-pin DGN provenance | `datasets/dgn-open-tiny/dataset_manifest.json` | `e68d70be7f5afc24d28bfec3537a3e23f9924fea` |
| `MR02-E03` | scope v1 | `configs/mvp/dexacquire-mvp-v1.yaml` | `47fbf8f0bb786473993f9196d442a8e0fe08f8e7bdcdec4117a12cfd70aff11d` |
| `MR02-E04` | eval manifest v1 | `configs/mvp/dexacquire-mvp-v1.eval-manifest.json` | `733ef3a04073337b82e586b60e442cdc7bdf386e38578a4d389decfa78f8dc12` |
| `MR02-E05` | scope v0 không đổi | `configs/mvp/dexacquire-mvp-v0.yaml` | `a897e36084c9ab11cbf8046e446ef318c48b7463d8b2db4af4d4ca0593109b8b` |
| `MR02-E06` | release gate chặn artifact v0 | `scripts/check_mvp.py --release --runs runs/mvp` | 53/104, `blocked_with_evidence` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| `MR02-T01` | `.venv/bin/python -m pytest -q` | 0 | 1496 passed, 1 skipped, 105 subtests |
| `MR02-T02` | `.venv/bin/python scripts/check_static_core.py` | 0 | Ruff 136 file, Mypy 32 contract |
| `MR02-T03` | `.venv/bin/python scripts/check_docs.py --root .` | 0 | 158 tài liệu |
| `MR02-T04` | `.venv/bin/python scripts/check_wheel.py` | 0 | Wheel gate PASS |
| `MR02-T05` | `.venv/bin/python scripts/check_phase0.py` | 0 | Phase 0 PASS; version đọc từ `VERSION` thay vì literal |
| `MR02-T06` | `.venv/bin/python scripts/check_version_identity.py` | 0 | `0.1.0a2` / `0.1.0-alpha.2` / `v0.1.0-alpha.2` |
| `MR02-T07` | `.venv/bin/python scripts/lock_mvp_scope.py --check` | 0 | manifest v0 khớp scope v0 |
| `MR02-T08` | `.venv/bin/python scripts/lock_mvp_scope.py --scope configs/mvp/dexacquire-mvp-v1.yaml --check` | 0 | manifest v1 khớp scope v1 |
| `MR02-T09` | `.venv/bin/python scripts/check_mvp.py --root . --runs runs/mvp --release` | 1 | expected: artifact v0 không đạt contract v1 |

## Việc chưa hoàn tất

- Miền Tier D chưa tồn tại: `configs/mvp/dexacquire-mvp-v1.challenge.json` là
  việc của MR-03, và release gate hiện báo `challenge_domain_present` fail đúng
  như thiết kế.
- `docs/reports/MVP-GRASP-POLICY-MODEL-CARD-V1.md` chưa tồn tại; đó là MR-06.
- `.venv/bin/ruff check .` toàn repository vẫn còn 357 finding có sẵn, không
  finding nào thuộc file của MR-01/MR-02. Ma trận release MR-07 liệt kê lệnh này
  với yêu cầu exit 0; khoảng cách vẫn mở và phải xử lý ở MR-07.
- `scripts/generate_dgn_open_tiny.py` còn stamp `generator_version="0.1.0a1"`.
  Đó là trường provenance của một dataset đã publish, không phải version của
  package, nên cố ý không đổi: sửa nó sẽ làm sai lệch bản ghi của artifact cũ.

## Sửa đổi phiên trước

`REV-20260902-001` (N3) ghi hai việc. Thứ nhất, MVP v0 được kết luận là không
đủ điều kiện phát hành — không phải vì số liệu nào sai, mà vì trên A/B/C
controller prior đã bão hòa nên không có chỗ để đo đóng góp của phần learned;
đọc lại chính artifact v0 dưới contract mới cho thấy trên Tier C candidate thua
prior một success theo paired count, trong khi experimental gate vẫn báo tier đó
pass vì gate ấy chỉ cầm một arm. Không artifact, số liệu hay kết luận nào của v0
bị sửa.

Thứ hai, `datasets/dgn-open-tiny/dataset_manifest.json` được re-pin. `VERSION`
literal nằm trong `qdgrasp/__init__.py`, mà file đó nằm trong import closure
DGN-Open-Tiny ghi hash, nên bump version làm corpus mất khả năng tự chứng minh
và 6 test chuyển đỏ. Maintainer chọn re-pin closure thay vì regenerate dataset.
Bốn entry đổi (`qdgrasp/__init__.py`, `engine/checkpoint.py`, `engine/runner.py`
và `qdgrasp/version.py` mới vào closure); closure không bị thu hẹp; không sample,
shard hay object manifest nào bị chạm; hash lấy trên cây sạch tại đúng commit mà
manifest ghi.

## Bàn giao

MR-02 đạt exit gate: contract, threshold, seed policy và version mapping đều có
test; docs validator pass; và không tiêu chí nào phụ thuộc kết quả chưa chạy —
mọi ngưỡng tồn tại trước run mà nó phán xét, điều được bảo đảm bởi việc chưa có
locked evaluation v1 nào tồn tại.

Work package tiếp theo là MR-03: dùng development seed root
`QDGrasp-DexAcquire-MVP-v1/challenge-development/2026-09-02` để khảo sát tối đa
ba cấu hình challenge, tìm miền mà controller prior đạt `40%–85%` với ít nhất 50
failure đo được và zero safety violation, rồi khóa miền đó vào
`configs/mvp/dexacquire-mvp-v1.challenge.json` trước khi train candidate cuối.
Nếu sau ba cấu hình không tìm được miền hợp lệ, verdict là `NO-GO`.
