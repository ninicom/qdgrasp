---
document_id: SESSION-20260821-001
document_type: session_report
title: Lưu PLAN-V2 và khởi tạo bộ tài liệu kiểm chứng
status: complete
date: 2026-08-21
session_id: SESSION-20260821-001
author: codex-primary-agent
revises: LEGACY-PLAN-PRE-V2
related_plan: PLAN-V2
revision_reason: Kế hoạch legacy không được chấp nhận hoàn toàn và không đáp ứng yêu cầu source clone chuẩn, kiến trúc YAML mới cùng khả năng chạy CPU/CUDA.
necessity: N3
impact: Thay kế hoạch và quy trình quản trị trước khi có code model; không sửa checkpoint, dữ liệu hoặc kết quả benchmark.
---

# SESSION-20260821-001 — Lưu plan và khởi tạo tài liệu kiểm chứng

## Mục tiêu phiên

- Lưu kế hoạch kiến trúc mới vào `PLAN.md` và không làm mất kế hoạch legacy.
- Thiết lập quy tắc báo số liệu, báo cáo kiểm tra độc lập, ghi việc hoàn tất theo
  phiên và sửa việc của phiên trước.
- Cung cấp validator có thể chạy bằng Python standard library.
- Chỉ đóng phiên khi cấu trúc tài liệu pass và có báo cáo audit độc lập nội bộ;
  không coi đây là kiểm chứng runtime/model.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
|---|---|---|---|
| W-01 | Bảo tồn nguyên byte kế hoạch legacy trước khi thay | `docs/archive/PLAN.pre-v2.md` | verified bằng SHA-256 |
| W-02 | Ghi kế hoạch V2 gồm kiến trúc thư viện, YAML backbone/neck/head, model flow, CPU/CUDA, export và roadmap | `PLAN.md` | verified cấu trúc và hash |
| W-03 | Thiết lập chính sách tài liệu, phiên, review và revision N0–N3 | `docs/governance/` | validator pass; đã qua audit v1 |
| W-04 | Thiết lập Metrics Registry và mẫu báo cáo số liệu có protocol, sample size, CI, artifact và SHA-256 | `docs/metrics/` và `docs/templates/METRICS_REPORT.md` | validator pass |
| W-05 | Thiết lập mẫu session, revision và independent review | `docs/templates/` | validator pass; negative test pass |
| W-06 | Ghi ADR truy xuất bằng chứng và các chỉ mục lưu trữ | `docs/decisions/0001-documentation-and-traceability.md`, `docs/README.md` | verified bằng kiểm tra thủ công |
| W-07 | Tạo validator không phụ thuộc thư viện ngoài | `scripts/check_docs.py`, `docs/schemas/README.md` | validator self-check và py_compile pass |
| W-08 | Lập hồ sơ N3 cho việc thay kế hoạch legacy | `docs/revisions/REV-20260821-001-plan-v2.md` | validator pass |
| W-09 | Thực hiện audit nội bộ độc lập trên snapshot v1 | `docs/reviews/TPR-20260821-001-documentation-suite.md` | `conditional_pass`, bốn finding đã được ghi bất biến |
| W-10 | Sửa bốn finding audit: khóa session ở scope kế tiếp, đồng bộ metadata/status, chặn placeholder và thêm archive sidecar | policy, validator, test và `docs/archive/README.md` | local gate pass; chờ delta-review trên snapshot v2 |

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
|---|---|---|---|
| E-01 | archive | `docs/archive/PLAN.pre-v2.md` | `d4d8b1d51723c830bab041575388cd58984e60b77e06013844ca02f03220d36d` |
| E-02 | plan | `PLAN.md` | `797c3b9ae17f8eb5975f6018b377e3b30c8bc5419920000502055271cf38384d` |
| E-03 | revision | `docs/revisions/REV-20260821-001-plan-v2.md` | xem front matter và mục Xác minh của revision record |
| E-04 | snapshot v1 | `docs/reviews/evidence/TPR-20260821-001-scope.sha256` | `6f845d453287ea332958c7b343cc2b55c92c324b6d3e4eed8c27a9b9d1b1abeb` |
| E-05 | validator | `scripts/check_docs.py` | `2f87c77d70025412ae44f4665071f4b646c4d907fa81a2177ee2bf5f7d6895c3` |
| E-06 | audit v1 | `docs/reviews/TPR-20260821-001-documentation-suite.md` | `2f5e995851d683f6db30717e2ab628e5ba973a8d10aef4bc2f9cbb2331dfa5b5` |
| E-07 | regression test | `scripts/tests/test_check_docs.py` | `fd298aaa25cea581f773b71501169de3f5ac1327b663a93dae51736b543eb6f8` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
|---|---|---:|---|
| T-01 | `sha256sum PLAN.md docs/archive/PLAN.pre-v2.md` | 0 | Plan V2 có hash `797c…384d`; archive giữ hash legacy `d4d8…d36d` |
| T-02 | `PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_docs.py --root .` | 0 | `Documentation check passed: 16 file đã kiểm tra.` |
| T-03 | `PYTHONDONTWRITEBYTECODE=1 python3 -c "import py_compile; py_compile.compile('scripts/check_docs.py', cfile='/tmp/dexgrasp-check_docs.pyc', doraise=True)"` | 0 | Validator biên dịch được; bytecode nằm ngoài workspace |
| T-04 | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` | 0 | 3/3 regression test pass, gồm negative placeholder test |
| T-05 | `sha256sum PLAN.md docs/archive/PLAN.pre-v2.md` | 0 | Hash plan mới và raw archive vẫn khớp revision record |

## Việc chưa hoàn tất

- Kiểm tra bởi tổ chức/người bên ngoài (`independence: external`) chưa được thực
  hiện. Phiên này chỉ có thể yêu cầu một agent khác làm
  `internal_independent`; hai loại không được đánh tráo.
- Chưa clone các upstream, chưa kiểm license thực tế, chưa viết code model và
  chưa chạy CPU/GPU benchmark. Các việc đó thuộc M0 trở đi của `PLAN-V2`.
- Các con số tham số/latency/success trong kế hoạch chỉ là target hoặc gate;
  chưa phải kết quả đo.
- Delta-review snapshot v2 được thực hiện sau khi record phiên này được đóng băng;
  kết quả nằm trong report review mới và không được chèn ngược vào record này.

## Sửa đổi phiên trước

Có. Hồ sơ `docs/revisions/REV-20260821-001-plan-v2.md` thay kế hoạch legacy bằng
`PLAN-V2`:

- claim cũ: phụ thuộc Ultralytics runtime và cải tiến pipeline DGN2 theo kiến
  trúc legacy;
- claim mới: package độc lập dựa trên source pin, model YAML mới thuần PyTorch,
  core CPU/CUDA và gate license trước port/phân phối;
- nguyên nhân: yêu cầu người dùng và các giả định kỹ thuật/pháp lý không còn phù
  hợp;
- mức cần thiết: `N3`, vì thay đổi quyết định kiến trúc và điều kiện phát hành;
- tác động: plan/roadmap và tiêu chí nghiệm thu; chưa có code hay số liệu runtime
  bị sửa.

## Bàn giao

- Nguồn tài liệu bắt đầu tại `docs/README.md`; plan active là `PLAN-V2`.
- Trước mỗi claim số liệu, đăng ký metric/protocol, lưu raw artifact và hash, rồi
  dùng `docs/templates/METRICS_REPORT.md`.
- Trước khi đóng một phiên, tạo session report và chạy
  validator cùng regression tests đã ghi ở T-02/T-04.
- Snapshot delta tiếp theo phải chứa chính file session này để đóng finding F-001
  của audit v1.
- Bước triển khai hợp lệ tiếp theo là M0: xác minh license/URL/SHA, clone source
  sạch vào `.references/`, tạo `references.lock.yaml` và đo baseline. Không được
  ghi M0 “hoàn tất” trước khi có manifest và log kiểm chứng.
