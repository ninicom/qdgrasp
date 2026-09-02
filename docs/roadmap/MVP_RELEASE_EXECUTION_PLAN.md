---
document_id: ROADMAP-MVP-RELEASE-001
document_type: plan
title: Kế hoạch thi công bắt buộc — MVP đến public release đầu tiên
version: 1.0.0
status: active
date: 2026-09-02
revises: none
related_plan:
  - PLAN-V2
  - ROADMAP-MVP-001
depends_on:
  - GOV-DOC-001
  - ROADMAP-MVP-001
release_target: 0.1.0-alpha.2
release_scope: mujoco_cpu_leap_state_cuboid_only
release_policy: no_release_before_mvp_release_gate
necessity: N3
impact: Public release bị chặn cho tới khi MVP v1 có provenance đầy đủ, chứng minh đóng góp của learned policy, zero safety violation, clean-install canary và independent PASS trên exact candidate.
---

# Kế hoạch thi công bắt buộc — MVP đến public release đầu tiên

## 0. Quyết định ràng buộc

Tài liệu này là execution contract duy nhất cho critical path từ trạng thái hiện
tại tới public release đầu tiên. Thứ tự work package ở §5 là bắt buộc. Không mở
workstream ngoài critical path, không chạy locked evaluation để thăm dò, không
đổi threshold sau khi thấy kết quả và không tạo release branch trước `MR-07`.

Quyết định phát hành:

1. Không phát hành package-only trước MVP.
2. Không đổi nhãn checkpoint/evidence v0 thành v1. Artifact tại
   `evidence/mvp/round-{1,2,3}` là lịch sử bất biến và
   `experimental_non_release`.
3. Public release đầu tiên chỉ hỗ trợ phạm vi hữu hạn:
   - MuJoCo CPU;
   - LEAP Hand profile đã pin;
   - privileged state observation;
   - một cuboid trên bàn trong miền được khóa bởi scope v1.
4. Robot thật, Allegro, Shadow, RGB/depth/point cloud, clutter, raw mesh,
   contact-rich dataset, MJX/GPU physics và sim-to-real đều **không được hỗ trợ**
   trong release này. Nếu nhu cầu đầu tiên là robot thật, plan dừng ở `MR-00`
   và phải có safety plan/hardware gate riêng trước khi tiếp tục.
5. Learned candidate phải chứng minh đóng góp trên challenge tier khóa trước.
   Việc controller prior một mình đạt A/B/C không đủ để gọi learned MVP là sẵn
   sàng phát hành.
6. Không có fallback âm thầm sang controller prior. Nếu learned candidate không
   đạt contribution gate, verdict là `NO-GO`; thay đổi sản phẩm thành
   controller-only cần quyết định N3 và plan/release claim mới.
7. Independent review phải là `PASS`, `max_severity: NONE`, trên exact clean
   commit và exact evidence manifest. `conditional_pass` không mở release.

Plan này không gỡ blocker của P3.4.3, P3.5, P4 hoặc P5 và không cho phép dùng
artifact đang `release_blocked`. Nó chỉ định nghĩa một release hẹp, có thể kiểm
chứng, cho MVP LEAP state-based.

## 1. Baseline tại thời điểm lập plan

| Hạng mục | Trạng thái quan sát | Ý nghĩa |
| --- | --- | --- |
| Full test suite | `1415 passed, 1 skipped, 93 subtests` | Code baseline hiện tại không có regression đã biết |
| Active-core static gate | Ruff + Mypy pass | Bề mặt core đủ sạch để đóng code candidate |
| Docs gate | 154 tài liệu pass | Governance checker đang hoạt động |
| Wheel/Phase 0 | pass | Wheel dựng và cài ngoài source tree được |
| Corrective registry | 12/13 finding closed | `COR-02` còn mở ngoài phạm vi MVP release hẹp |
| MVP artifact gate hiện tại | `50/75`, blocked | Demo/checkpoint/report v0 không đạt schema/lineage mới |
| MVP v0 behavior | A/B/C pass nhưng policy không hơn prior | Không đủ contribution claim |
| Worktree | dirty; feature branch chưa đồng bộ remote | Chưa đủ tư cách sinh release evidence |
| Independent review MVP | chưa có | Hard blocker |
| Release identity | `0.1.0a1` và `0.1.0-alpha.1` đang trộn | Phải sửa trước release branch |

Baseline pass không phải release evidence. Mọi số release phải được tạo lại từ
clean commit sau `MR-02`.

## 2. Định nghĩa “MVP hoàn tất để release”

MVP chỉ nhận verdict `GO` khi đồng thời thỏa tất cả điều kiện sau.

### 2.1 Source và artifact identity

- Worktree sạch; candidate commit tồn tại trên remote.
- Scope, eval manifest, prior, demonstration content, BC/PPO config, checkpoint,
  parent checkpoint, environment và code commit đều có SHA-256/digest trong
  lineage.
- Load dùng safe loader; mutation từng trường identity hoặc tensor bị từ chối
  trước episode đầu tiên.
- BC rollback artifact được giữ và hash trong release packet.
- Evidence được publish vào run ID mới; không ghi đè `round-1`…`round-3`.

### 2.2 Functional gate

| Tier | Mục đích | Episode tối thiểu | Gate |
| --- | --- | ---: | --- |
| A | canonical regression | 100 | candidate `>=95%`, không ít success hơn prior |
| B | randomized train domain | 300 | candidate `>=85%`, Wilson lower `>=80%`, không ít success hơn prior |
| C | held-out cuboid sizes | 200 | candidate `>=70%`, không ít success hơn prior |
| D | challenge domain nơi prior chưa bão hòa | 300 | candidate hơn prior ít nhất `5` percentage point và paired 95% CI lower `>0` |

Mọi tier bắt buộc có:

- `invalid_state = 0`;
- `safety_violation = 0`;
- `checkpoint_reload_mismatch = 0`;
- đúng số ledger row;
- exact checkpoint hash và fingerprint verdict `match`;
- failure bucket được tính lại từ raw ledger, không tin summary có sẵn.

Tier D được thiết kế bằng development seeds ở `MR-03`, rồi khóa trước training
candidate cuối và trước mọi locked evaluation. Phép ước lượng paired confidence
interval phải được cài thành script deterministic, có unit test và ghi seed.

### 2.3 Contribution gate

Candidate được gọi là learned MVP chỉ khi:

1. Tier D thỏa uplift gate ở §2.2.
2. A/B/C không regression theo paired success count.
3. Residual không suy biến về zero: model card phải báo distribution magnitude,
   saturation rate và ablation `prior` so với `prior + learned residual`.
4. Tắt learned residual phải làm mất phần improvement ở Tier D; nếu không,
   improvement không được quy cho model.
5. PPO chỉ được promote khi tốt hơn hoặc bằng BC trên từng A/B/C và thỏa Tier D.
   Dung sai cũ “không thấp hơn BC quá 2 pp” không được dùng cho release.

### 2.4 Package và user gate

- Full suite, active-core static, docs, security, wheel content và isolated
  install đều pass trên candidate commit.
- Fresh-environment canary chạy end-to-end: install wheel → load exact artifact
  → 100 canonical episodes → report; kết quả phải khớp contract và zero safety.
- Public API không tự tải artifact mutable hoặc artifact ngoài manifest.
- README/model card nêu đúng supported scope và hard limitations.
- Có rollback procedure đã rehearsal: pin previous tag, gỡ candidate artifact,
  dùng BC rollback hoặc dừng learned inference.

### 2.5 Review gate

- Review packet hash toàn bộ source/evidence/config/checkpoint cần thiết.
- Reviewer không viết/sửa candidate revision.
- Reviewer tái chạy tối thiểu manifest verification, safe-load/tamper probes,
  A canary, metric recomputation và wheel clean-install.
- Review report dùng `docs/templates/THIRD_PARTY_REVIEW_REPORT.md` và có verdict
  `pass`, `max_severity: NONE`.

## 3. Invariant không được thương lượng

1. Raw artifact bất biến; rerun tạo run ID mới.
2. Không sửa threshold, seed, tier hoặc scope sau khi xem locked result.
3. Không dùng locked seeds cho tuning, early stopping hoặc lựa chọn candidate.
4. Không migrate checkpoint v0 qua schema v1 như cùng policy.
5. Không chạy evidence từ dirty worktree hoặc commit chưa push.
6. Không đánh dấu pass bằng cách sửa report/summary; checker phải tái tính từ raw
   artifact.
7. Không nới safety budget để cứu success rate.
8. Không bỏ failed test, `xfail` hay gate để đạt lịch release.
9. Không phát hành nếu reviewer còn finding S0–S3 hoặc verdict có điều kiện.
10. Không tuyên bố robot thật/GPU/multi-hand ngoài supported scope.

Vi phạm một invariant làm invalid toàn bộ artifact sinh sau điểm vi phạm.

## 4. Critical path và WIP limit

```text
MR-00 Scope confirmation
  → MR-01 Close current lineage implementation
  → MR-02 Freeze MVP v1 release contract
  → MR-03 Calibrate and lock challenge tier D
  → MR-04 Generate demonstrations and train candidates
  → MR-05 Run locked evaluation once
  → MR-06 Seal evidence and independent review
  → MR-07 Build and canary the release candidate
  → MR-08 Merge, tag and publish
  → MR-09 Post-release verification and rollback watch
```

WIP limit là **một work package in progress**. Chỉ các test không mutate cùng
artifact mới được chạy song song bên trong một package. Không bắt đầu package
kế tiếp khi exit gate của package hiện tại chưa pass.

## 5. Work breakdown bắt buộc

### MR-00 — Xác nhận phạm vi sử dụng và hard stop

**Entry:** plan này được maintainer chấp nhận.

**Thực hiện:**

1. Xác nhận first-user workload nằm hoàn toàn trong supported scope §0.
2. Thêm release-scope statement vào release report draft.
3. Ghi rõ robot thật là unsupported và không có actuator command path công khai
   từ MVP artifact.
4. Chụp `git status`, branch topology và baseline gate output.

**Exit:** workload không cần robot thật hoặc tính năng ngoài scope; nếu có, dừng
plan và mở hardware/scope plan mới. Không được tự thu hẹp nhu cầu thực để pass.

**Output:** session report baseline, release report draft.

### MR-01 — Đóng implementation lineage hiện tại

**Entry:** MR-00 pass.

**Thực hiện:**

1. Review diff của tám file MVP đang sửa; tách mọi thay đổi ngoài COR-11.
2. Giữ demonstration schema/index/manifest v1 và checkpoint lineage aggregate.
3. Bổ sung negative tests tối thiểu:
   - đổi một byte observation/action/episode index;
   - sửa ledger, seed hoặc variant ID;
   - sửa scope/prior/environment fingerprint trong index;
   - sửa training config/dataset hash;
   - thay parent path hoặc parent checkpoint bytes;
   - archive có array thừa/thiếu hoặc object pickle.
4. Chạy target tests, full suite, Ruff, static core, docs và wheel gate.
5. Commit theo Conventional Commits; push feature candidate.

**Exit:** worktree sạch, full gate pass, commit có trên remote, rollback là parent
commit rõ ràng.

**Không được làm:** generate release evidence trước exit.

### MR-02 — Khóa contract MVP release v1

**Entry:** MR-01 clean/pushed.

**Thực hiện:**

1. Tạo scope mới `configs/mvp/dexacquire-mvp-v1.yaml`; không sửa scope v0.
2. Tạo eval manifest v1 bằng checker/locker, không copy rồi sửa hash thủ công.
3. Chọn release class đọc được bằng máy, tách khỏi
   `experimental_non_release`; cập nhật schema và parser fail-closed.
4. Thêm release-mode checker riêng hoặc mode rõ ràng cho `check_mvp.py`:
   experimental gate không được dùng làm release gate.
5. Khóa candidate selection, paired comparison, Tier D, ablation và safety
   criteria trước khi train.
6. Viết revision record N3 cho việc v0 không đủ release qualification; không
   sửa model card/report complete cũ.
7. Sửa version identity contract:
   - distribution `0.1.0a2`;
   - release/tag `0.1.0-alpha.2` / `v0.1.0-alpha.2`;
   - checker xác minh ánh xạ thay vì yêu cầu hai chuỗi giống nhau.
8. Thêm tests cho release script và version source duy nhất.

**Exit:** contract, threshold, seed policy và version mapping được test; docs
validator pass; review nội bộ xác nhận không có tiêu chí nào phụ thuộc kết quả
chưa chạy.

### MR-03 — Hiệu chuẩn rồi khóa Tier D

**Entry:** MR-02 pass; chưa train release candidate và chưa dùng locked seeds.

**Thực hiện:**

1. Dùng development-only seed root để khảo sát các trục đã khai báo trước:
   cuboid width/depth/height, yaw, friction, density và pose offset.
2. Không mở rộng sang raw mesh, clutter, tay khác hoặc observation khác.
3. Chọn challenge domain nơi controller prior đạt trong khoảng `40%–85%`, có
   ít nhất 50 failure đo được và zero safety violation.
4. Giải thích từng failure bucket; loại domain fail do simulator invalid hoặc
   safety budget không hợp lệ.
5. Khóa Tier D 300 seed mới, manifest và hash trước training cuối.
6. Test rằng train/dev/locked seed sets rời nhau hoàn toàn.

**Budget/stop rule:** tối đa ba cấu hình challenge development. Nếu không tìm
được domain hợp lệ sau ba cấu hình, `NO-GO`; không mở scope ngẫu hứng.

**Exit:** Tier D manifest immutable, prior report development có failure đủ để
đo uplift, zero safety/invalid và không có seed leakage.

### MR-04 — Sinh demo, train và chọn candidate bằng dev-only evidence

**Entry:** MR-03 pass; code commit sạch và đã push.

**Thực hiện theo đúng thứ tự:**

```bash
.venv/bin/python scripts/lock_mvp_scope.py --check
.venv/bin/python scripts/build_mvp_prior.py
.venv/bin/python scripts/generate_mvp_demos.py
.venv/bin/python scripts/train_mvp_policy.py
```

Các script phải nhận explicit scope/output/run ID; lệnh trên chỉ là interface
tối thiểu, session report phải ghi lệnh thực tế đầy đủ.

1. Generator tạo train/dev demonstrations v1 và content manifest.
2. Trainer xác minh index + cả split trước khi tạo output directory.
3. Train BC; xác minh reload probe và dev gates.
4. Train tối đa một PPO candidate từ exact BC parent.
5. Chọn candidate chỉ bằng train/dev/challenge-dev, không đọc locked A–D.
6. So BC, PPO và prior trên cùng dev seeds; lưu paired ledger.
7. Nếu PPO không hơn hoặc bằng BC ở mọi regression tier, chọn BC. Nếu không
   candidate learned nào có tín hiệu uplift Tier D development, `NO-GO`.

**Budget/stop rule:** một BC run và một PPO run cho release attempt. Bug làm run
invalid được phép rerun sau fix/commit mới; kết quả kém không phải bug và không
được mở tuning loop ngoài plan.

**Exit:** candidate path được khóa, checkpoint hash được ghi, BC rollback giữ
nguyên, lineage chain pass và candidate chưa từng thấy locked seeds.

### MR-05 — Locked evaluation một lần

**Entry:** MR-04 candidate đã khóa trên clean pushed commit.

**Thực hiện:**

1. Đánh giá controller prior trên A–D.
2. Đánh giá BC rollback trên A–D.
3. Đánh giá final candidate trên A–D đúng một lần.
4. Chạy ablation tắt learned residual trên exact candidate trajectory contract.
5. Tái tính aggregate, Wilson/paired CI và failure buckets từ ledger.
6. Chạy release-mode MVP checker; không chỉnh artifact sau khi checker đọc.

**Exit `GO`:** toàn bộ §2.1–§2.3 pass và release checker không có failure.

**Exit `NO-GO`:** bất kỳ functional, contribution, safety hoặc identity gate
fail. Giữ nguyên evidence và viết failure report; không tune trên locked result.
Một attempt mới cần scope/candidate version mới và revision record.

### MR-06 — Seal evidence và independent review

**Entry:** MR-05 `GO`.

**Thực hiện:**

1. Publish evidence vào `evidence/mvp/release-v1/<run-id>/`.
2. Manifest gồm path, bytes, SHA-256, generator command, timestamp, code commit.
3. Model card mới không sửa model card v0; báo prior/BC/PPO/ablation và mọi
   limitation ở phần đầu.
4. Tạo review packet không self-hash, bind exact candidate commit và manifest.
5. Reviewer độc lập tái chạy checklist §2.5.
6. Finding phải được sửa trên commit mới, regenerate evidence bị ảnh hưởng và
   delta/full re-review theo severity.

**Exit:** signed/authenticated review report `pass/NONE`; packet verifier pass;
không open condition.

### MR-07 — Build, clean-install và first-user canary

**Entry:** MR-06 pass.

**Thực hiện:**

1. Merge feature vào local `develop` tracking `origin/develop` theo GitFlow.
2. Tạo `release/0.1.0-alpha.2` từ exact reviewed develop commit.
3. Chỉ thay release metadata/docs; code behavior change làm invalid review.
4. Build wheel và sdist trong clean temporary source tree.
5. Cài wheel `--no-deps` vào isolated target để kiểm content, rồi cài đầy đủ
   trong fresh environment để chạy CLI/API canary.
6. Chạy full release matrix:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/check_static_core.py
.venv/bin/python scripts/check_docs.py --root .
.venv/bin/python scripts/check_wheel.py
.venv/bin/python scripts/check_phase0.py
.venv/bin/python scripts/check_mvp.py --release --runs <sealed-evidence>
scripts/release_gate.sh 0.1.0-alpha.2
```

7. Fresh-install canary chạy 100 Tier A episode bằng exact packaged code và
   exact candidate artifact; so checkpoint hash và zero safety.
8. Rehearse rollback trong môi trường tạm, không sửa release artifact.

**Exit:** mọi lệnh exit 0, canary pass, wheel/sdist hash có trong release report,
worktree sạch.

### MR-08 — Merge, tag và publish

**Entry:** MR-07 pass; maintainer ra quyết định `release` trong release report.

**Thực hiện:**

1. Merge release branch vào `main` bằng workflow đã kiểm.
2. Tạo annotated tag `v0.1.0-alpha.2` trên exact reviewed release commit.
3. Merge ngược release branch vào `develop`.
4. Push branch/tag; xác minh remote tag trỏ đúng commit.
5. Publish GitHub Release gồm wheel/sdist, checksum, model card, supported scope,
   limitations và rollback instructions.
6. Publish registry/package index chỉ từ artifact đã hash nếu credentials và
   quyền sở hữu đã xác minh. Không rebuild artifact ở bước upload.

**Exit:** remote source/tag/artifact hashes khớp release report; clean install từ
nguồn public pass.

### MR-09 — Hậu kiểm và rollback watch

**Entry:** MR-08 publish thành công.

**Thực hiện:**

1. Cài từ public endpoint trong fresh environment và chạy smoke/canary cuối.
2. Ghi mọi first-user run bằng version, environment fingerprint, checkpoint hash
   và failure bucket; không ghi secret hoặc dữ liệu cá nhân.
3. Kích hoạt rollback nếu xảy ra một trong các điều kiện:
   - safety violation;
   - artifact hash/fingerprint mismatch;
   - clean install/import/load thất bại;
   - success canonical dưới gate;
   - regression không có trong release limitations.
4. Rollback bằng cách đánh dấu release/artifact bị ảnh hưởng, pin previous safe
   version hoặc dừng learned inference; không xóa tag hay rewrite artifact.
5. Mở N3 revision nếu ảnh hưởng an toàn/release claim.

**Exit:** public artifact tái lập được và không có rollback trigger mở. P3–P7
tiếp tục theo roadmap riêng; alpha.2 không tự động mở rộng supported scope.

## 6. Ma trận artifact bắt buộc

| Artifact | Tạo ở | Checker | Bất biến sau |
| --- | --- | --- | --- |
| MVP scope v1 | MR-02 | scope/schema/hash tests | trước MR-03 |
| Eval manifests A–D | MR-02/MR-03 | lock checker + seed-disjoint test | trước MR-04 |
| Demonstration arrays/ledger/manifest | MR-04 | content hash + raw reload | khi generator kết thúc |
| BC checkpoint | MR-04 | safe load + reload probe + lineage | khi BC kết thúc |
| PPO checkpoint | MR-04 | BC parent hash + config/data lineage | khi PPO kết thúc |
| Candidate selection record | MR-04 | dev-only audit | trước MR-05 |
| Locked ledgers A–D | MR-05 | row count + recomputation | mỗi evaluation kết thúc |
| Evidence manifest | MR-06 | manifest verifier | trước review |
| Independent review | MR-06 | review/packet verifier | trước release branch |
| Wheel/sdist | MR-07 | content + isolated install | trước upload |
| Release report/tag | MR-08 | release gate + remote ref check | lúc publish |

## 7. Ma trận quyết định GO/NO-GO

| Tình huống | Quyết định bắt buộc |
| --- | --- |
| A/B/C pass, D không uplift | `NO-GO`; learned MVP chưa chứng minh giá trị |
| Candidate tốt hơn D nhưng kém prior ở A/B/C | `NO-GO`; không đổi reliability lấy novelty |
| Bất kỳ safety/invalid/reload mismatch | `NO-GO`; N3 audit nếu artifact đã chia sẻ |
| PPO kém BC | rollback candidate về BC chỉ khi BC vẫn đạt toàn bộ gate, kể cả D |
| BC và PPO đều không đạt D | `NO-GO`; không fallback prior âm thầm |
| Gate pass nhưng review conditional/fail | `NO-GO` |
| Review pass nhưng release branch đổi behavior | review hết hiệu lực; quay lại MR-06 |
| Clean install fail | `NO-GO`; sửa package rồi chạy lại MR-07 |
| Robot thật được yêu cầu | plan không áp dụng; quay lại MR-00 và mở hardware plan |

## 8. Kiểm soát thời gian và chống làm lại

1. Mỗi package có một owner, một entry gate và một exit verdict; không mở task
   “tiện thể”.
2. Mọi thay đổi contract hoàn tất ở MR-02. Từ MR-03 trở đi chỉ sửa bug làm run
   invalid; thay đổi semantics quay lại MR-02 và bump artifact version.
3. Challenge exploration tối đa ba cấu hình; training tối đa BC + một PPO;
   locked evaluation một lần cho mỗi final candidate.
4. Cache chỉ được dùng khi content hash và environment fingerprint khớp; cache
   miss làm chạy lại, không sửa manifest.
5. Log lệnh, thời gian, exit code và hash ngay khi chạy; không tái dựng bằng trí
   nhớ cuối phiên.
6. Review packet sinh bằng script, không chép tay số liệu.
7. Release artifact build đúng một lần; upload lại cùng bytes nếu endpoint lỗi.
8. Mọi `NO-GO` giữ evidence để quyết định nguyên nhân; không lặp cùng một run
   không thay đổi giả thuyết.

## 9. Checklist release cuối cùng

- [ ] MR-00 supported scope được first user xác nhận.
- [ ] MR-01 lineage code sạch, test đầy đủ, commit đã push.
- [ ] MR-02 scope/schema/version/release checker v1 khóa trước kết quả.
- [ ] MR-03 Tier D khóa và không seed leakage.
- [ ] MR-04 demo/BC/PPO/candidate có complete lineage.
- [ ] MR-05 A–D, contribution, safety và identity đều pass.
- [ ] MR-06 evidence sealed; independent review `PASS/NONE`.
- [ ] MR-07 full matrix, clean install, canary và rollback rehearsal pass.
- [ ] MR-08 release report kết luận `release`; tag/artifact remote khớp hash.
- [ ] MR-09 public install verification pass; zero rollback trigger mở.

Thiếu một checkbox thì kết luận duy nhất hợp lệ là `do_not_release`.
