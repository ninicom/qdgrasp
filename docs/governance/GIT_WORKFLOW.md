---
document_id: GOV-GIT-001
document_type: policy
title: Quy trình Git theo feature, develop và release
version: 1.1.0
status: active
date: 2026-08-21
owner: project-maintainer
revises: none
latest_revision_record: docs/revisions/REV-20260821-006-audit-remediation.md
---

# Quy trình Git theo feature, develop và release

## 1. Mô hình branch

| Branch | Nguồn tạo | Mục đích | Đích merge |
|---|---|---|---|
| `main` | khởi tạo một lần | Chỉ commit đã phát hành; mỗi trạng thái release có tag | Không nhận feature trực tiếp |
| `develop` | `main` | Tích hợp feature đã qua gate | `release/*` |
| `feature/<slug>` | `develop` | Một tính năng hoặc thay đổi độc lập | `develop` bằng `--no-ff` |
| `release/<semver>` | `develop` | Ổn định, version, changelog, release report | `main`, rồi merge ngược `develop` |
| `hotfix/<semver>` | `main` | Sửa lỗi release khẩn cấp | `main`, rồi merge ngược `develop` |

Không commit trực tiếp lên `main` hoặc `develop`. Không merge feature này vào
feature khác để né integration gate. Nếu có phụ thuộc, ghi rõ trong session
report và rebase/merge từ `develop` theo quyết định của maintainer.

## 2. Khi nào công việc được coi là xong

Một đơn vị công việc chỉ được ghi “hoàn tất” khi đồng thời:

1. output đã tồn tại trên `feature/*`;
2. test/evidence tương ứng đã pass;
3. session report đã ghi file, lệnh, exit code và phần chưa hoàn tất;
4. sửa việc phiên trước đã có `REV-...`, reason, `N0`–`N3` và impact;
5. thay đổi đã được commit bằng Conventional Commit;
6. working tree sạch.

Mỗi đơn vị hoàn tất phải có commit riêng có ý nghĩa. Không gom nhiều feature
không liên quan vào một commit. WIP có thể commit để bảo toàn công việc nhưng
không được dùng làm bằng chứng “đã xong” hoặc merge vào `develop`.

## 3. Vòng đời feature

```bash
scripts/git/start_feature.sh <slug>
# triển khai và kiểm tra
git add <đúng-path-thuộc-feature>
git commit -m "feat(scope): mô tả kết quả"
scripts/git/finish_feature.sh
```

`finish_feature.sh` từ chối working tree bẩn, chạy validator/unit tests, kiểm
whitespace từ merge-base, rồi merge `--no-ff` vào `develop` và chạy lại gate.
Branch feature được giữ tới khi release ổn định để audit; có thể xóa sau đó vì
merge commit và commit feature vẫn còn trong lịch sử.

## 4. Quy ước commit

Subject dùng:

```text
<type>(<scope>): <kết quả đã hoàn tất>
```

Type hợp lệ: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`,
`chore`, `revert`. Merge commit được tạo bởi workflow là ngoại lệ. Hook
`.githooks/commit-msg` kiểm subject; `.githooks/pre-commit` chạy whitespace,
documentation validator và regression tests.

## 5. Gate của develop

Một feature chỉ được merge khi:

- branch bắt nguồn từ `develop` và working tree sạch;
- mọi commit hoàn tất có session/evidence tương ứng;
- `git diff --check` pass;
- `python3 scripts/check_references.py --lock-only` pass;
- `python3 scripts/check_train_args.py --registry-only` pass;
- `python3 scripts/check_docs.py --root .` pass;
- `python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v` pass;
- finding S0/S1 không còn mở; finding còn lại có disposition rõ.

Khi `.references/` có mặt, hook và `run_project_checks()` bắt buộc chạy cả
`scripts/check_references.py --source-root .references` và full argument audit
với `--source .references/ultralytics`. Nếu clone bị loại khỏi source archive,
lock-only/registry-only vẫn chạy; CI release có source checkout phải cung cấp
hai clone bắt buộc. Session/release evidence phải lưu output full check.

Merge bắt buộc `--no-ff` để giữ ranh giới feature. Nếu gate sau merge thất bại,
không tạo release branch; sửa bằng feature/fix branch mới, không rewrite lịch sử
`develop` đã chia sẻ.

## 6. Vòng đời release

```bash
scripts/git/start_release.sh <semver>
# chỉ sửa version, changelog, tài liệu release và lỗi ổn định
git commit -m "chore(release): prepare <semver>"
scripts/git/finish_release.sh <semver>
```

Release branch phải có:

- `VERSION` khớp tên branch;
- mục `## [<semver>]` trong `CHANGELOG.md`;
- release report theo `docs/templates/RELEASE_REPORT.md`;
- toàn bộ gate feature/develop chạy lại trên commit release;
- phạm vi/giới hạn được ghi rõ, đặc biệt không biến target model thành kết quả.

`finish_release.sh` chạy gate, merge `--no-ff` vào `main`, tạo annotated tag
`v<semver>`, merge release ngược vào `develop`, rồi kiểm lại. Release alpha của
bootstrap tài liệu không được mô tả như bản model/runtime hoàn chỉnh.

## 7. Hotfix

Hotfix tạo từ `main` với tên `hotfix/<semver>`, có revision/session/report như
feature, được test đầy đủ, merge vào `main`, tag và merge ngược `develop`. Không
cherry-pick riêng lẻ rồi bỏ quên nhánh tích hợp.

## 8. Cấu hình repository và remote

Cấu hình local bắt buộc:

```bash
git config core.hooksPath .githooks
git config commit.template .gitmessage
git config merge.ff false
git config pull.ff only
```

Khi có remote, bật branch protection cho `main` và `develop`: cấm force-push,
cấm xóa, yêu cầu pull request, ít nhất một reviewer không phải tác giả, yêu cầu
status checks của docs/tests/release gate, và yêu cầu branch cập nhật trước
merge. Workspace hiện chưa có remote nên các rule server-side phải được cấu
hình khi remote được thêm; local hooks không thay thế branch protection.

## 9. Truy xuất release

Release report phải liệt kê feature merge commits, source commit của `develop`,
release commit/tag, lệnh gate, artifact hash, finding còn lại và rollback. Nếu
sửa một release report `complete/accepted`, tạo revision record; không sửa tag
hoặc rewrite `main` để che lịch sử.
