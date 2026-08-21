# Đóng góp vào DexGrasp

Mọi thay đổi đi qua branch riêng và commit có bằng chứng. Quy trình đầy đủ nằm
trong `docs/governance/GIT_WORKFLOW.md`.

## Thiết lập local

```bash
git config core.hooksPath .githooks
git config commit.template .gitmessage
git config merge.ff false
git config pull.ff only
```

## Feature

```bash
scripts/git/start_feature.sh yaml-parser
# triển khai, kiểm tra, ghi session/revision nếu cần
git add <paths>
git commit -m "feat(parser): add explicit YAML registry"
scripts/git/finish_feature.sh
```

Không ghi “hoàn tất” nếu chưa commit và chưa có test/evidence. Không commit trực
tiếp lên `main` hoặc `develop`.

## Release

```bash
scripts/git/start_release.sh 0.1.0
# cập nhật VERSION, CHANGELOG.md và release report; chỉ sửa lỗi ổn định
git commit -m "chore(release): prepare 0.1.0"
scripts/git/finish_release.sh 0.1.0
```

`main` chỉ chứa merge từ `release/*` hoặc `hotfix/*`; mỗi release có annotated
tag `vX.Y.Z`. Branch protection trên remote phải yêu cầu review và các gate ghi
trong tài liệu workflow.
