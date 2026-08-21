#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_directory/common.sh"

if [[ $# -ne 1 ]]; then
  printf '%s\n' "Usage: scripts/git/finish_release.sh <semver>" >&2
  exit 2
fi

version="$1"
release_branch="release/$version"
current_branch="$(git branch --show-current)"

if [[ "$current_branch" != "$release_branch" ]]; then
  printf '%s\n' "Phải đứng trên $release_branch; hiện tại: $current_branch" >&2
  exit 1
fi

require_clean_worktree
require_branch main
require_branch develop

if git rev-parse "v$version" >/dev/null 2>&1; then
  printf '%s\n' "Tag đã tồn tại: v$version" >&2
  exit 1
fi

scripts/release_gate.sh "$version"

git switch main
git merge --no-ff "$release_branch" \
  -m "Merge $release_branch into main"
git tag -a "v$version" -m "Release $version"

git switch develop
git merge --no-ff "$release_branch" \
  -m "Merge $release_branch back into develop"
run_project_checks

printf '%s\n' "Release v$version đã merge vào main, tag và merge ngược develop."
