#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_directory/common.sh"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  printf '%s\n' "Usage: scripts/git/start_release.sh <semver>" >&2
  exit 2
fi

version="$1"
release_branch="release/$version"

require_clean_worktree
require_branch develop
run_project_checks

if git show-ref --verify --quiet "refs/heads/$release_branch"; then
  printf '%s\n' "Branch đã tồn tại: $release_branch" >&2
  exit 1
fi

git switch develop
git switch -c "$release_branch"
printf '%s\n' "Đã tạo $release_branch từ develop."
