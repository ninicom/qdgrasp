#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_directory/common.sh"

if [[ $# -ne 1 || ! "$1" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  printf '%s\n' "Usage: scripts/git/start_feature.sh <feature-slug>" >&2
  exit 2
fi

require_clean_worktree
require_branch develop

feature_branch="feature/$1"
if git show-ref --verify --quiet "refs/heads/$feature_branch"; then
  printf '%s\n' "Branch đã tồn tại: $feature_branch" >&2
  exit 1
fi

git switch develop
git switch -c "$feature_branch"
printf '%s\n' "Đã tạo $feature_branch từ develop."
