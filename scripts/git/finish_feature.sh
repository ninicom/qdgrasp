#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_directory/common.sh"

feature_branch="$(git branch --show-current)"
if [[ ! "$feature_branch" =~ ^feature/ ]]; then
  printf '%s\n' "Lệnh này chỉ chạy trên feature/*; hiện tại: $feature_branch" >&2
  exit 1
fi

require_clean_worktree
require_branch develop
run_project_checks

merge_base="$(git merge-base develop HEAD)"
git diff --check "$merge_base" HEAD

git switch develop
git merge --no-ff "$feature_branch" \
  -m "Merge $feature_branch into develop"
run_project_checks

printf '%s\n' \
  "Đã merge $feature_branch vào develop." \
  "Branch feature được giữ lại để audit; chỉ xóa sau khi release ổn định."
