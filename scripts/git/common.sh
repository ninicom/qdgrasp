#!/usr/bin/env bash
set -euo pipefail

project_root="$(git rev-parse --show-toplevel)"
cd "$project_root"

require_clean_worktree() {
  local status_output
  status_output="$(git status --porcelain)"
  if [[ -n "$status_output" ]]; then
    printf '%s\n' "Working tree chưa sạch; hãy commit hoặc xử lý thay đổi:" >&2
    printf '%s\n' "$status_output" >&2
    exit 1
  fi
}

require_branch() {
  local branch_name="$1"
  if ! git show-ref --verify --quiet "refs/heads/$branch_name"; then
    printf '%s\n' "Thiếu branch bắt buộc: $branch_name" >&2
    exit 1
  fi
}

run_project_checks() {
  git diff --check
  PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_references.py --lock-only
  PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_train_args.py --registry-only
  if [[ -d .references ]]; then
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_references.py \
      --source-root .references
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_train_args.py \
      --source .references/ultralytics
  fi
  PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_static_core.py
  PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_docs.py --root .
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
    -s scripts/tests -p 'test_*.py' -v
}
