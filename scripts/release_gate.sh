#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_directory/git/common.sh"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  printf '%s\n' "Usage: scripts/release_gate.sh <semver>" >&2
  exit 2
fi

version="$1"
expected_branch="release/$version"
current_branch="$(git branch --show-current)"

if [[ "$current_branch" != "$expected_branch" ]]; then
  printf '%s\n' "Release gate chỉ chạy trên $expected_branch; hiện tại: $current_branch" >&2
  exit 1
fi

require_clean_worktree
require_branch main
require_branch develop

if [[ ! -f VERSION || "$(tr -d '[:space:]' < VERSION)" != "$version" ]]; then
  printf '%s\n' "VERSION phải chứa chính xác: $version" >&2
  exit 1
fi

if [[ ! -f CHANGELOG.md ]] || ! grep -Fq "## [$version]" CHANGELOG.md; then
  printf '%s\n' "CHANGELOG.md thiếu mục: ## [$version]" >&2
  exit 1
fi

if ! git merge-base --is-ancestor develop HEAD; then
  printf '%s\n' "Release branch không chứa HEAD hiện tại của develop." >&2
  exit 1
fi

run_project_checks
git diff --check main...HEAD

printf '%s\n' "Release gate $version: PASS"
