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

# VERSION holds the PEP 440 distribution version and "$version" is the SemVer
# release string; they are the same release in two notations that cannot be
# string-equal.  The checker verifies the mapping between them, and that the
# tree has exactly one place that declares it.
if ! PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_version_identity.py --release "$version"; then
  printf '%s\n' "Version identity không khớp release $version" >&2
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
