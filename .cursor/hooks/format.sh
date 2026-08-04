#!/bin/bash
# afterFileEdit hook: runs black (and isort, for pennylane/tests files) on
# edited Python files, mirroring the commands documented in AGENTS.md.
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.file_path // empty')

# Only format Python files that still exist (e.g. skip deleted files).
if [[ -z "$file_path" || "$file_path" != *.py || ! -f "$file_path" ]]; then
  exit 0
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

# Prefer the project's virtualenv so tool versions match what's pinned there.
black_bin="black"
isort_bin="isort"
if [[ -x "$root/.venv/bin/black" ]]; then
  black_bin="$root/.venv/bin/black"
fi
if [[ -x "$root/.venv/bin/isort" ]]; then
  isort_bin="$root/.venv/bin/isort"
fi

"$black_bin" --config ./pyproject.toml "$file_path" >/dev/null 2>&1 || true

# isort is only configured for pennylane/ and tests/ (see AGENTS.md).
rel_path=${file_path#"$root"/}
if [[ "$rel_path" == pennylane/* || "$rel_path" == tests/* ]]; then
  "$isort_bin" --settings-path ./pyproject.toml "$file_path" >/dev/null 2>&1 || true
fi

exit 0
