#!/bin/bash
# Shared editor-agent format hook: runs black (and isort, for pennylane/tests
# files) on edited Python files, mirroring .pre-commit-config.yaml.
#
# Used by both:
#   - Cursor  .cursor/hooks.json     afterFileEdit
#   - Claude  .claude/settings.json  PostToolUse (Write|Edit)
#
# The two send different event JSON on stdin, so accept either shape.
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '
  .file_path                # Cursor
  // .tool_response.filePath  # Claude, post-edit
  // .tool_input.file_path    # Claude, fallback
  // empty
')

# Only format Python files that still exist (e.g. skip deleted files).
if [[ -z "$file_path" || "$file_path" != *.py || ! -f "$file_path" ]]; then
  exit 0
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
rel_path=${file_path#"$root"/}

# Prefer the project's virtualenv so tool versions match what's pinned there.
black_bin="black"
isort_bin="isort"
if [[ -x "$root/.venv/bin/black" ]]; then
  black_bin="$root/.venv/bin/black"
fi
if [[ -x "$root/.venv/bin/isort" ]]; then
  isort_bin="$root/.venv/bin/isort"
fi

# black skips doc/ (pre-commit: exclude ^doc/).
if [[ "$rel_path" != doc/* ]]; then
  "$black_bin" --config ./pyproject.toml "$file_path" >/dev/null 2>&1 || true
fi

# isort only covers pennylane/ and tests/ (pre-commit: files ^(pennylane/|tests/)).
if [[ "$rel_path" == pennylane/* || "$rel_path" == tests/* ]]; then
  "$isort_bin" --settings-path ./pyproject.toml "$file_path" >/dev/null 2>&1 || true
fi

exit 0
