#!/bin/bash
# Install git hooks that strip notebook outputs on commit and push.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if ! command -v pre-commit &>/dev/null; then
  echo "pre-commit not found. Install dev dependencies first:" >&2
  echo "  pip install -e '.[dev]'" >&2
  exit 1
fi
pre-commit install --hook-type pre-commit --hook-type pre-push
echo "Installed pre-commit hooks (strip notebook outputs on commit and push)."
