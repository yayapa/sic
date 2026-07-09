#!/usr/bin/env bash
# Pre-commit entrypoint: run without requiring `sic-strip-notebooks` on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$ROOT/src/sic/strip_notebooks.py" "$@"
