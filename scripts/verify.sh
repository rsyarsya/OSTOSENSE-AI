#!/usr/bin/env bash
# Compatibility entry point; the implementation is shared across platforms.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/ai/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/ai/.venv/bin/python"
else
  PYTHON_BIN=python3
fi
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/verify.py" "$@"
