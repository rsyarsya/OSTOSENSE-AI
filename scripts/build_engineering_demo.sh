#!/usr/bin/env bash
# Legacy positional-output interface for the portable reproduction command.
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
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/reproduce.py" --output "${1:-$ROOT_DIR/outputs/engineering-demo-v0.3}"
