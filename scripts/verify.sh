#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/ai/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python tidak ditemukan di $PYTHON_BIN" >&2
  echo "Buat environment lalu install: cd ai && python3.11 -m venv .venv && .venv/bin/pip install -e '.[pipeline,quality]'" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("OSTOSENSE AI memerlukan Python 3.11 atau lebih baru")

missing = []
for module in ("build", "jsonschema", "pyright", "ruff"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Quality tools belum lengkap: " + ", ".join(missing)
        + ". Jalankan: cd ai && .venv/bin/pip install -e '.[pipeline,quality]'"
    )
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT_DIR/ai/src"

echo "[1/6] Python tests"
"$PYTHON_BIN" -W error -m unittest discover -s "$ROOT_DIR/ai/tests"

echo "[2/6] Ruff correctness gate"
(cd "$ROOT_DIR/ai" && "$PYTHON_BIN" -m ruff check src tests)

echo "[3/6] Pyright"
(cd "$ROOT_DIR/ai" && "$PYTHON_BIN" -m pyright --pythonpath "$PYTHON_BIN")

echo "[4/6] Python package build and clean-wheel import"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
(cd "$ROOT_DIR/ai" && "$PYTHON_BIN" -m build --outdir "$TEMP_DIR/dist")
"$PYTHON_BIN" -m venv "$TEMP_DIR/smoke-venv"
"$TEMP_DIR/smoke-venv/bin/python" -m pip install --no-deps "$TEMP_DIR"/dist/*.whl >/dev/null
(cd "$TEMP_DIR" && "$TEMP_DIR/smoke-venv/bin/python" -c "import ostosense_ai, ostosense_contract")

echo "[5/6] Portable C++17 tests"
for name in data_contract ordinal_inference capacitive_features; do
  g++ -std=c++17 -Wall -Wextra -Werror -pedantic \
    -I "$ROOT_DIR/firmware/include" \
    "$ROOT_DIR/firmware/tests/${name}_test.cpp" \
    -o "$TEMP_DIR/${name}_test"
  "$TEMP_DIR/${name}_test"
done

echo "[6/6] Repository hygiene"
(cd "$ROOT_DIR" && git diff --check)
(cd "$ROOT_DIR" && git diff --cached --check)
(cd "$ROOT_DIR" && bash -n scripts/*.sh)
if (cd "$ROOT_DIR" && git ls-files --cached --others --exclude-standard | grep -Eq '(^|/)(\.env($|\.)|.*\.(pem|key|p12|pfx)$)'); then
  echo "Tracked secret-like filename ditemukan" >&2
  exit 1
fi
if (cd "$ROOT_DIR" && git ls-files --cached --others --exclude-standard | grep -Eq '(^|/)P00[1-7](_[0-9]+)?\.csv$'); then
  echo "Raw P001-P007 CSV tidak boleh berada di repository publik" >&2
  exit 1
fi
if (cd "$ROOT_DIR" && git ls-files --cached --others --exclude-standard | grep -Eq '(^|/)(ordinal_model_params|golden_vectors|reconstructed_windows)\.hpp$'); then
  echo "Generated model/golden header tidak boleh dilacak di repository" >&2
  exit 1
fi

echo "Semua quality gates lulus."
