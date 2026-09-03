#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/ai/.venv/bin/python}"
OUTPUT_DIR="${1:-$ROOT_DIR/outputs/engineering-demo-v0.3}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python tidak ditemukan di $PYTHON_BIN" >&2
  echo "Siapkan environment [pipeline] sesuai README.md." >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output sudah ada: $OUTPUT_DIR" >&2
  echo "Pilih path baru atau hapus output lama secara manual setelah diperiksa." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT_DIR/ai/src"

"$PYTHON_BIN" -m ostosense_ai.synthetic \
  --config "$ROOT_DIR/ai/configs/synthetic-v0.3.json" \
  --output "$OUTPUT_DIR/raw" \
  --seed 20260722

"$PYTHON_BIN" -m ostosense_ai.features \
  --input "$OUTPUT_DIR/raw" \
  --config "$ROOT_DIR/ai/configs/features-v0.1.json" \
  --output "$OUTPUT_DIR/features"

"$PYTHON_BIN" -m ostosense_ai.labeling \
  --input "$OUTPUT_DIR/raw" \
  --protocol-manifest "$ROOT_DIR/ai/tests/fixtures/ostosense-evaluation-v0.1/protocol_manifest.csv" \
  --partition-manifest "$ROOT_DIR/ai/tests/fixtures/ostosense-evaluation-v0.1/partition_manifest.csv" \
  --boundary-config "$ROOT_DIR/ai/tests/fixtures/ostosense-labeling-v0.1/boundary-engineering-test-only-v0.1.json" \
  --features "$OUTPUT_DIR/features" \
  --output "$OUTPUT_DIR/labels"

"$PYTHON_BIN" -m ostosense_ai.matrix \
  --features "$OUTPUT_DIR/features" \
  --labels "$OUTPUT_DIR/labels" \
  --output "$OUTPUT_DIR/matrix"

"$PYTHON_BIN" -m ostosense_ai.training \
  --matrix "$OUTPUT_DIR/matrix" \
  --config "$ROOT_DIR/ai/configs/training-v0.1.json" \
  --output "$OUTPUT_DIR/model"

"$PYTHON_BIN" -m ostosense_ai.runtime_output predict-test-v2 \
  --model "$OUTPUT_DIR/model/ordinal_model.json" \
  --features "$ROOT_DIR/ai/contracts/examples/feature-input-v0.1/synthetic-capacitive.json" \
  --output "$OUTPUT_DIR/runtime-engineering-test.json"

echo "Demo engineering selesai: $OUTPUT_DIR"
echo "Output TEST_ONLY ini hanya untuk uji integrasi software, bukan kinerja OSTOSENSE."
