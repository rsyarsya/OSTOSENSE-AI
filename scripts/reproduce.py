"""Reproduce the public synthetic experiment using the active Python interpreter."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new output directory.")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        import mord  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        parser.error('Install the pipeline extra: python -m pip install -e "./ai[pipeline]"')
    # Refuse a reused destination before any pipeline stage can write to it.
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {output}. Choose a new directory.")

    configs = ROOT / "ai/configs"
    fixtures = ROOT / "ai/tests/fixtures"
    stages = [
        (
            "synthetic",
            [
                "--config",
                configs / "synthetic-v0.3.json",
                "--seed",
                "20260722",
                "--output",
                output / "raw",
            ],
        ),
        (
            "features",
            [
                "--input",
                output / "raw",
                "--config",
                configs / "features-v0.1.json",
                "--output",
                output / "features",
            ],
        ),
        (
            "labeling",
            [
                "--input",
                output / "raw",
                "--protocol-manifest",
                fixtures / "ostosense-evaluation-v0.1/protocol_manifest.csv",
                "--partition-manifest",
                fixtures / "ostosense-evaluation-v0.1/partition_manifest.csv",
                "--boundary-config",
                fixtures / "ostosense-labeling-v0.1/boundary-engineering-test-only-v0.1.json",
                "--features",
                output / "features",
                "--output",
                output / "labels",
            ],
        ),
        (
            "matrix",
            [
                "--features",
                output / "features",
                "--labels",
                output / "labels",
                "--output",
                output / "matrix",
            ],
        ),
        (
            "training",
            [
                "--matrix",
                output / "matrix",
                "--config",
                configs / "training-v0.1.json",
                "--output",
                output / "model",
            ],
        ),
        (
            "model_evaluation",
            [
                "--matrix",
                output / "matrix",
                "--model",
                output / "model",
                "--output",
                output / "validation",
            ],
        ),
        (
            "edge_export",
            [
                "--matrix",
                output / "matrix",
                "--model",
                output / "model",
                "--output",
                output / "edge",
            ],
        ),
        (
            "runtime_output",
            [
                "predict-test-v2",
                "--model",
                output / "model/ordinal_model.json",
                "--features",
                ROOT / "ai/contracts/examples/feature-input-v0.1/synthetic-capacitive.json",
                "--output",
                output / "runtime-engineering-test.json",
            ],
        ),
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "ai/src"), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        for name, arguments in stages:
            print(f"\nRunning {name}", flush=True)
            subprocess.run(
                [sys.executable, "-m", f"ostosense_ai.{name}", *map(str, arguments)],
                cwd=ROOT,
                env=env,
                check=True,
            )
        compare_predictions(output)
    except (subprocess.CalledProcessError, ValueError, OSError) as error:
        print(
            f"Reproduction failed: {error}. Intermediate files remain at {output}", file=sys.stderr
        )
        return 1
    print(f"\nReproduction complete: {output}")
    print("Synthetic validation only; see docs/research.md for interpretation.")
    return 0


def compare_predictions(output: Path) -> None:
    """Compare reproduced labels to the published OLR validation predictions."""
    published = ROOT / "docs/ai-evaluation/2026-09-09/tables/synthetic_predictions.csv"
    with published.open(encoding="utf-8", newline="") as handle:
        reference = list(csv.DictReader(handle))
    with (output / "validation/validation_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        actual = list(csv.DictReader(handle))
    # Keys are validated below against the public CSV, not inferred from row order.
    expected = {
        row["window_id"]: row["predicted_class"]
        for row in reference
        if row["model"] == "OLR" and row["partition"] == "validation"
    }
    observed = {row["window_id"]: row["predicted_label"] for row in actual}
    if not expected or expected != observed:
        raise ValueError("Validation predictions differ from the published OLR snapshot")
    (output / "reproduction_check.json").write_text(
        json.dumps(
            {"published_prediction_match": True, "validation_windows": len(observed)}, indent=2
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
