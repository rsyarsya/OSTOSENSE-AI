"""Recompute the published comparison metrics from public prediction CSVs."""

import csv
import math
from pathlib import Path

from ostosense_ai.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "docs/ai-evaluation/2026-09-09/tables"


def main() -> int:
    with (TABLES / "synthetic_predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    with (TABLES / "model_metrics.csv").open(encoding="utf-8", newline="") as handle:
        published = list(csv.DictReader(handle))
    errors = []
    for reference in published:
        key = (reference["model"], reference["partition"])
        rows = [row for row in predictions if (row["model"], row["partition"]) == key]
        if len(rows) != int(reference["n_windows"]) or not rows:
            errors.append(f"{key}: inconsistent window count")
            continue
        truth = [int(row["true_index"]) for row in rows]
        predicted = [int(row["predicted_index"]) for row in rows]
        metrics = evaluate_predictions(truth, predicted)
        computed = {
            "accuracy": sum(a == b for a, b in zip(truth, predicted)) / len(rows),
            "macro_f1": metrics["macro_f1"],
            "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
        }
        for name, actual in computed.items():
            if not math.isclose(actual, float(reference[name]), rel_tol=0, abs_tol=1e-12):
                errors.append(f"{key}: {name} differs from the published value")
        print(f"{key[0]} / {key[1]}: {len(rows)} windows, accuracy={computed['accuracy']:.6f}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("All published comparison metrics match their source predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
