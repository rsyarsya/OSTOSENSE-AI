# Reproducibility

## Public synthetic experiment

From an activated Python 3.11/3.12 environment at the repository root:

```bash
python -m pip install -e "./ai[pipeline]"
python scripts/reproduce.py --output outputs/synthetic-run
```

The command uses the current interpreter on Windows, Linux, and macOS. It runs
the existing versioned pipeline with seed `20260722`, evaluates the held-out
validation sessions, and compares OLR class predictions by window ID against
the published snapshot. It does not need private sensor data or a C++ compiler.

Choose a fresh output directory for each run. Existing directories are refused;
failed stages leave intermediate artifacts for inspection.

| Output | Contents |
|---|---|
| `raw/` | Synthetic sessions, samples, events, and provenance |
| `features/` | Window features and exclusions |
| `labels/` | Event-derived labels and exclusions |
| `matrix/` | Audited training/validation matrix |
| `model/` | Scaler, OLR parameters, training manifest |
| `validation/` | Predictions and `validation_evaluation.json` |
| `edge/` | C++ model parameters, golden vectors, export manifest |
| `runtime-engineering-test.json` | A synthetic application payload |
| `reproduction_check.json` | Comparison with published validation classes |

Expected OLR validation: 38 windows, 36 correct, Macro F1 approximately
0.952922, and quadratic weighted kappa approximately 0.973258. Accuracy is
36/38 = 94.7368%. Two Monitor windows are predicted as Safe.
See [research notes](research.md) for the small-sample and synthetic-data limits.

### Numerical reproducibility

The bundled artifact is immutable and its loader checks exact SHA-256 hashes.
Seeded generation and serialization are deterministic for fixed inputs.
Retraining depends on the numerical stack; different platforms can introduce
small floating-point differences. Compare class predictions, metric values,
and parameter/probability tolerances rather than requiring a new training
manifest to have an identical hash.

The public reproduction command verifies OLR classes. The three-model
comparison's predictions and hyperparameters are published separately in the
[September evaluation bundle](ai-evaluation/2026-09-09/README.md).
That historical full-report script also analyzes private data and requires
access to those recordings.

## Recompute the published comparison metrics

This uses only public prediction CSVs and verifies the reported metrics:

```bash
python scripts/check_published_metrics.py
```

It recomputes accuracy, Macro F1, and quadratic weighted kappa for each model
and partition. This verifies published predictions and arithmetic; it does
not retrain the logistic-regression or decision-tree comparators.

## Portable inference

```bash
python -m pip install -c ai/requirements-serve.lock -e "./ai[pipeline,quality,serve,serve-test]"
python scripts/verify.py
```

The full check requires a C++17 compiler. It includes synthetic Python–C++
feature/inference parity, standalone host tests, package installation in a
clean environment, service tests, lint/type checks, and repository hygiene.
See [Contributing](../CONTRIBUTING.md) for compiler configuration.

## Real sensor characterization

Raw P001–P007 CSV files are not published. Their identities and hashes are
recorded in the [inventory](../ai/data-manifests/real-pilot-v0.1.inventory.json).
Published statistics and figures can be inspected without those files; complete
reproduction of physical-sensor characterization requires authorized access.

With the exact private source files, the original evaluation script can be run:

```bash
python docs/ai-evaluation/2026-09-09/reproduce.py --repo . --raw /path/to/private-csv --output /path/to/new-report
```

The historical script expects the original project's `ai/.venv/bin/python`
environment. Use the public commands above for portable synthetic reproduction.
No raw data, participant documents, or claims of access are included here.

## Recording a research run

Retain the commit from `git rev-parse HEAD`, the Python version, installed
dependency versions, input hashes, stage manifests, and output directory.
Keep generated results outside tracked source. Cite the software revision and
explain whether a result came from synthetic validation or physical recordings.
