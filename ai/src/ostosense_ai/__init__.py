"""OSTOSENSE offline AI pipeline and integration tools.

This package will host the offline AI pipeline for the OSTOSENSE ordinal
leakage-risk classifier: deterministic synthetic time-series generation,
window feature extraction, and all-threshold ordinal logistic regression with
L2 regularization and exportable parameters (scaler, coefficients, ordered
thresholds, class mapping). Evaluation will use a confusion matrix, Macro F1,
and quadratic weighted Cohen's kappa.

Implemented so far (all pipeline-mechanics only, never OSTOSENSE performance
evidence): a deterministic synthetic raw-data generator (`synthetic`), a
dependency-free rolling-window capacitive feature extractor (`features`), a
canonical ordinal evaluation-metrics API (`evaluation`, which lazily imports
scikit-learn), a deterministic ENGINEERING_TEST_ONLY four-class ordinal labeler
(`labeling`, which reuses the `features` windowing), and a leakage-safe
feature-label matrix builder (`matrix`, which joins the feature and label
artifacts), an ENGINEERING_TEST_ONLY ordinal trainer/export (`training`, `StandardScaler` +
`mord.LogisticAT`, lazily importing the pipeline stack), a standard-library
exported-model forward reference (`inference`), and a grouped-validation
evaluator (`model_evaluation`, which scores the validation partition with
`inference` + `evaluation`), and a deterministic edge-bundle exporter
(`edge_export`) that emits float32 C++ headers (model params + golden vectors)
for host-side portable C++17 inference parity against
`firmware/include/ostosense/ordinal_inference.hpp`. A dependency-free
`runtime_output` module exposes versioned software-facing states: v0.1 remains
immutable, while v0.2 adds an explicitly `UNVALIDATED` class from real `Kap_7`
features for engineering integration. It emits no probabilities, countdown,
LIG state, bag-fill value, notification, or clinical action. The `synthetic`,
`evaluation-metric`, `training`, and `model_evaluation` (metric step) paths rely
on the optional `[pipeline]` dependencies; `features`, `labeling`, `matrix`,
`training`, `inference`, `edge_export`, `runtime_output`, and this package import
without them.

The `labeling` module derives ground-truth `Safe/Monitor/Caution/Urgent` labels
for synthetic data only, using an ENGINEERING_TEST_ONLY boundary fixture under
`ai/tests/fixtures/` guarded for `SYNTHETIC_PIPELINE_TEST_ONLY` input. The
`matrix` module joins validated feature-valid + label-valid rows into a single
`model_matrix.csv`, preserving the labeling `dataset_partition` and keeping the
five capacitive features structurally separate from audit/grouping/target
columns. Numeric *production* B1/B2/B3 boundaries remain `PILOT_PENDING` and are
never introduced in the repository.

The `training` module fits `StandardScaler` + `mord.LogisticAT` on the matrix
`development` rows only (`validation` ignored, any `final_test` row rejected) and
exports `ordinal_model.json` + `training_manifest.json` with no evaluation
metrics; its parity check uses the canonical `inference` forward. The
`model_evaluation` module scores the `validation` partition only (never
development/training, never Final Test) and writes deterministic
`validation_predictions.csv` + `validation_evaluation.json` with metrics nested
under `pipeline_mechanics_metrics` and no target/pass-fail fields. All of this is
deterministic and SYNTHETIC_PIPELINE_TEST_ONLY. Numeric production B1/B2/B3
boundaries remain `PILOT_PENDING`.

Host-side portable C++ ordinal-inference parity is implemented (the exported
synthetic model runs through `firmware/include/ostosense/ordinal_inference.hpp`
with exact class labels and <=1e-5 probability agreement), as is host-side
portable C++ capacitive-feature parity
(`firmware/include/ostosense/capacitive_features.hpp`). The `raw_qc` module is a
standard-library-only, deterministic real-data intake/QC gate that classifies
each logger session as contract/protocol `PASS`, `FAIL`, or `PARTIAL` (readiness
mechanics only, never OSTOSENSE performance). The standard-library-only
`pilot_data` module separately prepares the current flat 10 Hz P001-P007 logger
CSVs into 1 Hz summaries, provisional-baseline normalized features, unlabeled
120-second windows, and session-weighted descriptive correlation artifacts.
`pilot_reporting` lazily uses the optional pipeline/reporting stack to combine
that real unlabeled descriptive evidence with a separately identified
`SYNTHETIC_PIPELINE_TEST_ONLY` optimizer trace in deterministic 300-dpi progress
figures. Not implemented yet: training/evaluation on labeled real sealed data
and an on-device ESP32 feature/inference/transmission loop. P001-P007 are real
five-channel logger data, but remain unlabeled and cannot support classifier
performance claims.
The design future batches follow is documented
in docs/ostosense-ai-foundation.md and the structure-locked
`docs/ai-label-rulebook-v0.3.md`; the dependency-free Tier 1 contract lives in
the separate ostosense_contract package.
"""

__version__ = "0.1.0"

__all__: list[str] = []
