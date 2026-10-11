# Research notes

This repository supports inspection and reproduction of a prototype AI pipeline.
The current evidence separates a synthetic classification experiment from
unlabeled physical sensor characterization.

## Data and provenance

| Dataset | Contents | Role | Public availability |
|---|---|---|---|
| Synthetic v0.3 | Generated sessions, controlled events, four ordinal labels | Model development and internal validation | Generator, configs, fixtures, bundled model, predictions |
| Real pilot P001–P007 | 11 logger files; 21,887 raw readings, 2,184 one-second summaries, 94 windows | Sensor characterization | Inventory, hashes, aggregate tables, figures; raw CSVs private |

The real recordings have relative timestamps and multiple channels, but no
verified per-window risk labels. They are not used to train or score the
four-class model. The final project report also describes limited wear testing
on two patients; those observations are not classifier evaluation data in this
repository.

The term `SYNTHETIC_PIPELINE_TEST_ONLY` is the machine-readable provenance tag
for generated data. `ENGINEERING_TEST_ONLY` marks synthetic labeling/training
fixtures. `PILOT_PENDING` means real-data labeling boundaries have not been
established. These identifiers are retained in artifacts for compatibility.

## Experimental method

1. Generate synthetic v0.3 with seed `20260722`.
2. Extract five baseline-adjusted capacitive features in 120-second windows
   with a 10-second stride.
3. Derive ordinal labels from independent synthetic event timing and the
   versioned boundary fixture.
4. Join eligible features and labels, validating hashes and group assignments.
5. Fit the scaler and OLR on 76 development windows from four eligible sessions.
6. Score 38 validation windows from two separate sessions.

The input manifest includes additional scenarios that are excluded by the
labeling rules. Session, bag, and sensor identities cannot cross partitions.
Do not create a random window-level split: adjacent windows overlap and share
most of their readings.

The implemented OLR is `mord.LogisticAT` with L2 strength `alpha=1.0` and
`max_iter=10000`. The comparison models use fixed settings recorded in
[model_specifications.json](ai-evaluation/2026-09-09/model_specifications.json).
The published comparison is a small internal experiment, not a model-selection
study on an independent real-world test set.

## Label semantics

For gradual-leak fixture sessions, labels depend on the time from the window
end to an independently recorded physical-leak event. Three ordered boundaries
define Safe, Monitor, Caution, and Urgent. Urgent covers the pre-leak interval;
post-leak windows are excluded.

Safe scenarios require the observation and horizon conditions specified by the
[label rulebook](ai-label-rulebook-v0.3.md). Sudden-leak and field scenarios are
excluded from ordinal training. Invalid quality and incomplete windows are
tracked as exclusions. Production boundary values are unset; synthetic fixture
values must not be interpreted as clinical thresholds or real leakage times.

LIG measurements do not determine the model's feature vector or serve as the
sole ground truth. Physical leak observations and LIG detection timestamps
represent different events.

## Metrics and results

[Published metrics](ai-evaluation/2026-09-09/tables/model_metrics.csv) and
[predictions](ai-evaluation/2026-09-09/tables/synthetic_predictions.csv)
are available as CSV.

| Model | Validation accuracy | Macro F1 | Quadratic weighted kappa |
|---|---:|---:|---:|
| OLR | 0.9474 | 0.9529 | 0.9733 |
| Logistic regression | 0.9474 | 0.9529 | 0.9733 |
| Decision tree | 0.9737 | 0.9775 | 0.9865 |

Macro F1 averages the four fixed classes. Confusion matrices use true classes
as rows and predicted classes as columns. Quadratic weighted kappa penalizes
errors according to ordinal distance. The evaluator reports no project
acceptance or clinical pass/fail decision.

For OLR, class support is 21 Safe, 8 Monitor, 6 Caution, and 3 Urgent.
Both errors are Monitor predicted as Safe. Monitor recall is 75%; the perfect
Caution/Urgent scores have only six and three examples, respectively.
OLR and logistic regression have identical validation predictions; the tree has
one additional correct prediction. OLR is retained for its ordinal formulation
and compact portable representation.

## Limitations

- Synthetic signal patterns and class boundaries reflect generator assumptions.
  Generalization to real sensor distributions has not been established.
- The validation sample is small and contains overlapping windows from only
  two sessions. Window counts are not independent subject counts.
- Real measurements lack verified risk labels, physical volume calibration,
  and complete event annotations for classifier evaluation.
- The report describes motion-related spikes and inconsistent LIG response
  with layered hydrocolloid. These remain sensor/integration limitations.
- Python–C++ agreement establishes host-side implementation consistency.
  It does not establish ESP32 timing, memory use, battery life, or deployment.
- No measured patient notification accuracy, warning lead time, false-alarm
  rate, or clinical outcome follows from the classification metrics.
- The separate skin-exposure scoring concept in historical project documents
  is not the evaluated OLR model.

## Reproducing and extending the work

The [reproduction guide](reproducibility.md) separates public synthetic work
from analysis that requires private recordings. New research should report
the exact commit, environment, sensor configuration, calibration, labeling
procedure, partitioning unit, and exclusions. A new physical protocol requires
its own permissions and validation; the historical bench proposal is not a
ready-to-use human-study protocol.

## Citation

Use [CITATION.cff](../CITATION.cff) and record the commit used in your methods or
software availability statement. There is no assigned DOI or verified
peer-reviewed publication attached to this repository. Citation and reuse
permission are separate; see [REUSE.md](../REUSE.md).
