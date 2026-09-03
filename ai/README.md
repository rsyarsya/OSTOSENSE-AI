# OSTOSENSE AI Tools

This folder contains the dependency-free Tier 1 CSV contract/logger and the
offline AI pipeline. Deterministic synthetic raw-data generation, rolling-window
capacitive feature extraction, canonical ordinal evaluation metrics,
ENGINEERING_TEST_ONLY four-class labeling, a leakage-safe feature-label matrix,
an ENGINEERING_TEST_ONLY ordinal trainer/export, grouped validation evaluation,
host-side portable C++ inference parity, host-side portable C++
capacitive-feature parity, and a deterministic real-data intake/QC gate are
implemented for pipeline testing and real-data readiness. A deterministic
software-facing runtime-output contract is also implemented for safe integration
demos. Training/evaluation on labeled real sealed data, an approved LIVE model,
and ESP32 hardware/sensor integration are not implemented yet.

## Minimal usage

```python
from ostosense_contract import (
    CapQuality,
    LigQuality,
    SampleRecord,
    Tier1CsvLogger,
)

logger = Tier1CsvLogger("tier1-data/session-batch-001")
logger.append_sample(
    SampleRecord.create(
        timestamp=1750000000000,
        session_id="session-001",
        capacitance_raw=12.5,
        lig_raw=410.0,
        cap_quality=CapQuality.OK,
        lig_quality=LigQuality.OK,
    )
)
```

`SampleRecord.create()` always derives `system_quality` from the two channel
qualities. Direct construction with an inconsistent aggregate is rejected.

Run tests from `ai/src`:

```bash
cd ai/src
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -W error \
  -m unittest discover -s ../tests
```

The full field and event metadata definitions are in
`../docs/ai-data-contract-v1.1.md`.

## Optional pipeline environment

The Tier 1 contract remains dependency-free. The offline pipeline uses a
separate, pinned optional dependency set. The pipeline extra requires Python
3.11 or newer and is verified with the project's Python 3.12 environment:

```bash
cd ai
.venv/bin/python -m pip install -e ".[pipeline]"
```

The selected trainer is `mord.LogisticAT`, an all-threshold ordinal logistic
model with L2 regularization and exportable `coef_`/`theta_` parameters.
`scikit-learn` provides scaling and established metrics; dataset partitions
come from validated manifests and are never generated randomly by the trainer.
The primary ordinal agreement metric is quadratic weighted Cohen's kappa;
Macro F1 and confusion matrices always use the fixed class order
`Safe`, `Monitor`, `Caution`, `Urgent`.

Any synthetic-data metric is a pipeline-mechanics check only. It is never an
OSTOSENSE performance result; headline results require real sealed Final Test
data under the project contract and rulebook.

## Synthetic raw-data generator (pipeline test only)

`ostosense_ai.synthetic` writes contract-valid `sessions.csv`, `samples.csv`,
and `events.csv` through the existing `ostosense_contract` logger, plus a
deterministic `manifest.json` sidecar. It exists only to exercise pipeline
mechanics (CSV generation, contract compliance, determinism, scenario and
quality-state coverage). Output is `SYNTHETIC_PIPELINE_TEST_ONLY` and can never
support any performance, sensor, or clinical claim.

Run it from `ai/src` (same convention as the test suite), or from the repo
root with `PYTHONPATH=ai/src`:

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.synthetic \
  --config ../configs/synthetic-v0.2.json \
  --output /tmp/ostosense-synthetic-smoke \
  --seed 20260722
```

The command refuses to replace existing artifacts by default. Use
`--overwrite` only when replacement is intentional.

Re-running with the same `--config` and `--seed` produces byte-identical files.
Every numeric config parameter is classified in the manifest provenance map
(`CONTRACT_DERIVED`, `ENGINEERING_TEST_ONLY`, `LITERATURE_VERIFIED`,
`PILOT_PENDING`). No features, four-class labels, or B1/B2/B3 boundaries are
produced.

## Feature extraction (pipeline test only)

`ostosense_ai.features` converts contract-valid `sessions.csv` + `samples.csv`
into an auditable `features.csv` plus a deterministic `feature_manifest.json`.
It is dependency-free (standard library + `ostosense_contract`). The only AI
signal is the capacitive channel; it emits five baseline-normalized features
(`cap_delta_mean`, `cap_delta_last`, `cap_delta_slope_per_s`,
`cap_delta_variance`, `cap_delta_range`) in the fixed `FEATURE_COLUMNS` tuple.
LIG, events, arm, timing, and identifiers are never features. It produces no
Safe/Monitor/Caution/Urgent labels, no model, and no B1/B2/B3 boundaries.

Locked window convention (Data Collection Protocol v0.1 and Label Rulebook
v0.3): interval `(t-W,t]`, `W`=120 s, stride 10 s, 1 Hz, `t_ref` = first
sample of the session. Rulebook v0.3 has resolved the earlier off-by-one
notation.

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.features \
  --input /tmp/ostosense-synthetic-raw \
  --config ../configs/features-v0.1.json \
  --output /tmp/ostosense-features
```

One row is written per candidate window (valid and excluded); excluded windows
leave the feature cells empty and carry a single priority `exclusion_reason`
(`DUPLICATE_TIMESTAMP` > `PARTIAL_WINDOW` > `TIMING_OUT_OF_TOLERANCE` >
`INVALID_CAP_QUALITY`). The command refuses to replace existing artifacts unless
`--overwrite` is passed, and the same input + config produce byte-identical
outputs. A passing run proves feature-pipeline mechanics only, never AI
accuracy, sensor validity, early-warning performance, or clinical value.

## Evaluation metrics (pipeline test only)

`ostosense_ai.evaluation.evaluate_predictions(y_true, y_pred)` is the one
canonical evaluator for the four-class ordinal classifier. The class order is
fixed and never inferred from observations: `0=Safe, 1=Monitor, 2=Caution,
3=Urgent`. The confusion matrix is oriented **rows = ground truth, columns =
predicted**. It returns per-class precision/recall/F1/support, **Macro F1** over
all four fixed classes (absent classes contribute F1 = 0, `zero_division=0`),
and the **quadratic weighted Cohen's kappa** (`quadratic_weighted_kappa`).

scikit-learn is the authoritative implementation and is imported lazily, so the
package imports without the optional `[pipeline]` dependencies; requesting a
metric without scikit-learn raises an actionable `RuntimeError`. Inputs must be
equal-length, non-empty integer labels in `0..3` (booleans, strings, and floats
are rejected); the degenerate case where quadratic kappa is mathematically
undefined (all labels a single identical class) raises `ValueError`.

This validates metric mechanics only. It emits **no project-target pass/fail**
field and is **not** an OSTOSENSE performance, notification-accuracy,
early-warning, sensor, or clinical claim unless predictions come from the
approved real-data evaluation protocol.

## Four-class labeling (ENGINEERING_TEST_ONLY, pipeline test only)

`ostosense_ai.labeling` derives canonical ordinal ground-truth labels
(`0=Safe, 1=Monitor, 2=Caution, 3=Urgent`) for the synthetic dataset, per
structure-locked Label Rulebook v0.3. It produces **labels only** — no model, no
metrics, no confusion matrix. It is dependency-free and reuses the exact
`features` windowing (`(t-W,t]`, `W`=120 s, stride 10 s, same `window_id`), so
labels and features share one window definition. Labels come only from arm,
recorded events, and timing; capacitance/LIG raw values, feature values, and
model predictions never determine a label (LIG *quality* is used only for the
protocol-required SAFE observation-start anchor).

Exclusion precedence: structural/capacitive-quality (`DUPLICATE_TIMESTAMP`,
`PARTIAL_WINDOW`, `TIMING_OUT_OF_TOLERANCE`, `INVALID_CAP_QUALITY`) → arm
(`SUDDEN_ARM`, `FIELD_ARM_EXCLUDED`) → scenario (`CENSORED_NO_SAFE_HORIZON`,
`POST_LEAK`) → otherwise an ordinal label. Malformed/contradictory required
events fail the whole run (`MALFORMED_REQUIRED_EVENTS`), leaving outputs
untouched.

Numeric boundaries come from an `ENGINEERING_TEST_ONLY` fixture under
`ai/tests/fixtures/`, guarded so it only applies to `SYNTHETIC_PIPELINE_TEST_ONLY`
input; production B1/B2/B3 remain `PILOT_PENDING` and are never introduced.

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.labeling \
  --input /tmp/ostosense-synthetic-raw \
  --protocol-manifest ../tests/fixtures/ostosense-labeling-v0.1/protocol_manifest.csv \
  --partition-manifest ../tests/fixtures/ostosense-labeling-v0.1/partition_manifest.csv \
  --boundary-config ../tests/fixtures/ostosense-labeling-v0.1/boundary-engineering-test-only-v0.1.json \
  --features /tmp/ostosense-features \
  --output /tmp/ostosense-labels
```

`--overwrite` is required to replace existing artifacts; same inputs produce
byte-identical `labels.csv` and `label_manifest.json`. A passing run proves
labeling-pipeline mechanics only, never AI accuracy, sensor validity,
early-warning performance, or clinical value.

## Feature-label matrix (pipeline test only)

`ostosense_ai.matrix` joins the canonical feature artifacts (`features.csv` +
`feature_manifest.json`) and label artifacts (`labels.csv` +
`label_manifest.json`) into one `model_matrix.csv` plus a deterministic
`matrix_manifest.json`, ready for a later trainer. It is dependency-free, reuses
the canonical feature/class constants (never redefining them), and performs **no
training, metrics, or splitting**. It only keeps rows that are both
`feature_valid` and `label_valid`.

Column separation is explicit: audit/grouping (`window_id`, `session_id`,
`bag_id`, `sensor_id`, `dataset_partition`) → the five features (`cap_delta_*`) →
target (`risk_label`, `risk_label_index`, with `Safe=0..Urgent=3`). The
manifest declares numeric `risk_label_index` as the trainer target and
`risk_label` as its human-readable companion. Feature values never enter
audit/grouping columns and forbidden metadata never enters the five-feature
allowlist (guarded at import).

Before writing anything it validates: canonical headers; both manifests are JSON
objects; SHA-256 of `features.csv`/`feature_manifest.json` match the label
manifest's `feature_artifact_sha256`; raw `sessions.csv`/`samples.csv` hashes and
dataset origin (`SYNTHETIC_PIPELINE_TEST_ONLY`) agree across manifests; window
convention matches; contract/rulebook/boundary versions, strict boolean fields,
class and exclusion distributions, and candidate/valid/excluded counts reconcile
with both CSVs; window sets and per-row identity match; partitions use the
canonical enum; and no session/bag/sensor spans multiple `dataset_partition`
values (leakage). Malformed records fail the run; nothing is silently deduped,
repaired, or dropped.

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.matrix \
  --features /tmp/ostosense-features \
  --labels /tmp/ostosense-labels \
  --output /tmp/ostosense-matrix
```

`--overwrite` is required to replace existing artifacts (staged then swapped in;
the two-file replace is not a single OS transaction); same inputs produce
byte-identical outputs. A passing run proves matrix-construction mechanics only,
never AI accuracy, notification accuracy, early-warning performance, sensor
validity, or clinical value.

## Ordinal trainer / export (ENGINEERING_TEST_ONLY, pipeline test only)

`ostosense_ai.training` fits the project-selected trainer only —
`sklearn.preprocessing.StandardScaler` + `mord.LogisticAT` (`alpha=1.0`,
`max_iter=10000`) — on a validated `model_matrix.csv` and exports a portable
parameter set. It computes **no evaluation metrics** (no confusion matrix, Macro
F1, kappa, accuracy, or pass/fail). The package imports without the optional
`[pipeline]` dependencies (NumPy/scikit-learn/mord/SciPy are imported lazily; a
missing stack raises an actionable `RuntimeError` pointing at `.[pipeline]`).

It fits on `development` rows **only**; `validation` rows may exist but never
influence scaling/fitting/weighting; any `final_test` row rejects the whole
input. There is no random split and no seed. Before fitting it validates the
matrix (`model_matrix_sha256`, builder/contract/rulebook versions, origin,
five-feature order, `risk_label_index` target, `Safe=0..Urgent=3` mapping,
count/partition/class reconciliation, unique
`window_id`, finite features, label/index consistency, no group crossing
partitions). Exported forward inference uses the cumulative all-threshold form
`P(Y<=k|x) = sigmoid(theta[k] - beta^T z)` (`z` = StandardScaler-transformed) and
is checked to reproduce mord's probabilities within tolerance and its predicted
labels exactly.

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.training \
  --matrix /tmp/ostosense-matrix \
  --config ../configs/training-v0.1.json \
  --output /tmp/ostosense-model
```

Outputs `ordinal_model.json` (scaler `mean`/`scale`, `beta`, `theta`, class
mapping, cumulative formula, hyperparameters) and `training_manifest.json`
(separate input/output hashes, dependency versions, fit-partition policy,
row/class counts,
model-sanity/parity results, and an `optimizer_convergence_status` noting mord
0.7 does not expose convergence). `--overwrite` is required to replace existing
artifacts (staged then swapped in; not a single OS transaction); same
input/config/environment produce byte-identical outputs. A passing run proves
deterministic synthetic training and parameter-export mechanics only, never AI
accuracy, notification accuracy, early-warning performance, sensor validity, or
clinical value.

## Exported-model inference (standard library, pipeline test only)

`ostosense_ai.inference.predict_exported_model(model_artifact, feature_rows)` is
the single forward reference for the exported `ordinal_model.json`. It uses only
`math` (no NumPy/scikit-learn/mord/SciPy), validates the artifact (version,
family, synthetic origin, feature/class order, scaler/beta/theta lengths, finite
parameters, positive scales, strictly increasing theta) and each five-value
feature row, and computes `z=(x-mean)/scale`, `eta=beta^T z`,
`P(Y<=k)=sigmoid(theta[k]-eta)` with a numerically stable sigmoid, deriving the
four class probabilities by cumulative differences. Argmax ties select the
lowest class index (matching NumPy/mord); it rejects (never silently clamps)
non-finite, out-of-range, or non-unit-sum probabilities beyond tolerance. The
trainer's parity check reuses this function so there is one forward formula.

## Software-facing AI runtime output

`ostosense_ai.runtime_output` is the canonical boundary between this repository
and downstream backend, mobile, or web software. It deliberately exposes only
the ordinal class needed by software and keeps direct LIG leak status, bag fill,
sensor quality, identity, timestamps, and notification policy in their own
system contracts.

Version `0.1.0` supports two honest states:

- `LIVE` + `UNAVAILABLE`: no approved real-data model is shipped, so
  `prediction_available=false` and the class/index are `null`.
- `ENGINEERING_TEST` + `TEST_ONLY`: a synthetic model and synthetic feature row
  may produce `Safe`, `Monitor`, `Caution`, or `Urgent` for integration testing.
  It must never trigger a patient notification.

Generate the current LIVE state from `ai/src`:

```bash
../.venv/bin/python -m ostosense_ai.runtime_output unavailable \
  --output /tmp/ostosense-ai-live.json
```

Run an explicitly synthetic integration test:

```bash
../.venv/bin/python -m ostosense_ai.runtime_output predict-test \
  --model /path/to/ordinal_model.json \
  --features /path/to/synthetic_feature.json \
  --output /tmp/ostosense-ai-test.json
```

The machine-readable schema and checked examples are in `contracts/`; the
software display rules, domain separation, and `MUST FIX` handoff list are in
`../docs/ai-software-integration-contract-v0.1.md`. The payload contains no
probabilities, risk percentage, countdown, LIG state, bag-fill estimate,
humidity, notification, or clinical action.

## Grouped validation evaluation (pipeline test only)

`ostosense_ai.model_evaluation` scores the matrix `validation` partition with the
exported model (`inference`) and the canonical `evaluation.evaluate_predictions`.
It touches **only** validation rows — never development/training metrics, never
Final Test, never random splits. Before evaluating it validates the matrix
against its manifest (SHA, versions, origin, feature/target/class, counts, group
separation), the model SHA against `training_manifest.output_sha256`, the
training matrix hashes against `training_manifest.input_sha256`, the model
artifact, and the training origin/fit-partition policy; it rejects any
`final_test` row or source candidate and requires at least one validation row.

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.model_evaluation \
  --matrix /tmp/ostosense-matrix \
  --model /tmp/ostosense-model \
  --output /tmp/ostosense-validation
```

Outputs `validation_predictions.csv` (in matrix row order, with per-class
probabilities) and `validation_evaluation.json` (input/output hashes, validation
counts + ground-truth support, and canonical metrics nested under
`pipeline_mechanics_metrics` with an explicit confusion-matrix orientation). It
emits **no** `meets_target`, pass/fail, notification, lead-time, false-alarm,
event-level, firmware, or clinical fields. `--overwrite` is required; same inputs
produce byte-identical outputs (staged then swapped in; not a single OS
transaction). A passing run proves grouped synthetic validation, exported
inference, and metric-pipeline mechanics only — never OSTOSENSE accuracy,
notification accuracy, early-warning performance, sensor validity, firmware
parity, or clinical value.

## Host-side portable C++ inference parity (pipeline test only)

`firmware/include/ostosense/ordinal_inference.hpp` is a dependency-free,
header-only C++17 `PredictOrdinal(model, features, &out)` implementing the same
cumulative all-threshold forward (`z=(x-mean)/scale`, `eta=beta^T z`,
`P(Y<=k)=sigmoid(theta[k]-eta)`) with an overflow-safe sigmoid, strict `>`
argmax (ties keep the lowest class), model/feature validation, and typed
`InferenceStatus`. It uses no dynamic allocation, exceptions, JSON, PlatformIO,
ESP-IDF, or Arduino. `firmware/tests/ordinal_inference_test.cpp` covers it
standalone.

`ostosense_ai.edge_export` re-validates the full model/training/matrix provenance
chain (synthetic origin only, no Final Test), then writes a temporary,
**test-only** edge bundle: `ordinal_model_params.hpp`
(`ostosense::generated::kOrdinalModel`), `golden_vectors.hpp` (validation feature
rows + Python-reference probabilities + expected class index, from
`inference.predict_exported_model`, never ground truth), and
`edge_export_manifest.json` — all as exact float32 hexadecimal literals with
input/output SHA-256 and no metric/target/pass-fail fields.

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.edge_export \
  --model /tmp/ostosense-model \
  --matrix /tmp/ostosense-matrix \
  --output /tmp/ostosense-edge
```

An automated integration test runs synthetic-v0.3 → matrix → training → export,
compiles a harness with `g++ -std=c++17 -Wall -Wextra -pedantic`, and confirms
every validation vector matches the Python class exactly with maximum absolute
probability difference ≤ 1e-5. Generated model/vector headers are
ENGINEERING_TEST_ONLY and are never committed into the production firmware
include path. A passing run proves synthetic cross-language inference mechanics
only — not ESP32 deployment, firmware parity on hardware, or OSTOSENSE
performance.

## Host-side portable C++ capacitive-feature parity (pipeline test only)

`firmware/include/ostosense/capacitive_features.hpp` is a dependency-free,
header-only C++17 `ComputeCapacitiveFeatures(samples, count, baseline, &out)`
that reproduces the five baseline-normalized capacitive features from
`features.py` over one `(t-W,t]` window of 120 raw samples (`cap_delta_mean`,
`cap_delta_last`, `cap_delta_slope_per_s`, `cap_delta_variance`,
`cap_delta_range`). It consumes only the capacitive channel (never
LIG/labels/events/partitions), uses double accumulators and emits direct finite
`float` model inputs. Deterministic six-decimal rounding remains part of the
offline Python CSV serialization; the C++ kernel avoids locale-sensitive string
conversion and is checked against serialized features within `1e-5`. It enforces
the same structural precedence (duplicate timestamp → partial window → timing
tolerance → capacitive quality) plus typed statuses for
null/descending/non-finite/overflow
inputs, and resets its output on any failure. It uses no dynamic allocation,
exceptions, JSON, PlatformIO, ESP-IDF, Arduino, or embedded model.
`firmware/tests/capacitive_features_test.cpp` covers it standalone.

`ai/tests/test_capacitive_features.py` closes the portable path. It runs
synthetic-v0.3 → features → labels → matrix → training → edge export,
reconstructs each of the 38 eligible validation windows from `samples.csv` /
`sessions.csv` using the exact `(t-W,t]` convention, feeds the raw capacitance /
timestamps / quality / session baseline into the C++ kernel, and compiles a
harness with `g++ -std=c++17 -Wall -Wextra -pedantic`. It confirms all 38
windows return `kOk`, the C++ features match `features.csv` within ≤ 1e-5, and
those C++ features fed to `PredictOrdinal` reproduce the Python inference class
38/38 (probability difference ≤ 1e-5). All reconstructed/generated headers are
written to a temporary directory and never committed into the production
firmware include path. A passing run proves host-side synthetic
capacitive-feature and ordinal-inference parity mechanics only — not ESP32
deployment, firmware parity on hardware, sensor validation, or OSTOSENSE
performance.

## Real-data intake and QC gate (readiness mechanics only)

`ostosense_ai.raw_qc` is a standard-library-only, deterministic quality-control
gate for future integrated ESP32 logger outputs. It reads `sessions.csv` +
`samples.csv` + `events.csv` (plus an optional input `manifest.json` and an
optional `protocol_manifest.csv`) and decides, per session, whether it satisfies
the AI Data Contract v1.1 record semantics (`contract_status`), the currently
evaluable Data Collection Protocol v0.1 checks (`protocol_status`), and an
`overall_status` of `PASS`, `FAIL`, or `PARTIAL`. `PARTIAL` marks a
structurally usable session that could only be partially evaluated because no
protocol manifest was supplied (`protocol_status = NOT_EVALUATED`). It reuses the
canonical field/enum definitions from `ostosense_contract` and never redefines
schema constants.

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.raw_qc \
  --input /tmp/ostosense-logger-output \
  --config ../configs/raw-qc-v0.1.json \
  --output /tmp/ostosense-qc \
  [--protocol-manifest .../protocol_manifest.csv] [--overwrite]
```

Fatal, actionable `RawQcError`s (missing mandatory files, wrong canonical
headers, malformed CSV/JSON, unparseable numbers, unknown enums, non-finite
required values, duplicate `session_id`, orphan session references, or
protocol-manifest rows for unknown sessions) are raised
before any output is touched. Semantic issues are reported, never raised, as
stable issue codes with `ERROR`/`WARNING` severity into `qc_issues.csv`; a
per-session summary lands in `qc_sessions.csv` (including a median/population-std
baseline recomputation whose difference from the recorded baseline is reported
but never thresholded); provenance/hashes/scope go into `qc_report.json`. The
timing tolerance is 800–1200 ms inclusive, so every interval outside that range
fails provisional protocol QC. An interval `>= 2000 ms` must additionally carry
a `DATA_GAP` mark on the next sample or it also raises `UNMARKED_DATA_GAP`.
Baseline samples must cover the complete 60-second window within the same timing
tolerance. Injection sessions need `>= 120 s` of
pre-injection dry after both channels first read `OK`. Missing LIG flag events
are warnings only (LIG behavior is pending hardware calibration). Outputs are
byte-identical for identical inputs, existing artifacts are refused without
`--overwrite`, and inputs are never mutated. CLI exit codes: `0` when every
session is `PASS`, `2` when any session is `FAIL`/`PARTIAL`, `1` on fatal
invocation/input failure. The QC config values are `PROPOSED_PILOT_SETTING`
DRAFT Protocol v0.1 working values — not clinical thresholds, not validated
OSTOSENSE performance criteria, and not label boundaries; production B1/B2/B3
remain `PILOT_PENDING`. A passing run proves deterministic intake and provisional
QC mechanics only — never AI accuracy, notification accuracy, sensor validity,
firmware validity, or clinical value.

QC tool `0.2.1` also strictly validates each supplied `protocol_manifest.csv`
row (per-session `INVALID_PROTOCOL_MANIFEST`): a versioned `protocol_version`
(`^v0.1-<label>$`), a valid `planned_arm` equal to `sessions.csv`, matching
CSV-safe identities, a non-empty `target_fill_or_volume`, and arm-specific
`planned_safe_horizon_s`/injection/observation fields; duplicate or
unknown-session manifest rows stay fatal. Injection events are validated in
timestamp order against the locked scenarios (`MALFORMED_REQUIRED_EVENTS` for an
`INJECTION_END` without an open `INJECTION_START`, overlapping or unclosed
`INJECTION_START`, more than one `PHYSICAL_LEAK_OBSERVED`, a gradual physical
leak before injection, `LEAK_CONFIRMED` without a physical-leak event, or
injection in a SAFE dry session). Only a valid supplied manifest distinguishes a
planned gradual leak from a non-leaking fill; without one, that sub-scenario is
unknown and the absence of a physical leak is not itself an error. A planned
non-leaking fill (`LEAK_GRADUAL` with a positive horizon) keeps an unexpected
leak as the WARNING `UNPLANNED_PHYSICAL_LEAK`. Missing LIG flags are warnings
only when a physical leak was actually recorded. Any `DEVICE_RESTART` inside a
session is `DEVICE_RESTART_DURING_SESSION`. The
operational `docs/ai-shakedown-runbook-v0.1.md` and the
`docs/templates/protocol_manifest-shakedown-v0.1.example.csv` template cover the
first 3–5 two-channel engineering shakedown sessions (shakedown data never enters
model training or evaluation).

## Current real-pilot preparation and progress figures

`ostosense_ai.pilot_data` prepares the current flat P001-P007 ESP32 logger CSVs
without changing the source files. It strictly checks the 12-column schema,
consecutive sample numbers, 100 ms timing, finite sensor values, and known
status values. Each complete group of 10 raw samples is summarized by its median
to 1 Hz; an incomplete final group is reported and excluded. A provisional
per-session baseline uses the first 20 complete seconds. The module then emits
120-second windows every 10 seconds and calculates five temporal features for
each of the three capacitive candidates (`Kap_4`, `Kap_5`, `Kap_7`), for 15
unlabeled features total. The resistive channels remain descriptive/fail-safe
channels and never enter this model-feature table.

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.pilot_data \
  --input /path/to/OSTOSENSE_LIG_Manual_Data_Collection_v0.1 \
  --config ../configs/real-pilot-v0.1.json \
  --output /tmp/ostosense-real-pilot-v0.1
```

The outputs are `qc_sessions.csv`, `samples_1hz.csv`,
`window_features_unlabeled.csv`, `sensor_correlation_median.csv`,
`sensor_correlation_iqr.csv`, and `pilot_manifest.json`. Correlation is computed
within each operational session using Spearman ranks and then summarized by the
median so every session has equal weight. P006 is excluded because it is a
sensor-fault test. The first 20 seconds have not been independently verified as
dry, and the files have no exact per-window risk labels; therefore these outputs
must not be used to train or evaluate the four-class classifier.

`ostosense_ai.pilot_reporting` produces four deterministic 300-dpi PNGs and an
inspectable optimizer trace. It keeps evidence sources explicit: feature flow
and correlation use the real unlabeled pilot; the L-BFGS-B optimization trace
uses only the existing `SYNTHETIC_PIPELINE_TEST_ONLY` development/validation
matrix. The instrumented optimizer must reproduce canonical `mord.LogisticAT`
parameters, probabilities, and classes within the configured parity tolerance
before any figure is written.

```bash
../.venv/bin/python -m ostosense_ai.pilot_reporting \
  --pilot /tmp/ostosense-real-pilot-v0.1 \
  --synthetic-matrix /tmp/ostosense-synthetic-matrix \
  --training-config ../configs/training-v0.1.json \
  --output /tmp/ostosense-real-pilot-v0.1
```

The PNGs are `01_alur_ekstraksi_fitur.png`,
`02_matriks_korelasi_sensor_real.png`,
`03_jejak_optimisasi_olr_sintetis.png`, and
`04_panel_ai_ostosense.png`. The synthetic Macro F1 curve is a pipeline-mechanics
illustration, not OSTOSENSE performance on real data. No production label
boundaries, real-data accuracy, notification accuracy, or clinical claim is
created by either module.
