# Architecture

OSTOSENSE AI is the processing and inference component of an experimental
ostomy-monitoring prototype. It has two entry paths: an offline experiment
pipeline and an optional stateless HTTP adapter. Both use the same feature
ordering, exported model representation, and ordinal class names.

## Components

| Component | Responsibility | Implementation |
|---|---|---|
| Data contract | Validate and write session, sample, and event records | `ostosense_contract` |
| Offline processing | Generate fixtures, extract features, derive labels, validate partitions | `synthetic`, `features`, `labeling`, `matrix` |
| Modeling | Fit a scaler and OLR, evaluate held-out groups, export parameters | `training`, `model_evaluation`, `edge_export` |
| Inference | Load a checked model and compute ordered probabilities/classes | `inference`, `bundled_model` |
| Application adapter | Validate raw windows, generate runtime payloads, serve HTTP | `live_window`, `runtime_output`, `service` |
| Portable kernels | Run feature extraction and inference on a C++17 host | `firmware/include/ostosense/` |

The repository does not include a complete ESP32 firmware application, MQTT
broker, database, mobile app, or clinical dashboard. Portable C++ tests establish
host behavior; they do not establish on-device deployment, latency, or power use.

## Signal processing

The recorded prototype has three capacitive and two resistive channels. The
classifier consumes **five features from one capacitive channel**, not five
sensor channels. The raw-window adapter currently names that channel `Kap_7`.
This identifier is part of the input contract, not a verified mapping of every
physical strip or a calibrated unit.

Raw requests are sampled at 10 Hz. Ten consecutive readings are summarized by
their median to produce one value per second. A confirmed dry baseline is fixed
for the acquisition; the adapter computes it from the first 60 one-second medians.
The model operates on `delta = reading - baseline`.

Each feature window covers 120 seconds and advances by 10 seconds. The canonical
offline interval is `(t - 120 s, t]`. Missing samples, invalid quality, and timing
violations are handled explicitly instead of being interpolated silently.

| Position | Feature | Definition over the window |
|---|---|---|
| 1 | `cap_delta_mean` | Arithmetic mean of delta |
| 2 | `cap_delta_last` | Last delta value |
| 3 | `cap_delta_slope_per_s` | Ordinary least-squares slope against elapsed seconds |
| 4 | `cap_delta_variance` | Population variance of delta |
| 5 | `cap_delta_range` | Maximum delta minus minimum delta |

Feature serialization uses six decimal places. Feature names, order, timing,
and quality settings are versioned in [features-v0.1.json](../ai/configs/features-v0.1.json).
The older descriptive pilot analysis uses a provisional 20-second baseline and
normalized deltas; those columns are not interchangeable with the model input.

## Ordinal model

Training uses `StandardScaler` followed by `mord.LogisticAT` with `alpha=1.0`
and `max_iter=10000`. The scaler and model fit only development rows.
Session, bag, and sensor identifiers cannot cross partitions.

The exported artifact stores five scaler means and scales, five coefficients
(`beta`), and three ordered thresholds (`theta`). For standardized features `z`:

```text
P(Y <= k | z) = sigmoid(theta[k] - beta · z), k = 0, 1, 2
```

Adjacent cumulative probabilities yield the four class probabilities. The
selected class is the probability argmax, ordered as Safe, Monitor, Caution,
Urgent. These probabilities describe the synthetic fitted model; they are not
calibrated probabilities of leakage in a patient.

The [model card](../ai/src/ostosense_ai/model_assets/engineering_v03/MODEL_CARD.md)
records training provenance, intended use, and known limitations.

## Runtime states

The adapter returns a versioned state as well as a class. Consumers must retain
that state when storing or displaying a result.

| State | Meaning |
|---|---|
| `UNAVAILABLE` | No usable prediction; class fields are empty |
| `TEST_ONLY` | A synthetic demonstration result |
| `UNVALIDATED` | An experimental result from real sensor input |

Direct LIG contact detection, sensor quality, relative bag filling, and connection
status remain separate signals. None can be inferred from the OLR class.
The service has no validated clinical operating mode.

## Integrity and determinism

Stage manifests record input hashes, versions, feature order, group assignments,
and exclusions. The bundled model verifies its JSON assets before inference.
Generation is seeded; downstream processing is deterministic for fixed inputs
and an equivalent numerical environment. Floating-point results across platforms
should be compared using the documented tolerances rather than assumed byte
identical. The [reproduction guide](reproducibility.md) distinguishes these checks.
