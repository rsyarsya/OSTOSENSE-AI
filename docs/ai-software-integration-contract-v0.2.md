# Runtime output contract v0.2

This is the current application-facing contract. The
[JSON Schema](../ai/contracts/ai-runtime-output-v0.2.schema.json) and
[TypeScript definition](../ai/contracts/typescript/ai-runtime-output-v0.2.ts)
specify the payload. The [service guide](ai-service-integration-v0.1.md)
explains transport and raw-window processing.

## Valid states

| Mode | Data source | Model status | Meaning |
|---|---|---|---|
| `LIVE` | `NONE` | `UNAVAILABLE` | No usable prediction |
| `ENGINEERING_TEST` | `SYNTHETIC_FIXTURE` | `TEST_ONLY` | Synthetic demonstration |
| `LIVE_EXPERIMENTAL` | `REAL_SENSOR` | `UNVALIDATED` | Experimental real-sensor input |

Class order is `Safe=0`, `Monitor=1`, `Caution=2`, `Urgent=3`.
The class is an ordered model output, not a calibrated probability or time
estimate. Neither available mode is a validated clinical prediction.

## Fields

| Field | Meaning |
|---|---|
| `runtime_output_version` | `0.2.0` |
| `mode`, `data_source`, `model_status` | State combination from the table above |
| `prediction_available` | Whether the class fields are available |
| `risk_class`, `risk_class_index` | Consistent class name/index, or null |
| `source_window_end_ms` | Window end relative to acquisition start, not UTC |
| `model_input_channel` | `Kap_7` for real input |
| `model_artifact_version`, `model_artifact_sha256` | Model identity |
| `evidence_scope`, `warning` | Interpretation/provenance metadata to retain |

Use the complete [example payloads](../ai/contracts/examples/v0.2/), not a
hand-built subset. The schema rejects extra fields and incompatible states.

The backend adds authenticated device, session, acquisition, and receive-time
metadata in an outer envelope. It must not insert those fields into the `ai`
object or derive them from the predicted class.

## Features

The [feature-input schema](../ai/contracts/ai-feature-input-v0.1.schema.json)
specifies five ordered values using `RAW_MINUS_SESSION_BASELINE`:

1. `cap_delta_mean`
2. `cap_delta_last`
3. `cap_delta_slope_per_s`
4. `cap_delta_variance`
5. `cap_delta_range`

The real-input adapter uses a confirmed fixed dry baseline, median aggregation
from 10 Hz to 1 Hz, 120-second windows, and a 10-second stride.
Normalized deltas from the descriptive pilot report are not compatible inputs.
See [feature definitions](architecture.md).

## Consumer requirements

- Validate the schema and state combinations before accepting a result.
- When unavailable, clear the current class rather than retaining a stale one.
- Label test output as simulation and real output as experimental/unvalidated.
- Retain model identity, source window, and provenance alongside the result.
- Determine freshness using acquisition/window time and authorized device
  identity. A new fetch or insert must not make old measurements appear fresh.
- Keep LIG contact detection, filling estimates, sensor faults, and connection
  state separate from AI.
- Do not convert these classes into patient alerts, risk percentages, leakage
  countdowns, or skin-condition assessments.

Zero-valued feature examples and repeated placeholder hashes in schema fixtures
test formatting only. For computed examples tied to the actual bundled model,
run:

```bash
python -m ostosense_ai.integration_cli examples --output outputs/examples.json
```

## Compatibility

Runtime v0.1 remains available in the schema directory. Its original
[specification](archive/ai-software-integration-contract-v0.1.md) is archived.
Changes to payload fields or state meanings require a new contract version.

Consumer acceptance tests should cover all four classes in both available modes,
unavailable transitions, malformed versions/hashes, extra fields, class-index
mismatches, wrong channels, replayed windows, and transport failures.
