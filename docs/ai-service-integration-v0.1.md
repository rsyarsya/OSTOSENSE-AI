# HTTP service and raw-window integration

The optional service exposes the bundled synthetic model through a stateless
FastAPI adapter. Applications can use it without installing training packages.
For the model's intended use and limitations, see the
[model card](../ai/src/ostosense_ai/model_assets/engineering_v03/MODEL_CARD.md).

The Python distribution is version 0.2.0, model artifact format 0.1.0, runtime
payload 0.2.0, and service interface 0.1.0. These versions identify separate
interfaces; they do not indicate clinical validation.

## Install and start

After activating the repository environment:

```bash
python -m pip install -c ai/requirements-serve.lock -e "./ai[serve]"
```

For a local test, generate a token in the shell without writing it to a file.

Linux / macOS:

```bash
export OSTOSENSE_SERVICE_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export OSTOSENSE_ALLOW_TEST=true
python -m ostosense_ai.service
```

Windows PowerShell:

```powershell
$env:OSTOSENSE_SERVICE_TOKEN = python -c "import secrets; print(secrets.token_hex(32))"
$env:OSTOSENSE_ALLOW_TEST = "true"
python -m ostosense_ai.service
```

Keep the token in the backend environment. In another terminal configured with
the same token, run:

```bash
python scripts/probe_ai_service.py --url http://127.0.0.1:8000
```

The probe compares all four synthetic responses against the bundled references.
Interactive API documentation is disabled; use the checked-in JSON Schemas.

| Variable | Default / meaning |
|---|---|
| `OSTOSENSE_SERVICE_TOKEN` | Required token, at least 32 characters |
| `OSTOSENSE_ALLOW_TEST` | `false`; enables synthetic endpoints when `true` |
| `OSTOSENSE_ENABLE_EXPERIMENTAL` | `false`; enables operator-confirmed real input |
| `HOST` | `127.0.0.1`; the container binds to `0.0.0.0` |
| `PORT` | `8000` |

A non-root [Dockerfile](../Dockerfile) is supplied:

```bash
docker build -t ostosense-ai .
docker run --rm -p 127.0.0.1:8000:8000 -e OSTOSENSE_SERVICE_TOKEN -e OSTOSENSE_ALLOW_TEST ostosense-ai
```

Network deployment requires authentication and encrypted transport or a managed
isolated network. The adapter is a backend component; do not embed its token
in a browser or mobile application.

## Endpoints

All `/v1/` requests require `Authorization: Bearer <token>`.
Requests are limited to 512 KiB.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Unauthenticated model readiness; no sensor data |
| `GET /v1/examples` | Four synthetic examples; test mode required |
| `POST /v1/predict-features` | Synthetic feature input; returns `TEST_ONLY` |
| `POST /v1/evaluate-window` | Raw sensor input; returns `UNVALIDATED` or unavailable |

POST responses follow the
[service-response schema](../ai/contracts/ai-service-response-v0.1.schema.json):

- `ai`: the runtime v0.2 object, with no extra fields.
- `diagnostics`: availability reason, quality checks, and baseline information.
- `feature_input`: the actual feature vector, or null when unavailable.

See the [runtime contract](ai-software-integration-contract-v0.2.md) and
[TypeScript types](../ai/contracts/typescript/ai-service-v0.1.ts).
Resolve JSON Schema references with a local registry; `ostosense.local`
identifies schemas and is not a network dependency.

| HTTP status | Meaning |
|---|---|
| 200 | Request processed; inspect `prediction_available` |
| 401 | Missing or invalid authentication |
| 403 | Requested operating mode is disabled |
| 413 | Request too large |
| 422 | Invalid request shape or values |
| 503 | Model unavailable |

A normal waiting response can be HTTP 200 with no class. On errors, stale data,
or invalid responses, clear the current class. Historical predictions may remain
in a timestamped history.

## Raw-window input

Use the [raw-window schema](../ai/contracts/ai-raw-window-input-v0.1.schema.json).
The backend owns device/session authorization and creates a fresh acquisition
identifier for each confirmed measurement. Normalize device time once to that
acquisition; do not reset it per request or replace it with Wi-Fi receipt time.

The current model input channel is `Kap_7`. Keep all recorded channels in
the raw archive, but submit only the specified capacitive input to this adapter.

| Input | Required interval |
|---|---|
| `baseline_samples` | 600 dry samples in `[0, 60000)` ms |
| `window_samples` | 1,200 samples in `[end - 120000, end)` ms |
| Window endpoints | 120000 ms, then increments of 10000 ms |

The baseline is the median of 60 one-second medians. Each one-second summary
requires ten samples; its timestamp is the last sample in that bin. Raw intervals
of 80–120 ms and summary intervals of 800–1200 ms are provisional engineering
settings. The same baseline is sent for subsequent windows, and overlapping
baseline/window samples must agree.

The five features use raw minus baseline, six-decimal rounding, and the order
defined in the [architecture guide](architecture.md). Do not substitute
normalized pilot-report deltas or add LIG/other channels.

Wait for the device-time watermark before requesting a window. Backends with
a 1,000-row query limit must paginate the 1,200-sample window.

## Quality and consumer behavior

Operator confirmation of sensor connection and dry baseline is required.
`NOT_EVALUATED` may be accepted only for this confirmed demonstration path,
with `quality_verification=OPERATOR_ONLY`; it is not rewritten to `OK`.
`AUTOMATIC` indicates firmware-provided `OK` states, not independent AI
verification of the sensor.

Missing, duplicate, nonfinite, misordered, or faulty readings prevent prediction.
A faulty baseline needs a new confirmed dry acquisition; a faulty window may
recover once a complete clean window is available.

| Condition | Consumer behavior |
|---|---|
| Collecting baseline/window | Show unavailable and the waiting reason |
| Synthetic example | Label as simulation, separate from patient data |
| Real experimental input | Label as experimental/unvalidated |
| Sensor/transport fault or stale result | Clear the current class and show the reason |

The original consumer policy refreshes every 5 seconds and considers results
older than 30 seconds stale. Determine freshness from acquisition time and
window identity, not the most recent database insert or screen refresh.
Reject future timestamps, replayed windows, wrong devices, and late responses.

Do not derive patient notifications, clinical actions, risk percentages, leakage
countdowns, volume, or skin status from the experimental class. LIG contact
detection and connection quality are independent states.

## Verification

```bash
python scripts/verify.py
```

The full environment is described in [Contributing](../CONTRIBUTING.md).
Container tests run separately in CI. Application/database/device integration
requires tests in those systems; this service does not implement their storage,
authorization, or firmware behavior.
