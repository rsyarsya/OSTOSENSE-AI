# Python package

The package contains the CSV data contract, offline modeling pipeline, portable
model loader, and optional HTTP service. Install it from the repository root:

```bash
python -m pip install -e ./ai
python -m ostosense_ai.bundled_model
```

The distribution name `ostosense-contract` is retained for compatibility.
It installs both `ostosense_contract` and `ostosense_ai`. Core data processing
and exported-model inference use the standard library. Training and serving
dependencies are optional.

## Installation options

| Extra | Purpose |
|---|---|
| `pipeline` | NumPy, SciPy, scikit-learn, mord, and Matplotlib |
| `serve` | FastAPI and Uvicorn |
| `serve-test` | HTTP client for service tests |
| `quality` | Build, Ruff, Pyright, and JSON Schema validation |

```bash
python -m pip install -c ai/requirements-serve.lock -e "./ai[pipeline,quality,serve,serve-test]"
```

Run commands below from the repository root with the environment activated.
For environment setup and supported versions, see [Contributing](../CONTRIBUTING.md).

## Inference API

```python
from ostosense_ai.bundled_model import load_bundled_model
from ostosense_ai.inference import predict_exported_model

bundle = load_bundled_model()
example = bundle["examples"][0]
result = predict_exported_model(bundle["model"], [example["input"]["features"]])
print(result["predicted_labels"])  # ['Safe'] for this synthetic example
```

The five features must follow the model's declared order and scaling basis.
Use the [architecture guide](../docs/architecture.md) for their definitions.
The included examples are synthetic, and real input remains unvalidated.

## File-based integration

```bash
python -m ostosense_ai.integration_cli examples --output outputs/examples.json
python -m ostosense_ai.integration_cli evaluate-window --input ai/contracts/examples/service-v0.1/waiting-input.json --output outputs/waiting.json
```

The second command intentionally returns exit code **2**: the sample request
has no readings, so its reason is `COLLECTING_BASELINE`. Exit code 0 indicates a
prediction or successful example export; 1 indicates invalid input or an I/O
failure. Existing outputs are refused unless `--overwrite` is specified.

The [HTTP service guide](../docs/ai-service-integration-v0.1.md) documents the
same model behind an authenticated API.

## Offline pipeline

The simplest complete run is:

```bash
python scripts/reproduce.py --output outputs/synthetic-run
```

For custom experiments, each stage also has a `python -m` interface.
Use `--help` for complete arguments.

| Module | Input | Output |
|---|---|---|
| `synthetic` | Versioned config and seed | Sessions, samples, events, provenance |
| `features` | Contract CSVs and feature config | Five features per eligible window and exclusions |
| `labeling` | Events, protocol/partition manifests, boundary fixture | Ordinal labels and exclusions |
| `matrix` | Features and labels with matching provenance | Audited feature-label matrix |
| `training` | Matrix and training config | Scaler, OLR parameters, training manifest |
| `model_evaluation` | Exported model and matrix | Validation predictions and metrics |
| `edge_export` | Model and matrix | C++ model parameters and golden vectors |
| `raw_qc` | Real contract CSVs and protocol/config | Intake quality report |
| `pilot_data` | Authorized private logger CSVs | Unlabeled sensor summaries |
| `pilot_reporting` | Pilot summaries and synthetic matrix | Descriptive figures and optimizer trace |

Example: evaluate an existing run without retraining:

```bash
python -m ostosense_ai.model_evaluation --matrix outputs/synthetic-run/matrix --model outputs/synthetic-run/model --output outputs/validation-repeat
```

Outputs carry hashes and versioned manifests. Labels derive from events and
timing, while model inputs use capacitive measurements only. The trainer fits
development rows, ignores validation rows during fitting, and rejects sealed
final-test rows. Synthetic boundary fixtures cannot be used as real-data labels.

## CSV logger

```python
from ostosense_contract import CapQuality, LigQuality, SampleRecord, Tier1CsvLogger

logger = Tier1CsvLogger("outputs/session-001")
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

The example demonstrates sample serialization; a complete experiment also
requires session metadata and events. Aggregate quality is derived from channel
qualities. See the [CSV contract](../docs/ai-data-contract-v1.1.md).

## Validation

```bash
python -W error -m unittest discover -s ai/tests
python scripts/verify.py
```

Tests that require optional dependencies or a C++ compiler can be skipped in a
minimal installation. The full verifier requires the complete development stack
and compiler. Research assumptions, data availability, and interpretation are
documented in [Research notes](../docs/research.md).
