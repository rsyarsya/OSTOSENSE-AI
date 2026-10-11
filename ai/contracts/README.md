# Application contracts

These versioned interfaces connect the AI package to application consumers.
Validate payloads at runtime; TypeScript declarations alone do not validate JSON.

| Schema | Purpose |
|---|---|
| [Runtime output v0.2](ai-runtime-output-v0.2.schema.json) | Current availability, model status, and ordinal class |
| [Feature input v0.1](ai-feature-input-v0.1.schema.json) | Five ordered capacitive features |
| [Raw-window input v0.1](ai-raw-window-input-v0.1.schema.json) | Baseline, raw window, and operator confirmation |
| [Service response v0.1](ai-service-response-v0.1.schema.json) | Runtime output, diagnostics, and computed features |
| [Runtime output v0.1](ai-runtime-output-v0.1.schema.json) | Legacy interface retained for compatibility |

[TypeScript definitions](typescript/) accompany the schemas.
[Examples](examples/) cover waiting, synthetic, and experimental real-input
states. The current display and handling rules are in the
[runtime guide](../../docs/ai-software-integration-contract-v0.2.md).

The five features always follow `feature_order` and
`RAW_MINUS_SESSION_BASELINE`. Normalized pilot-report deltas are descriptive
statistics, not this model's inputs.

Repeated placeholder hashes and zero-valued vectors in format fixtures are not
measured samples. The [bundled examples](../src/ostosense_ai/model_assets/engineering_v03/examples.json)
contain actual synthetic inputs, the real model hash, and computed outputs.

Resolve schema references locally. The `ostosense.local` identifiers do not
require network access. Breaking field or state changes need a new schema
version; existing consumers must retain their versioned interface.
