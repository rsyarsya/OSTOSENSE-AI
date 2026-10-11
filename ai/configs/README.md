# Pipeline configuration

These files define the existing experiments and data-processing settings.
Their version numbers identify configurations, not clinical validation stages.

| File | Purpose |
|---|---|
| [synthetic-v0.2.json](synthetic-v0.2.json) | Original nine-session deterministic fixture |
| [synthetic-v0.3.json](synthetic-v0.3.json) | Adds two validation sessions while retaining the original nine sessions |
| [features-v0.1.json](features-v0.1.json) | 120-second windows, 10-second stride, 1 Hz, five capacitive features |
| [training-v0.1.json](training-v0.1.json) | StandardScaler + mord.LogisticAT; development-only fitting |
| [raw-qc-v0.1.json](raw-qc-v0.1.json) | Provisional timing, baseline, and quality-check settings |
| [real-pilot-v0.1.json](real-pilot-v0.1.json) | Private logger inventory and descriptive preparation settings |

Synthetic v0.3 uses group-separated manifests in
[the evaluation fixtures](../tests/fixtures/ostosense-evaluation-v0.1/).
Its original nine sessions are assigned to development; the two added sessions
to validation. Only eligible labeled windows proceed to the model matrix.

Feature extraction uses `(t-W,t]` windows, a 200 ms timing tolerance, and the
declared feature order. Training uses `alpha=1.0`, `max_iter=10000`, and
uniform window weights. It accepts synthetic provenance, fits development only,
and rejects final-test rows.

The real-pilot preparation uses a provisional 20-second baseline and descriptive
normalized deltas. This differs from the confirmed 60-second baseline used by
the raw-window application adapter. P006 is retained as a sensor-fault recording
and excluded from the operational correlation summary.

## Provenance and boundaries

Numeric settings carry provenance such as `CONTRACT_DERIVED`,
`ENGINEERING_TEST_ONLY`, or `PROPOSED_PILOT_SETTING`. A provisional value is
not a calibrated physical threshold.

Real-data labeling boundaries B1/B2/B3 remain unset under the
[labeling specification](../../docs/ai-label-rulebook-v0.3.md). Numeric boundary
fixtures are permitted only for synthetic tests; the labeler rejects a mismatched
data origin. Configuration validation rejects missing/unknown keys, unsupported
versions, invalid types, and inconsistent timing.
