# Ordinal labeling specification v0.3

**Status:** labeling structure specified; numerical boundaries for real data
remain `PILOT_PENDING`. The current implementation labels synthetic fixtures.
The [original specification](archive/ai-label-rulebook-v0.3-original.md) is
retained for historical detail; this reference describes the applicable rules
without changing the version or algorithms.

## Scope and terminology

| Term | Meaning |
|---|---|
| Risk label | Offline target derived from experiment arm, events, and timing |
| Physical leak time | Independently observed `PHYSICAL_LEAK_OBSERVED` event |
| LIG flag | Direct contact-detection event, separate from the physical observation |
| Quality state | Eligibility of a reading, not a risk class |
| Notification | An application policy, evaluated separately from classification |

Capacitive values, LIG magnitude, and model predictions must not determine the
target label. Conversely, label-generating metadata must not enter the feature
vector. The current model uses only the five declared capacitive features.

## Windows

Labels and features use the same full window `(t - W, t]`, with `W=120 s`,
stride `S=10 s`, and 1 Hz samples. The left boundary is excluded and the right
boundary included. For the first sample timestamp `t_ref`, candidate ends are
`t_ref + (W + k*S)*1000` milliseconds for `k=0,1,...`.

Use the condition at window end, not a majority vote across phases. Partial
windows are excluded. A single non-`OK` capacitive sample invalidates the
window. Duplicate timestamps and timing violations have explicit exclusion
reasons. Excluded windows remain in the audit output.

## Event-derived classes

For a gradual-leak window ending before the observed leak,
`tau = T_physical_leak - t`. Boundaries satisfy `0 < B1 < B2 < B3`.

| Class | Condition |
|---|---|
| Safe | Dry baseline before injection, or `tau > B3` |
| Monitor | `B2 < tau <= B3` |
| Caution | `B1 < tau <= B2` |
| Urgent | `0 < tau <= B1` |

Windows ending at or after the physical leak are `POST_LEAK` and excluded.
Urgent does not include an ongoing leak. Direct detection remains a separate
sensor function.

The primary boundary scheme uses fixed bench-time intervals, stored in seconds.
Real values require development/pilot evidence and must be fixed before final
test access. Historical hour-scale proposals are not valid fixture or clinical
thresholds. Fractional time-to-leak based on total session duration is an offline
sensitivity analysis only because it uses future information.

Numeric test boundaries live under `ai/tests/fixtures/`, identify themselves
as `ENGINEERING_TEST_ONLY`, and accept only `SYNTHETIC_PIPELINE_TEST_ONLY`
inputs. They cannot be applied to real recordings.

## Session rules

| Scenario | Rule |
|---|---|
| `SAFE` | Valid windows are Safe only if a predeclared non-leaking protocol and safe horizon are satisfied. The horizon begins at the first sample where both channels are `OK`. |
| `LEAK_GRADUAL` with observed leak | Dry windows are Safe; subsequent pre-leak windows use the boundary table; post-leak windows are excluded. |
| Non-leaking fill in gradual arm | Post-injection windows are Safe only when a predeclared horizon is satisfied after the final `INJECTION_END`; otherwise they are censored. Dry windows remain Safe. |
| Planned non-leaking fill that leaks | Use the observed leak endpoint and record `UNPLANNED_PHYSICAL_LEAK` as a protocol deviation. |
| `LEAK_SUDDEN` | Exclude from ordinal training/evaluation as `SUDDEN_ARM`. |
| `FIELD` | Exclude as `FIELD_ARM_EXCLUDED`; the bench rule does not establish field labels. |
| Movement without fluid | Safe only within a predeclared non-leaking protocol; movement during gradual filling does not override event-derived labels. |

Missing or unsatisfied observation horizons yield
`CENSORED_NO_SAFE_HORIZON`. LIG quality is used for the safe-arm observation
anchor, not as a continuous risk target. LIG flags do not replace visual ground
truth; their order relative to physical observations must be recorded rather
than assumed.

## Inputs and validation

The labeler uses six inputs:

1. `sessions.csv`: identity, baseline, arm, and end reason.
2. `samples.csv`: timestamps and channel quality.
3. `events.csv`: injection, leak, and other experiment events.
4. `protocol_manifest.csv`: predeclared plan, horizon, bag/sensor identities,
   and observation method.
5. Versioned boundary configuration.
6. `partition_manifest.csv`: group assignments.

Optional feature files may align candidate windows and verify source hashes.
Feature values never determine labels.

Malformed required events fail the complete labeling run before output is
created or replaced. Examples include a confirmed-leak session without a leak
observation, injection end without a preceding start, duplicate physical-leak
endpoints, events outside the session, or an unresolved leak in a safe arm.
`MALFORMED_REQUIRED_EVENTS` is a run error, not a per-window exclusion.

## Outputs and exclusion precedence

The labeler writes one row per candidate window, including IDs, index,
start/end timestamps, class/index, validity, exclusion reason, rulebook/boundary
versions, partition, and protocol-deviation fields. Invalid rows have empty
class fields.

Exclusion precedence is:

1. `DUPLICATE_TIMESTAMP`
2. `PARTIAL_WINDOW`
3. `TIMING_OUT_OF_TOLERANCE`
4. `INVALID_CAP_QUALITY`
5. Arm exclusions: `SUDDEN_ARM`, `FIELD_ARM_EXCLUDED`
6. Scenario exclusions: `CENSORED_NO_SAFE_HORIZON`, `POST_LEAK`

Raw records are retained. Invalid readings are never relabeled as low or high
risk. Run manifests capture input hashes, configuration, counts, and exclusions.

## Partitioning

Every session belongs to one partition. Sessions sharing a bag or sensor also
belong to the same partition. The manifest establishes development, validation,
and sealed final-test groups before fitting.

Final-test data must not influence preprocessing, exclusion policy, boundaries,
simulator fitting, or hyperparameter selection. The current synthetic trainer
fits development rows, ignores validation rows during fitting, and rejects
final-test rows.

## References

- [CSV data contract](ai-data-contract-v1.1.md)
- [Feature definitions](architecture.md)
- [Research method and limitations](research.md)
- [Original v0.3 specification](archive/ai-label-rulebook-v0.3-original.md)
- [Earlier v0.2 specification](archive/ai-label-rulebook-v0.2.md)
