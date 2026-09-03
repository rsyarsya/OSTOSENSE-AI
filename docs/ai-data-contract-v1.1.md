# OSTOSENSE AI Data Contract v1.1

This document is the implementation-level companion to the locked AI contract.
It defines storage types and event metadata for Tier 1 collection. Training and
Tier 2 generation are intentionally out of scope.

## Common rules

- Timestamps are Unix epoch milliseconds in UTC and are stored as integers.
- IDs are non-empty UTF-8 strings without commas or line breaks.
- Empty optional CSV values are serialized as an empty field, never as `null`.
- Raw sensor values remain in readout-circuit/ADC units until hardware
  calibration defines physical units.
- `wetness_proxy` is never reported as `%RH` while LIG is uncalibrated.
- `system_quality` is derived from `cap_quality` and `lig_quality`; producers
  must not choose it independently.

## sessions.csv

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `session_id` | string | yes | Unique session identifier |
| `arm` | enum | yes | `LEAK_GRADUAL`, `SAFE`, `LEAK_SUDDEN`, or `FIELD` |
| `bag_id` | string | yes | Physical bag identifier |
| `sensor_id` | string | yes | Sensor assembly identifier |
| `device_id` | string | yes | ESP32/device identifier |
| `fluid_type` | string | yes | Test fluid and mixture identifier |
| `operator_id` | string | yes | Experiment operator identifier |
| `baseline_value` | float | yes | Median 60-second capacitance baseline |
| `baseline_std` | float | yes | Standard deviation during baseline |
| `start_timestamp` | int64 | yes | Session start, epoch ms |
| `end_timestamp` | int64 | yes | Session end, epoch ms |
| `end_reason` | enum | yes | `CEILING_REACHED`, `LEAK_CONFIRMED`, or `MANUAL_STOP` |
| `model_version` | string | no | Empty before an inference model is deployed |
| `firmware_version` | string | yes | Firmware build identifier |

Session rows are appended when a session closes. Samples and events may be
written while the session is active.

## samples.csv

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `timestamp` | int64 | yes | Sample timestamp, epoch ms |
| `session_id` | string | yes | Parent session |
| `capacitance_raw` | float | yes | Differential capacitance readout |
| `lig_raw` | float | yes | Uncalibrated LIG wetness proxy |
| `cap_quality` | enum | yes | Capacitance-channel quality |
| `lig_quality` | enum | yes | LIG-channel quality |
| `system_quality` | enum | yes | Derived system quality |
| `activity_state` | string | no | Controlled activity label |
| `orientation_position` | string | no | Controlled bag/device orientation |

Channel quality values are:

```text
cap_quality: OK, DISCONNECTED, ADC_SATURATED, BASELINE_INVALID, DATA_GAP,
             WARMING_UP
lig_quality: OK, DISCONNECTED, ADC_SATURATED, BASELINE_INVALID, DATA_GAP,
             WARMING_UP

system_quality:
  INITIALIZING        at least one channel is still WARMING_UP
  NORMAL             cap valid, LIG valid
  ML_UNAVAILABLE      cap invalid, LIG valid
  FAILSAFE_DEGRADED   cap valid, LIG invalid
  UNSAFE              cap invalid, LIG invalid
```

Only `OK` is considered valid for channel availability decisions. While either
channel is `WARMING_UP`, aggregate quality is `INITIALIZING`; the four-state
steady-state matrix is evaluated only after both channels resolve to `OK` or a
fault state. Predictive output is not published during `INITIALIZING`, although
the firmware may fill its rolling buffer.

### Pending LIG calibration state machine

The LIG readout requires an initial calibration before `leak_flag` is valid.
The schema already supports the required quality states, but firmware timing
and retry behavior are not locked until the hardware team confirms:

1. LIG calibration duration.
2. The concrete validity test and thresholds.
3. Whether mid-session recalibration is physically meaningful.

The provisional transition is:

```text
Boot/new session -> WARMING_UP
                 -> OK                 when calibration is valid
                 -> BASELINE_INVALID   when calibration is invalid
```

Until item 3 is confirmed, `BASELINE_INVALID` should remain stable rather
than silently retrying. This is a provisional safety default, not a locked
session-lifetime invariant. Any later retry policy must expose calibration
state to the user and must never report direct leak detection as available
before the LIG channel returns to `OK`.

The existing long-press action starts a full new session: it resets the
capacitance baseline and restarts LIG calibration together. It is valid only
for the new/empty-bag workflow. It must not be presented as a general
mid-session LIG recovery action because resetting capacitance while a bag is
partially filled would erase the current fill reference. If hardware later
supports contact repair without replacing the bag, that recovery requires an
explicit recalibration path that preserves the capacitance baseline.

## events.csv

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `event_id` | string | yes | Unique event identifier |
| `session_id` | string | yes | Parent session |
| `timestamp` | int64 | yes | Event timestamp, epoch ms |
| `event_type` | enum | yes | Event name below |
| `event_metadata` | JSON object | yes | Compact, structured event payload |

Supported event types:

```text
INJECTION_START, INJECTION_END,
PHYSICAL_LEAK_OBSERVED, LEAK_FLAG_FIRST, LEAK_FLAG_CONFIRMED,
LIG_CALIBRATION_STARTED, LIG_CALIBRATION_PASSED, LIG_CALIBRATION_FAILED,
BAG_EMPTIED_LOGGED,
ALERT_RAISED, ALERT_ACKNOWLEDGED, ALERT_IGNORED,
MANUAL_SESSION_RESET, DEVICE_RESTART
```

Recommended metadata:

- `INJECTION_START`: `delivery_mode`, `target_flow_ml_min`.
- `INJECTION_END`: `delivered_volume_ml`, `cumulative_volume_ml`,
  `measured_flow_ml_min`.
- `PHYSICAL_LEAK_OBSERVED`: `observation_method`, `operator_id`, optional
  `video_reference`.
- `LIG_CALIBRATION_STARTED`: `attempt`, `reason`, and configured duration if
  known.
- `LIG_CALIBRATION_PASSED`/`LIG_CALIBRATION_FAILED`: `duration_ms`,
  `sample_count`, measured baseline statistics, `validity_rule_version`, and
  optional `failure_reason`. Exact statistics remain pending hardware input.
- `BAG_EMPTIED_LOGGED`: `source="mobile_app"`.
- Alert events: `alert_level`, `model_version`, and related alert/event ID.

The three leak timestamps are represented by three independent events:

```text
T_physical_leak = PHYSICAL_LEAK_OBSERVED.timestamp
T_flag          = LEAK_FLAG_FIRST.timestamp
T_confirm       = LEAK_FLAG_CONFIRMED.timestamp
```

`T_physical_leak` is the independent ground truth. `T_flag` and `T_confirm`
measure LIG detection latency and must not replace it.

## Dataset partition invariant

Tier 1 Final Test session, bag, and sensor identifiers must be sealed before
simulator fitting, preprocessing selection, boundary/window selection, or
hyperparameter tuning. No Final Test record may be used to fit Tier 2.
