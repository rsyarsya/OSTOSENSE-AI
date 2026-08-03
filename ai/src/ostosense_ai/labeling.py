"""Deterministic ENGINEERING_TEST_ONLY four-class ordinal labeler (pipeline test only).

This module derives canonical ordinal ground-truth labels
(`0=Safe, 1=Monitor, 2=Caution, 3=Urgent`) for the synthetic OSTOSENSE dataset,
per structure-locked Label Rulebook v0.3. It produces *ground-truth labels only*:
no model, no metrics, no confusion matrix, no firmware inference.

It is dependency-free (standard library + ``ostosense_contract``) and reuses the
canonical windowing already implemented in ``features.py`` (interval ``(t-W,t]``,
``W``=120 s, stride 10 s, 1 Hz, ``t_ref``=first sample timestamp,
``window_id={session_id}-win-{index:04d}``) so labels and features share exactly
one window definition. Labels are derived only from arm, recorded events, and
timing; capacitance/LIG raw values, feature values, and model predictions never
determine a label. LIG *quality* is consulted only for the protocol-required
SAFE observation-start anchor.

Numeric boundaries come from an ENGINEERING_TEST_ONLY fixture under
``ai/tests/fixtures/`` guarded for ``SYNTHETIC_PIPELINE_TEST_ONLY`` input;
production B1/B2/B3 remain PILOT_PENDING and are never introduced here. A passing
run proves labeling-pipeline mechanics only, never AI accuracy, sensor validity,
early-warning performance, or clinical value.

CLI::

    PYTHONPATH=ai/src ai/.venv/bin/python -m ostosense_ai.labeling \\
        --input /tmp/ostosense-synthetic-raw \\
        --protocol-manifest .../protocol_manifest.csv \\
        --partition-manifest .../partition_manifest.csv \\
        --boundary-config .../boundary-engineering-test-only-v0.1.json \\
        --features /tmp/ostosense-features \\
        --output /tmp/ostosense-labels
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from ostosense_contract import (
    Arm,
    CapQuality,
    EndReason,
    EventRecord,
    EventType,
    LigQuality,
    SampleRecord,
    SessionRecord,
    SystemQuality,
)

from ostosense_ai import features

LABELER_VERSION = "0.1.1"
RULEBOOK_VERSION = "v0.3"
DATA_CONTRACT_VERSION = "v1.1"
ALLOWED_INPUT_ORIGIN = "SYNTHETIC_PIPELINE_TEST_ONLY"
ENGINEERING_TEST_ONLY = "ENGINEERING_TEST_ONLY"
NO_PERFORMANCE_WARNING = (
    "ENGINEERING_TEST_ONLY synthetic pipeline testing. These are ground-truth "
    "labels for mechanical pipeline checks only; not a model, not a metric, and "
    "not an OSTOSENSE performance, sensor, or clinical result."
)

CLASS_NAMES = ("Safe", "Monitor", "Caution", "Urgent")
CLASS_NAME_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

STRUCTURAL_EXCLUSIONS = features.EXCLUSION_PRIORITY  # DUPLICATE/PARTIAL/TIMING/INVALID_CAP
ARM_EXCLUSIONS = ("SUDDEN_ARM", "FIELD_ARM_EXCLUDED")
SCENARIO_EXCLUSIONS = ("CENSORED_NO_SAFE_HORIZON", "POST_LEAK")
ALL_EXCLUSION_REASONS = STRUCTURAL_EXCLUSIONS + ARM_EXCLUSIONS + SCENARIO_EXCLUSIONS

LABELS_CSV_COLUMNS = (
    "window_id",
    "session_id",
    "window_index",
    "window_start",
    "window_end",
    "risk_label",
    "risk_label_index",
    "label_valid",
    "exclusion_reason",
    "rulebook_version",
    "boundary_config_version",
    "dataset_partition",
    "protocol_deviation",
    "protocol_deviation_reason",
)
_OUTPUT_FILES = ("labels.csv", "label_manifest.json")

PROTOCOL_MANIFEST_FIELDS = (
    "session_id",
    "protocol_version",
    "planned_arm",
    "planned_safe_horizon_s",
    "target_fill_or_volume",
    "injection_profile",
    "injection_method",
    "planned_flow_ml_min",
    "physical_leak_observation_method",
    "operator_id",
    "bag_id",
    "sensor_id",
    "device_id",
)
PARTITION_MANIFEST_FIELDS = (
    "session_id",
    "dataset_partition",
    "partition_version",
    "bag_id",
    "sensor_id",
)
PARTITION_VALUES = ("development", "validation", "final_test")

_BOUNDARY_KEYS = {
    "boundary_config_version",
    "status",
    "rulebook_version",
    "unit",
    "allowed_dataset_origin",
    "b1_s",
    "b2_s",
    "b3_s",
    "warning",
}

_MS = features._MS_PER_SECOND
_WINDOW_MS = features._WORKING_WINDOW_SECONDS * _MS
_STRIDE_MS = features._WORKING_STRIDE_SECONDS * _MS
_EXPECTED_FULL = features._WORKING_WINDOW_SECONDS * features._WORKING_SAMPLING_RATE_HZ
_JITTER = features._WORKING_JITTER_TOLERANCE_MS

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES_DIR = (_PROJECT_ROOT / "ai" / "tests" / "fixtures").resolve()


class LabelingError(ValueError):
    """Run-level labeling validation failure (leaves outputs untouched)."""


class MalformedRequiredEvents(LabelingError):
    """Required events are missing, duplicated, out of bounds, or contradictory."""


# --------------------------------------------------------------------------- #
# Boundary fixture (ENGINEERING_TEST_ONLY)
# --------------------------------------------------------------------------- #
def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelingError(f"boundary {name} must be an integer")
    if value <= 0:
        raise LabelingError(f"boundary {name} must be positive")
    return value


def _load_boundary_config(path: str | Path, input_origin: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise LabelingError(f"boundary config not found: {path}")
    if not resolved.is_relative_to(_FIXTURES_DIR):
        raise LabelingError(
            "boundary config must live under ai/tests/fixtures/ "
            f"(got {resolved})"
        )
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise LabelingError(f"invalid boundary config JSON: {error}") from error
    if not isinstance(data, dict):
        raise LabelingError("boundary config must be a JSON object")

    missing = sorted(_BOUNDARY_KEYS - data.keys())
    if missing:
        raise LabelingError(f"boundary config missing keys: {missing}")
    unknown = sorted(data.keys() - _BOUNDARY_KEYS)
    if unknown:
        raise LabelingError(f"boundary config has unsupported keys: {unknown}")

    version = data["boundary_config_version"]
    if not isinstance(version, str) or ENGINEERING_TEST_ONLY not in version:
        raise LabelingError("boundary_config_version must contain ENGINEERING_TEST_ONLY")
    if data["status"] != ENGINEERING_TEST_ONLY:
        raise LabelingError("boundary status must be exactly ENGINEERING_TEST_ONLY")
    if data["rulebook_version"] != RULEBOOK_VERSION:
        raise LabelingError(f"boundary rulebook_version must be {RULEBOOK_VERSION}")
    if data["unit"] != "seconds":
        raise LabelingError("boundary unit must be 'seconds'")
    if data["allowed_dataset_origin"] != ALLOWED_INPUT_ORIGIN:
        raise LabelingError(
            f"boundary allowed_dataset_origin must be {ALLOWED_INPUT_ORIGIN}"
        )
    if input_origin != ALLOWED_INPUT_ORIGIN:
        raise LabelingError(
            "raw dataset origin must be SYNTHETIC_PIPELINE_TEST_ONLY for this "
            f"ENGINEERING_TEST_ONLY fixture (got {input_origin!r})"
        )

    b1 = _require_positive_int("b1_s", data["b1_s"])
    b2 = _require_positive_int("b2_s", data["b2_s"])
    b3 = _require_positive_int("b3_s", data["b3_s"])
    if not b1 < b2 < b3:
        raise LabelingError("boundaries must satisfy 0 < b1_s < b2_s < b3_s")

    return {
        "version": version,
        "b1_ms": b1 * _MS,
        "b2_ms": b2 * _MS,
        "b3_ms": b3 * _MS,
        "sha256": _sha256_file(resolved),
        "path": resolved,
    }


# --------------------------------------------------------------------------- #
# Input parsing (strict, contract-aware)
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_sessions_full(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise LabelingError(f"missing required input file: {path.name}")
    sessions: dict[str, dict[str, Any]] = {}
    for row in features._load_contract_rows(path, SessionRecord.FIELDS):
        label = row.get("session_id") or "<empty>"
        try:
            record = SessionRecord(
                session_id=row["session_id"],
                arm=Arm(row["arm"]),
                bag_id=row["bag_id"],
                sensor_id=row["sensor_id"],
                device_id=row["device_id"],
                fluid_type=row["fluid_type"],
                operator_id=row["operator_id"],
                baseline_value=features._parse_float(
                    f"session {label} baseline_value", row["baseline_value"]
                ),
                baseline_std=features._parse_float(
                    f"session {label} baseline_std", row["baseline_std"]
                ),
                start_timestamp=features._parse_int(
                    f"session {label} start_timestamp", row["start_timestamp"]
                ),
                end_timestamp=features._parse_int(
                    f"session {label} end_timestamp", row["end_timestamp"]
                ),
                end_reason=EndReason(row["end_reason"]),
                model_version=row["model_version"],
                firmware_version=row["firmware_version"],
            )
        except (TypeError, ValueError) as error:
            raise LabelingError(f"invalid sessions.csv row for {label}: {error}") from error
        if record.session_id in sessions:
            raise LabelingError(f"duplicate session_id in sessions.csv: {record.session_id}")
        sessions[record.session_id] = {
            "session_id": record.session_id,
            "arm": record.arm.value,
            "bag_id": record.bag_id,
            "sensor_id": record.sensor_id,
            "device_id": record.device_id,
            "baseline_value": record.baseline_value,
            "start_timestamp": record.start_timestamp,
            "end_timestamp": record.end_timestamp,
            "end_reason": record.end_reason.value,
        }
    if not sessions:
        raise LabelingError("sessions.csv contains no sessions")
    return sessions


def _read_samples_full(
    path: Path, sessions: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise LabelingError(f"missing required input file: {path.name}")
    grouped: dict[str, list[dict[str, Any]]] = {sid: [] for sid in sessions}
    last_seen: dict[str, int] = {}
    for row in features._load_contract_rows(path, SampleRecord.FIELDS):
        session_id = row["session_id"]
        session = sessions.get(session_id)
        if session is None:
            raise LabelingError(f"samples.csv references unknown session_id: {session_id}")
        try:
            record = SampleRecord(
                timestamp=features._parse_int("sample timestamp", row["timestamp"]),
                session_id=session_id,
                capacitance_raw=features._parse_float("capacitance_raw", row["capacitance_raw"]),
                lig_raw=features._parse_float("lig_raw", row["lig_raw"]),
                cap_quality=CapQuality(row["cap_quality"]),
                lig_quality=LigQuality(row["lig_quality"]),
                system_quality=SystemQuality(row["system_quality"]),
                activity_state=row["activity_state"],
                orientation_position=row["orientation_position"],
            )
        except (TypeError, ValueError) as error:
            raise LabelingError(
                f"invalid samples.csv row in session {session_id}: {error}"
            ) from error
        if not (session["start_timestamp"] <= record.timestamp <= session["end_timestamp"]):
            raise LabelingError(f"sample in {session_id} falls outside session bounds")
        if session_id in last_seen and record.timestamp < last_seen[session_id]:
            raise LabelingError(f"samples in {session_id} are not in ascending time order")
        last_seen[session_id] = record.timestamp
        grouped[session_id].append(
            {
                "timestamp": record.timestamp,
                "capacitance_raw": record.capacitance_raw,
                "cap_quality": record.cap_quality.value,
                "lig_quality": record.lig_quality.value,
            }
        )
    return grouped


def _read_events(
    path: Path, sessions: dict[str, dict[str, Any]]
) -> dict[str, list[tuple[int, str]]]:
    if not path.exists():
        raise LabelingError(f"missing required input file: {path.name}")
    grouped: dict[str, list[tuple[int, str]]] = {sid: [] for sid in sessions}
    seen_event_ids: set[str] = set()
    for row in features._load_contract_rows(path, EventRecord.FIELDS):
        session_id = row["session_id"]
        session = sessions.get(session_id)
        if session is None:
            raise LabelingError(f"events.csv references unknown session_id: {session_id}")
        try:
            metadata = json.loads(row["event_metadata"])
            record = EventRecord(
                event_id=row["event_id"],
                session_id=session_id,
                timestamp=features._parse_int("event timestamp", row["timestamp"]),
                event_type=EventType(row["event_type"]),
                event_metadata=metadata,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise LabelingError(
                f"invalid events.csv row in session {session_id}: {error}"
            ) from error
        if record.event_id in seen_event_ids:
            raise MalformedRequiredEvents(
                "MALFORMED_REQUIRED_EVENTS: duplicate event_id "
                f"{record.event_id!r}"
            )
        seen_event_ids.add(record.event_id)
        if not (session["start_timestamp"] <= record.timestamp <= session["end_timestamp"]):
            raise MalformedRequiredEvents(
                f"MALFORMED_REQUIRED_EVENTS: event in {session_id} is outside session bounds"
            )
        grouped[session_id].append((record.timestamp, record.event_type.value))
    return grouped


def _load_manifest_rows(
    path: str | Path, expected_fields: tuple[str, ...], label: str
) -> dict[str, dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise LabelingError(f"missing {label}: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_fields):
            raise LabelingError(
                f"{label} header does not match Protocol v0.1: {reader.fieldnames!r}"
            )
        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise LabelingError(f"{label} line {line_number} has a malformed row")
            session_id = row["session_id"]
            if not session_id:
                raise LabelingError(f"{label} line {line_number} has an empty session_id")
            if session_id in rows:
                raise LabelingError(f"duplicate session_id in {label}: {session_id}")
            rows[session_id] = row
    return rows


def _read_input_origin(input_dir: Path) -> str:
    return features._read_input_origin(input_dir)


# --------------------------------------------------------------------------- #
# Cross-input validation
# --------------------------------------------------------------------------- #
def _validate_cross_inputs(
    sessions: dict[str, dict[str, Any]],
    protocol: dict[str, dict[str, str]],
    partition: dict[str, dict[str, str]],
) -> None:
    session_ids = set(sessions)
    if set(protocol) != session_ids:
        raise LabelingError("protocol_manifest sessions do not exactly cover sessions.csv")
    if set(partition) != session_ids:
        raise LabelingError("partition_manifest sessions do not exactly cover sessions.csv")

    bag_partition: dict[str, str] = {}
    sensor_partition: dict[str, str] = {}
    for session_id, session in sessions.items():
        prow = protocol[session_id]
        if prow["planned_arm"] != session["arm"]:
            raise LabelingError(f"planned_arm mismatch for {session_id}")
        for field in ("bag_id", "sensor_id", "device_id"):
            if prow[field] != session[field]:
                raise LabelingError(f"protocol_manifest {field} mismatch for {session_id}")

        part = partition[session_id]
        for field in ("bag_id", "sensor_id"):
            if part[field] != session[field]:
                raise LabelingError(f"partition_manifest {field} mismatch for {session_id}")
        assigned = part["dataset_partition"]
        if assigned not in PARTITION_VALUES:
            raise LabelingError(f"invalid dataset_partition for {session_id}: {assigned!r}")

        for group, mapping in (("bag", bag_partition), ("sensor", sensor_partition)):
            key = session["bag_id"] if group == "bag" else session["sensor_id"]
            if key in mapping and mapping[key] != assigned:
                raise LabelingError(
                    f"partition leakage: {group} {key!r} spans multiple partitions"
                )
            mapping[key] = assigned


def _parse_planned_horizon(session_id: str, raw: str) -> int | None:
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise LabelingError(
            f"planned_safe_horizon_s for {session_id} must be an integer or empty"
        ) from error
    if value <= 0:
        raise LabelingError(f"planned_safe_horizon_s for {session_id} must be positive")
    return value


# --------------------------------------------------------------------------- #
# Per-session event summary + labeling
# --------------------------------------------------------------------------- #
def _summarize_events(
    session: dict[str, Any], events: list[tuple[int, str]]
) -> dict[str, Any]:
    injections: list[tuple[int, int]] = []  # (timestamp, rank: START=0, END=1)
    physical_leaks: list[int] = []
    for timestamp, event_type in events:
        if event_type == EventType.INJECTION_START.value:
            injections.append((timestamp, 0))
        elif event_type == EventType.INJECTION_END.value:
            injections.append((timestamp, 1))
        elif event_type == EventType.PHYSICAL_LEAK_OBSERVED.value:
            physical_leaks.append(timestamp)

    session_id = session["session_id"]

    # Injection pairing: no overlap, every START closes, and every END has a START.
    depth = 0
    starts: list[int] = []
    ends: list[int] = []
    for timestamp, rank in sorted(injections):
        if rank == 0:
            depth += 1
            starts.append(timestamp)
            if depth > 1:
                raise MalformedRequiredEvents(
                    f"MALFORMED_REQUIRED_EVENTS: overlapping injection pair in {session_id}"
                )
        else:
            ends.append(timestamp)
            if depth == 0:
                raise MalformedRequiredEvents(
                    f"MALFORMED_REQUIRED_EVENTS: INJECTION_END without INJECTION_START in {session_id}"
                )
            depth -= 1
    if depth != 0:
        raise MalformedRequiredEvents(
            f"MALFORMED_REQUIRED_EVENTS: INJECTION_START without INJECTION_END in {session_id}"
        )

    if len(physical_leaks) > 1:
        raise MalformedRequiredEvents(
            f"MALFORMED_REQUIRED_EVENTS: more than one PHYSICAL_LEAK_OBSERVED in {session_id}"
        )
    has_leak = len(physical_leaks) == 1
    if session["end_reason"] == EndReason.LEAK_CONFIRMED.value and not has_leak:
        raise MalformedRequiredEvents(
            f"MALFORMED_REQUIRED_EVENTS: {session_id} end_reason LEAK_CONFIRMED without physical leak"
        )

    arm = session["arm"]
    if arm == Arm.SAFE.value and injections:
        raise MalformedRequiredEvents(
            f"MALFORMED_REQUIRED_EVENTS: SAFE session {session_id} contains injection events"
        )
    if arm == Arm.SAFE.value and has_leak:
        raise MalformedRequiredEvents(
            f"MALFORMED_REQUIRED_EVENTS: SAFE session {session_id} contains a physical leak"
        )
    if arm == Arm.LEAK_GRADUAL.value:
        if not starts:
            raise MalformedRequiredEvents(
                f"MALFORMED_REQUIRED_EVENTS: LEAK_GRADUAL session {session_id} has no INJECTION_START"
            )
        if not ends:  # Defensive; an unmatched START is already rejected above.
            raise MalformedRequiredEvents(
                f"MALFORMED_REQUIRED_EVENTS: LEAK_GRADUAL session {session_id} has no INJECTION_END"
            )
        if has_leak and starts[0] > physical_leaks[0]:
            raise MalformedRequiredEvents(
                f"MALFORMED_REQUIRED_EVENTS: physical leak precedes injection in {session_id}"
            )

    # LIG flag events are intentionally absent here. They are fail-safe outputs,
    # not ordinal ground truth; a flag without physical leak is a false-alarm
    # observation for system-level evaluation, not a labeling error.
    return {
        "first_injection_start": starts[0] if starts else None,
        "last_injection_end": ends[-1] if ends else None,
        "physical_leak": physical_leaks[0] if has_leak else None,
    }


def _safe_anchor(samples: list[dict[str, Any]]) -> int | None:
    for sample in samples:
        if (
            sample["cap_quality"] == CapQuality.OK.value
            and sample["lig_quality"] == LigQuality.OK.value
        ):
            return sample["timestamp"]
    return None


def _tau_class(tau_ms: int, boundary: dict[str, Any]) -> str:
    if tau_ms > boundary["b3_ms"]:
        return "Safe"
    if tau_ms > boundary["b2_ms"]:
        return "Monitor"
    if tau_ms > boundary["b1_ms"]:
        return "Caution"
    return "Urgent"


def _label_session(
    session: dict[str, Any],
    windows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    events: list[tuple[int, str]],
    planned_horizon: int | None,
    partition: str,
    boundary: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool, str]:
    arm = session["arm"]
    summary = _summarize_events(session, events)
    first_inj = summary["first_injection_start"]
    last_inj_end = summary["last_injection_end"]
    t_leak = summary["physical_leak"]
    anchor = _safe_anchor(samples)
    end_ts = session["end_timestamp"]

    protocol_deviation = arm == Arm.LEAK_GRADUAL.value and t_leak is not None and planned_horizon is not None
    deviation_reason = "UNPLANNED_PHYSICAL_LEAK" if protocol_deviation else ""

    safe_horizon_completed = (
        planned_horizon is not None
        and anchor is not None
        and end_ts >= anchor + planned_horizon * _MS
    )
    fill_horizon_completed = (
        planned_horizon is not None
        and last_inj_end is not None
        and end_ts >= last_inj_end + planned_horizon * _MS
    )

    rows: list[dict[str, Any]] = []
    for window in windows:
        structural = window["exclusion_reason"]
        risk: str | None
        reason: str
        if structural:  # precedence 1: structural / capacitive-quality
            risk, reason = None, structural
        elif arm == Arm.LEAK_SUDDEN.value:  # precedence 2: arm
            risk, reason = None, "SUDDEN_ARM"
        elif arm == Arm.FIELD.value:
            risk, reason = None, "FIELD_ARM_EXCLUDED"
        else:
            window_end = window["window_end"]
            if arm == Arm.SAFE.value:
                if safe_horizon_completed:
                    risk, reason = "Safe", ""
                else:
                    risk, reason = None, "CENSORED_NO_SAFE_HORIZON"
            elif arm == Arm.LEAK_GRADUAL.value:
                if t_leak is not None:
                    if window_end >= t_leak:
                        risk, reason = None, "POST_LEAK"
                    elif first_inj is not None and window_end < first_inj:
                        risk, reason = "Safe", ""
                    else:
                        risk, reason = _tau_class(t_leak - window_end, boundary), ""
                else:  # non-leaking fill
                    if first_inj is not None and window_end < first_inj:
                        risk, reason = "Safe", ""
                    elif fill_horizon_completed:
                        risk, reason = "Safe", ""
                    else:
                        risk, reason = None, "CENSORED_NO_SAFE_HORIZON"
            else:  # pragma: no cover - guarded by contract enum
                raise LabelingError(f"unhandled arm {arm!r}")

        rows.append(
            {
                "window_id": window["window_id"],
                "session_id": window["session_id"],
                "window_index": window["window_index"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "risk_label": risk or "",
                "risk_label_index": CLASS_NAME_TO_INDEX[risk] if risk else "",
                "label_valid": risk is not None,
                "exclusion_reason": reason,
                "rulebook_version": RULEBOOK_VERSION,
                "boundary_config_version": boundary["version"],
                "dataset_partition": partition,
                "protocol_deviation": protocol_deviation,
                "protocol_deviation_reason": deviation_reason,
            }
        )
    return rows, protocol_deviation, deviation_reason


# --------------------------------------------------------------------------- #
# Optional feature-artifact cross-check
# --------------------------------------------------------------------------- #
def _verify_features(
    features_dir: Path,
    sessions_sha: str,
    samples_sha: str,
    structural_rows: list[dict[str, Any]],
) -> dict[str, str]:
    manifest_path = features_dir / "feature_manifest.json"
    csv_path = features_dir / "features.csv"
    if not manifest_path.is_file() or not csv_path.is_file():
        raise LabelingError("features directory must contain features.csv and feature_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise LabelingError(f"invalid feature_manifest.json: {error}") from error
    if not isinstance(manifest, dict):
        raise LabelingError("feature_manifest.json must contain a JSON object")
    if manifest.get("input_sessions_sha256") != sessions_sha:
        raise LabelingError("feature_manifest sessions hash does not match labeling input")
    if manifest.get("input_samples_sha256") != samples_sha:
        raise LabelingError("feature_manifest samples hash does not match labeling input")
    if manifest.get("input_dataset_origin") != ALLOWED_INPUT_ORIGIN:
        raise LabelingError("feature_manifest dataset origin does not match labeling input")

    expected_convention = {
        "interval": features._WINDOW_INTERVAL,
        "window_seconds": features._WORKING_WINDOW_SECONDS,
        "stride_seconds": features._WORKING_STRIDE_SECONDS,
        "sampling_rate_hz": features._WORKING_SAMPLING_RATE_HZ,
        "jitter_tolerance_ms": _JITTER,
        "expected_full_samples": _EXPECTED_FULL,
    }
    convention = manifest.get("window_convention")
    if not isinstance(convention, dict) or any(
        convention.get(key) != value for key, value in expected_convention.items()
    ):
        raise LabelingError("feature_manifest window convention is not canonical")

    feature_rows: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(features.FEATURES_CSV_COLUMNS):
            raise LabelingError("features.csv header does not match the canonical extractor")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise LabelingError(f"features.csv line {line_number} is malformed")
            window_id = row["window_id"]
            if not window_id:
                raise LabelingError(f"features.csv line {line_number} has empty window_id")
            if window_id in feature_rows:
                raise LabelingError(f"duplicate window_id in features.csv: {window_id}")
            if row["feature_valid"] not in ("true", "false"):
                raise LabelingError(
                    f"features.csv line {line_number} has invalid feature_valid"
                )
            feature_rows[window_id] = row

    if manifest.get("candidate_window_count") != len(feature_rows):
        raise LabelingError("feature_manifest candidate count does not match features.csv")
    valid_feature_count = sum(
        1 for row in feature_rows.values() if row["feature_valid"] == "true"
    )
    if manifest.get("valid_window_count") != valid_feature_count:
        raise LabelingError("feature_manifest valid count does not match features.csv")
    if manifest.get("excluded_window_count") != len(feature_rows) - valid_feature_count:
        raise LabelingError("feature_manifest excluded count does not match features.csv")

    expected = {row["window_id"]: row for row in structural_rows}
    if set(feature_rows) != set(expected):
        raise LabelingError("feature window_id set does not match labeling windows")
    for window_id, ours in expected.items():
        theirs = feature_rows[window_id]
        try:
            their_index = int(theirs["window_index"])
            their_start = int(theirs["window_start"])
            their_end = int(theirs["window_end"])
        except ValueError as error:
            raise LabelingError(f"invalid feature audit integer for {window_id}") from error
        if their_index != ours["window_index"]:
            raise LabelingError(f"window_index mismatch for {window_id}")
        if their_start != ours["window_start"]:
            raise LabelingError(f"window_start mismatch for {window_id}")
        if their_end != ours["window_end"]:
            raise LabelingError(f"window_end mismatch for {window_id}")
        their_valid = theirs["feature_valid"] == "true"
        if their_valid != (ours["exclusion_reason"] == ""):
            raise LabelingError(f"validity mismatch for {window_id}")
        if theirs["exclusion_reason"] != ours["exclusion_reason"]:
            raise LabelingError(f"structural exclusion mismatch for {window_id}")
    return {
        "features_csv_sha256": _sha256_file(csv_path),
        "feature_manifest_sha256": _sha256_file(manifest_path),
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _check_output_targets(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    existing = [output_dir / name for name in _OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"output already contains label artifacts: {names}; "
            "pass overwrite=True or --overwrite to replace them"
        )


def _format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_labels_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABELS_CSV_COLUMNS)
        for row in rows:
            writer.writerow([_format_cell(row[column]) for column in LABELS_CSV_COLUMNS])


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _write_output_artifacts(
    output_dir: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Stage both artifacts completely before replacing any existing output."""
    _check_output_targets(output_dir, overwrite=overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ostosense-labeling-", dir=output_dir.parent
    ) as temporary:
        stage = Path(temporary)
        _write_labels_csv(stage / "labels.csv", rows)
        _write_manifest(stage / "label_manifest.json", manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            (stage / name).replace(output_dir / name)


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def label_dataset(
    input_dir: str | Path,
    protocol_manifest_path: str | Path,
    partition_manifest_path: str | Path,
    boundary_config_path: str | Path,
    output_dir: str | Path,
    *,
    features_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Derive ordinal ground-truth labels; return the label manifest dict."""
    input_dir = Path(input_dir)
    sessions_path = input_dir / "sessions.csv"
    samples_path = input_dir / "samples.csv"
    events_path = input_dir / "events.csv"

    # ---- validate & compute everything BEFORE mutating any output ----
    input_origin = _read_input_origin(input_dir)
    boundary = _load_boundary_config(boundary_config_path, input_origin)

    sessions = _read_sessions_full(sessions_path)
    samples = _read_samples_full(samples_path, sessions)
    events = _read_events(events_path, sessions)
    protocol = _load_manifest_rows(
        protocol_manifest_path, PROTOCOL_MANIFEST_FIELDS, "protocol_manifest.csv"
    )
    partition = _load_manifest_rows(
        partition_manifest_path, PARTITION_MANIFEST_FIELDS, "partition_manifest.csv"
    )
    _validate_cross_inputs(sessions, protocol, partition)

    structural_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    deviation_sessions = 0
    for session_id, session in sessions.items():
        three_tuples = [
            (s["timestamp"], s["capacitance_raw"], s["cap_quality"]) for s in samples[session_id]
        ]
        windows = features._windows_for_session(
            session, three_tuples, _WINDOW_MS, _STRIDE_MS, _EXPECTED_FULL, _JITTER
        )
        structural_rows.extend(windows)
        planned_horizon = _parse_planned_horizon(
            session_id, protocol[session_id]["planned_safe_horizon_s"]
        )
        rows, deviated, _ = _label_session(
            session,
            windows,
            samples[session_id],
            events[session_id],
            planned_horizon,
            partition[session_id]["dataset_partition"],
            boundary,
        )
        label_rows.extend(rows)
        if deviated:
            deviation_sessions += 1

    sessions_sha = _sha256_file(sessions_path)
    samples_sha = _sha256_file(samples_path)
    feature_hashes: dict[str, str] = {}
    if features_dir is not None:
        feature_hashes = _verify_features(
            Path(features_dir), sessions_sha, samples_sha, structural_rows
        )

    valid_count = sum(1 for row in label_rows if row["label_valid"])
    risk_counts = {name: 0 for name in CLASS_NAMES}
    exclusion_counts = {reason: 0 for reason in ALL_EXCLUSION_REASONS}
    for row in label_rows:
        if row["label_valid"]:
            risk_counts[row["risk_label"]] += 1
        else:
            exclusion_counts[row["exclusion_reason"]] += 1

    manifest = {
        "labeler_version": LABELER_VERSION,
        "rulebook_version": RULEBOOK_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "boundary_config_version": boundary["version"],
        "boundary_config_sha256": boundary["sha256"],
        "dataset_origin": input_origin,
        "input_sha256": {
            "sessions_csv": sessions_sha,
            "samples_csv": samples_sha,
            "events_csv": _sha256_file(events_path),
            "protocol_manifest_csv": _sha256_file(Path(protocol_manifest_path)),
            "partition_manifest_csv": _sha256_file(Path(partition_manifest_path)),
            "boundary_config_json": boundary["sha256"],
        },
        "feature_artifact_sha256": feature_hashes,
        "window_convention": {
            "interval": features._WINDOW_INTERVAL,
            "window_seconds": features._WORKING_WINDOW_SECONDS,
            "stride_seconds": features._WORKING_STRIDE_SECONDS,
            "sampling_rate_hz": features._WORKING_SAMPLING_RATE_HZ,
            "jitter_tolerance_ms": _JITTER,
            "expected_full_samples": _EXPECTED_FULL,
        },
        "session_count": len(sessions),
        "candidate_window_count": len(label_rows),
        "valid_window_count": valid_count,
        "excluded_window_count": len(label_rows) - valid_count,
        "risk_class_counts": risk_counts,
        "exclusion_reason_counts": exclusion_counts,
        "protocol_deviation_session_count": deviation_sessions,
        "warning": NO_PERFORMANCE_WARNING,
    }

    output_dir = Path(output_dir)
    _write_output_artifacts(
        output_dir, label_rows, manifest, overwrite=overwrite
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.labeling",
        description="Deterministic ENGINEERING_TEST_ONLY four-class ordinal labeler.",
    )
    parser.add_argument("--input", required=True, help="Directory with sessions/samples/events + manifest.json.")
    parser.add_argument("--protocol-manifest", required=True, help="protocol_manifest.csv path.")
    parser.add_argument("--partition-manifest", required=True, help="partition_manifest.csv path.")
    parser.add_argument("--boundary-config", required=True, help="ENGINEERING_TEST_ONLY boundary JSON under ai/tests/fixtures/.")
    parser.add_argument("--features", default=None, help="Optional features directory to cross-check window identities.")
    parser.add_argument("--output", required=True, help="Output directory for label artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing label artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = label_dataset(
        args.input,
        args.protocol_manifest,
        args.partition_manifest,
        args.boundary_config,
        args.output,
        features_dir=args.features,
        overwrite=args.overwrite,
    )
    print(
        f"labels: {manifest['candidate_window_count']} windows "
        f"({manifest['valid_window_count']} valid, {manifest['excluded_window_count']} excluded) "
        f"from {manifest['session_count']} sessions to {args.output}"
    )
    print(f"risk_class_counts: {manifest['risk_class_counts']}")
    print(NO_PERFORMANCE_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
