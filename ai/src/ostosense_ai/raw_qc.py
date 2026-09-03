"""Deterministic real-data intake and QC gate (real-data readiness, mechanics only).

``ostosense_ai.raw_qc`` inspects a directory of integrated logger outputs
(``sessions.csv`` + ``samples.csv`` + ``events.csv``, optional ``manifest.json``
and ``protocol_manifest.csv``) and decides, per session, whether it:

1. satisfies the AI Data Contract v1.1 record semantics (``contract_status``),
2. satisfies the currently evaluable Data Collection Protocol v0.1 checks
   (``protocol_status``),
3. must be repeated (``overall_status = FAIL``), or
4. is structurally usable but only partially evaluated because no protocol
   manifest was supplied (``overall_status = PARTIAL``).

It is a real-data *readiness* tool. It is standard-library only, deterministic,
and reuses the canonical field/enum definitions from ``ostosense_contract`` (it
never redefines schema constants). It performs no model training, AI evaluation,
notification evaluation, firmware deployment, or clinical validation, and emits
no performance metric or target pass/fail field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from ostosense_contract import (
    Arm,
    CapQuality,
    EndReason,
    EventType,
    LigQuality,
    SystemQuality,
    aggregate_system_quality,
)
from ostosense_contract.schema import EventRecord, SampleRecord, SessionRecord
from ostosense_ai import features
from ostosense_ai.labeling import PROTOCOL_MANIFEST_FIELDS

QC_TOOL_VERSION = "0.2.1"
CONFIG_ID = "raw-qc-v0.1"
CONFIG_STATUS = "PROPOSED_PILOT_SETTING"
CONTRACT_VERSION = "v1.1"
PROTOCOL_VERSION = "v0.1"
UNDECLARED_ORIGIN = "UNDECLARED"
_PROVENANCE_TAG = "DRAFT_PROTOCOL_V0.1"

MANDATORY_FILES = ("sessions.csv", "samples.csv", "events.csv")
_OUTPUT_FILES = ("qc_sessions.csv", "qc_issues.csv", "qc_report.json")

_NUMERIC_CONFIG_KEYS = (
    "expected_interval_ms",
    "jitter_tolerance_ms",
    "unmarked_gap_threshold_ms",
    "minimum_pre_injection_dry_s",
    "baseline_window_s",
)
_CONFIG_KEYS = frozenset(
    (
        "config_id",
        "status",
        "contract_version",
        "protocol_version",
        "provenance",
        "warning",
        *_NUMERIC_CONFIG_KEYS,
    )
)

# Status vocabulary.
PASS = "PASS"
FAIL = "FAIL"
PARTIAL = "PARTIAL"
NOT_EVALUATED = "NOT_EVALUATED"

# Severities.
ERROR = "ERROR"
WARNING = "WARNING"

REPORT_WARNING = (
    "Provisional real-data QC. This report proves deterministic intake and QC "
    "mechanics only. It is not AI accuracy, notification accuracy, sensor "
    "validation, firmware validation, or clinical evidence."
)

QC_SESSION_COLUMNS = (
    "session_id",
    "contract_status",
    "protocol_status",
    "overall_status",
    "sample_count",
    "event_count",
    "duration_s",
    "duplicate_timestamp_count",
    "out_of_order_interval_count",
    "out_of_tolerance_interval_count",
    "unmarked_gap_count",
    "cap_non_ok_count",
    "lig_non_ok_count",
    "first_both_ok_timestamp",
    "first_injection_start_timestamp",
    "pre_injection_dry_s",
    "baseline_sample_count",
    "baseline_recomputed_median",
    "baseline_recomputed_std",
    "baseline_value_abs_diff",
    "baseline_std_abs_diff",
    "error_count",
    "warning_count",
    "issue_codes",
)
QC_ISSUE_COLUMNS = ("session_id", "severity", "code", "timestamp", "detail")


class RawQcError(ValueError):
    """Fatal QC invocation/input failure (leaves any existing outputs untouched)."""


# --------------------------------------------------------------------------- #
# Deterministic formatting helpers (shared repository conventions)
# --------------------------------------------------------------------------- #
def _round6(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0.0 else rounded  # normalize -0.0 for byte-stable output


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(_round6(value))
    return str(value)


def _dump_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RawQcError(f"invalid QC config {path}: {error}") from error


def validate_qc_config(config: Any) -> dict[str, Any]:
    """Reject missing/unknown keys, bad types/versions, and inconsistent timing."""
    if not isinstance(config, dict):
        raise RawQcError("QC config must be a JSON object")
    missing = sorted(_CONFIG_KEYS - config.keys())
    if missing:
        raise RawQcError(f"QC config is missing required keys: {missing}")
    unknown = sorted(config.keys() - _CONFIG_KEYS)
    if unknown:
        raise RawQcError(f"QC config contains unsupported keys: {unknown}")

    if config["config_id"] != CONFIG_ID:
        raise RawQcError(f"config_id must be {CONFIG_ID!r}")
    if config["status"] != CONFIG_STATUS:
        raise RawQcError(f"status must be {CONFIG_STATUS!r}")
    if config["contract_version"] != CONTRACT_VERSION:
        raise RawQcError(f"contract_version must be {CONTRACT_VERSION!r}")
    if config["protocol_version"] != PROTOCOL_VERSION:
        raise RawQcError(f"protocol_version must be {PROTOCOL_VERSION!r}")

    values: dict[str, int] = {}
    for key in _NUMERIC_CONFIG_KEYS:
        raw = config[key]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise RawQcError(f"{key} must be an integer")
        if raw <= 0:
            raise RawQcError(f"{key} must be a positive integer")
        values[key] = raw

    if values["jitter_tolerance_ms"] >= values["expected_interval_ms"]:
        raise RawQcError("jitter_tolerance_ms must be smaller than expected_interval_ms")
    if values["unmarked_gap_threshold_ms"] <= (
        values["expected_interval_ms"] + values["jitter_tolerance_ms"]
    ):
        raise RawQcError(
            "unmarked_gap_threshold_ms must exceed expected_interval_ms + jitter_tolerance_ms"
        )
    if values["minimum_pre_injection_dry_s"] < values["baseline_window_s"]:
        raise RawQcError(
            "minimum_pre_injection_dry_s must be >= baseline_window_s"
        )

    provenance = config["provenance"]
    if not isinstance(provenance, dict):
        raise RawQcError("provenance must be a JSON object")
    expected_provenance = {key: _PROVENANCE_TAG for key in _NUMERIC_CONFIG_KEYS}
    if provenance != expected_provenance:
        raise RawQcError(
            f"provenance must map every numeric setting to {_PROVENANCE_TAG!r}"
        )

    if not isinstance(config["warning"], str) or not config["warning"].strip():
        raise RawQcError("warning must be non-empty text")
    return config


def _resolve_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        config = load_config(config)
    return validate_qc_config(config)


# --------------------------------------------------------------------------- #
# Strict input parsing (fatal on unparseable/enum/finite/header failures)
# --------------------------------------------------------------------------- #
def _load_rows(path: Path, expected: tuple[str, ...]) -> list[tuple[int, dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                raise RawQcError(f"{path.name} is empty")
            if header != list(expected):
                raise RawQcError(
                    f"{path.name} header does not match Data Contract v1.1: {header!r}"
                )
            rows: list[tuple[int, dict[str, str]]] = []
            for line_number, raw in enumerate(reader, start=2):
                if len(raw) != len(expected):
                    raise RawQcError(
                        f"{path.name} line {line_number} has {len(raw)} fields, "
                        f"expected {len(expected)}"
                    )
                rows.append((line_number, dict(zip(expected, raw))))
            return rows
    except OSError as error:
        raise RawQcError(f"cannot read {path.name}: {error}") from error


def _to_int(name: str, raw: str, *, line: int, file: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise RawQcError(f"{file} line {line}: {name} is not an integer: {raw!r}") from error


def _to_finite_float(name: str, raw: str, *, line: int, file: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise RawQcError(f"{file} line {line}: {name} is not a number: {raw!r}") from error
    if not math.isfinite(value):
        raise RawQcError(f"{file} line {line}: {name} must be finite: {raw!r}")
    return value


def _to_enum(name: str, raw: str, enum_type: type, *, line: int, file: str):
    try:
        return enum_type(raw)
    except ValueError as error:
        raise RawQcError(
            f"{file} line {line}: {name} has unknown value {raw!r}"
        ) from error


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _parse_sessions(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    sessions: dict[str, dict[str, Any]] = {}
    for line, row in _load_rows(path, SessionRecord.FIELDS):
        session_id = row["session_id"]
        # Enum + numeric validity are fatal; ID/semantics are contract findings.
        arm = _to_enum("arm", row["arm"], Arm, line=line, file="sessions.csv")
        end_reason = _to_enum(
            "end_reason", row["end_reason"], EndReason, line=line, file="sessions.csv"
        )
        baseline_value = _to_finite_float(
            "baseline_value", row["baseline_value"], line=line, file="sessions.csv"
        )
        baseline_std = _to_finite_float(
            "baseline_std", row["baseline_std"], line=line, file="sessions.csv"
        )
        start_ts = _to_int("start_timestamp", row["start_timestamp"], line=line, file="sessions.csv")
        end_ts = _to_int("end_timestamp", row["end_timestamp"], line=line, file="sessions.csv")
        if session_id in sessions:
            raise RawQcError(f"duplicate session_id in sessions.csv: {session_id!r}")
        order.append(session_id)
        sessions[session_id] = {
            "session_id": session_id,
            "arm": arm.value,
            "bag_id": row["bag_id"],
            "sensor_id": row["sensor_id"],
            "device_id": row["device_id"],
            "operator_id": row["operator_id"],
            "fluid_type": row["fluid_type"],
            "firmware_version": row["firmware_version"],
            "model_version": row["model_version"],
            "end_reason": end_reason.value,
            "baseline_value": baseline_value,
            "baseline_std": baseline_std,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "line": line,
            "samples": [],
            "events": [],
        }
    if not sessions:
        raise RawQcError("sessions.csv contains no sessions")
    return order, sessions


def _parse_samples(path: Path, sessions: dict[str, dict[str, Any]]) -> None:
    for line, row in _load_rows(path, SampleRecord.FIELDS):
        session_id = row["session_id"]
        session = sessions.get(session_id)
        if session is None:
            raise RawQcError(
                f"samples.csv line {line} references unknown session_id: {session_id!r}"
            )
        timestamp = _to_int("timestamp", row["timestamp"], line=line, file="samples.csv")
        capacitance = _to_finite_float(
            "capacitance_raw", row["capacitance_raw"], line=line, file="samples.csv"
        )
        _to_finite_float("lig_raw", row["lig_raw"], line=line, file="samples.csv")
        cap_quality = _to_enum("cap_quality", row["cap_quality"], CapQuality, line=line, file="samples.csv")
        lig_quality = _to_enum("lig_quality", row["lig_quality"], LigQuality, line=line, file="samples.csv")
        system_quality = _to_enum(
            "system_quality", row["system_quality"], SystemQuality, line=line, file="samples.csv"
        )
        session["samples"].append(
            {
                "timestamp": timestamp,
                "capacitance_raw": capacitance,
                "cap_quality": cap_quality,
                "lig_quality": lig_quality,
                "system_quality": system_quality,
                "activity_state": row["activity_state"],
                "orientation_position": row["orientation_position"],
                "line": line,
            }
        )


def _parse_events(path: Path, sessions: dict[str, dict[str, Any]]) -> None:
    for line, row in _load_rows(path, EventRecord.FIELDS):
        session_id = row["session_id"]
        session = sessions.get(session_id)
        if session is None:
            raise RawQcError(
                f"events.csv line {line} references unknown session_id: {session_id!r}"
            )
        timestamp = _to_int("timestamp", row["timestamp"], line=line, file="events.csv")
        event_type = _to_enum("event_type", row["event_type"], EventType, line=line, file="events.csv")
        raw_metadata = row["event_metadata"]
        try:
            metadata = json.loads(
                raw_metadata,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise RawQcError(
                f"events.csv line {line}: event_metadata is not valid finite JSON"
            ) from error
        if not isinstance(metadata, dict):
            raise RawQcError(
                f"events.csv line {line}: event_metadata must be a JSON object"
            )
        session["events"].append(
            {
                "event_id": row["event_id"],
                "timestamp": timestamp,
                "event_type": event_type.value,
                "line": line,
            }
        )


def _read_origin(input_dir: Path) -> str:
    try:
        origin = features._read_input_origin(input_dir)
    except ValueError as error:
        raise RawQcError(f"invalid input manifest.json: {error}") from error
    return UNDECLARED_ORIGIN if origin == features.UNDECLARED_INPUT_ORIGIN else origin


# --------------------------------------------------------------------------- #
# Protocol manifest (optional)
# --------------------------------------------------------------------------- #
def _load_protocol_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise RawQcError(f"protocol manifest not found: {path}")
    rows = _load_rows(path, PROTOCOL_MANIFEST_FIELDS)
    manifest: dict[str, dict[str, str]] = {}
    for line, row in rows:
        session_id = row["session_id"]
        if session_id in manifest:
            raise RawQcError(
                f"protocol_manifest line {line}: duplicate session_id {session_id!r}"
            )
        manifest[session_id] = row
    return manifest


# --------------------------------------------------------------------------- #
# Per-session QC
# --------------------------------------------------------------------------- #
class _IssueSink:
    """Collects issues for a session in stable detection order."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.issues: list[dict[str, Any]] = []
        self.contract_error = False
        self.protocol_error = False

    def add(self, severity: str, code: str, *, timestamp: int | None, detail: str, domain: str) -> None:
        self.issues.append(
            {
                "session_id": self.session_id,
                "severity": severity,
                "code": code,
                "timestamp": "" if timestamp is None else timestamp,
                "detail": detail,
            }
        )
        if severity == ERROR:
            if domain == "contract":
                self.contract_error = True
            elif domain == "protocol":
                self.protocol_error = True


_REQUIRED_ID_FIELDS = (
    "session_id",
    "bag_id",
    "sensor_id",
    "device_id",
    "fluid_type",
    "operator_id",
    "firmware_version",
)
_PLANNED_LEAK_ARMS = (Arm.LEAK_GRADUAL.value, Arm.LEAK_SUDDEN.value)

# A supplied protocol_manifest row must carry an explicit, versioned protocol tag
# (for example ``v0.1-shakedown-a``); a bare ``v0.1`` is rejected on purpose.
_PROTOCOL_VERSION_RE = re.compile(r"^v0\.1-[A-Za-z0-9][A-Za-z0-9._-]*$")
_INJECTION_PROFILES = ("stepwise", "continuous")
_INJECTION_METHODS = ("manual_syringe", "pump")


class _GradualPlanMode(str, Enum):
    UNKNOWN = "UNKNOWN"
    PLANNED_LEAK = "PLANNED_LEAK"
    NONLEAKING_FILL = "NONLEAKING_FILL"


def _invalid_csv_id(value: str, *, optional: bool = False) -> bool:
    if optional and value == "":
        return False
    return value == "" or any(character in value for character in (",", "\n", "\r"))


def _parse_positive_int(raw: str) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _parse_positive_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _contract_checks(session: dict[str, Any], sink: _IssueSink, seen_event_ids: dict[str, str]) -> None:
    # Session-level record semantics.
    invalid_ids = [name for name in _REQUIRED_ID_FIELDS if _invalid_csv_id(session[name])]
    if _invalid_csv_id(session["model_version"], optional=True):
        invalid_ids.append("model_version")
    if invalid_ids:
        sink.add(ERROR, "INVALID_RECORD", timestamp=None,
                 detail=f"invalid CSV-safe id field(s): {','.join(invalid_ids)}", domain="contract")
    if session["start_timestamp"] < 0 or session["end_timestamp"] < 0:
        sink.add(ERROR, "INVALID_RECORD", timestamp=None,
                 detail="negative session timestamp", domain="contract")
    if session["end_timestamp"] < session["start_timestamp"]:
        sink.add(ERROR, "INVALID_RECORD", timestamp=None,
                 detail="end_timestamp precedes start_timestamp", domain="contract")
    if session["baseline_std"] < 0:
        sink.add(ERROR, "INVALID_RECORD", timestamp=None,
                 detail="negative baseline_std", domain="contract")

    start, end = session["start_timestamp"], session["end_timestamp"]

    # Samples outside bounds (summarized per session) and system_quality mismatch.
    outside = [s for s in session["samples"] if not (start <= s["timestamp"] <= end)]
    if outside:
        sink.add(ERROR, "SAMPLE_OUTSIDE_SESSION", timestamp=outside[0]["timestamp"],
                 detail=f"{len(outside)} sample(s) outside [start,end]", domain="contract")
    mismatched = [
        s for s in session["samples"]
        if s["system_quality"] is not aggregate_system_quality(s["cap_quality"], s["lig_quality"])
    ]
    if mismatched:
        first = mismatched[0]
        expected = aggregate_system_quality(first["cap_quality"], first["lig_quality"]).value
        sink.add(ERROR, "SYSTEM_QUALITY_MISMATCH", timestamp=first["timestamp"],
                 detail=f"{len(mismatched)} sample(s); first expected {expected}", domain="contract")
    invalid_sample_metadata = [
        sample
        for sample in session["samples"]
        if _invalid_csv_id(sample["activity_state"], optional=True)
        or _invalid_csv_id(sample["orientation_position"], optional=True)
    ]
    if invalid_sample_metadata:
        sink.add(
            ERROR,
            "INVALID_RECORD",
            timestamp=invalid_sample_metadata[0]["timestamp"],
            detail=f"{len(invalid_sample_metadata)} sample(s) contain invalid audit metadata",
            domain="contract",
        )

    # Events outside bounds and duplicate event ids.
    for event in session["events"]:
        if _invalid_csv_id(event["event_id"]):
            sink.add(ERROR, "INVALID_RECORD", timestamp=event["timestamp"],
                     detail="invalid event_id", domain="contract")
        if not (start <= event["timestamp"] <= end):
            sink.add(ERROR, "EVENT_OUTSIDE_SESSION", timestamp=event["timestamp"],
                     detail=f"event {event['event_type']} outside [start,end]", domain="contract")
        event_id = event["event_id"]
        if not _invalid_csv_id(event_id) and event_id in seen_event_ids:
            sink.add(ERROR, "DUPLICATE_EVENT_ID", timestamp=event["timestamp"],
                     detail=f"event_id {event_id!r} first seen in {seen_event_ids[event_id]}",
                     domain="contract")
        elif not _invalid_csv_id(event_id):
            seen_event_ids[event_id] = session["session_id"]


def _timing_metrics(session: dict[str, Any], config: dict[str, Any], sink: _IssueSink) -> dict[str, Any]:
    samples = session["samples"]
    expected = config["expected_interval_ms"]
    jitter = config["jitter_tolerance_ms"]
    low, high = expected - jitter, expected + jitter
    gap_threshold = config["unmarked_gap_threshold_ms"]

    metrics = {
        "duplicate_timestamp_count": 0,
        "out_of_order_interval_count": 0,
        "out_of_tolerance_interval_count": 0,
        "unmarked_gap_count": 0,
    }
    if not samples:
        sink.add(ERROR, "NO_SAMPLES", timestamp=None, detail="session has no samples", domain="protocol")
        return metrics

    seen: set[int] = set()
    first_duplicate_ts: int | None = None
    for sample in samples:
        ts = sample["timestamp"]
        if ts in seen:
            metrics["duplicate_timestamp_count"] += 1
            if first_duplicate_ts is None:
                first_duplicate_ts = ts
        seen.add(ts)
    if metrics["duplicate_timestamp_count"]:
        sink.add(ERROR, "DUPLICATE_SAMPLE_TIMESTAMP", timestamp=first_duplicate_ts,
                 detail=f"{metrics['duplicate_timestamp_count']} duplicate timestamp(s)", domain="protocol")

    first_out_of_order: int | None = None
    first_out_of_tolerance: int | None = None
    for earlier, later in zip(samples, samples[1:]):
        delta = later["timestamp"] - earlier["timestamp"]
        if delta < 0:
            metrics["out_of_order_interval_count"] += 1
            if first_out_of_order is None:
                first_out_of_order = later["timestamp"]
            continue
        if delta == 0:
            continue  # counted as a duplicate timestamp already
        if delta < low or delta > high:
            metrics["out_of_tolerance_interval_count"] += 1
            if first_out_of_tolerance is None:
                first_out_of_tolerance = later["timestamp"]
        if delta >= gap_threshold:
            marked = (
                later["cap_quality"] is CapQuality.DATA_GAP
                or later["lig_quality"] is LigQuality.DATA_GAP
            )
            if not marked:
                metrics["unmarked_gap_count"] += 1
                sink.add(ERROR, "UNMARKED_DATA_GAP", timestamp=later["timestamp"],
                         detail=f"{delta} ms gap without DATA_GAP on the following sample",
                         domain="protocol")
    if metrics["out_of_order_interval_count"]:
        sink.add(ERROR, "TIMESTAMP_OUT_OF_ORDER", timestamp=first_out_of_order,
                 detail=f"{metrics['out_of_order_interval_count']} descending interval(s)", domain="protocol")
    if metrics["out_of_tolerance_interval_count"]:
        sink.add(ERROR, "INTERVAL_OUT_OF_TOLERANCE", timestamp=first_out_of_tolerance,
                 detail=f"{metrics['out_of_tolerance_interval_count']} interval(s) outside "
                        f"[{low},{high}] ms", domain="protocol")
    return metrics


def _baseline_metrics(session: dict[str, Any], config: dict[str, Any], sink: _IssueSink) -> dict[str, Any]:
    start = session["start_timestamp"]
    window_end = start + config["baseline_window_s"] * 1000
    window = [s for s in session["samples"] if start <= s["timestamp"] < window_end]
    result: dict[str, Any] = {
        "baseline_sample_count": len(window),
        "baseline_recomputed_median": None,
        "baseline_recomputed_std": None,
        "baseline_value_abs_diff": None,
        "baseline_std_abs_diff": None,
    }
    if not window:
        sink.add(ERROR, "BASELINE_WINDOW_INCOMPLETE", timestamp=None,
                 detail=f"no samples in [start, start+{config['baseline_window_s']}s)", domain="protocol")
        return result
    ordered = sorted(window, key=lambda sample: sample["timestamp"])
    low = config["expected_interval_ms"] - config["jitter_tolerance_ms"]
    high = config["expected_interval_ms"] + config["jitter_tolerance_ms"]
    coverage_reasons: list[str] = []
    if len(ordered) < 2:
        coverage_reasons.append("fewer than two samples")
    if ordered[0]["timestamp"] - start > high:
        coverage_reasons.append("baseline starts too late")
    if window_end - ordered[-1]["timestamp"] > high:
        coverage_reasons.append("baseline ends too early")
    if any(
        later["timestamp"] - earlier["timestamp"] < low
        or later["timestamp"] - earlier["timestamp"] > high
        for earlier, later in zip(ordered, ordered[1:])
    ):
        coverage_reasons.append("baseline contains an invalid sampling interval")
    if coverage_reasons:
        sink.add(
            ERROR,
            "BASELINE_WINDOW_INCOMPLETE",
            timestamp=ordered[0]["timestamp"],
            detail="; ".join(coverage_reasons),
            domain="protocol",
        )
    values = [s["capacitance_raw"] for s in window]
    median = _round6(statistics.median(values))
    std = _round6(statistics.pstdev(values))
    result["baseline_recomputed_median"] = median
    result["baseline_recomputed_std"] = std
    result["baseline_value_abs_diff"] = _round6(abs(median - session["baseline_value"]))
    result["baseline_std_abs_diff"] = _round6(abs(std - session["baseline_std"]))
    return result


def _validate_injection_pairs(injection_events: list[dict[str, Any]], sink: _IssueSink) -> int:
    """Validate INJECTION_START/END events in timestamp order; return valid-pair count.

    Emits a single MALFORMED_REQUIRED_EVENTS error listing every structural fault
    (END without open START, overlapping START, unclosed START).
    """
    open_start: int | None = None
    pairs = 0
    reasons: list[str] = []
    for event in injection_events:
        if event["event_type"] == EventType.INJECTION_START.value:
            if open_start is not None:
                reasons.append("overlapping INJECTION_START")
            open_start = event["timestamp"]
        else:  # INJECTION_END
            if open_start is None:
                reasons.append("INJECTION_END without an open INJECTION_START")
            else:
                pairs += 1
                open_start = None
    if open_start is not None:
        reasons.append("unclosed INJECTION_START")
    if reasons:
        first_ts = injection_events[0]["timestamp"] if injection_events else None
        sink.add(ERROR, "MALFORMED_REQUIRED_EVENTS", timestamp=first_ts,
                 detail="; ".join(reasons), domain="protocol")
    return pairs


def _event_metrics(
    session: dict[str, Any],
    config: dict[str, Any],
    sink: _IssueSink,
    gradual_plan_mode: _GradualPlanMode,
) -> dict[str, Any]:
    arm = session["arm"]
    events = session["events"]
    types = {event["event_type"] for event in events}
    injection_events = sorted(
        [e for e in events if e["event_type"] in (
            EventType.INJECTION_START.value, EventType.INJECTION_END.value)],
        key=lambda e: e["timestamp"],
    )
    physical_leaks = sorted(
        [e for e in events if e["event_type"] == EventType.PHYSICAL_LEAK_OBSERVED.value],
        key=lambda e: e["timestamp"],
    )
    leak_count = len(physical_leaks)
    device_restarts = [e for e in events if e["event_type"] == EventType.DEVICE_RESTART.value]
    injection_starts = [e["timestamp"] for e in events if e["event_type"] == EventType.INJECTION_START.value]
    first_injection = min(injection_starts) if injection_starts else None

    first_both_ok = None
    for sample in session["samples"]:
        if sample["cap_quality"] is CapQuality.OK and sample["lig_quality"] is LigQuality.OK:
            first_both_ok = sample["timestamp"]
            break

    result = {
        "first_both_ok_timestamp": first_both_ok,
        "first_injection_start_timestamp": first_injection,
        "pre_injection_dry_s": None,
    }

    leak_confirmed_without_event = (
        session["end_reason"] == EndReason.LEAK_CONFIRMED.value and leak_count == 0
    )
    if leak_confirmed_without_event:
        sink.add(
            ERROR,
            "MALFORMED_REQUIRED_EVENTS",
            timestamp=None,
            detail="end_reason LEAK_CONFIRMED without PHYSICAL_LEAK_OBSERVED",
            domain="protocol",
        )

    # A logger must open a fresh session after any restart.
    if device_restarts:
        sink.add(ERROR, "DEVICE_RESTART_DURING_SESSION", timestamp=device_restarts[0]["timestamp"],
                 detail="DEVICE_RESTART inside a session; the logger must start a new session after restart",
                 domain="protocol")

    if arm == Arm.FIELD.value:
        sink.add(ERROR, "FIELD_ARM_NOT_SUPPORTED", timestamp=None,
                 detail="FIELD arm is outside Data Collection Protocol v0.1 bench scope",
                 domain="protocol")
    elif arm == Arm.SAFE.value:
        if injection_events:
            sink.add(ERROR, "MALFORMED_REQUIRED_EVENTS", timestamp=injection_events[0]["timestamp"],
                     detail="injection events present in a SAFE dry session", domain="protocol")
        if physical_leaks:
            sink.add(ERROR, "UNEXPECTED_PHYSICAL_LEAK_SAFE", timestamp=physical_leaks[0]["timestamp"],
                     detail="SAFE arm contains PHYSICAL_LEAK_OBSERVED", domain="protocol")
    elif arm == Arm.LEAK_GRADUAL.value:
        _validate_injection_pairs(injection_events, sink)
        for event_name in (EventType.INJECTION_START.value, EventType.INJECTION_END.value):
            if event_name not in types:
                sink.add(ERROR, "MISSING_REQUIRED_EVENT", timestamp=None,
                         detail=f"LEAK_GRADUAL arm missing {event_name}", domain="protocol")
        if leak_count > 1:
            sink.add(ERROR, "MALFORMED_REQUIRED_EVENTS", timestamp=physical_leaks[1]["timestamp"],
                     detail="more than one PHYSICAL_LEAK_OBSERVED", domain="protocol")
        elif leak_count == 1 and first_injection is not None:
            if physical_leaks[0]["timestamp"] < first_injection:
                sink.add(ERROR, "MALFORMED_REQUIRED_EVENTS",
                         timestamp=physical_leaks[0]["timestamp"],
                         detail="physical leak precedes INJECTION_START", domain="protocol")

        if gradual_plan_mode is _GradualPlanMode.NONLEAKING_FILL:
            # A valid predeclared non-leaking fill may unexpectedly leak; retain it
            # as a protocol deviation instead of silently changing its plan.
            if leak_count == 1:
                sink.add(WARNING, "UNPLANNED_PHYSICAL_LEAK", timestamp=physical_leaks[0]["timestamp"],
                         detail="planned non-leaking fill recorded a physical leak; data retained",
                         domain="protocol")
        elif gradual_plan_mode is _GradualPlanMode.PLANNED_LEAK:
            if leak_count == 0 and not leak_confirmed_without_event:
                sink.add(ERROR, "MISSING_REQUIRED_EVENT", timestamp=None,
                         detail="LEAK_GRADUAL arm missing PHYSICAL_LEAK_OBSERVED", domain="protocol")
    elif arm == Arm.LEAK_SUDDEN.value:
        if injection_events:
            _validate_injection_pairs(injection_events, sink)
        if leak_count == 0 and not leak_confirmed_without_event:
            sink.add(ERROR, "MISSING_REQUIRED_EVENT", timestamp=None,
                     detail="LEAK_SUDDEN arm missing PHYSICAL_LEAK_OBSERVED", domain="protocol")
        elif leak_count > 1:
            sink.add(ERROR, "MALFORMED_REQUIRED_EVENTS", timestamp=physical_leaks[1]["timestamp"],
                     detail="more than one PHYSICAL_LEAK_OBSERVED", domain="protocol")

    if arm in _PLANNED_LEAK_ARMS and leak_count == 1:
        missing_flags = [
            name for name in (EventType.LEAK_FLAG_FIRST.value, EventType.LEAK_FLAG_CONFIRMED.value)
            if name not in types
        ]
        if missing_flags:
            sink.add(WARNING, "MISSING_LIG_FLAG_EVENT", timestamp=None,
                     detail=f"missing {','.join(missing_flags)} (LIG pending hardware calibration)",
                     domain="protocol")

    # Pre-injection dry phase (only sessions that actually inject).
    if first_injection is not None:
        if first_both_ok is None:
            sink.add(ERROR, "PREINJECTION_DRY_TOO_SHORT", timestamp=first_injection,
                     detail="no both-channel-OK sample precedes INJECTION_START", domain="protocol")
        else:
            dry_ms = first_injection - first_both_ok
            result["pre_injection_dry_s"] = _round6(dry_ms / 1000.0)
            if dry_ms < config["minimum_pre_injection_dry_s"] * 1000:
                sink.add(ERROR, "PREINJECTION_DRY_TOO_SHORT", timestamp=first_injection,
                         detail=f"dry phase {dry_ms} ms < "
                                f"{config['minimum_pre_injection_dry_s']*1000} ms", domain="protocol")
    return result


def _validate_manifest_row(
    session: dict[str, Any], manifest_row: dict[str, str] | None, sink: _IssueSink
) -> _GradualPlanMode:
    """Validate one supplied manifest row and return trusted scenario context.

    Structural manifest failures (header/duplicate/unknown session) are already
    fatal upstream. Here per-row semantic failures become the per-session ERROR
    ``INVALID_PROTOCOL_MANIFEST``, while identity/arm disagreements keep their own
    ``MANIFEST_ID_MISMATCH`` / ``MANIFEST_ARM_MISMATCH`` codes. Invalid or
    mismatched rows never determine whether a gradual session was a planned leak
    or a non-leaking fill.
    """
    if manifest_row is None:
        sink.add(ERROR, "MANIFEST_ID_MISMATCH", timestamp=None,
                 detail="session has no protocol_manifest row", domain="protocol")
        return _GradualPlanMode.UNKNOWN

    arm = session["arm"]
    problems: list[str] = []

    if not _PROTOCOL_VERSION_RE.match(manifest_row["protocol_version"]):
        problems.append("protocol_version")

    try:
        planned_arm = Arm(manifest_row["planned_arm"]).value
    except ValueError:
        planned_arm = None
        problems.append("planned_arm")
    arm_mismatch = planned_arm is not None and planned_arm != arm
    if arm_mismatch:
        sink.add(ERROR, "MANIFEST_ARM_MISMATCH", timestamp=None,
                 detail=f"planned_arm {manifest_row['planned_arm']!r} != session arm {arm!r}",
                 domain="protocol")

    id_mismatch: list[str] = []
    for field in ("operator_id", "bag_id", "sensor_id", "device_id"):
        value = manifest_row[field]
        if _invalid_csv_id(value):
            problems.append(field)
        elif value != session[field]:
            id_mismatch.append(field)
    if id_mismatch:
        sink.add(ERROR, "MANIFEST_ID_MISMATCH", timestamp=None,
                 detail=f"manifest mismatch on: {','.join(id_mismatch)}", domain="protocol")

    if _invalid_csv_id(manifest_row["target_fill_or_volume"]):
        problems.append("target_fill_or_volume")

    horizon_raw = manifest_row["planned_safe_horizon_s"]
    profile = manifest_row["injection_profile"]
    method = manifest_row["injection_method"]
    flow = manifest_row["planned_flow_ml_min"]
    observation = manifest_row["physical_leak_observation_method"]

    gradual_mode = _GradualPlanMode.UNKNOWN
    if arm == Arm.SAFE.value:
        parsed = _parse_positive_int(horizon_raw)
        if parsed is None:
            problems.append("planned_safe_horizon_s")
        if profile != "" or method != "" or flow != "":
            problems.append("injection_fields_must_be_empty")
        if observation != "" and _invalid_csv_id(observation):
            problems.append("physical_leak_observation_method")
    elif arm == Arm.LEAK_GRADUAL.value:
        if horizon_raw == "":
            gradual_mode = _GradualPlanMode.PLANNED_LEAK
        else:
            parsed = _parse_positive_int(horizon_raw)
            if parsed is None:
                problems.append("planned_safe_horizon_s")
            else:
                gradual_mode = _GradualPlanMode.NONLEAKING_FILL
        if profile not in _INJECTION_PROFILES:
            problems.append("injection_profile")
        if method not in _INJECTION_METHODS:
            problems.append("injection_method")
        if flow != "" and _parse_positive_float(flow) is None:
            problems.append("planned_flow_ml_min")
        if _invalid_csv_id(observation):
            problems.append("physical_leak_observation_method")
    elif arm == Arm.LEAK_SUDDEN.value:
        if horizon_raw != "":
            problems.append("planned_safe_horizon_s")
        if _invalid_csv_id(observation):
            problems.append("physical_leak_observation_method")
        if profile not in ("", *_INJECTION_PROFILES):
            problems.append("injection_profile")
        if method not in ("", *_INJECTION_METHODS):
            problems.append("injection_method")
        if flow != "" and _parse_positive_float(flow) is None:
            problems.append("planned_flow_ml_min")
    else:  # FIELD: outside bench scope; the horizon must stay empty.
        if horizon_raw != "":
            problems.append("planned_safe_horizon_s")

    if problems:
        sink.add(ERROR, "INVALID_PROTOCOL_MANIFEST", timestamp=None,
                 detail=f"invalid manifest field(s): {','.join(sorted(set(problems)))}",
                 domain="protocol")
    valid = not problems and not arm_mismatch and not id_mismatch
    return gradual_mode if valid else _GradualPlanMode.UNKNOWN


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _evaluate(
    order: list[str],
    sessions: dict[str, dict[str, Any]],
    config: dict[str, Any],
    manifest: dict[str, dict[str, str]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    seen_event_ids: dict[str, str] = {}

    for session_id in order:
        session = sessions[session_id]
        sink = _IssueSink(session_id)

        _contract_checks(session, sink, seen_event_ids)
        timing = _timing_metrics(session, config, sink)
        baseline = _baseline_metrics(session, config, sink)
        # Only a valid supplied manifest may determine the gradual sub-scenario.
        gradual_plan_mode = _GradualPlanMode.UNKNOWN
        if manifest is not None:
            gradual_plan_mode = _validate_manifest_row(session, manifest.get(session_id), sink)
        events = _event_metrics(session, config, sink, gradual_plan_mode)

        contract_status = FAIL if sink.contract_error else PASS
        if sink.protocol_error:
            protocol_status = FAIL
        elif manifest is None:
            protocol_status = NOT_EVALUATED
        else:
            protocol_status = PASS

        if contract_status == FAIL or protocol_status == FAIL:
            overall_status = FAIL
        elif protocol_status == NOT_EVALUATED:
            overall_status = PARTIAL
        else:
            overall_status = PASS

        error_count = sum(1 for issue in sink.issues if issue["severity"] == ERROR)
        warning_count = sum(1 for issue in sink.issues if issue["severity"] == WARNING)
        codes = sorted({issue["code"] for issue in sink.issues})

        samples = session["samples"]
        cap_non_ok = sum(1 for s in samples if s["cap_quality"] is not CapQuality.OK)
        lig_non_ok = sum(1 for s in samples if s["lig_quality"] is not LigQuality.OK)

        session_rows.append(
            {
                "session_id": session_id,
                "contract_status": contract_status,
                "protocol_status": protocol_status,
                "overall_status": overall_status,
                "sample_count": len(samples),
                "event_count": len(session["events"]),
                "duration_s": _round6((session["end_timestamp"] - session["start_timestamp"]) / 1000.0),
                "duplicate_timestamp_count": timing["duplicate_timestamp_count"],
                "out_of_order_interval_count": timing["out_of_order_interval_count"],
                "out_of_tolerance_interval_count": timing["out_of_tolerance_interval_count"],
                "unmarked_gap_count": timing["unmarked_gap_count"],
                "cap_non_ok_count": cap_non_ok,
                "lig_non_ok_count": lig_non_ok,
                "first_both_ok_timestamp": events["first_both_ok_timestamp"],
                "first_injection_start_timestamp": events["first_injection_start_timestamp"],
                "pre_injection_dry_s": events["pre_injection_dry_s"],
                "baseline_sample_count": baseline["baseline_sample_count"],
                "baseline_recomputed_median": baseline["baseline_recomputed_median"],
                "baseline_recomputed_std": baseline["baseline_recomputed_std"],
                "baseline_value_abs_diff": baseline["baseline_value_abs_diff"],
                "baseline_std_abs_diff": baseline["baseline_std_abs_diff"],
                "error_count": error_count,
                "warning_count": warning_count,
                "issue_codes": "|".join(codes),
            }
        )
        issue_rows.extend(sink.issues)

    # Dataset-level warning (empty session_id), appended after per-session issues.
    if manifest is None:
        issue_rows.append(
            {
                "session_id": "",
                "severity": WARNING,
                "code": "PROTOCOL_MANIFEST_NOT_PROVIDED",
                "timestamp": "",
                "detail": "no protocol_manifest.csv supplied; protocol checks partially evaluated",
            }
        )
    return session_rows, issue_rows


def _issue_counts(issue_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_severity = {ERROR: 0, WARNING: 0}
    by_code: dict[str, int] = {}
    for issue in issue_rows:
        by_severity[issue["severity"]] = by_severity.get(issue["severity"], 0) + 1
        by_code[issue["code"]] = by_code.get(issue["code"], 0) + 1
    return {"by_severity": by_severity, "by_code": dict(sorted(by_code.items()))}


# --------------------------------------------------------------------------- #
# Output (staged, refuse-by-default)
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(row.get(column, "")) for column in columns])


def _check_output_targets(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise RawQcError(f"output path is not a directory: {output_dir}")
    existing = [name for name in _OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output already contains QC artifacts: {', '.join(existing)}; "
            "pass overwrite=True or --overwrite to replace them"
        )


def _write_outputs(
    output_dir: Path,
    session_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    report_base: dict[str, Any],
    input_hashes: dict[str, str],
    config: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_targets(output_dir, overwrite=overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ostosense-qc-", dir=output_dir.parent) as tmp:
        stage = Path(tmp)
        _write_csv(stage / "qc_sessions.csv", QC_SESSION_COLUMNS, session_rows)
        _write_csv(stage / "qc_issues.csv", QC_ISSUE_COLUMNS, issue_rows)
        config_bytes = json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        report = {
            **report_base,
            "input_sha256": {
                **input_hashes,
                "raw_qc_config": _sha256_bytes(config_bytes),
            },
            "output_sha256": {
                "qc_sessions_csv": _sha256_file(stage / "qc_sessions.csv"),
                "qc_issues_csv": _sha256_file(stage / "qc_issues.csv"),
            },
        }
        (stage / "qc_report.json").write_text(_dump_json(report), encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            (stage / name).replace(output_dir / name)
    return report


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def run_raw_qc(
    input_dir: str | Path,
    config: str | Path | dict[str, Any],
    output_dir: str | Path,
    *,
    protocol_manifest: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the deterministic real-data QC gate; return the QC report dict."""
    resolved_config = _resolve_config(config)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    input_paths = {name: input_dir / name for name in MANDATORY_FILES}
    for name, path in input_paths.items():
        if not path.exists():
            raise RawQcError(f"missing required input file: {name}")

    dataset_origin = _read_origin(input_dir)
    order, sessions = _parse_sessions(input_paths["sessions.csv"])
    _parse_samples(input_paths["samples.csv"], sessions)
    _parse_events(input_paths["events.csv"], sessions)

    manifest: dict[str, dict[str, str]] | None = None
    manifest_path = Path(protocol_manifest) if protocol_manifest is not None else None
    if manifest_path is not None:
        manifest = _load_protocol_manifest(manifest_path)
        extra_manifest_sessions = [session_id for session_id in manifest if session_id not in sessions]
        if extra_manifest_sessions:
            raise RawQcError(
                "protocol_manifest.csv references unknown session_id(s): "
                + ", ".join(repr(session_id) for session_id in extra_manifest_sessions)
            )

    session_rows, issue_rows = _evaluate(order, sessions, resolved_config, manifest)

    status_counts = {PASS: 0, FAIL: 0, PARTIAL: 0}
    for row in session_rows:
        status_counts[row["overall_status"]] += 1

    total_samples = sum(len(sessions[sid]["samples"]) for sid in order)
    total_events = sum(len(sessions[sid]["events"]) for sid in order)

    input_hashes = {
        "sessions_csv": _sha256_file(input_paths["sessions.csv"]),
        "samples_csv": _sha256_file(input_paths["samples.csv"]),
        "events_csv": _sha256_file(input_paths["events.csv"]),
    }
    if manifest_path is not None:
        input_hashes["protocol_manifest_csv"] = _sha256_file(manifest_path)

    report_base = {
        "qc_tool_version": QC_TOOL_VERSION,
        "config_id": resolved_config["config_id"],
        "config_status": resolved_config["status"],
        "contract_version": resolved_config["contract_version"],
        "protocol_version": resolved_config["protocol_version"],
        "dataset_origin": dataset_origin,
        "protocol_manifest_provided": manifest is not None,
        "session_count": len(order),
        "sample_count": total_samples,
        "event_count": total_events,
        "status_counts": status_counts,
        "issue_counts": _issue_counts(issue_rows),
        "provisional_settings_warning": resolved_config["warning"],
        "warning": REPORT_WARNING,
    }

    return _write_outputs(
        output_dir, session_rows, issue_rows, report_base, input_hashes, resolved_config,
        overwrite=overwrite,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.raw_qc",
        description="Deterministic real-data intake and QC gate (pipeline/readiness mechanics only).",
    )
    parser.add_argument("--input", required=True, help="Logger output directory.")
    parser.add_argument("--config", required=True, help="raw-qc config JSON path.")
    parser.add_argument("--output", required=True, help="Output directory for QC artifacts.")
    parser.add_argument("--protocol-manifest", default=None, help="Optional protocol_manifest.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing QC artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = run_raw_qc(
            args.input, args.config, args.output,
            protocol_manifest=args.protocol_manifest, overwrite=args.overwrite,
        )
    except FileExistsError as error:
        print(f"raw_qc: {error}")
        return 1
    except (RawQcError, OSError) as error:
        print(f"raw_qc: {error}")
        return 1

    counts = report["status_counts"]
    print(
        f"raw_qc: {report['session_count']} session(s) -> "
        f"PASS={counts[PASS]} FAIL={counts[FAIL]} PARTIAL={counts[PARTIAL]} "
        f"(dataset_origin={report['dataset_origin']})"
    )
    print(REPORT_WARNING)
    return 0 if counts[FAIL] == 0 and counts[PARTIAL] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
