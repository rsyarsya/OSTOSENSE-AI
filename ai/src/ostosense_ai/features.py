"""Deterministic rolling-window capacitive feature extractor (pipeline test only).

This module converts contract-valid ``sessions.csv`` + ``samples.csv`` into an
auditable ``features.csv`` plus a deterministic ``feature_manifest.json``. It is
dependency-free (standard library + ``ostosense_contract`` only) so it runs in
the dependency-free environment too.

Scope guardrails (Batch 3A):

- The only AI signal here is the capacitive channel. LIG, events, arm, scenario
  metadata, ground-truth timing, and any future information must never become a
  model feature. The five baseline-normalized capacitive features live in the
  fixed ``FEATURE_COLUMNS`` tuple; identifiers/timestamps in ``features.csv`` are
  audit metadata kept structurally separate from ``FEATURE_COLUMNS``.
- No Safe/Monitor/Caution/Urgent labels, no model, and no B1/B2/B3 boundaries
  are produced. A passing run proves feature-pipeline mechanics only, never AI
  accuracy, sensor validity, early-warning performance, or clinical value.

Working window convention (Data Collection Protocol v0.1, §5): interval
``(t - W, t]`` with ``W`` = 120 s, stride ``S`` = 10 s, nominal 1 Hz,
``t_ref`` = first sample timestamp of the session, candidate endpoints
``t_k = t_ref + (W + k*S) * 1000 ms`` while ``t_k <= session.end_timestamp``.
The structure-locked Label Rulebook v0.3 uses the same ``(t - W, t]``
convention, resolving the earlier off-by-one discrepancy.

CLI::

    PYTHONPATH=ai/src ai/.venv/bin/python -m ostosense_ai.features \\
        --input /tmp/ostosense-synthetic-raw \\
        --config ai/configs/features-v0.1.json \\
        --output /tmp/ostosense-features
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ostosense_contract import (
    Arm,
    CapQuality,
    EndReason,
    LigQuality,
    SampleRecord,
    SessionRecord,
    SystemQuality,
)

EXTRACTOR_VERSION = "0.1.1"
DATA_CONTRACT_VERSION = "v1.1"
UNDECLARED_INPUT_ORIGIN = "UNDECLARED_INPUT_ORIGIN"
NO_LABEL_WARNING = (
    "Feature-pipeline mechanics only: this file contains no labels, no model, "
    "and no OSTOSENSE performance, sensor, or clinical claim."
)

FEATURE_COLUMNS = (
    "cap_delta_mean",
    "cap_delta_last",
    "cap_delta_slope_per_s",
    "cap_delta_variance",
    "cap_delta_range",
)
AUDIT_COLUMNS = (
    "window_id",
    "session_id",
    "bag_id",
    "sensor_id",
    "window_index",
    "window_start",
    "window_end",
    "sample_count",
    "feature_valid",
    "exclusion_reason",
)
FEATURES_CSV_COLUMNS = AUDIT_COLUMNS + FEATURE_COLUMNS

# Priority order: the first matching reason wins for a candidate window.
EXCLUSION_PRIORITY = (
    "DUPLICATE_TIMESTAMP",
    "PARTIAL_WINDOW",
    "TIMING_OUT_OF_TOLERANCE",
    "INVALID_CAP_QUALITY",
)

# Names that must never appear as a model feature (see leakage guard below).
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "arm",
        "lig_raw",
        "lig_quality",
        "system_quality",
        "cap_quality",
        "capacitance_raw",
        "timestamp",
        "event_type",
        "event_id",
        "event_metadata",
        "injection_start",
        "injection_end",
        "physical_leak_observed",
        "leak_flag_first",
        "leak_flag_confirmed",
        "lig_calibration_started",
        "lig_calibration_passed",
        "lig_calibration_failed",
        "delivered_volume_ml",
        "cumulative_volume_ml",
        "measured_flow_ml_min",
        "target_flow_ml_min",
        "injection_volume_ml_per_step",
        "t_physical_leak",
        "t_flag",
        "t_confirm",
        "tau",
        "b1",
        "b2",
        "b3",
        "boundary_config_version",
        "risk_label",
        "dataset_partition",
        "scenario_id",
        "kind",
        "session_id",
        "bag_id",
        "sensor_id",
        "window_id",
        "window_index",
        "window_start",
        "window_end",
        "start_timestamp",
        "end_timestamp",
        "end_reason",
        "time_remaining",
        "time_since_empty",
        "model_version",
        "operator_id",
        "device_id",
        "firmware_version",
        "fluid_type",
    }
)

_MS_PER_SECOND = 1000
_VALUE_DECIMALS = 6
_OUTPUT_FILES = ("features.csv", "feature_manifest.json")

_SUPPORTED_FEATURES = (
    "cap_delta_mean",
    "cap_delta_last",
    "cap_delta_slope_per_s",
    "cap_delta_variance",
    "cap_delta_range",
)
_FEATURE_CONFIG_KEYS = {
    "config_id",
    "status",
    "data_contract_version",
    "window_seconds",
    "stride_seconds",
    "sampling_rate_hz",
    "jitter_tolerance_ms",
    "window_interval",
    "features",
    "working_source",
}
_CONFIG_ID = "features-v0.1"
_CONFIG_STATUS = "DRAFT working pipeline config"
_WORKING_SOURCE = "docs/ai-data-collection-protocol-v0.1.md"
_WINDOW_INTERVAL = "(t-W,t]"
_WORKING_WINDOW_SECONDS = 120
_WORKING_STRIDE_SECONDS = 10
_WORKING_SAMPLING_RATE_HZ = 1
_WORKING_JITTER_TOLERANCE_MS = 200


def _assert_feature_columns_safe() -> None:
    """Fail import if the feature set ever drifts or overlaps forbidden/audit names."""
    if FEATURE_COLUMNS != _SUPPORTED_FEATURES:
        raise AssertionError("FEATURE_COLUMNS drifted from the supported feature set")
    forbidden_overlap = set(FEATURE_COLUMNS) & FORBIDDEN_FEATURE_NAMES
    if forbidden_overlap:
        raise AssertionError(
            "forbidden names present in FEATURE_COLUMNS: "
            f"{forbidden_overlap}"
        )
    audit_overlap = set(FEATURE_COLUMNS) & set(AUDIT_COLUMNS)
    if audit_overlap:
        raise AssertionError(f"audit names present in FEATURE_COLUMNS: {audit_overlap}")


_assert_feature_columns_safe()


# --------------------------------------------------------------------------- #
# Config loading and validation
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def validate_features_config(config: dict[str, Any]) -> None:
    """Reject missing/unknown keys and inconsistent numeric values before any I/O."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")

    missing = sorted(_FEATURE_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")
    unknown = sorted(config.keys() - _FEATURE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"config contains unsupported keys: {unknown}")

    if config["config_id"] != _CONFIG_ID:
        raise ValueError(f"config_id must be {_CONFIG_ID!r}")
    if config["status"] != _CONFIG_STATUS:
        raise ValueError(f"status must be {_CONFIG_STATUS!r}")

    if config["data_contract_version"] != DATA_CONTRACT_VERSION:
        raise ValueError(f"data_contract_version must be {DATA_CONTRACT_VERSION}")

    window_seconds = _require_int("window_seconds", config["window_seconds"], minimum=1)
    stride_seconds = _require_int("stride_seconds", config["stride_seconds"], minimum=1)
    sampling_rate = _require_int(
        "sampling_rate_hz", config["sampling_rate_hz"], minimum=1
    )
    jitter = _require_int(
        "jitter_tolerance_ms", config["jitter_tolerance_ms"], minimum=1
    )

    if window_seconds != _WORKING_WINDOW_SECONDS:
        raise ValueError(
            f"features-v0.1 requires window_seconds={_WORKING_WINDOW_SECONDS}"
        )
    if stride_seconds != _WORKING_STRIDE_SECONDS:
        raise ValueError(
            f"features-v0.1 requires stride_seconds={_WORKING_STRIDE_SECONDS}"
        )
    if sampling_rate != _WORKING_SAMPLING_RATE_HZ:
        raise ValueError(
            f"features-v0.1 requires sampling_rate_hz="
            f"{_WORKING_SAMPLING_RATE_HZ}"
        )
    if jitter != _WORKING_JITTER_TOLERANCE_MS:
        raise ValueError(
            f"features-v0.1 requires jitter_tolerance_ms="
            f"{_WORKING_JITTER_TOLERANCE_MS}"
        )

    if config["window_interval"] != _WINDOW_INTERVAL:
        raise ValueError(f"window_interval must be {_WINDOW_INTERVAL!r}")

    features = config["features"]
    if not isinstance(features, list) or tuple(features) != _SUPPORTED_FEATURES:
        raise ValueError(
            "features must be exactly the supported five names in fixed order"
        )
    if config["working_source"] != _WORKING_SOURCE:
        raise ValueError(f"working_source must be {_WORKING_SOURCE!r}")


def _canonical_config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Input parsing (strict, contract-aware)
# --------------------------------------------------------------------------- #
def _load_contract_rows(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected):
            raise ValueError(
                f"{path.name} header does not match Data Contract v1.1: "
                f"{reader.fieldnames!r}"
            )
        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"{path.name} line {line_number} contains extra CSV fields"
                )
            missing = sorted(key for key, value in row.items() if value is None)
            if missing:
                raise ValueError(
                    f"{path.name} line {line_number} is missing fields: {missing}"
                )
            rows.append(row)
        return rows


def _parse_float(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a valid number: {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite: {raw!r}")
    return value


def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a valid integer: {raw!r}") from error


def _read_sessions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required input file: {path.name}")
    sessions: dict[str, dict[str, Any]] = {}
    for row in _load_contract_rows(path, SessionRecord.FIELDS):
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
                baseline_value=_parse_float(
                    f"session {label} baseline_value", row["baseline_value"]
                ),
                baseline_std=_parse_float(
                    f"session {label} baseline_std", row["baseline_std"]
                ),
                start_timestamp=_parse_int(
                    f"session {label} start_timestamp", row["start_timestamp"]
                ),
                end_timestamp=_parse_int(
                    f"session {label} end_timestamp", row["end_timestamp"]
                ),
                end_reason=EndReason(row["end_reason"]),
                model_version=row["model_version"],
                firmware_version=row["firmware_version"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid sessions.csv row for {label}: {error}"
            ) from error
        if record.session_id in sessions:
            raise ValueError(
                f"duplicate session_id in sessions.csv: {record.session_id}"
            )
        sessions[record.session_id] = {
            "session_id": record.session_id,
            "bag_id": record.bag_id,
            "sensor_id": record.sensor_id,
            "baseline_value": record.baseline_value,
            "start_timestamp": record.start_timestamp,
            "end_timestamp": record.end_timestamp,
        }
    if not sessions:
        raise ValueError("sessions.csv contains no sessions")
    return sessions


def _read_samples(
    path: Path, sessions: dict[str, dict[str, Any]]
) -> dict[str, list[tuple[int, float, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required input file: {path.name}")
    grouped: dict[str, list[tuple[int, float, str]]] = {sid: [] for sid in sessions}
    last_seen: dict[str, int] = {}
    for row in _load_contract_rows(path, SampleRecord.FIELDS):
        session_id = row["session_id"]
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(
                f"samples.csv references unknown session_id: {session_id}"
            )
        try:
            record = SampleRecord(
                timestamp=_parse_int("sample timestamp", row["timestamp"]),
                session_id=session_id,
                capacitance_raw=_parse_float(
                    "capacitance_raw", row["capacitance_raw"]
                ),
                lig_raw=_parse_float("lig_raw", row["lig_raw"]),
                cap_quality=CapQuality(row["cap_quality"]),
                lig_quality=LigQuality(row["lig_quality"]),
                system_quality=SystemQuality(row["system_quality"]),
                activity_state=row["activity_state"],
                orientation_position=row["orientation_position"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid samples.csv row in session {session_id}: {error}"
            ) from error
        if not (
            session["start_timestamp"]
            <= record.timestamp
            <= session["end_timestamp"]
        ):
            raise ValueError(f"sample in {session_id} falls outside session bounds")
        if session_id in last_seen and record.timestamp < last_seen[session_id]:
            raise ValueError(
                f"samples in {session_id} are not in ascending time order"
            )
        last_seen[session_id] = record.timestamp
        grouped[session_id].append(
            (
                record.timestamp,
                record.capacitance_raw,
                record.cap_quality.value,
            )
        )
    return grouped


def _read_input_origin(input_dir: Path) -> str:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return UNDECLARED_INPUT_ORIGIN
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid input manifest.json: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("input manifest.json must contain a JSON object")
    if "dataset_origin" not in data:
        return UNDECLARED_INPUT_ORIGIN
    origin = data["dataset_origin"]
    if not isinstance(origin, str) or not origin:
        raise ValueError("input manifest.json dataset_origin must be non-empty text")
    return origin


# --------------------------------------------------------------------------- #
# Windowing and feature computation
# --------------------------------------------------------------------------- #
def _round_value(raw: float) -> float:
    value = round(float(raw), _VALUE_DECIMALS)
    if value == 0.0:  # normalize -0.0 for byte-stable output
        return 0.0
    return value


def _classify(
    members: list[tuple[int, float, str]], expected_full: int, jitter: int
) -> str | None:
    timestamps = [ts for ts, _, _ in members]
    if len(timestamps) != len(set(timestamps)):
        return "DUPLICATE_TIMESTAMP"
    if len(timestamps) != expected_full:
        return "PARTIAL_WINDOW"
    low, high = _MS_PER_SECOND - jitter, _MS_PER_SECOND + jitter
    for earlier, later in zip(timestamps, timestamps[1:]):
        if not low <= later - earlier <= high:
            return "TIMING_OUT_OF_TOLERANCE"
    if any(quality != CapQuality.OK.value for _, _, quality in members):
        return "INVALID_CAP_QUALITY"
    return None


def _compute_features(
    members: list[tuple[int, float, str]], baseline: float
) -> dict[str, float]:
    deltas = [capacitance - baseline for _, capacitance, _ in members]
    first_ts = members[0][0]
    elapsed = [(ts - first_ts) / _MS_PER_SECOND for ts, _, _ in members]
    n = len(deltas)

    delta_mean = sum(deltas) / n
    x_mean = sum(elapsed) / n
    sxx = sum((x - x_mean) ** 2 for x in elapsed)
    sxy = sum((x - x_mean) * (d - delta_mean) for x, d in zip(elapsed, deltas))
    slope = sxy / sxx  # sxx > 0: a valid window has 120 unique, ordered timestamps
    variance = sum((d - delta_mean) ** 2 for d in deltas) / n  # population, ddof=0

    return {
        "cap_delta_mean": _round_value(delta_mean),
        "cap_delta_last": _round_value(deltas[-1]),
        "cap_delta_slope_per_s": _round_value(slope),
        "cap_delta_variance": _round_value(variance),
        "cap_delta_range": _round_value(max(deltas) - min(deltas)),
    }


def _windows_for_session(
    session: dict[str, Any],
    samples: list[tuple[int, float, str]],
    window_ms: int,
    stride_ms: int,
    expected_full: int,
    jitter: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not samples:
        return rows
    t_ref = samples[0][0]
    end = session["end_timestamp"]
    baseline = session["baseline_value"]

    k = 0
    while True:
        t_k = t_ref + window_ms + k * stride_ms
        if t_k > end:
            break
        window_start = t_k - window_ms
        members = [s for s in samples if window_start < s[0] <= t_k]
        reason = _classify(members, expected_full, jitter)
        row = {
            "window_id": f"{session['session_id']}-win-{k:04d}",
            "session_id": session["session_id"],
            "bag_id": session["bag_id"],
            "sensor_id": session["sensor_id"],
            "window_index": k,
            "window_start": window_start,
            "window_end": t_k,
            "sample_count": len(members),
            "feature_valid": reason is None,
            "exclusion_reason": "" if reason is None else reason,
        }
        if reason is None:
            row.update(_compute_features(members, baseline))
        rows.append(row)
        k += 1
    return rows


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / name
        for name in _OUTPUT_FILES
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"output already contains feature artifacts: {names}; "
            "pass overwrite=True or --overwrite to replace them"
        )
    for target in existing:
        target.unlink()


def _format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_features_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FEATURES_CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                [_format_cell(row.get(column, "")) for column in FEATURES_CSV_COLUMNS]
            )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")


def extract(
    input_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract windowed features; return the feature manifest dict."""
    validate_features_config(config)

    input_dir = Path(input_dir)
    sessions_path = input_dir / "sessions.csv"
    samples_path = input_dir / "samples.csv"

    # Validate config + input and compute everything BEFORE mutating any output.
    sessions = _read_sessions(sessions_path)
    samples_by_session = _read_samples(samples_path, sessions)
    input_origin = _read_input_origin(input_dir)

    window_ms = config["window_seconds"] * _MS_PER_SECOND
    stride_ms = config["stride_seconds"] * _MS_PER_SECOND
    expected_full = config["window_seconds"] * config["sampling_rate_hz"]
    jitter = config["jitter_tolerance_ms"]

    rows: list[dict[str, Any]] = []
    for session_id in sessions:  # sessions.csv order is deterministic
        rows.extend(
            _windows_for_session(
                sessions[session_id],
                samples_by_session[session_id],
                window_ms,
                stride_ms,
                expected_full,
                jitter,
            )
        )

    exclusion_counts = {reason: 0 for reason in EXCLUSION_PRIORITY}
    valid_count = 0
    for row in rows:
        if row["feature_valid"]:
            valid_count += 1
        else:
            exclusion_counts[row["exclusion_reason"]] += 1

    manifest = {
        "extractor_version": EXTRACTOR_VERSION,
        "config_id": config["config_id"],
        "config_sha256": _canonical_config_hash(config),
        "data_contract_version": DATA_CONTRACT_VERSION,
        "input_sessions_sha256": _sha256_file(sessions_path),
        "input_samples_sha256": _sha256_file(samples_path),
        "input_dataset_origin": input_origin,
        "window_convention": {
            "interval": config["window_interval"],
            "window_seconds": config["window_seconds"],
            "stride_seconds": config["stride_seconds"],
            "sampling_rate_hz": config["sampling_rate_hz"],
            "jitter_tolerance_ms": jitter,
            "expected_full_samples": expected_full,
            "working_source": config["working_source"],
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "candidate_window_count": len(rows),
        "valid_window_count": valid_count,
        "excluded_window_count": len(rows) - valid_count,
        "exclusion_reason_counts": exclusion_counts,
        "session_count": len(sessions),
        "warning": NO_LABEL_WARNING,
    }

    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, overwrite=overwrite)
    _write_features_csv(output_dir / "features.csv", rows)
    _write_manifest(output_dir / "feature_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.features",
        description="Deterministic capacitive feature extractor (pipeline test only).",
    )
    parser.add_argument(
        "--input", required=True, help="Directory with sessions.csv + samples.csv."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a feature config JSON."
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for feature artifacts."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing feature artifacts in the output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = extract(
        args.input, load_config(args.config), args.output, overwrite=args.overwrite
    )
    print(
        f"features: {manifest['candidate_window_count']} candidate windows "
        f"({manifest['valid_window_count']} valid, "
        f"{manifest['excluded_window_count']} excluded) from "
        f"{manifest['session_count']} sessions to {args.output}"
    )
    print(NO_LABEL_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
