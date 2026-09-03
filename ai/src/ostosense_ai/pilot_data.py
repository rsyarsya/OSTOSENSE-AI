"""Deterministic preparation of the current unlabeled OSTOSENSE pilot CSVs.

This module is deliberately separate from the canonical labeled training
pipeline.  It accepts the flat 10 Hz ESP32 logger files collected in P001-P007,
performs strict structural checks, aggregates complete groups of ten samples to
1 Hz with a median, and emits auditable *unlabeled* window features.

The real pilot output is suitable for descriptive feature-flow and sensor-
correlation figures.  It is not suitable for classifier fitting or accuracy
claims because the current files do not contain verified per-window risk labels
or event timestamps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

PILOT_PREPARER_VERSION = "0.1.0"
DATASET_ORIGIN = "REAL_PILOT_UNLABELED"

SENSOR_CHANNELS = ("Res_15", "Res_16", "Kap_4", "Kap_5", "Kap_7")
MODEL_CHANNELS = ("Kap_4", "Kap_5", "Kap_7")
AUDIT_ONLY_CHANNELS = ("Res_15", "Res_16")
FEATURE_KINDS = (
    "delta_mean",
    "delta_last",
    "delta_slope_per_s",
    "delta_variance",
    "delta_range",
)

RAW_COLUMNS = (
    "sample_no",
    "elapsed_ms",
    "Res_15",
    "Res_15_status",
    "Res_16",
    "Res_16_status",
    "Kap_4",
    "Kap_4_status",
    "Kap_5",
    "Kap_5_status",
    "Kap_7",
    "Kap_7_status",
)

SAMPLES_1HZ_COLUMNS = (
    "session_id",
    "source_file",
    "scenario",
    "position",
    "second_index",
    "elapsed_ms",
    "elapsed_s",
    "raw_sample_start",
    "raw_sample_end",
    *tuple(
        item
        for channel in SENSOR_CHANNELS
        for item in (
            channel,
            f"{channel}_status",
            f"{channel}_baseline",
            f"{channel}_delta_norm",
        )
    ),
)

MODEL_FEATURE_COLUMNS = tuple(
    f"{channel.lower()}_{kind}" for channel in MODEL_CHANNELS for kind in FEATURE_KINDS
)

WINDOW_COLUMNS = (
    "window_id",
    "session_id",
    "source_file",
    "scenario",
    "position",
    "window_index",
    "window_start_exclusive_s",
    "window_end_inclusive_s",
    "sample_count",
    *MODEL_FEATURE_COLUMNS,
)

QC_COLUMNS = (
    "session_id",
    "source_file",
    "scenario",
    "position",
    "qc_status",
    "correlation_included",
    "raw_row_count",
    "full_1hz_bin_count",
    "partial_final_raw_sample_count",
    "window_count",
    "first_elapsed_ms",
    "last_elapsed_ms",
    "interval_min_ms",
    "interval_max_ms",
    *tuple(
        item
        for channel in SENSOR_CHANNELS
        for item in (
            f"{channel}_baseline",
            f"{channel}_baseline_mad",
            f"{channel}_normalization_scale",
            f"{channel}_status_values",
        )
    ),
)

CORRELATION_COLUMNS = ("sensor", *SENSOR_CHANNELS)

OUTPUT_FILES = (
    "qc_sessions.csv",
    "samples_1hz.csv",
    "window_features_unlabeled.csv",
    "sensor_correlation_median.csv",
    "sensor_correlation_iqr.csv",
    "pilot_manifest.json",
)

_CONFIG_KEYS = {
    "config_id",
    "status",
    "expected_interval_ms",
    "samples_per_second",
    "baseline_seconds",
    "window_seconds",
    "stride_seconds",
    "window_interval",
    "allowed_statuses",
    "model_channels",
    "audit_only_channels",
    "sessions",
    "warning",
}
_SESSION_KEYS = {
    "session_id",
    "file_name",
    "scenario",
    "position",
    "include_in_correlation",
}


class PilotDataError(ValueError):
    """Fatal pilot-input or configuration error; outputs remain untouched."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_number(value: float | int) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise PilotDataError("attempted to serialize a non-finite numeric value")
    if number == 0.0:
        number = 0.0
    return format(number, ".12g")


def _dump_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotDataError(f"invalid pilot config: {error}") from error
    if not isinstance(value, dict):
        raise PilotDataError("pilot config must be a JSON object")
    return value


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(_CONFIG_KEYS - config.keys())
    unknown = sorted(config.keys() - _CONFIG_KEYS)
    if missing:
        raise PilotDataError(f"pilot config missing keys: {missing}")
    if unknown:
        raise PilotDataError(f"pilot config has unsupported keys: {unknown}")

    if config["config_id"] != "real-pilot-v0.1":
        raise PilotDataError("config_id must be 'real-pilot-v0.1'")
    if config["status"] != DATASET_ORIGIN:
        raise PilotDataError(f"status must be {DATASET_ORIGIN}")
    if config["window_interval"] != "(t-W,t]":
        raise PilotDataError("window_interval must be '(t-W,t]'")
    if config["model_channels"] != list(MODEL_CHANNELS):
        raise PilotDataError("model_channels must be exactly Kap_4, Kap_5, Kap_7")
    if config["audit_only_channels"] != list(AUDIT_ONLY_CHANNELS):
        raise PilotDataError("audit_only_channels must be exactly Res_15, Res_16")

    positive_ints = (
        "expected_interval_ms",
        "samples_per_second",
        "baseline_seconds",
        "window_seconds",
        "stride_seconds",
    )
    for key in positive_ints:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PilotDataError(f"{key} must be a positive integer")
    if config["expected_interval_ms"] * config["samples_per_second"] != 1000:
        raise PilotDataError("expected_interval_ms * samples_per_second must equal 1000")
    if config["baseline_seconds"] > config["window_seconds"]:
        raise PilotDataError("baseline_seconds must not exceed window_seconds")

    statuses = config["allowed_statuses"]
    if (
        not isinstance(statuses, list)
        or not statuses
        or any(not isinstance(item, str) or not item for item in statuses)
        or len(set(statuses)) != len(statuses)
    ):
        raise PilotDataError("allowed_statuses must be a unique non-empty string list")
    if "NOT_EVALUATED" not in statuses:
        raise PilotDataError("allowed_statuses must contain NOT_EVALUATED")

    sessions = config["sessions"]
    if not isinstance(sessions, list) or not sessions:
        raise PilotDataError("sessions must be a non-empty list")
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for index, session in enumerate(sessions):
        if not isinstance(session, dict) or set(session) != _SESSION_KEYS:
            raise PilotDataError(f"sessions[{index}] must contain exactly {_SESSION_KEYS}")
        for key in ("session_id", "file_name", "scenario", "position"):
            if not isinstance(session[key], str) or not session[key]:
                raise PilotDataError(f"sessions[{index}].{key} must be a non-empty string")
        file_name = session["file_name"]
        if Path(file_name).name != file_name or not file_name.endswith(".csv"):
            raise PilotDataError(f"sessions[{index}].file_name must be a plain CSV file name")
        if not isinstance(session["include_in_correlation"], bool):
            raise PilotDataError(
                f"sessions[{index}].include_in_correlation must be boolean"
            )
        if session["session_id"] in seen_ids:
            raise PilotDataError(f"duplicate session_id: {session['session_id']}")
        if file_name in seen_files:
            raise PilotDataError(f"duplicate session file: {file_name}")
        seen_ids.add(session["session_id"])
        seen_files.add(file_name)
    if not isinstance(config["warning"], str) or not config["warning"]:
        raise PilotDataError("warning must be a non-empty string")


def _parse_int(raw: str, *, label: str) -> int:
    if raw == "":
        raise PilotDataError(f"{label} is empty")
    try:
        value = int(raw)
    except ValueError as error:
        raise PilotDataError(f"{label} must be an integer, got {raw!r}") from error
    return value


def _parse_number(raw: str, *, label: str) -> float:
    if raw == "":
        raise PilotDataError(f"{label} is empty")
    try:
        value = float(raw)
    except ValueError as error:
        raise PilotDataError(f"{label} must be numeric, got {raw!r}") from error
    if not math.isfinite(value):
        raise PilotDataError(f"{label} must be finite, got {raw!r}")
    return value


def _read_raw_session(
    path: Path,
    session_id: str,
    expected_interval_ms: int,
    allowed_statuses: set[str],
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PilotDataError(f"missing session CSV: {path.name}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(RAW_COLUMNS):
                raise PilotDataError(
                    f"{path.name} header is not canonical: {reader.fieldnames!r}"
                )
            first_elapsed_ms: int | None = None
            for line_number, raw_row in enumerate(reader, start=2):
                if None in raw_row or any(value is None for value in raw_row.values()):
                    raise PilotDataError(f"{path.name} line {line_number} is malformed")
                sample_no = _parse_int(
                    raw_row["sample_no"], label=f"{path.name}:{line_number}:sample_no"
                )
                elapsed_ms = _parse_int(
                    raw_row["elapsed_ms"], label=f"{path.name}:{line_number}:elapsed_ms"
                )
                expected_sample_no = len(rows)
                if sample_no != expected_sample_no:
                    raise PilotDataError(
                        f"{path.name} sample_no must be consecutive from 0; "
                        f"expected {expected_sample_no}, got {sample_no}"
                    )
                if first_elapsed_ms is None:
                    if elapsed_ms not in (0, expected_interval_ms):
                        raise PilotDataError(
                            f"{path.name} first elapsed_ms must be 0 or "
                            f"{expected_interval_ms}, got {elapsed_ms}"
                        )
                    first_elapsed_ms = elapsed_ms
                expected_elapsed = first_elapsed_ms + sample_no * expected_interval_ms
                if elapsed_ms != expected_elapsed:
                    raise PilotDataError(
                        f"{path.name} sample {sample_no} elapsed_ms must be "
                        f"{expected_elapsed}, got {elapsed_ms}"
                    )
                row: dict[str, Any] = {
                    "session_id": session_id,
                    "sample_no": sample_no,
                    "elapsed_ms": elapsed_ms,
                }
                for channel in SENSOR_CHANNELS:
                    row[channel] = _parse_number(
                        raw_row[channel],
                        label=f"{path.name}:{line_number}:{channel}",
                    )
                    status = raw_row[f"{channel}_status"]
                    if status not in allowed_statuses:
                        raise PilotDataError(
                            f"{path.name}:{line_number}:{channel}_status has unknown "
                            f"value {status!r}"
                        )
                    row[f"{channel}_status"] = status
                rows.append(row)
    except UnicodeDecodeError as error:
        raise PilotDataError(f"{path.name} is not valid UTF-8 CSV") from error
    if not rows:
        raise PilotDataError(f"{path.name} contains no data rows")
    return rows


def _aggregate_1hz(
    raw_rows: list[dict[str, Any]],
    session: dict[str, Any],
    samples_per_second: int,
) -> tuple[list[dict[str, Any]], int]:
    full_count = len(raw_rows) // samples_per_second
    partial_count = len(raw_rows) % samples_per_second
    aggregated: list[dict[str, Any]] = []
    for bin_index in range(full_count):
        group = raw_rows[
            bin_index * samples_per_second : (bin_index + 1) * samples_per_second
        ]
        output: dict[str, Any] = {
            "session_id": session["session_id"],
            "source_file": session["file_name"],
            "scenario": session["scenario"],
            "position": session["position"],
            "second_index": bin_index + 1,
            "elapsed_ms": group[-1]["elapsed_ms"],
            "elapsed_s": float(bin_index + 1),
            "raw_sample_start": group[0]["sample_no"],
            "raw_sample_end": group[-1]["sample_no"],
        }
        for channel in SENSOR_CHANNELS:
            output[channel] = float(statistics.median(row[channel] for row in group))
            statuses = sorted({row[f"{channel}_status"] for row in group})
            output[f"{channel}_status"] = (
                statuses[0] if len(statuses) == 1 else "MIXED:" + "+".join(statuses)
            )
        aggregated.append(output)
    return aggregated, partial_count


def _baseline_stats(
    samples: list[dict[str, Any]], baseline_seconds: int
) -> dict[str, dict[str, float]]:
    if len(samples) < baseline_seconds:
        raise PilotDataError(
            f"session {samples[0]['session_id']} has only {len(samples)} complete seconds; "
            f"{baseline_seconds} are required for the provisional baseline"
        )
    result: dict[str, dict[str, float]] = {}
    baseline_rows = samples[:baseline_seconds]
    for channel in SENSOR_CHANNELS:
        values = [float(row[channel]) for row in baseline_rows]
        baseline = float(statistics.median(values))
        mad = float(statistics.median(abs(value - baseline) for value in values))
        scale = max(abs(baseline), 1.4826 * mad, 1.0)
        result[channel] = {"baseline": baseline, "mad": mad, "scale": scale}
    return result


def _add_normalized_deltas(
    samples: list[dict[str, Any]], baseline: dict[str, dict[str, float]]
) -> None:
    for row in samples:
        for channel in SENSOR_CHANNELS:
            stats = baseline[channel]
            row[f"{channel}_baseline"] = stats["baseline"]
            row[f"{channel}_delta_norm"] = (
                float(row[channel]) - stats["baseline"]
            ) / stats["scale"]


def _feature_values(values: list[float], times: list[float]) -> dict[str, float]:
    if not values or len(values) != len(times):
        raise PilotDataError("feature window values/times must be non-empty and aligned")
    count = len(values)
    mean_value = sum(values) / count
    mean_time = sum(times) / count
    denominator = sum((time - mean_time) ** 2 for time in times)
    if denominator <= 0.0:
        raise PilotDataError("feature window has no usable time variation")
    slope = sum(
        (time - mean_time) * (value - mean_value)
        for time, value in zip(times, values)
    ) / denominator
    variance = sum((value - mean_value) ** 2 for value in values) / count
    return {
        "delta_mean": mean_value,
        "delta_last": values[-1],
        "delta_slope_per_s": slope,
        "delta_variance": variance,
        "delta_range": max(values) - min(values),
    }


def _build_windows(
    samples: list[dict[str, Any]],
    session: dict[str, Any],
    window_seconds: int,
    stride_seconds: int,
) -> list[dict[str, Any]]:
    by_second = {int(row["elapsed_s"]): row for row in samples}
    final_second = int(samples[-1]["elapsed_s"])
    output: list[dict[str, Any]] = []
    for end_second in range(window_seconds, final_second + 1, stride_seconds):
        expected_seconds = list(range(end_second - window_seconds + 1, end_second + 1))
        if any(second not in by_second for second in expected_seconds):
            raise PilotDataError(
                f"session {session['session_id']} has an incomplete 1 Hz window ending "
                f"at {end_second} s"
            )
        rows = [by_second[second] for second in expected_seconds]
        window_index = len(output)
        result: dict[str, Any] = {
            "window_id": f"{session['session_id']}-w{window_index:04d}",
            "session_id": session["session_id"],
            "source_file": session["file_name"],
            "scenario": session["scenario"],
            "position": session["position"],
            "window_index": window_index,
            "window_start_exclusive_s": end_second - window_seconds,
            "window_end_inclusive_s": end_second,
            "sample_count": len(rows),
        }
        times = [float(row["elapsed_s"]) for row in rows]
        for channel in MODEL_CHANNELS:
            values = [float(row[f"{channel}_delta_norm"]) for row in rows]
            channel_features = _feature_values(values, times)
            for kind, value in channel_features.items():
                result[f"{channel.lower()}_{kind}"] = value
        output.append(result)
    return output


def _rankdata(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rankdata(left), _rankdata(right))


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise PilotDataError("cannot calculate a quantile from an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _correlation_matrices(
    session_samples: list[tuple[dict[str, Any], list[dict[str, Any]]]]
) -> tuple[list[list[float]], list[list[float]], dict[str, int]]:
    pair_values: dict[tuple[str, str], list[float]] = {
        (left, right): [] for left in SENSOR_CHANNELS for right in SENSOR_CHANNELS
    }
    included_ids: list[str] = []
    for session, rows in session_samples:
        if not session["include_in_correlation"]:
            continue
        included_ids.append(session["session_id"])
        values = {
            channel: [float(row[channel]) for row in rows] for channel in SENSOR_CHANNELS
        }
        ranks = {channel: _rankdata(channel_values) for channel, channel_values in values.items()}
        for left in SENSOR_CHANNELS:
            for right in SENSOR_CHANNELS:
                correlation = 1.0 if left == right else _pearson(ranks[left], ranks[right])
                if correlation is not None:
                    pair_values[(left, right)].append(correlation)

    if not included_ids:
        raise PilotDataError("no sessions are enabled for the correlation analysis")
    median_matrix: list[list[float]] = []
    iqr_matrix: list[list[float]] = []
    valid_counts: dict[str, int] = {}
    for left in SENSOR_CHANNELS:
        median_row: list[float] = []
        iqr_row: list[float] = []
        for right in SENSOR_CHANNELS:
            pair = pair_values[(left, right)]
            if not pair:
                raise PilotDataError(f"correlation {left}/{right} is undefined in all sessions")
            median_row.append(float(statistics.median(pair)))
            iqr_row.append(_quantile(pair, 0.75) - _quantile(pair, 0.25))
            valid_counts[f"{left}__{right}"] = len(pair)
        median_matrix.append(median_row)
        iqr_matrix.append(iqr_row)
    return median_matrix, iqr_matrix, valid_counts


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    columns = tuple(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            serialized = {
                column: _format_number(row[column])
                if isinstance(row[column], (int, float)) and not isinstance(row[column], bool)
                else ("true" if row[column] is True else "false" if row[column] is False else row[column])
                for column in columns
            }
            writer.writerow(serialized)


def _matrix_rows(matrix: list[list[float]]) -> list[dict[str, Any]]:
    return [
        {"sensor": channel, **dict(zip(SENSOR_CHANNELS, row))}
        for channel, row in zip(SENSOR_CHANNELS, matrix)
    ]


def _check_outputs(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output already contains pilot artifacts: "
            + ", ".join(existing)
            + "; pass overwrite=True or --overwrite to replace them"
        )


def prepare_pilot_dataset(
    input_dir: str | Path,
    config: str | Path | dict[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and transform the flat real-pilot logger files.

    All inputs are validated and all outputs are built in a staging directory
    before any destination artifact is replaced.
    """

    config_object = load_config(config) if not isinstance(config, dict) else config
    validate_config(config_object)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise PilotDataError(f"input directory does not exist: {input_dir}")
    _check_outputs(output_dir, overwrite)

    configured_files = {session["file_name"] for session in config_object["sessions"]}
    actual_files = {path.name for path in input_dir.glob("P*.csv") if path.is_file()}
    missing = sorted(configured_files - actual_files)
    unexpected = sorted(actual_files - configured_files)
    if missing:
        raise PilotDataError(f"configured session files are missing: {missing}")
    if unexpected:
        raise PilotDataError(f"unconfigured P*.csv session files are present: {unexpected}")

    all_1hz: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    correlation_inputs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    input_hashes: dict[str, str] = {}
    raw_row_count = 0
    partial_count = 0

    for session in config_object["sessions"]:
        path = input_dir / session["file_name"]
        input_hashes[session["file_name"]] = _sha256_file(path)
        raw = _read_raw_session(
            path,
            session["session_id"],
            config_object["expected_interval_ms"],
            set(config_object["allowed_statuses"]),
        )
        one_hz, session_partial = _aggregate_1hz(
            raw, session, config_object["samples_per_second"]
        )
        baseline = _baseline_stats(one_hz, config_object["baseline_seconds"])
        _add_normalized_deltas(one_hz, baseline)
        windows = _build_windows(
            one_hz,
            session,
            config_object["window_seconds"],
            config_object["stride_seconds"],
        )
        intervals = [
            raw[index]["elapsed_ms"] - raw[index - 1]["elapsed_ms"]
            for index in range(1, len(raw))
        ]
        qc: dict[str, Any] = {
            "session_id": session["session_id"],
            "source_file": session["file_name"],
            "scenario": session["scenario"],
            "position": session["position"],
            "qc_status": "STRUCTURAL_PASS",
            "correlation_included": session["include_in_correlation"],
            "raw_row_count": len(raw),
            "full_1hz_bin_count": len(one_hz),
            "partial_final_raw_sample_count": session_partial,
            "window_count": len(windows),
            "first_elapsed_ms": raw[0]["elapsed_ms"],
            "last_elapsed_ms": raw[-1]["elapsed_ms"],
            "interval_min_ms": min(intervals) if intervals else 0,
            "interval_max_ms": max(intervals) if intervals else 0,
        }
        for channel in SENSOR_CHANNELS:
            qc[f"{channel}_baseline"] = baseline[channel]["baseline"]
            qc[f"{channel}_baseline_mad"] = baseline[channel]["mad"]
            qc[f"{channel}_normalization_scale"] = baseline[channel]["scale"]
            qc[f"{channel}_status_values"] = "+".join(
                sorted({row[f"{channel}_status"] for row in raw})
            )

        raw_row_count += len(raw)
        partial_count += session_partial
        all_1hz.extend(one_hz)
        all_windows.extend(windows)
        qc_rows.append(qc)
        correlation_inputs.append((session, one_hz))

    correlation_median, correlation_iqr, valid_pair_counts = _correlation_matrices(
        correlation_inputs
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ostosense-pilot-", dir=output_dir.parent) as name:
        stage = Path(name)
        _write_csv(stage / "qc_sessions.csv", QC_COLUMNS, qc_rows)
        _write_csv(stage / "samples_1hz.csv", SAMPLES_1HZ_COLUMNS, all_1hz)
        _write_csv(
            stage / "window_features_unlabeled.csv", WINDOW_COLUMNS, all_windows
        )
        _write_csv(
            stage / "sensor_correlation_median.csv",
            CORRELATION_COLUMNS,
            _matrix_rows(correlation_median),
        )
        _write_csv(
            stage / "sensor_correlation_iqr.csv",
            CORRELATION_COLUMNS,
            _matrix_rows(correlation_iqr),
        )
        output_hashes = {
            file_name: _sha256_file(stage / file_name)
            for file_name in OUTPUT_FILES
            if file_name != "pilot_manifest.json"
        }
        included_sessions = [
            session["session_id"]
            for session in config_object["sessions"]
            if session["include_in_correlation"]
        ]
        excluded_sessions = [
            session["session_id"]
            for session in config_object["sessions"]
            if not session["include_in_correlation"]
        ]
        manifest = {
            "pilot_preparer_version": PILOT_PREPARER_VERSION,
            "config_id": config_object["config_id"],
            "dataset_origin": DATASET_ORIGIN,
            "scope": "unlabeled real-pilot preprocessing and descriptive analysis",
            "input_sha256": input_hashes,
            "output_sha256": output_hashes,
            "session_count": len(config_object["sessions"]),
            "raw_row_count": raw_row_count,
            "full_1hz_bin_count": len(all_1hz),
            "partial_final_raw_sample_count": partial_count,
            "unlabeled_window_count": len(all_windows),
            "channel_roles": {
                "model_candidates": list(MODEL_CHANNELS),
                "descriptive_and_fail_safe_only": list(AUDIT_ONLY_CHANNELS),
            },
            "aggregation": {
                "source_rate_hz": config_object["samples_per_second"],
                "output_rate_hz": 1,
                "method": "median of each consecutive complete 10-sample group",
                "partial_group_policy": "exclude and report",
                "outlier_replacement": "none",
            },
            "provisional_baseline": {
                "seconds": config_object["baseline_seconds"],
                "center": "median",
                "mad": "median absolute deviation",
                "scale": "max(abs(baseline), 1.4826*MAD, 1)",
                "normalization": "(value-baseline)/scale",
                "verified_dry": False,
            },
            "windowing": {
                "window_seconds": config_object["window_seconds"],
                "stride_seconds": config_object["stride_seconds"],
                "interval": config_object["window_interval"],
                "feature_columns": list(MODEL_FEATURE_COLUMNS),
                "label_status": "UNLABELED",
            },
            "correlation": {
                "method": "Spearman within each session, then median across sessions",
                "session_weighting": "equal",
                "channels": list(SENSOR_CHANNELS),
                "included_sessions": included_sessions,
                "excluded_sessions": excluded_sessions,
                "valid_session_pair_counts": valid_pair_counts,
                "dispersion_output": "per-pair interquartile range",
            },
            "warning": config_object["warning"],
        }
        (stage / "pilot_manifest.json").write_text(
            _dump_json(manifest), encoding="utf-8"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_name in OUTPUT_FILES:
            (stage / file_name).replace(output_dir / file_name)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.pilot_data",
        description="Prepare unlabeled OSTOSENSE real-pilot logger CSVs.",
    )
    parser.add_argument("--input", required=True, help="Directory containing P001-P007 CSVs.")
    parser.add_argument("--config", required=True, help="real-pilot-v0.1 JSON config.")
    parser.add_argument("--output", required=True, help="Output artifact directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing pilot artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = prepare_pilot_dataset(
        args.input, args.config, args.output, overwrite=args.overwrite
    )
    print(
        "pilot-data: "
        f"{manifest['session_count']} sessions, {manifest['raw_row_count']} raw rows, "
        f"{manifest['full_1hz_bin_count']} complete 1 Hz bins, "
        f"{manifest['unlabeled_window_count']} unlabeled windows"
    )
    print(manifest["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
