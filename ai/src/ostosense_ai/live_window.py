"""Stateless raw-window adapter for operator-confirmed, UNVALIDATED integration.

This is deliberately separate from the locked Tier 1 quality classifier.
NOT_EVALUATED remains NOT_EVALUATED; an operator confirmation is not calibration.
"""

import math
import statistics
from typing import Any

from ostosense_ai import features, runtime_output

REQUEST_FIELDS = {
    "request_version",
    "device_id",
    "session_id",
    "acquisition_id",
    "operator_confirmed",
    "source_window_end_ms",
    "baseline_samples",
    "window_samples",
}
SAMPLE_FIELDS = {"sample_no", "elapsed_ms", "kap_7_raw", "kap_7_status"}
MAX_RAW_VALUE = 4294967295


class LiveWindowError(ValueError):
    """Structurally malformed live request; never a fabricated fallback class."""


class WindowUnavailable(ValueError):
    """Well-formed request that cannot produce a usable experimental prediction."""


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0 or value > 2**53 - 1:
        raise LiveWindowError(f"{name} must be a non-negative safe integer")
    return value


def _samples(value: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > limit:
        raise LiveWindowError(f"samples must be a list of at most {limit} items")
    for row in value:
        if not isinstance(row, dict) or set(row) != SAMPLE_FIELDS:
            raise LiveWindowError("sample fields do not match the raw-window contract")
        _integer(row["sample_no"], "sample_no")
        _integer(row["elapsed_ms"], "elapsed_ms")
        if (
            type(row["kap_7_raw"]) not in (int, float)
            or not 0 <= row["kap_7_raw"] <= MAX_RAW_VALUE
            or not math.isfinite(row["kap_7_raw"])
        ):
            raise LiveWindowError("kap_7_raw must be a finite uint32-range raw reading")
        if not isinstance(row["kap_7_status"], str) or not row["kap_7_status"]:
            raise LiveWindowError("kap_7_status must be non-empty text")
    return value


def _reduce(rows: list[dict[str, Any]], start: int, seconds: int) -> list[tuple[int, float, str]]:
    if len(rows) != seconds * 10:
        raise WindowUnavailable("INCOMPLETE_DATA")
    times = [row["elapsed_ms"] for row in rows]
    numbers = [row["sample_no"] for row in rows]
    if len(set(times)) != len(times) or len(set(numbers)) != len(numbers):
        raise WindowUnavailable("DUPLICATE_SAMPLE")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise WindowUnavailable("OUT_OF_ORDER")
    if any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
        raise WindowUnavailable("DATA_GAP")
    if any(not 80 <= b - a <= 120 for a, b in zip(times, times[1:])):
        raise WindowUnavailable("TIMING_OUT_OF_TOLERANCE")
    if any(row["kap_7_status"] not in ("OK", "NOT_EVALUATED") for row in rows):
        raise WindowUnavailable("SENSOR_PROBLEM")
    bins: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        index = (row["elapsed_ms"] - start) // 1000
        if not 0 <= index < seconds:
            raise LiveWindowError("sample falls outside the requested time range")
        bins.setdefault(index, []).append(row)
    if len(bins) != seconds or any(len(rows) != 10 for rows in bins.values()):
        raise WindowUnavailable("INCOMPLETE_DATA")
    reduced = [
        (
            group[-1]["elapsed_ms"],
            statistics.median(row["kap_7_raw"] for row in group),
            "NOT_EVALUATED" if any(row["kap_7_status"] != "OK" for row in group) else "OK",
        )
        for group in (bins[i] for i in range(seconds))
    ]
    if any(not 800 <= b[0] - a[0] <= 1200 for a, b in zip(reduced, reduced[1:])):
        raise WindowUnavailable("TIMING_OUT_OF_TOLERANCE")
    return reduced


def unavailable(reason: str, quality: str = "UNKNOWN") -> dict[str, Any]:
    return {
        "service_response_version": "0.1.0",
        "ai": runtime_output.unavailable_output_v2(),
        "diagnostics": {
            "availability_reason": reason,
            "quality_verification": quality,
            "baseline_value": None,
        },
        "feature_input": None,
    }


def evaluate_window(request: Any, model: dict[str, Any], model_sha256: str) -> dict[str, Any]:
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        raise LiveWindowError("request fields do not match raw-window-input-v0.1")
    if request["request_version"] != "0.1.0":
        raise LiveWindowError("unsupported request_version")
    for key in ("device_id", "session_id", "acquisition_id"):
        value = request[key]
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise LiveWindowError(f"{key} must be non-empty text, maximum 128 characters")
    if type(request["operator_confirmed"]) is not bool:
        raise LiveWindowError("operator_confirmed must be boolean")
    end = _integer(request["source_window_end_ms"], "source_window_end_ms")
    if end % 10000:
        raise LiveWindowError("source_window_end_ms must align with the 10-second stride")
    base = _samples(request["baseline_samples"], 600)
    window = _samples(request["window_samples"], 1200)
    if not request["operator_confirmed"]:
        return unavailable("OPERATOR_CONFIRMATION_REQUIRED")
    if end < 60000:
        return unavailable("COLLECTING_BASELINE")
    try:
        baseline_rows = _reduce(base, 0, 60)
    except WindowUnavailable as error:
        return unavailable("BASELINE_" + str(error))
    if end < 120000:
        return unavailable("COLLECTING_WINDOW")
    try:
        members = _reduce(window, end - 120000, 120)
    except WindowUnavailable as error:
        return unavailable(str(error))
    # The initial baseline and a later overlapping window must describe the same raw readings.
    baseline_by_time = {row["elapsed_ms"]: row for row in base}
    if any(
        row != baseline_by_time.get(row["elapsed_ms"])
        for row in window
        if row["elapsed_ms"] < 60000
    ):
        raise LiveWindowError("baseline/window overlap has conflicting readings")
    quality = (
        "OPERATOR_ONLY" if any(row[2] != "OK" for row in baseline_rows + members) else "AUTOMATIC"
    )
    baseline = statistics.median(row[1] for row in baseline_rows)
    computed = features._compute_features(members, baseline)
    values = [computed[name] for name in features.FEATURE_COLUMNS]
    if not all(math.isfinite(value) for value in values):
        return unavailable("NUMERICAL_ERROR", quality)
    feature_input = {
        "feature_input_version": "0.1.0",
        "data_source": "REAL_SENSOR",
        "model_input_channel": "Kap_7",
        "source_window_end_ms": end,
        "feature_basis": "RAW_MINUS_SESSION_BASELINE",
        "feature_order": list(features.FEATURE_COLUMNS),
        "features": values,
    }
    prediction = runtime_output.predict_live_experimental_output(
        model, values, source_window_end_ms=end, model_artifact_sha256=model_sha256
    )
    return {
        "service_response_version": "0.1.0",
        "ai": prediction,
        "diagnostics": {
            "availability_reason": "READY",
            "quality_verification": quality,
            "baseline_value": baseline,
        },
        "feature_input": feature_input,
    }
