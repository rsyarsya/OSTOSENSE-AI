"""Canonical AI output for downstream software integration.

Version 0.1 remains the immutable unavailable/synthetic-test contract. Version
0.2 additionally permits an explicitly unvalidated engineering class from real
Kap_7 features. Neither version converts model output into a leakage percentage,
countdown, LIG state, bag-fill estimate, notification, or clinical action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ostosense_ai import inference

RUNTIME_OUTPUT_VERSION = "0.1.0"
RUNTIME_OUTPUT_V2_VERSION = "0.2.0"
FEATURE_INPUT_VERSION = "0.1.0"

LIVE_MODE = "LIVE"
ENGINEERING_TEST_MODE = "ENGINEERING_TEST"
LIVE_EXPERIMENTAL_MODE = "LIVE_EXPERIMENTAL"
NO_DATA_SOURCE = "NONE"
SYNTHETIC_DATA_SOURCE = "SYNTHETIC_FIXTURE"
REAL_SENSOR_DATA_SOURCE = "REAL_SENSOR"
UNAVAILABLE_STATUS = "UNAVAILABLE"
TEST_ONLY_STATUS = "TEST_ONLY"
UNVALIDATED_STATUS = "UNVALIDATED"

NO_PREDICTION_SCOPE = "NO_PREDICTION"
PIPELINE_MECHANICS_SCOPE = "PIPELINE_MECHANICS_ONLY"
EXPERIMENTAL_UNVALIDATED_SCOPE = "EXPERIMENTAL_UNVALIDATED"

KAP_7_INPUT_CHANNEL = "Kap_7"
SYNTHETIC_INPUT_CHANNEL = "SYNTHETIC_CAPACITIVE"
RAW_BASELINE_FEATURE_BASIS = "RAW_MINUS_SESSION_BASELINE"
FEATURE_INPUT_FIELDS = (
    "feature_input_version",
    "data_source",
    "model_input_channel",
    "source_window_end_ms",
    "feature_basis",
    "feature_order",
    "features",
)

LIVE_UNAVAILABLE_WARNING = (
    "No approved real-data model is available; no OSTOSENSE risk class was produced."
)
LIVE_UNAVAILABLE_WARNING_V2 = (
    "No usable AI prediction is available; no OSTOSENSE risk class was produced."
)
ENGINEERING_TEST_WARNING = (
    "ENGINEERING_TEST_ONLY synthetic result; not a patient-risk assessment or clinical output."
)
LIVE_EXPERIMENTAL_WARNING = (
    "UNVALIDATED experimental class from a synthetic-trained model applied to real Kap_7 "
    "sensor features; not for patient notification or clinical action."
)

OUTPUT_FIELDS = (
    "runtime_output_version",
    "mode",
    "data_source",
    "model_status",
    "prediction_available",
    "risk_class",
    "risk_class_index",
    "model_artifact_version",
    "evidence_scope",
    "warning",
)

OUTPUT_FIELDS_V2 = (
    "runtime_output_version",
    "mode",
    "data_source",
    "model_status",
    "prediction_available",
    "risk_class",
    "risk_class_index",
    "source_window_end_ms",
    "model_input_channel",
    "model_artifact_version",
    "model_artifact_sha256",
    "evidence_scope",
    "warning",
)


class RuntimeOutputError(ValueError):
    """Runtime output or its inference inputs violate the integration contract."""


def _require_exact_fields(payload: dict[str, Any], fields: Sequence[str]) -> None:
    expected = set(fields)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeOutputError(
            f"runtime output fields differ from the contract; missing={missing}, extra={extra}"
        )


def _validate_common(payload: dict[str, Any]) -> None:
    if not isinstance(payload["warning"], str) or not payload["warning"].strip():
        raise RuntimeOutputError("warning must be a non-empty string")
    if not isinstance(payload["prediction_available"], bool):
        raise RuntimeOutputError("prediction_available must be boolean")


def _validate_class_pair(payload: dict[str, Any]) -> tuple[str, int]:
    risk_class = payload["risk_class"]
    risk_index = payload["risk_class_index"]
    if risk_class not in inference.CLASS_NAME_TO_INDEX:
        raise RuntimeOutputError("risk_class must be Safe, Monitor, Caution, or Urgent")
    if isinstance(risk_index, bool) or not isinstance(risk_index, int):
        raise RuntimeOutputError("risk_class_index must be an integer")
    if risk_index != inference.CLASS_NAME_TO_INDEX[risk_class]:
        raise RuntimeOutputError("risk_class and risk_class_index are inconsistent")
    return risk_class, risk_index


def _validate_runtime_output_v1(payload: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(payload, OUTPUT_FIELDS)
    _validate_common(payload)

    if payload["prediction_available"] is False:
        expected = {
            "mode": LIVE_MODE,
            "data_source": NO_DATA_SOURCE,
            "model_status": UNAVAILABLE_STATUS,
            "risk_class": None,
            "risk_class_index": None,
            "model_artifact_version": None,
            "evidence_scope": NO_PREDICTION_SCOPE,
            "warning": LIVE_UNAVAILABLE_WARNING,
        }
    else:
        risk_class, risk_index = _validate_class_pair(payload)
        expected = {
            "mode": ENGINEERING_TEST_MODE,
            "data_source": SYNTHETIC_DATA_SOURCE,
            "model_status": TEST_ONLY_STATUS,
            "risk_class": risk_class,
            "risk_class_index": risk_index,
            "model_artifact_version": inference.EXPECTED_MODEL_ARTIFACT_VERSION,
            "evidence_scope": PIPELINE_MECHANICS_SCOPE,
            "warning": ENGINEERING_TEST_WARNING,
        }

    for field, expected_value in expected.items():
        if payload[field] != expected_value:
            raise RuntimeOutputError(
                f"{field} must be {expected_value!r} for this runtime state"
            )
    return payload


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_runtime_output_v2(payload: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(payload, OUTPUT_FIELDS_V2)
    _validate_common(payload)

    if payload["prediction_available"] is False:
        expected = {
            "mode": LIVE_MODE,
            "data_source": NO_DATA_SOURCE,
            "model_status": UNAVAILABLE_STATUS,
            "risk_class": None,
            "risk_class_index": None,
            "source_window_end_ms": None,
            "model_input_channel": None,
            "model_artifact_version": None,
            "model_artifact_sha256": None,
            "evidence_scope": NO_PREDICTION_SCOPE,
            "warning": LIVE_UNAVAILABLE_WARNING_V2,
        }
    else:
        risk_class, risk_index = _validate_class_pair(payload)
        source_window_end_ms = payload["source_window_end_ms"]
        if (
            isinstance(source_window_end_ms, bool)
            or not isinstance(source_window_end_ms, int)
            or source_window_end_ms < 0
        ):
            raise RuntimeOutputError("source_window_end_ms must be a non-negative integer")
        if not _valid_sha256(payload["model_artifact_sha256"]):
            raise RuntimeOutputError("model_artifact_sha256 must be a lowercase SHA-256 digest")

        if payload["mode"] == ENGINEERING_TEST_MODE:
            state = {
                "data_source": SYNTHETIC_DATA_SOURCE,
                "model_status": TEST_ONLY_STATUS,
                "model_input_channel": SYNTHETIC_INPUT_CHANNEL,
                "evidence_scope": PIPELINE_MECHANICS_SCOPE,
                "warning": ENGINEERING_TEST_WARNING,
            }
        elif payload["mode"] == LIVE_EXPERIMENTAL_MODE:
            state = {
                "data_source": REAL_SENSOR_DATA_SOURCE,
                "model_status": UNVALIDATED_STATUS,
                "model_input_channel": KAP_7_INPUT_CHANNEL,
                "evidence_scope": EXPERIMENTAL_UNVALIDATED_SCOPE,
                "warning": LIVE_EXPERIMENTAL_WARNING,
            }
        else:
            raise RuntimeOutputError(
                "prediction mode must be ENGINEERING_TEST or LIVE_EXPERIMENTAL"
            )

        expected = {
            "mode": payload["mode"],
            "risk_class": risk_class,
            "risk_class_index": risk_index,
            "source_window_end_ms": source_window_end_ms,
            "model_artifact_version": inference.EXPECTED_MODEL_ARTIFACT_VERSION,
            "model_artifact_sha256": payload["model_artifact_sha256"],
            **state,
        }

    for field, expected_value in expected.items():
        if payload[field] != expected_value:
            raise RuntimeOutputError(
                f"{field} must be {expected_value!r} for this runtime state"
            )
    return payload


def validate_runtime_output(payload: Any) -> dict[str, Any]:
    """Validate v0.1 or v0.2 and return the payload without mutating it."""
    if not isinstance(payload, dict):
        raise RuntimeOutputError("runtime output must be a JSON object")
    version = payload.get("runtime_output_version")
    if version == RUNTIME_OUTPUT_VERSION:
        return _validate_runtime_output_v1(payload)
    if version == RUNTIME_OUTPUT_V2_VERSION:
        return _validate_runtime_output_v2(payload)
    raise RuntimeOutputError(
        "runtime_output_version must be 0.1.0 or 0.2.0"
    )


def unavailable_output(
    *, mode: str = LIVE_MODE, data_source: str = NO_DATA_SOURCE
) -> dict[str, Any]:
    """Return the only valid live state while no approved real model exists."""
    if mode != LIVE_MODE or data_source != NO_DATA_SOURCE:
        raise RuntimeOutputError("unavailable output requires mode=LIVE and data_source=NONE")
    return validate_runtime_output(
        {
            "runtime_output_version": RUNTIME_OUTPUT_VERSION,
            "mode": LIVE_MODE,
            "data_source": NO_DATA_SOURCE,
            "model_status": UNAVAILABLE_STATUS,
            "prediction_available": False,
            "risk_class": None,
            "risk_class_index": None,
            "model_artifact_version": None,
            "evidence_scope": NO_PREDICTION_SCOPE,
            "warning": LIVE_UNAVAILABLE_WARNING,
        }
    )


def unavailable_output_v2() -> dict[str, Any]:
    """Return the v0.2 live state when no usable prediction is available."""
    return validate_runtime_output(
        {
            "runtime_output_version": RUNTIME_OUTPUT_V2_VERSION,
            "mode": LIVE_MODE,
            "data_source": NO_DATA_SOURCE,
            "model_status": UNAVAILABLE_STATUS,
            "prediction_available": False,
            "risk_class": None,
            "risk_class_index": None,
            "source_window_end_ms": None,
            "model_input_channel": None,
            "model_artifact_version": None,
            "model_artifact_sha256": None,
            "evidence_scope": NO_PREDICTION_SCOPE,
            "warning": LIVE_UNAVAILABLE_WARNING_V2,
        }
    )


def predict_test_output(
    model_artifact: Any,
    feature_row: Sequence[Any],
    *,
    mode: str = ENGINEERING_TEST_MODE,
    data_source: str = SYNTHETIC_DATA_SOURCE,
) -> dict[str, Any]:
    """Return one explicitly synthetic ordinal class for engineering demos."""
    if mode != ENGINEERING_TEST_MODE:
        raise RuntimeOutputError(
            "the current synthetic model cannot run in LIVE mode"
        )
    if data_source != SYNTHETIC_DATA_SOURCE:
        raise RuntimeOutputError(
            "engineering-test prediction requires data_source=SYNTHETIC_FIXTURE"
        )
    try:
        prediction = inference.predict_exported_model(model_artifact, [feature_row])
    except (inference.InferenceError, TypeError) as error:
        raise RuntimeOutputError(str(error)) from error

    risk_class = prediction["predicted_labels"][0]
    risk_index = prediction["predicted_indices"][0]
    return validate_runtime_output(
        {
            "runtime_output_version": RUNTIME_OUTPUT_VERSION,
            "mode": ENGINEERING_TEST_MODE,
            "data_source": SYNTHETIC_DATA_SOURCE,
            "model_status": TEST_ONLY_STATUS,
            "prediction_available": True,
            "risk_class": risk_class,
            "risk_class_index": risk_index,
            "model_artifact_version": model_artifact.get("model_artifact_version")
            if isinstance(model_artifact, dict)
            else None,
            "evidence_scope": PIPELINE_MECHANICS_SCOPE,
            "warning": ENGINEERING_TEST_WARNING,
        }
    )


def _predict_v2_output(
    model_artifact: Any,
    feature_row: Sequence[Any],
    *,
    source_window_end_ms: int,
    model_artifact_sha256: str,
    mode: str,
) -> dict[str, Any]:
    try:
        prediction = inference.predict_exported_model(model_artifact, [feature_row])
    except (inference.InferenceError, TypeError) as error:
        raise RuntimeOutputError(str(error)) from error

    if mode == ENGINEERING_TEST_MODE:
        data_source = SYNTHETIC_DATA_SOURCE
        model_status = TEST_ONLY_STATUS
        model_input_channel = SYNTHETIC_INPUT_CHANNEL
        evidence_scope = PIPELINE_MECHANICS_SCOPE
        warning = ENGINEERING_TEST_WARNING
    elif mode == LIVE_EXPERIMENTAL_MODE:
        data_source = REAL_SENSOR_DATA_SOURCE
        model_status = UNVALIDATED_STATUS
        model_input_channel = KAP_7_INPUT_CHANNEL
        evidence_scope = EXPERIMENTAL_UNVALIDATED_SCOPE
        warning = LIVE_EXPERIMENTAL_WARNING
    else:
        raise RuntimeOutputError("unsupported v0.2 prediction mode")

    return validate_runtime_output(
        {
            "runtime_output_version": RUNTIME_OUTPUT_V2_VERSION,
            "mode": mode,
            "data_source": data_source,
            "model_status": model_status,
            "prediction_available": True,
            "risk_class": prediction["predicted_labels"][0],
            "risk_class_index": prediction["predicted_indices"][0],
            "source_window_end_ms": source_window_end_ms,
            "model_input_channel": model_input_channel,
            "model_artifact_version": model_artifact.get("model_artifact_version")
            if isinstance(model_artifact, dict)
            else None,
            "model_artifact_sha256": model_artifact_sha256,
            "evidence_scope": evidence_scope,
            "warning": warning,
        }
    )


def predict_test_output_v2(
    model_artifact: Any,
    feature_row: Sequence[Any],
    *,
    source_window_end_ms: int,
    model_artifact_sha256: str,
) -> dict[str, Any]:
    """Return one versioned synthetic result for software contract tests."""
    return _predict_v2_output(
        model_artifact,
        feature_row,
        source_window_end_ms=source_window_end_ms,
        model_artifact_sha256=model_artifact_sha256,
        mode=ENGINEERING_TEST_MODE,
    )


def predict_live_experimental_output(
    model_artifact: Any,
    feature_row: Sequence[Any],
    *,
    source_window_end_ms: int,
    model_artifact_sha256: str,
) -> dict[str, Any]:
    """Apply the synthetic-trained model to real Kap_7 features for integration only."""
    return _predict_v2_output(
        model_artifact,
        feature_row,
        source_window_end_ms=source_window_end_ms,
        model_artifact_sha256=model_artifact_sha256,
        mode=LIVE_EXPERIMENTAL_MODE,
    )


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeOutputError(f"cannot read {name} JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeOutputError(f"{name} JSON must be an object")
    return value


def _read_feature_document(path: Path) -> list[Any]:
    document = _read_json_object(path, name="feature")
    if set(document) != {"data_source", "features"}:
        raise RuntimeOutputError(
            "feature JSON must contain exactly data_source and features"
        )
    if document["data_source"] != SYNTHETIC_DATA_SOURCE:
        raise RuntimeOutputError(
            "feature JSON data_source must be SYNTHETIC_FIXTURE"
        )
    features = document["features"]
    if not isinstance(features, list):
        raise RuntimeOutputError("feature JSON features must be an array")
    return features


def _read_v2_feature_document(
    path: Path, *, expected_data_source: str, expected_channel: str
) -> tuple[list[Any], int]:
    document = _read_json_object(path, name="feature")
    if set(document) != set(FEATURE_INPUT_FIELDS):
        raise RuntimeOutputError(
            "v0.2 feature JSON fields differ from the feature-input contract"
        )
    if document["feature_input_version"] != FEATURE_INPUT_VERSION:
        raise RuntimeOutputError(
            f"feature_input_version must be {FEATURE_INPUT_VERSION}"
        )
    if document["data_source"] != expected_data_source:
        raise RuntimeOutputError(f"feature JSON data_source must be {expected_data_source}")
    if document["model_input_channel"] != expected_channel:
        raise RuntimeOutputError(f"feature JSON model_input_channel must be {expected_channel}")
    if document["feature_basis"] != RAW_BASELINE_FEATURE_BASIS:
        raise RuntimeOutputError(
            f"feature_basis must be {RAW_BASELINE_FEATURE_BASIS}; scaled pilot "
            "delta_norm features are not model inputs"
        )
    if document["feature_order"] != list(inference.FEATURE_COLUMNS):
        raise RuntimeOutputError("feature_order must match the canonical five features")
    source_window_end_ms = document["source_window_end_ms"]
    if (
        isinstance(source_window_end_ms, bool)
        or not isinstance(source_window_end_ms, int)
        or source_window_end_ms < 0
    ):
        raise RuntimeOutputError("source_window_end_ms must be a non-negative integer")
    features = document["features"]
    if not isinstance(features, list):
        raise RuntimeOutputError("feature JSON features must be an array")
    return features, source_window_end_ms


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeOutputError(f"cannot hash model artifact: {error}") from error


def write_runtime_output(
    output_path: str | Path,
    payload: Any,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write one deterministic runtime JSON file."""
    validate_runtime_output(payload)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"runtime output already exists: {output}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".runtime-output-"
    ) as stage_dir:
        staged = Path(stage_dir) / output.name
        staged.write_text(encoded, encoding="utf-8", newline="")
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"runtime output already exists: {output}; pass --overwrite to replace it"
            )
        staged.replace(output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.runtime_output",
        description="Emit the canonical OSTOSENSE AI runtime output contract.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    unavailable = subparsers.add_parser(
        "unavailable", help="Emit the live no-approved-model state."
    )
    unavailable.add_argument("--output", required=True)
    unavailable.add_argument("--overwrite", action="store_true")

    predict_test = subparsers.add_parser(
        "predict-test",
        help="Emit one explicitly synthetic ENGINEERING_TEST_ONLY class.",
    )
    predict_test.add_argument("--model", required=True)
    predict_test.add_argument("--features", required=True)
    predict_test.add_argument("--output", required=True)
    predict_test.add_argument("--overwrite", action="store_true")

    unavailable_v2 = subparsers.add_parser(
        "unavailable-v2", help="Emit the v0.2 live no-prediction state."
    )
    unavailable_v2.add_argument("--output", required=True)
    unavailable_v2.add_argument("--overwrite", action="store_true")

    predict_test_v2 = subparsers.add_parser(
        "predict-test-v2", help="Emit one v0.2 synthetic contract-test class."
    )
    predict_test_v2.add_argument("--model", required=True)
    predict_test_v2.add_argument("--features", required=True)
    predict_test_v2.add_argument("--output", required=True)
    predict_test_v2.add_argument("--overwrite", action="store_true")

    predict_live = subparsers.add_parser(
        "predict-live-experimental",
        help="Emit an unvalidated class from real Kap_7 features.",
    )
    predict_live.add_argument("--model", required=True)
    predict_live.add_argument("--features", required=True)
    predict_live.add_argument("--output", required=True)
    predict_live.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "unavailable":
        payload = unavailable_output()
    elif args.command == "unavailable-v2":
        payload = unavailable_output_v2()
    elif args.command == "predict-test":
        model = _read_json_object(Path(args.model), name="model")
        feature_row = _read_feature_document(Path(args.features))
        payload = predict_test_output(model, feature_row)
    else:
        model_path = Path(args.model)
        model = _read_json_object(model_path, name="model")
        if args.command == "predict-test-v2":
            feature_row, source_window_end_ms = _read_v2_feature_document(
                Path(args.features),
                expected_data_source=SYNTHETIC_DATA_SOURCE,
                expected_channel=SYNTHETIC_INPUT_CHANNEL,
            )
            payload = predict_test_output_v2(
                model,
                feature_row,
                source_window_end_ms=source_window_end_ms,
                model_artifact_sha256=_sha256_file(model_path),
            )
        else:
            feature_row, source_window_end_ms = _read_v2_feature_document(
                Path(args.features),
                expected_data_source=REAL_SENSOR_DATA_SOURCE,
                expected_channel=KAP_7_INPUT_CHANNEL,
            )
            payload = predict_live_experimental_output(
                model,
                feature_row,
                source_window_end_ms=source_window_end_ms,
                model_artifact_sha256=_sha256_file(model_path),
            )
    write_runtime_output(args.output, payload, overwrite=args.overwrite)
    print(
        f"AI runtime output: model_status={payload['model_status']}, "
        f"prediction_available={str(payload['prediction_available']).lower()} "
        f"to {args.output}"
    )
    print(payload["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
