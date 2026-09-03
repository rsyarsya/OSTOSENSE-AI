"""Canonical AI output for downstream software integration.

The current repository has no approved real-data model. Consequently, this
module exposes either an unavailable live state or an explicitly synthetic
engineering-test class. It never converts model output into a leakage
percentage, countdown, LIG state, bag-fill estimate, or clinical action.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ostosense_ai import inference

RUNTIME_OUTPUT_VERSION = "0.1.0"

LIVE_MODE = "LIVE"
ENGINEERING_TEST_MODE = "ENGINEERING_TEST"
NO_DATA_SOURCE = "NONE"
SYNTHETIC_DATA_SOURCE = "SYNTHETIC_FIXTURE"
UNAVAILABLE_STATUS = "UNAVAILABLE"
TEST_ONLY_STATUS = "TEST_ONLY"

NO_PREDICTION_SCOPE = "NO_PREDICTION"
PIPELINE_MECHANICS_SCOPE = "PIPELINE_MECHANICS_ONLY"

LIVE_UNAVAILABLE_WARNING = (
    "No approved real-data model is available; no OSTOSENSE risk class was produced."
)
ENGINEERING_TEST_WARNING = (
    "ENGINEERING_TEST_ONLY synthetic result; not a patient-risk assessment or clinical output."
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


class RuntimeOutputError(ValueError):
    """Runtime output or its inference inputs violate the integration contract."""


def _require_exact_fields(payload: dict[str, Any]) -> None:
    expected = set(OUTPUT_FIELDS)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeOutputError(
            f"runtime output fields differ from the contract; missing={missing}, extra={extra}"
        )


def validate_runtime_output(payload: Any) -> dict[str, Any]:
    """Validate and return a runtime payload without mutating it."""
    if not isinstance(payload, dict):
        raise RuntimeOutputError("runtime output must be a JSON object")
    _require_exact_fields(payload)

    if payload["runtime_output_version"] != RUNTIME_OUTPUT_VERSION:
        raise RuntimeOutputError(
            f"runtime_output_version must be {RUNTIME_OUTPUT_VERSION}"
        )
    if not isinstance(payload["warning"], str) or not payload["warning"].strip():
        raise RuntimeOutputError("warning must be a non-empty string")
    if not isinstance(payload["prediction_available"], bool):
        raise RuntimeOutputError("prediction_available must be boolean")

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
        risk_class = payload["risk_class"]
        risk_index = payload["risk_class_index"]
        if risk_class not in inference.CLASS_NAME_TO_INDEX:
            raise RuntimeOutputError("risk_class must be Safe, Monitor, Caution, or Urgent")
        if isinstance(risk_index, bool) or not isinstance(risk_index, int):
            raise RuntimeOutputError("risk_class_index must be an integer")
        if risk_index != inference.CLASS_NAME_TO_INDEX[risk_class]:
            raise RuntimeOutputError("risk_class and risk_class_index are inconsistent")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "unavailable":
        payload = unavailable_output()
    else:
        model = _read_json_object(Path(args.model), name="model")
        feature_row = _read_feature_document(Path(args.features))
        payload = predict_test_output(model, feature_row)
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
