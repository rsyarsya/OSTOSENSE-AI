"""Canonical standard-library inference for the exported ordinal model (pipeline test only).

This is the *single* forward reference for the exported `ordinal_model.json`
artifact: given the model and rows of five capacitive features, it reproduces the
`mord.LogisticAT` cumulative all-threshold probabilities and predicted classes
without importing NumPy, scikit-learn, mord, or SciPy. The trainer's parity check
and the validation evaluator both call this function, so there is exactly one
forward formula.

Formula (matches mord.LogisticAT)::

    z = (x - scaler.mean) / scaler.scale
    eta = beta^T z
    P(Y <= k) = sigmoid(theta[k] - eta)            # k = 0, 1, 2
    P(Y = 0) = P(Y <= 0)
    P(Y = j) = P(Y <= j) - P(Y <= j-1)             # j = 1, 2
    P(Y = 3) = 1 - P(Y <= 2)
    predicted class = argmax_j P(Y = j)            # ties -> lowest index (NumPy/mord)

A passing run proves exported-inference mechanics only, never OSTOSENSE accuracy,
sensor validity, firmware parity, or clinical value.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ostosense_ai import matrix

INFERENCE_VERSION = "0.1.0"
EXPECTED_MODEL_ARTIFACT_VERSION = "0.1.0"
ALLOWED_ORIGIN = matrix.ALLOWED_ORIGIN
FEATURE_COLUMNS = matrix.FEATURE_COLUMNS
CLASS_NAMES = matrix.CLASS_NAMES
CLASS_NAME_TO_INDEX = matrix.CLASS_NAME_TO_INDEX
CLASS_INDEX_TO_NAME = {index: name for name, index in CLASS_NAME_TO_INDEX.items()}

# Documented numerical tolerances for the exported forward pass.
PROBABILITY_SUM_TOLERANCE = 1e-9
PROBABILITY_RANGE_TOLERANCE = 1e-9


class InferenceError(ValueError):
    """Model artifact or feature-row is not usable for exported inference."""


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceError(f"{name} must be a finite number (got {type(value).__name__})")
    result = float(value)
    if not math.isfinite(result):
        raise InferenceError(f"{name} must be finite: {value!r}")
    return result


def _finite_vector(name: str, value: Any, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise InferenceError(f"{name} must be a length-{length} sequence")
    return [_finite_number(f"{name}[{i}]", v) for i, v in enumerate(value)]


def _stable_sigmoid(x: float) -> float:
    # Overflow-free logistic sigmoid.
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def validate_model_artifact(model: Any) -> dict[str, list[float]]:
    """Validate the exported model artifact; return its parsed numeric parameters."""
    if not isinstance(model, dict):
        raise InferenceError("model artifact must be a JSON object")
    if model.get("model_artifact_version") != EXPECTED_MODEL_ARTIFACT_VERSION:
        raise InferenceError(
            f"model_artifact_version must be {EXPECTED_MODEL_ARTIFACT_VERSION}"
        )
    if model.get("model_family") != "mord.LogisticAT":
        raise InferenceError("model_family must be mord.LogisticAT")
    if model.get("dataset_origin") != ALLOWED_ORIGIN:
        raise InferenceError("model dataset_origin must be SYNTHETIC_PIPELINE_TEST_ONLY")
    if model.get("feature_order") != list(FEATURE_COLUMNS):
        raise InferenceError("model feature_order is not the canonical five")
    if model.get("class_order") != list(CLASS_NAMES):
        raise InferenceError("model class_order is not canonical")
    if model.get("class_mapping") != {name: index for name, index in CLASS_NAME_TO_INDEX.items()}:
        raise InferenceError("model class_mapping must be Safe=0, Monitor=1, Caution=2, Urgent=3")

    scaler = model.get("scaler")
    if not isinstance(scaler, dict):
        raise InferenceError("model scaler must be an object with mean/scale")
    mean = _finite_vector("scaler.mean", scaler.get("mean"), 5)
    scale = _finite_vector("scaler.scale", scaler.get("scale"), 5)
    if any(s <= 0.0 for s in scale):
        raise InferenceError("scaler.scale values must be positive")
    beta = _finite_vector("beta", model.get("beta"), 5)
    theta = _finite_vector("theta", model.get("theta"), 3)
    if not all(theta[i] < theta[i + 1] for i in range(len(theta) - 1)):
        raise InferenceError("theta must be strictly increasing")

    return {"mean": mean, "scale": scale, "beta": beta, "theta": theta}


def _row_probabilities(x: list[float], params: dict[str, list[float]]) -> list[float]:
    mean, scale, beta, theta = params["mean"], params["scale"], params["beta"], params["theta"]
    z = [(x[i] - mean[i]) / scale[i] for i in range(5)]
    eta = math.fsum(beta[i] * z[i] for i in range(5))
    cumulative = [_stable_sigmoid(theta[k] - eta) for k in range(3)]
    probabilities = [
        cumulative[0],
        cumulative[1] - cumulative[0],
        cumulative[2] - cumulative[1],
        1.0 - cumulative[2],
    ]
    low, high = -PROBABILITY_RANGE_TOLERANCE, 1.0 + PROBABILITY_RANGE_TOLERANCE
    for index, probability in enumerate(probabilities):
        if not math.isfinite(probability) or probability < low or probability > high:
            raise InferenceError(
                f"class {index} probability {probability!r} is outside [0, 1] beyond tolerance"
            )
    if abs(math.fsum(probabilities) - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise InferenceError("class probabilities do not sum to 1 within tolerance")
    return probabilities


def _argmax_lowest(probabilities: Sequence[float]) -> int:
    best = 0
    for index in range(1, len(probabilities)):
        if probabilities[index] > probabilities[best]:  # strict -> ties keep lowest index
            best = index
    return best


def predict_exported_model(
    model_artifact: Any, feature_rows: Sequence[Sequence[Any]]
) -> dict[str, Any]:
    """Return per-row probabilities, predicted indices, and predicted labels."""
    params = validate_model_artifact(model_artifact)
    probabilities: list[list[float]] = []
    predicted_indices: list[int] = []
    predicted_labels: list[str] = []
    for position, row in enumerate(feature_rows):
        try:
            values = list(row)
        except TypeError as error:
            raise InferenceError(f"feature row {position} must be an iterable") from error
        if len(values) != 5:
            raise InferenceError(f"feature row {position} must have exactly five values")
        x = [_finite_number(f"feature row {position}[{i}]", v) for i, v in enumerate(values)]
        row_probabilities = _row_probabilities(x, params)
        index = _argmax_lowest(row_probabilities)
        probabilities.append(row_probabilities)
        predicted_indices.append(index)
        predicted_labels.append(CLASS_INDEX_TO_NAME[index])
    return {
        "probabilities": probabilities,
        "predicted_indices": predicted_indices,
        "predicted_labels": predicted_labels,
    }
