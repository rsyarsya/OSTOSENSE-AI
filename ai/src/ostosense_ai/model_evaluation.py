"""Deterministic grouped-validation evaluation of the exported model (pipeline test only).

This stage predicts the matrix ``validation`` rows with the exported
``ordinal_model.json`` (via the canonical ``ostosense_ai.inference`` forward) and
scores them with the canonical ``ostosense_ai.evaluation.evaluate_predictions``.
It **only** touches the validation partition; it never computes development or
training metrics, never creates splits, never uses or exposes Final Test, and
emits no project-target / pass-fail / notification / lead-time / event-level /
firmware / clinical fields.

It reuses the canonical constants, the trainer's matrix validation, the exported
inference, and the metrics evaluator — nothing is redefined. Metrics live nested
under ``pipeline_mechanics_metrics`` with an explicit synthetic-only warning.

A passing run proves grouped synthetic validation, exported-inference, and
metric-pipeline mechanics only, never OSTOSENSE accuracy, notification accuracy,
early-warning performance, sensor validity, firmware parity, or clinical value.

CLI::

    PYTHONPATH=ai/src ai/.venv/bin/python -m ostosense_ai.model_evaluation \\
        --matrix /tmp/ostosense-matrix \\
        --model /tmp/ostosense-model \\
        --output /tmp/ostosense-validation
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from pathlib import Path
from typing import Any

from ostosense_ai import evaluation, inference, matrix, training

EVALUATOR_VERSION = "0.1.1"
ALLOWED_ORIGIN = matrix.ALLOWED_ORIGIN
FEATURE_COLUMNS = matrix.FEATURE_COLUMNS
CLASS_NAMES = matrix.CLASS_NAMES
CLASS_NAME_TO_INDEX = matrix.CLASS_NAME_TO_INDEX

VALIDATION_PARTITION = "validation"
DEVELOPMENT_PARTITION = "development"
FINAL_TEST_PARTITION = "final_test"
EVALUATION_SCOPE = (
    "pipeline-mechanics grouped-validation evaluation of the exported synthetic "
    "model only"
)
NO_PERFORMANCE_WARNING = (
    "SYNTHETIC_PIPELINE_TEST_ONLY. These grouped-validation metrics are "
    "pipeline-mechanics fixtures for the exported synthetic model. They are not "
    "OSTOSENSE accuracy, do not satisfy any project target, and must never be "
    "presented as real model, notification, early-warning, sensor, firmware, or "
    "clinical performance."
)

PREDICTIONS_COLUMNS = (
    "window_id",
    "session_id",
    "bag_id",
    "sensor_id",
    "dataset_partition",
    "ground_truth_label",
    "ground_truth_index",
    "predicted_label",
    "predicted_index",
    "prob_safe",
    "prob_monitor",
    "prob_caution",
    "prob_urgent",
)
_OUTPUT_FILES = ("validation_predictions.csv", "validation_evaluation.json")
_EXPECTED_FIT_POLICY = {
    "fit": DEVELOPMENT_PARTITION,
    "ignored": VALIDATION_PARTITION,
    "forbidden": FINAL_TEST_PARTITION,
}
_EXPECTED_CONFIG_ID = "training-v0.1"
_EXPECTED_PREPROCESSING = "sklearn.preprocessing.StandardScaler"
_EXPECTED_SAMPLE_WEIGHTING = "uniform_window"
_EXPECTED_DEPENDENCIES = {"numpy", "scikit-learn", "mord", "scipy"}
_EXPECTED_INPUT_HASHES = {
    "training_config_json",
    "model_matrix_csv",
    "matrix_manifest_json",
}
_SANITY_TRUE_FIELDS = {
    "scaler_mean_finite",
    "scaler_scale_finite",
    "scaler_scale_positive",
    "theta_strictly_increasing",
    "reference_probability_shape_valid",
    "reference_probabilities_finite",
    "reference_probabilities_within_unit_interval",
    "exported_probability_shape_valid",
    "exported_probabilities_finite",
    "exported_probabilities_within_unit_interval",
    "forward_label_parity",
}


class ModelEvaluationError(ValueError):
    """Validation-evaluation failure (leaves outputs untouched)."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelEvaluationError(f"training manifest {name} must be a non-negative integer")
    return value


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelEvaluationError(f"training manifest {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ModelEvaluationError(f"training manifest {name} must be finite")
    return result


def _validate_training_manifest(
    manifest: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    model_sha: str,
    matrix_csv_sha: str,
    matrix_manifest_sha: str,
) -> None:
    """Validate that model provenance is a complete canonical trainer output."""
    expected_scalars = {
        "trainer_version": training.TRAINER_VERSION,
        "config_id": _EXPECTED_CONFIG_ID,
        "model_artifact_version": training.MODEL_ARTIFACT_VERSION,
        "data_contract_version": matrix.DATA_CONTRACT_VERSION,
        "rulebook_version": matrix.RULEBOOK_VERSION,
        "dataset_origin": ALLOWED_ORIGIN,
        "preprocessing": _EXPECTED_PREPROCESSING,
        "sample_weighting": _EXPECTED_SAMPLE_WEIGHTING,
        "optimizer_convergence_status": training._OPTIMIZER_CONVERGENCE_STATUS,
        "warning": training.MANIFEST_WARNING,
    }
    for field, expected in expected_scalars.items():
        if manifest.get(field) != expected:
            raise ModelEvaluationError(
                f"training manifest {field} must match canonical trainer output"
            )
    if manifest.get("fit_partition_policy") != _EXPECTED_FIT_POLICY:
        raise ModelEvaluationError("training fit_partition_policy is not development-only")

    input_hashes = manifest.get("input_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != _EXPECTED_INPUT_HASHES:
        raise ModelEvaluationError("training input_sha256 schema is not canonical")
    if not all(_is_sha256(value) for value in input_hashes.values()):
        raise ModelEvaluationError("training input_sha256 values must be lowercase SHA-256")
    if input_hashes["model_matrix_csv"] != matrix_csv_sha:
        raise ModelEvaluationError("training was not fitted from this model_matrix.csv")
    if input_hashes["matrix_manifest_json"] != matrix_manifest_sha:
        raise ModelEvaluationError("training was not fitted from this matrix_manifest.json")

    output_hashes = manifest.get("output_sha256")
    if (
        not isinstance(output_hashes, dict)
        or set(output_hashes) != {"ordinal_model_json"}
        or output_hashes["ordinal_model_json"] != model_sha
    ):
        raise ModelEvaluationError("ordinal_model.json hash does not match training output_sha256")

    dependency_versions = manifest.get("dependency_versions")
    if (
        not isinstance(dependency_versions, dict)
        or set(dependency_versions) != _EXPECTED_DEPENDENCIES
        or any(not isinstance(value, str) or not value for value in dependency_versions.values())
    ):
        raise ModelEvaluationError("training dependency_versions schema is not canonical")

    development = [
        row for row in rows if row["dataset_partition"] == DEVELOPMENT_PARTITION
    ]
    ignored = [
        row for row in rows if row["dataset_partition"] == VALIDATION_PARTITION
    ]
    expected_counts = {
        "source_row_count": len(rows),
        "fitted_row_count": len(development),
        "ignored_row_count": len(ignored),
        "fitted_session_count": len({row["session_id"] for row in development}),
        "fitted_bag_count": len({row["bag_id"] for row in development}),
        "fitted_sensor_count": len({row["sensor_id"] for row in development}),
    }
    for field, expected in expected_counts.items():
        if _nonnegative_int(field, manifest.get(field)) != expected:
            raise ModelEvaluationError(f"training manifest {field} does not reconcile")

    expected_class_counts = {name: 0 for name in CLASS_NAMES}
    for row in development:
        expected_class_counts[row["risk_label"]] += 1
    class_counts = manifest.get("fitted_class_counts")
    if not isinstance(class_counts, dict) or set(class_counts) != set(CLASS_NAMES):
        raise ModelEvaluationError("training fitted_class_counts schema is not canonical")
    for name in CLASS_NAMES:
        _nonnegative_int(f"fitted_class_counts.{name}", class_counts[name])
    if class_counts != expected_class_counts:
        raise ModelEvaluationError("training fitted_class_counts do not reconcile")

    sanity = manifest.get("model_sanity")
    if not isinstance(sanity, dict):
        raise ModelEvaluationError("training model_sanity must be an object")
    if any(sanity.get(field) is not True for field in _SANITY_TRUE_FIELDS):
        raise ModelEvaluationError("training model_sanity contains a failed invariant")
    if sanity.get("beta_finite_count") != 5 or sanity.get("theta_finite_count") != 3:
        raise ModelEvaluationError("training model_sanity parameter counts are invalid")

    probability_tolerance = _finite_number(
        "model_sanity.probability_sum_tolerance",
        sanity.get("probability_sum_tolerance"),
    )
    parity_tolerance = _finite_number(
        "model_sanity.forward_parity_tolerance",
        sanity.get("forward_parity_tolerance"),
    )
    if probability_tolerance != training._PROBABILITY_SUM_TOLERANCE:
        raise ModelEvaluationError("training probability-sum tolerance is not canonical")
    if parity_tolerance != training._FORWARD_PARITY_TOLERANCE:
        raise ModelEvaluationError("training forward-parity tolerance is not canonical")
    for field in (
        "reference_max_probability_sum_error",
        "exported_max_probability_sum_error",
    ):
        if _finite_number(f"model_sanity.{field}", sanity.get(field)) > probability_tolerance:
            raise ModelEvaluationError(f"training model_sanity {field} exceeds tolerance")
    if (
        _finite_number(
            "model_sanity.forward_max_probability_difference",
            sanity.get("forward_max_probability_difference"),
        )
        > parity_tolerance
    ):
        raise ModelEvaluationError(
            "training model_sanity forward probability difference exceeds tolerance"
        )


def _prob_text(value: float) -> str:
    return str(training._round(value))


def _check_output_targets(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    existing = [output_dir / name for name in _OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"output already contains validation artifacts: {names}; "
            "pass overwrite=True or --overwrite to replace them"
        )


def _write_predictions_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTIONS_COLUMNS)
        writer.writerows(rows)


def _write_output_artifacts(
    output_dir: Path, prediction_rows: list[list[str]], manifest: dict[str, Any], *, overwrite: bool
) -> dict[str, Any]:
    """Stage both artifacts; hash the predictions CSV and record it in the manifest.

    The two ``replace`` calls are individually atomic but are not a single
    OS-level transaction across both files.
    """
    _check_output_targets(output_dir, overwrite=overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ostosense-validation-", dir=output_dir.parent) as tmp:
        stage = Path(tmp)
        _write_predictions_csv(stage / "validation_predictions.csv", prediction_rows)
        csv_sha = training._sha256_file(stage / "validation_predictions.csv")
        manifest = {
            **manifest,
            "output_sha256": {"validation_predictions_csv": csv_sha},
        }
        (stage / "validation_evaluation.json").write_text(
            training._dump_json(manifest), encoding="utf-8"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            (stage / name).replace(output_dir / name)
    return manifest


def evaluate_validation_partition(
    matrix_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate the exported model on the matrix validation partition; return manifest."""
    matrix_dir = Path(matrix_dir)
    model_dir = Path(model_dir)
    csv_path = matrix_dir / "model_matrix.csv"
    matrix_manifest_path = matrix_dir / "matrix_manifest.json"
    model_path = model_dir / "ordinal_model.json"
    training_manifest_path = model_dir / "training_manifest.json"

    # ---- validate everything BEFORE predicting/writing ----
    matrix_manifest = _load_json(matrix_manifest_path)
    # Reuse the trainer's canonical matrix validation (incl. SHA, versions, origin,
    # feature/target/class, count reconciliation, and group separation); surface its
    # failures as ModelEvaluationError.
    try:
        rows = training._read_matrix_rows(csv_path)
        training._validate_matrix(
            matrix_manifest, rows, csv_path, {"accepted_dataset_origin": ALLOWED_ORIGIN}
        )
    except training.TrainingError as error:
        raise ModelEvaluationError(str(error)) from error

    # No Final Test may exist as a row or as a source-candidate partition.
    if any(row["dataset_partition"] == FINAL_TEST_PARTITION for row in rows):
        raise ModelEvaluationError("final_test rows are present; refusing to evaluate")
    if FINAL_TEST_PARTITION in matrix_manifest.get("partition_values", []):
        raise ModelEvaluationError("matrix declares a final_test partition; refusing to evaluate")
    if matrix_manifest.get("source_partition_window_counts", {}).get(FINAL_TEST_PARTITION):
        raise ModelEvaluationError("matrix source partitions include final_test; refusing to evaluate")

    model_artifact = _load_json(model_path)
    training_manifest = _load_json(training_manifest_path)

    # The exported model must be the complete canonical output trained from THIS matrix.
    model_sha = training._sha256_file(model_path)
    _validate_training_manifest(
        training_manifest,
        rows,
        model_sha=model_sha,
        matrix_csv_sha=training._sha256_file(csv_path),
        matrix_manifest_sha=training._sha256_file(matrix_manifest_path),
    )

    # Validate the exported model artifact (feature/class order + parameters).
    try:
        inference.validate_model_artifact(model_artifact)
    except inference.InferenceError as error:
        raise ModelEvaluationError(f"invalid exported model: {error}") from error

    validation_rows = [r for r in rows if r["dataset_partition"] == VALIDATION_PARTITION]
    if not validation_rows:
        raise ModelEvaluationError("no eligible validation rows to evaluate")

    feature_rows = [[float(r[column]) for column in FEATURE_COLUMNS] for r in validation_rows]
    try:
        prediction = inference.predict_exported_model(model_artifact, feature_rows)
    except inference.InferenceError as error:
        raise ModelEvaluationError(f"exported inference failed: {error}") from error

    ground_truth_indices = [int(r["risk_label_index"]) for r in validation_rows]
    try:
        metrics = evaluation.evaluate_predictions(
            ground_truth_indices, prediction["predicted_indices"]
        )
    except ValueError as error:
        raise ModelEvaluationError(f"validation metrics are undefined: {error}") from error

    prediction_rows: list[list[str]] = []
    support = {name: 0 for name in CLASS_NAMES}
    for row, probabilities, predicted_index, predicted_label in zip(
        validation_rows,
        prediction["probabilities"],
        prediction["predicted_indices"],
        prediction["predicted_labels"],
    ):
        support[row["risk_label"]] += 1
        prediction_rows.append(
            [
                row["window_id"],
                row["session_id"],
                row["bag_id"],
                row["sensor_id"],
                row["dataset_partition"],
                row["risk_label"],
                row["risk_label_index"],
                predicted_label,
                str(predicted_index),
                *[_prob_text(p) for p in probabilities],
            ]
        )

    manifest = {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_scope": EVALUATION_SCOPE,
        "dataset_origin": ALLOWED_ORIGIN,
        "evaluated_partition": VALIDATION_PARTITION,
        "input_sha256": {
            "model_matrix_csv": training._sha256_file(csv_path),
            "matrix_manifest_json": training._sha256_file(matrix_manifest_path),
            "ordinal_model_json": model_sha,
            "training_manifest_json": training._sha256_file(training_manifest_path),
        },
        "feature_order": list(FEATURE_COLUMNS),
        "class_order": list(CLASS_NAMES),
        "validation_row_count": len(validation_rows),
        "validation_session_count": len({r["session_id"] for r in validation_rows}),
        "validation_bag_count": len({r["bag_id"] for r in validation_rows}),
        "validation_sensor_count": len({r["sensor_id"] for r in validation_rows}),
        "validation_ground_truth_support": support,
        "confusion_matrix_orientation": evaluation.CONFUSION_MATRIX_ORIENTATION,
        "pipeline_mechanics_metrics": metrics,
        "warning": NO_PERFORMANCE_WARNING,
    }

    return _write_output_artifacts(Path(output_dir), prediction_rows, manifest, overwrite=overwrite)


def _load_json(path: Path) -> dict[str, Any]:
    import json

    if not path.is_file():
        raise ModelEvaluationError(f"missing required input: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ModelEvaluationError(f"invalid JSON in {path.name}: {error}") from error
    if not isinstance(data, dict):
        raise ModelEvaluationError(f"{path.name} must be a JSON object")
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.model_evaluation",
        description="Deterministic grouped-validation evaluation of the exported model (pipeline test only).",
    )
    parser.add_argument("--matrix", required=True, help="Matrix artifact directory.")
    parser.add_argument("--model", required=True, help="Trained model artifact directory.")
    parser.add_argument("--output", required=True, help="Output directory for validation artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing validation artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = evaluate_validation_partition(args.matrix, args.model, args.output, overwrite=args.overwrite)
    print(
        f"validation: scored {manifest['validation_row_count']} rows "
        f"({manifest['validation_session_count']} sessions) to {args.output}"
    )
    print(f"support: {manifest['validation_ground_truth_support']}")
    print("synthetic fixture metrics were written to validation_evaluation.json")
    print(NO_PERFORMANCE_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
