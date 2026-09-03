"""Deterministic ENGINEERING_TEST_ONLY ordinal-regression trainer/export (pipeline test only).

This stage fits the canonical ordinal classifier on a validated
``model_matrix.csv`` and exports a portable parameter set. It uses the
project-selected trainer only: ``sklearn.preprocessing.StandardScaler`` +
``mord.LogisticAT`` (``alpha=1.0``, ``max_iter=10000``). It computes **no
evaluation metrics** (no confusion matrix, Macro F1, kappa, accuracy, pass/fail).

The package stays import-safe without the optional ``[pipeline]`` dependencies:
NumPy / scikit-learn / mord / SciPy are imported lazily and a concise, actionable
``RuntimeError`` is raised only when they are genuinely absent (a broken partial
install surfaces its own error).

Partition policy (from the training config): fit on ``development`` rows only;
``validation`` rows may exist but never influence scaling/fitting/weighting; any
``final_test`` row rejects the whole input. There is no internal random split
and no random seed — this training operation contains no random step.

Exported forward inference uses the cumulative all-threshold formulation, matching
``mord.LogisticAT``::

    z = (x - scaler.mean) / scaler.scale
    eta = beta^T z
    P(Y <= k | x) = sigmoid(theta[k] - eta)          # k = 0, 1, 2
    P(Y = 0) = P(Y <= 0)
    P(Y = j) = P(Y <= j) - P(Y <= j-1)               # j = 1, 2
    P(Y = 3) = 1 - P(Y <= 2)
    predicted class = argmax_j P(Y = j)

A passing run proves deterministic synthetic training and parameter-export
mechanics only. The dataset must be SYNTHETIC_PIPELINE_TEST_ONLY.

CLI::

    PYTHONPATH=ai/src ai/.venv/bin/python -m ostosense_ai.training \\
        --matrix /tmp/ostosense-matrix \\
        --config ai/configs/training-v0.1.json \\
        --output /tmp/ostosense-model
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from typing import Any, TypeGuard

from ostosense_ai import inference, matrix

TRAINER_VERSION = "0.1.2"
MODEL_ARTIFACT_VERSION = "0.1.0"
DATA_CONTRACT_VERSION = matrix.DATA_CONTRACT_VERSION
RULEBOOK_VERSION = matrix.RULEBOOK_VERSION
ALLOWED_ORIGIN = matrix.ALLOWED_ORIGIN

FEATURE_COLUMNS = matrix.FEATURE_COLUMNS
CLASS_NAMES = matrix.CLASS_NAMES
CLASS_NAME_TO_INDEX = matrix.CLASS_NAME_TO_INDEX
CLASS_INDEX_TO_NAME = {index: name for name, index in CLASS_NAME_TO_INDEX.items()}
MODEL_MATRIX_COLUMNS = matrix.MODEL_MATRIX_COLUMNS

FIT_PARTITION = "development"
IGNORED_PARTITION = "validation"
FORBIDDEN_PARTITION = "final_test"
_PARTITION_VALUES = (FIT_PARTITION, IGNORED_PARTITION, FORBIDDEN_PARTITION)

_OUTPUT_FILES = ("ordinal_model.json", "training_manifest.json")
_EXPORT_DECIMALS = 12
_PROBABILITY_SUM_TOLERANCE = 1e-9
_FORWARD_PARITY_TOLERANCE = 1e-6
_CUMULATIVE_FORMULA = (
    "z=(x-mean)/scale; eta=beta^T z; P(Y<=k)=sigmoid(theta[k]-eta) for k in {0,1,2}; "
    "P(Y=0)=P(Y<=0); P(Y=j)=P(Y<=j)-P(Y<=j-1); P(Y=3)=1-P(Y<=2); class=argmax_j P(Y=j)"
)
_SIGN_CONVENTION = "eta = beta^T z; cumulative logits use (theta[k] - eta)"
_OPTIMIZER_CONVERGENCE_STATUS = (
    "mord.LogisticAT 0.7 does not expose optimizer convergence; convergence is "
    "neither claimed nor checked."
)
MODEL_WARNING = (
    "ENGINEERING_TEST_ONLY synthetic model. Trained on SYNTHETIC_PIPELINE_TEST_ONLY "
    "data to exercise training/export mechanics only; not an OSTOSENSE performance, "
    "notification-accuracy, sensor, or clinical result."
)
MANIFEST_WARNING = (
    "SYNTHETIC_PIPELINE_TEST_ONLY. This manifest records training/export mechanics "
    "only. It contains no performance metric and no pass/fail target, and is not an "
    "OSTOSENSE performance result."
)

_PIPELINE_MODULES = ("numpy", "sklearn", "mord", "scipy")
_PIPELINE_HINT = (
    "install the optional pipeline dependencies with `pip install -e .[pipeline]` "
    "(numpy, scikit-learn, mord, scipy)"
)

_CONFIG_KEYS = {
    "config_id",
    "status",
    "accepted_dataset_origin",
    "fit_partition",
    "ignored_partition",
    "forbidden_partition",
    "preprocessing",
    "model",
    "alpha",
    "max_iter",
    "sample_weighting",
    "warning",
}


class TrainingError(ValueError):
    """Training validation failure (leaves outputs untouched)."""


# --------------------------------------------------------------------------- #
# Lazy optional dependencies
# --------------------------------------------------------------------------- #
def _load_pipeline():
    """Import the optional pipeline stack lazily; actionable error only if absent."""
    missing = [name for name in _PIPELINE_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"the optional [pipeline] dependencies are required for training "
            f"(missing: {', '.join(missing)}); {_PIPELINE_HINT}"
        )
    import numpy as np
    from mord import LogisticAT
    from scipy.special import expit
    from sklearn.preprocessing import StandardScaler

    return np, StandardScaler, LogisticAT, expit


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_training_config(config: dict[str, Any]) -> None:
    """Reject missing/unknown keys, wrong types, and unsupported values."""
    if not isinstance(config, dict):
        raise TrainingError("training config must be a JSON object")
    missing = sorted(_CONFIG_KEYS - config.keys())
    if missing:
        raise TrainingError(f"training config missing keys: {missing}")
    unknown = sorted(config.keys() - _CONFIG_KEYS)
    if unknown:
        raise TrainingError(f"training config has unsupported keys: {unknown}")

    expected = {
        "config_id": "training-v0.1",
        "status": "ENGINEERING_TEST_ONLY",
        "accepted_dataset_origin": ALLOWED_ORIGIN,
        "fit_partition": FIT_PARTITION,
        "ignored_partition": IGNORED_PARTITION,
        "forbidden_partition": FORBIDDEN_PARTITION,
        "preprocessing": "sklearn.preprocessing.StandardScaler",
        "model": "mord.LogisticAT",
        "sample_weighting": "uniform_window",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise TrainingError(f"training config {key} must be {value!r}")

    alpha = config["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TrainingError("training config alpha must be numeric")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise TrainingError("training config alpha must be a positive finite number")

    max_iter = config["max_iter"]
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise TrainingError("training config max_iter must be a positive integer")

    if not isinstance(config["warning"], str) or not config["warning"]:
        raise TrainingError("training config warning must be a non-empty string")


# --------------------------------------------------------------------------- #
# Matrix input validation
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"missing {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise TrainingError(f"invalid {label} JSON: {error}") from error
    if not isinstance(data, dict):
        raise TrainingError(f"{label} must be a JSON object")
    return data


def _read_matrix_rows(csv_path: Path) -> list[dict[str, str]]:
    import csv

    if not csv_path.is_file():
        raise TrainingError(f"missing model_matrix.csv: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(MODEL_MATRIX_COLUMNS):
            raise TrainingError(f"model_matrix.csv header is not canonical: {reader.fieldnames!r}")
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise TrainingError(f"model_matrix.csv line {line_number} is malformed")
            window_id = row["window_id"]
            if not window_id:
                raise TrainingError(f"model_matrix.csv line {line_number} has an empty window_id")
            if window_id in seen:
                raise TrainingError(f"duplicate window_id in model_matrix.csv: {window_id}")
            seen.add(window_id)
            rows.append(row)
    return rows


def _validate_matrix(
    manifest: dict[str, Any], rows: list[dict[str, str]], csv_path: Path, config: dict[str, Any]
) -> None:
    expected_manifest_fields = {
        "matrix_builder_version": matrix.MATRIX_BUILDER_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "rulebook_version": RULEBOOK_VERSION,
        "target_label_column": "risk_label",
        "audit_columns": list(matrix.AUDIT_COLUMNS),
        "grouping_columns": list(matrix.GROUPING_COLUMNS),
        "partition_column": matrix.PARTITION_COLUMN,
        "window_convention": dict(matrix._CANONICAL_CONVENTION),
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise TrainingError(
                f"matrix manifest {field} must match the canonical value {expected!r}"
            )

    if manifest.get("dataset_origin") != ALLOWED_ORIGIN:
        raise TrainingError("matrix dataset_origin must be SYNTHETIC_PIPELINE_TEST_ONLY")
    if config["accepted_dataset_origin"] != manifest.get("dataset_origin"):
        raise TrainingError("config accepted_dataset_origin does not match matrix origin")

    declared_sha = manifest.get("model_matrix_sha256")
    if not isinstance(declared_sha, str) or declared_sha != _sha256_file(csv_path):
        raise TrainingError("model_matrix_sha256 does not match model_matrix.csv")

    if manifest.get("feature_columns") != list(FEATURE_COLUMNS):
        raise TrainingError("matrix feature_columns is not exactly the canonical five")
    if manifest.get("target_column") != "risk_label_index":
        raise TrainingError("matrix target_column must be risk_label_index")
    if manifest.get("class_mapping") != {name: index for name, index in CLASS_NAME_TO_INDEX.items()}:
        raise TrainingError("matrix class_mapping must be Safe=0, Monitor=1, Caution=2, Urgent=3")
    if manifest.get("class_order") != list(CLASS_NAMES):
        raise TrainingError("matrix class_order is not canonical")

    partition_by_session: dict[str, str] = {}
    partition_by_bag: dict[str, str] = {}
    partition_by_sensor: dict[str, str] = {}
    class_counts = {name: 0 for name in CLASS_NAMES}
    partition_counts: dict[str, int] = {}
    class_counts_by_partition: dict[str, dict[str, int]] = {}

    for row in rows:
        partition = row["dataset_partition"]
        if partition not in _PARTITION_VALUES:
            raise TrainingError(f"unknown dataset_partition {partition!r} for {row['window_id']}")
        for field in ("session_id", "bag_id", "sensor_id"):
            if not row[field]:
                raise TrainingError(f"{row['window_id']} has an empty {field}")
        _assign_group(partition_by_session, row["session_id"], partition, "session")
        _assign_group(partition_by_bag, row["bag_id"], partition, "bag")
        _assign_group(partition_by_sensor, row["sensor_id"], partition, "sensor")

        for column in FEATURE_COLUMNS:
            raw = row[column]
            if raw == "":
                raise TrainingError(f"{row['window_id']} has an empty {column}")
            try:
                numeric = float(raw)
            except ValueError as error:
                raise TrainingError(f"{row['window_id']} has a malformed {column}: {raw!r}") from error
            if not math.isfinite(numeric):
                raise TrainingError(f"{row['window_id']} has a non-finite {column}: {raw!r}")

        risk_label = row["risk_label"]
        risk_index = row["risk_label_index"]
        if risk_label not in CLASS_NAME_TO_INDEX:
            raise TrainingError(f"{row['window_id']} has an unknown risk_label {risk_label!r}")
        if risk_index != str(CLASS_NAME_TO_INDEX[risk_label]):
            raise TrainingError(f"{row['window_id']} risk_label/index inconsistency")

        class_counts[risk_label] += 1
        partition_counts[partition] = partition_counts.get(partition, 0) + 1
        class_counts_by_partition.setdefault(
            partition, {name: 0 for name in CLASS_NAMES}
        )
        class_counts_by_partition[partition][risk_label] += 1

    if manifest.get("eligible_row_count") != len(rows):
        raise TrainingError("matrix eligible_row_count does not reconcile with model_matrix.csv")
    if manifest.get("class_counts") != class_counts:
        raise TrainingError("matrix class_counts does not reconcile with model_matrix.csv")
    if manifest.get("partition_row_counts") != partition_counts:
        raise TrainingError("matrix partition_row_counts does not reconcile with model_matrix.csv")
    if manifest.get("class_counts_by_partition") != class_counts_by_partition:
        raise TrainingError(
            "matrix class_counts_by_partition does not reconcile with model_matrix.csv"
        )

    source_candidate_count = manifest.get("source_candidate_window_count")
    excluded_count = manifest.get("excluded_row_count")
    if not _is_nonnegative_int(source_candidate_count):
        raise TrainingError("matrix source_candidate_window_count must be a non-negative integer")
    if not _is_nonnegative_int(excluded_count):
        raise TrainingError("matrix excluded_row_count must be a non-negative integer")
    if source_candidate_count != len(rows) + excluded_count:
        raise TrainingError(
            "matrix source_candidate_window_count must equal eligible plus excluded rows"
        )

    source_partition_counts = manifest.get("source_partition_window_counts")
    if not isinstance(source_partition_counts, dict) or not source_partition_counts:
        raise TrainingError("matrix source_partition_window_counts must be a non-empty object")
    if any(
        not isinstance(partition, str)
        or partition not in _PARTITION_VALUES
        or not _is_nonnegative_int(count)
        for partition, count in source_partition_counts.items()
    ):
        raise TrainingError("matrix source_partition_window_counts is invalid")
    if sum(source_partition_counts.values()) != source_candidate_count:
        raise TrainingError(
            "matrix source_partition_window_counts do not sum to source candidates"
        )
    if source_partition_counts.get(FORBIDDEN_PARTITION, 0) > 0:
        raise TrainingError(
            "final_test candidates are present in the matrix source; refusing to train"
        )
    if manifest.get("partition_values") != sorted(source_partition_counts):
        raise TrainingError("matrix partition_values do not match source partitions")
    for partition, eligible_count in partition_counts.items():
        if source_partition_counts.get(partition, 0) < eligible_count:
            raise TrainingError(
                f"matrix source count for {partition!r} is below its eligible-row count"
            )

    exclusion_counts = manifest.get("exclusion_counts")
    if not isinstance(exclusion_counts, dict):
        raise TrainingError("matrix exclusion_counts must be an object")
    if any(
        not isinstance(reason, str)
        or not reason
        or not _is_nonnegative_int(count)
        for reason, count in exclusion_counts.items()
    ):
        raise TrainingError("matrix exclusion_counts is invalid")
    if sum(exclusion_counts.values()) != excluded_count:
        raise TrainingError("matrix exclusion_counts do not sum to excluded_row_count")


def _is_nonnegative_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _assign_group(mapping: dict[str, str], key: str, partition: str, group: str) -> None:
    existing = mapping.get(key)
    if existing is not None and existing != partition:
        raise TrainingError(
            f"partition leakage: {group} {key!r} spans partitions {existing!r} and {partition!r}"
        )
    mapping[key] = partition


# --------------------------------------------------------------------------- #
# Deterministic rounding / output
# --------------------------------------------------------------------------- #
def _round(value: float) -> float:
    rounded = round(float(value), _EXPORT_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _round_tree(obj: Any) -> Any:
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return _round(obj)
    if isinstance(obj, dict):
        return {key: _round_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_round_tree(value) for value in obj]
    return obj


def _dump_json(obj: dict[str, Any]) -> str:
    return json.dumps(
        _round_tree(obj), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"


def _check_output_targets(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    existing = [output_dir / name for name in _OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"output already contains model artifacts: {names}; "
            "pass overwrite=True or --overwrite to replace them"
        )


def _write_output_artifacts(
    output_dir: Path, model: dict[str, Any], manifest: dict[str, Any], *, overwrite: bool
) -> dict[str, Any]:
    """Stage both artifacts; hash the model file and record it in the manifest.

    The two ``replace`` calls are individually atomic but are not a single
    OS-level transaction across both files.
    """
    _check_output_targets(output_dir, overwrite=overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ostosense-training-", dir=output_dir.parent) as tmp:
        stage = Path(tmp)
        (stage / "ordinal_model.json").write_text(_dump_json(model), encoding="utf-8")
        model_sha = _sha256_file(stage / "ordinal_model.json")
        manifest = {
            **manifest,
            "output_sha256": {"ordinal_model_json": model_sha},
        }
        (stage / "training_manifest.json").write_text(_dump_json(manifest), encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            (stage / name).replace(output_dir / name)
    return manifest


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def train_ordinal_model(
    matrix_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit StandardScaler + mord.LogisticAT on development rows; export params."""
    config = load_config(config_path)
    validate_training_config(config)

    matrix_dir = Path(matrix_dir)
    csv_path = matrix_dir / "model_matrix.csv"
    manifest_path = matrix_dir / "matrix_manifest.json"
    matrix_manifest = _load_json_object(manifest_path, "matrix_manifest.json")
    rows = _read_matrix_rows(csv_path)
    _validate_matrix(matrix_manifest, rows, csv_path, config)

    development = [r for r in rows if r["dataset_partition"] == FIT_PARTITION]
    ignored = [r for r in rows if r["dataset_partition"] == IGNORED_PARTITION]
    if any(r["dataset_partition"] == FORBIDDEN_PARTITION for r in rows):
        raise TrainingError("final_test rows are present; refusing to train")
    if not development:
        raise TrainingError("no development rows to fit")
    development_classes = {int(r["risk_label_index"]) for r in development}
    if development_classes != set(CLASS_INDEX_TO_NAME):
        raise TrainingError("the development subset must contain all four classes")

    np, StandardScaler, LogisticAT, expit = _load_pipeline()

    features = np.array(
        [[float(r[column]) for column in FEATURE_COLUMNS] for r in development], dtype=np.float64
    )
    targets = np.array([int(r["risk_label_index"]) for r in development], dtype=np.int64)

    scaler = StandardScaler().fit(features)  # development rows only
    scaled = scaler.transform(features)
    model = LogisticAT(alpha=float(config["alpha"]), max_iter=int(config["max_iter"]))
    model.fit(scaled, targets)  # uniform per-window weighting (no custom sample_weight)

    beta = np.asarray(model.coef_, dtype=np.float64).ravel()
    theta = np.asarray(model.theta_, dtype=np.float64).ravel()
    mean = np.asarray(scaler.mean_, dtype=np.float64).ravel()
    scale = np.asarray(scaler.scale_, dtype=np.float64).ravel()
    if beta.shape != (5,) or theta.shape != (3,) or mean.shape != (5,) or scale.shape != (5,):
        raise TrainingError("unexpected fitted parameter shapes")

    export_mean = [_round(v) for v in mean]
    export_scale = [_round(v) for v in scale]
    export_beta = [_round(v) for v in beta]
    export_theta = [_round(v) for v in theta]

    model_artifact = {
        "model_artifact_version": MODEL_ARTIFACT_VERSION,
        "model_family": "mord.LogisticAT",
        "dataset_origin": ALLOWED_ORIGIN,
        "feature_order": list(FEATURE_COLUMNS),
        "class_order": list(CLASS_NAMES),
        "class_mapping": dict(CLASS_NAME_TO_INDEX),
        "scaler": {"mean": export_mean, "scale": export_scale},
        "beta": export_beta,
        "theta": export_theta,
        "cumulative_probability_formula": _CUMULATIVE_FORMULA,
        "sign_convention": _SIGN_CONVENTION,
        "hyperparameters": {
            "preprocessing": "sklearn.preprocessing.StandardScaler",
            "model": "mord.LogisticAT",
            "alpha": float(config["alpha"]),
            "max_iter": int(config["max_iter"]),
            "sample_weighting": "uniform_window",
        },
        "warning": MODEL_WARNING,
    }

    mord_proba = np.asarray(model.predict_proba(scaled), dtype=np.float64)
    mord_pred = np.asarray(model.predict(scaled))

    # Single canonical forward reference: the exported model via ostosense_ai.inference.
    forward = inference.predict_exported_model(model_artifact, features.tolist())
    forward_proba = np.asarray(forward["probabilities"], dtype=np.float64)
    forward_pred = np.asarray(forward["predicted_indices"])

    sanity = _model_sanity(
        np, export_mean, export_scale, export_beta, export_theta,
        mord_proba, forward_proba, mord_pred, forward_pred,
    )

    dependency_versions = {
        "numpy": importlib.metadata.version("numpy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "mord": importlib.metadata.version("mord"),
        "scipy": importlib.metadata.version("scipy"),
    }

    fitted_class_counts = {name: 0 for name in CLASS_NAMES}
    for row in development:
        fitted_class_counts[row["risk_label"]] += 1

    manifest = {
        "trainer_version": TRAINER_VERSION,
        "config_id": config["config_id"],
        "model_artifact_version": MODEL_ARTIFACT_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "rulebook_version": RULEBOOK_VERSION,
        "dataset_origin": ALLOWED_ORIGIN,
        "input_sha256": {
            "training_config_json": _sha256_file(Path(config_path)),
            "model_matrix_csv": _sha256_file(csv_path),
            "matrix_manifest_json": _sha256_file(manifest_path),
        },
        "dependency_versions": dependency_versions,
        "fit_partition_policy": {
            "fit": FIT_PARTITION,
            "ignored": IGNORED_PARTITION,
            "forbidden": FORBIDDEN_PARTITION,
        },
        "source_row_count": len(rows),
        "fitted_row_count": len(development),
        "ignored_row_count": len(ignored),
        "fitted_class_counts": fitted_class_counts,
        "fitted_session_count": len({r["session_id"] for r in development}),
        "fitted_bag_count": len({r["bag_id"] for r in development}),
        "fitted_sensor_count": len({r["sensor_id"] for r in development}),
        "preprocessing": "sklearn.preprocessing.StandardScaler",
        "sample_weighting": "uniform_window",
        "model_sanity": sanity,
        "optimizer_convergence_status": _OPTIMIZER_CONVERGENCE_STATUS,
        "warning": MANIFEST_WARNING,
    }

    return _write_output_artifacts(Path(output_dir), model_artifact, manifest, overwrite=overwrite)


def _model_sanity(
    np, export_mean, export_scale, export_beta, export_theta,
    mord_proba, forward_proba, mord_pred, forward_pred,
) -> dict[str, Any]:
    mean_finite = all(math.isfinite(v) for v in export_mean)
    scale_finite = all(math.isfinite(v) for v in export_scale)
    scale_positive = all(v > 0.0 for v in export_scale)
    beta_finite = sum(1 for v in export_beta if math.isfinite(v))
    theta_finite = sum(1 for v in export_theta if math.isfinite(v))
    theta_increasing = all(
        export_theta[i] < export_theta[i + 1]
        for i in range(len(export_theta) - 1)
    )

    expected_shape = (len(mord_pred), len(CLASS_NAMES))
    reference_shape_valid = tuple(mord_proba.shape) == expected_shape
    forward_shape_valid = tuple(forward_proba.shape) == expected_shape

    reference_finite = bool(np.all(np.isfinite(mord_proba)))
    reference_within_unit = bool(
        np.all(mord_proba >= -1e-12) and np.all(mord_proba <= 1.0 + 1e-12)
    )
    reference_sum_error = (
        float(np.max(np.abs(mord_proba.sum(axis=1) - 1.0)))
        if reference_shape_valid and reference_finite
        else math.inf
    )

    forward_finite = bool(np.all(np.isfinite(forward_proba)))
    forward_within_unit = bool(
        np.all(forward_proba >= -1e-12)
        and np.all(forward_proba <= 1.0 + 1e-12)
    )
    forward_sum_error = (
        float(np.max(np.abs(forward_proba.sum(axis=1) - 1.0)))
        if forward_shape_valid and forward_finite
        else math.inf
    )
    forward_max_diff = (
        float(np.max(np.abs(forward_proba - mord_proba)))
        if reference_shape_valid
        and forward_shape_valid
        and reference_finite
        and forward_finite
        else math.inf
    )
    label_parity = bool(np.array_equal(forward_pred, mord_pred))

    if not (mean_finite and scale_finite and scale_positive):
        raise TrainingError("scaler parameters are not finite/positive")
    if beta_finite != 5 or theta_finite != 3:
        raise TrainingError("beta/theta do not have the required finite counts")
    if not theta_increasing:
        raise TrainingError("theta is not strictly increasing")
    if not (reference_shape_valid and forward_shape_valid):
        raise TrainingError("reference/exported probability arrays must have four columns")
    if not (reference_finite and reference_within_unit):
        raise TrainingError("mord reference probabilities are not finite within [0, 1]")
    if reference_sum_error > _PROBABILITY_SUM_TOLERANCE:
        raise TrainingError("mord reference probabilities do not sum to 1 within tolerance")
    if not (forward_finite and forward_within_unit):
        raise TrainingError("exported forward probabilities are not finite within [0, 1]")
    if forward_sum_error > _PROBABILITY_SUM_TOLERANCE:
        raise TrainingError("exported forward probabilities do not sum to 1 within tolerance")
    if not math.isfinite(forward_max_diff):
        raise TrainingError("exported forward probability difference is not finite")
    if forward_max_diff > _FORWARD_PARITY_TOLERANCE:
        raise TrainingError("exported forward inference does not reproduce mord probabilities")
    if not label_parity:
        raise TrainingError("exported forward inference does not reproduce mord predicted labels")

    return {
        "scaler_mean_finite": mean_finite,
        "scaler_scale_finite": scale_finite,
        "scaler_scale_positive": scale_positive,
        "beta_finite_count": beta_finite,
        "theta_finite_count": theta_finite,
        "theta_strictly_increasing": theta_increasing,
        "reference_probability_shape_valid": reference_shape_valid,
        "reference_probabilities_finite": reference_finite,
        "reference_probabilities_within_unit_interval": reference_within_unit,
        "reference_max_probability_sum_error": reference_sum_error,
        "exported_probability_shape_valid": forward_shape_valid,
        "exported_probabilities_finite": forward_finite,
        "exported_probabilities_within_unit_interval": forward_within_unit,
        "exported_max_probability_sum_error": forward_sum_error,
        "probability_sum_tolerance": _PROBABILITY_SUM_TOLERANCE,
        "forward_max_probability_difference": forward_max_diff,
        "forward_parity_tolerance": _FORWARD_PARITY_TOLERANCE,
        "forward_label_parity": label_parity,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.training",
        description="Deterministic ENGINEERING_TEST_ONLY ordinal-regression trainer (pipeline test only).",
    )
    parser.add_argument("--matrix", required=True, help="Matrix artifact directory.")
    parser.add_argument("--config", required=True, help="Training config JSON path.")
    parser.add_argument("--output", required=True, help="Output directory for model artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing model artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = train_ordinal_model(args.matrix, args.config, args.output, overwrite=args.overwrite)
    print(
        f"training: fitted {manifest['fitted_row_count']} development rows "
        f"(ignored {manifest['ignored_row_count']}) to {args.output}"
    )
    print(f"fitted_class_counts: {manifest['fitted_class_counts']}")
    print(MANIFEST_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
