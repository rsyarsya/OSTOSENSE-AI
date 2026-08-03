"""Deterministic, leakage-safe feature-label matrix builder (pipeline test only).

This module joins the canonical feature artifacts (`features.csv` +
`feature_manifest.json`) with the canonical label artifacts (`labels.csv` +
`label_manifest.json`) into a single `model_matrix.csv` plus a deterministic
`matrix_manifest.json`, ready to hand to a trainer in a later batch. It performs
**no training, no metrics, no splitting** — it only stitches validated rows
together and preserves the `dataset_partition` produced by labeling.

It is dependency-free (standard library + `ostosense_contract`) and reuses the
canonical feature/class constants from `features.py` and `labeling.py`; it never
redefines a feature or class. Only rows that are both `feature_valid` and
`label_valid` become matrix rows. Feature values never leak into the audit or
grouping columns, and audit/grouping metadata never enters the feature allowlist.

A passing run proves matrix-construction mechanics only, never AI accuracy,
notification accuracy, early-warning performance, sensor validity, or clinical
value. The dataset must be SYNTHETIC_PIPELINE_TEST_ONLY.

CLI::

    PYTHONPATH=ai/src ai/.venv/bin/python -m ostosense_ai.matrix \\
        --features /tmp/ostosense-features \\
        --labels /tmp/ostosense-labels \\
        --output /tmp/ostosense-matrix
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from ostosense_ai import features, labeling

MATRIX_BUILDER_VERSION = "0.1.2"
DATA_CONTRACT_VERSION = features.DATA_CONTRACT_VERSION
RULEBOOK_VERSION = labeling.RULEBOOK_VERSION
ALLOWED_ORIGIN = labeling.ALLOWED_INPUT_ORIGIN
NO_PERFORMANCE_WARNING = (
    "SYNTHETIC_PIPELINE_TEST_ONLY matrix-construction mechanics only. This "
    "feature-label matrix is not a model, not a metric, and not an OSTOSENSE "
    "performance, notification-accuracy, sensor, or clinical result."
)

FEATURE_COLUMNS = features.FEATURE_COLUMNS  # the exact five capacitive features
CLASS_NAMES = labeling.CLASS_NAMES
CLASS_NAME_TO_INDEX = labeling.CLASS_NAME_TO_INDEX

AUDIT_COLUMNS = ("window_id", "session_id", "bag_id", "sensor_id", "dataset_partition")
TARGET_COLUMNS = ("risk_label", "risk_label_index")
GROUPING_COLUMNS = ("session_id", "bag_id", "sensor_id")
PARTITION_COLUMN = "dataset_partition"
MODEL_MATRIX_COLUMNS = AUDIT_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
_OUTPUT_FILES = ("model_matrix.csv", "matrix_manifest.json")
_BOOLEAN_TEXT = frozenset(("true", "false"))

# Anything that must never be treated as a model feature.
_NON_FEATURE_NAMES = (
    frozenset(AUDIT_COLUMNS)
    | frozenset(TARGET_COLUMNS)
    | features.FORBIDDEN_FEATURE_NAMES
    | {
        "window_index",
        "window_start",
        "window_end",
        "sample_count",
        "feature_valid",
        "label_valid",
        "exclusion_reason",
        "arm",
        "protocol_deviation",
        "protocol_deviation_reason",
        "boundary_config_version",
        "rulebook_version",
    }
)

_CANONICAL_CONVENTION = {
    "interval": features._WINDOW_INTERVAL,
    "window_seconds": features._WORKING_WINDOW_SECONDS,
    "stride_seconds": features._WORKING_STRIDE_SECONDS,
    "sampling_rate_hz": features._WORKING_SAMPLING_RATE_HZ,
    "jitter_tolerance_ms": features._WORKING_JITTER_TOLERANCE_MS,
    "expected_full_samples": features._WORKING_WINDOW_SECONDS
    * features._WORKING_SAMPLING_RATE_HZ,
}
_IDENTITY_INT_FIELDS = ("window_index", "window_start", "window_end")


class MatrixError(ValueError):
    """Feature-label matrix validation failure (leaves outputs untouched)."""


def _assert_matrix_columns_safe() -> None:
    """Fail import if the feature allowlist ever drifts or admits forbidden metadata."""
    if tuple(FEATURE_COLUMNS) != features._SUPPORTED_FEATURES:
        raise AssertionError("matrix FEATURE_COLUMNS drifted from the canonical five")
    forbidden = set(FEATURE_COLUMNS) & _NON_FEATURE_NAMES
    if forbidden:
        raise AssertionError(f"forbidden metadata present in feature columns: {forbidden}")


_assert_matrix_columns_safe()


# --------------------------------------------------------------------------- #
# Input parsing / provenance
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise MatrixError(f"missing {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise MatrixError(f"invalid {label} JSON: {error}") from error
    if not isinstance(data, dict):
        raise MatrixError(f"{label} must be a JSON object")
    return data


def _read_csv_rows(
    path: Path, expected_header: tuple[str, ...], label: str
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise MatrixError(f"missing {label}: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_header):
            raise MatrixError(f"{label} header is not canonical: {reader.fieldnames!r}")
        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise MatrixError(f"{label} line {line_number} is malformed")
            window_id = row["window_id"]
            if not window_id:
                raise MatrixError(f"{label} line {line_number} has an empty window_id")
            if window_id in rows:
                raise MatrixError(f"duplicate window_id in {label}: {window_id}")
            rows[window_id] = row
        return rows


def _strict_bool(raw: str, field: str, row_label: str) -> bool:
    if raw not in _BOOLEAN_TEXT:
        raise MatrixError(
            f"{row_label} has invalid {field} {raw!r}; expected 'true' or 'false'"
        )
    return raw == "true"


def _require_nonempty(row: dict[str, str], fields: tuple[str, ...], row_label: str) -> None:
    for field in fields:
        if row[field] == "":
            raise MatrixError(f"{row_label} has an empty {field}")


def _reconcile_base_counts(
    manifest: dict[str, Any], total: int, valid: int, label: str
) -> None:
    if manifest.get("candidate_window_count") != total:
        raise MatrixError(f"{label} candidate count does not reconcile with its CSV")
    if manifest.get("valid_window_count") != valid:
        raise MatrixError(f"{label} valid count does not reconcile with its CSV")
    if manifest.get("excluded_window_count") != total - valid:
        raise MatrixError(f"{label} excluded count does not reconcile with its CSV")


def _validate_manifest_versions(
    feature_manifest: dict[str, Any], label_manifest: dict[str, Any]
) -> str:
    if feature_manifest.get("data_contract_version") != DATA_CONTRACT_VERSION:
        raise MatrixError(
            f"feature_manifest data_contract_version must be {DATA_CONTRACT_VERSION}"
        )
    if label_manifest.get("data_contract_version") != DATA_CONTRACT_VERSION:
        raise MatrixError(
            f"label_manifest data_contract_version must be {DATA_CONTRACT_VERSION}"
        )
    if label_manifest.get("rulebook_version") != RULEBOOK_VERSION:
        raise MatrixError(f"label_manifest rulebook_version must be {RULEBOOK_VERSION}")
    boundary_version = label_manifest.get("boundary_config_version")
    if not isinstance(boundary_version, str) or not boundary_version:
        raise MatrixError("label_manifest has no valid boundary_config_version")
    return boundary_version


def _canonical_convention(manifest: dict[str, Any], label: str) -> None:
    convention = manifest.get("window_convention")
    if not isinstance(convention, dict):
        raise MatrixError(f"{label} has no window_convention object")
    for key, value in _CANONICAL_CONVENTION.items():
        if convention.get(key) != value:
            raise MatrixError(f"{label} window convention mismatch on {key!r}")


def _assign_group(mapping: dict[str, str], key: str, partition: str, group: str) -> None:
    existing = mapping.get(key)
    if existing is not None and existing != partition:
        raise MatrixError(
            f"partition leakage: {group} {key!r} spans partitions "
            f"{existing!r} and {partition!r}"
        )
    mapping[key] = partition


def _validate_feature_values(window_id: str, feature_row: dict[str, str]) -> list[str]:
    values: list[str] = []
    for column in FEATURE_COLUMNS:
        raw = feature_row[column]
        if raw == "":
            raise MatrixError(f"feature-valid window {window_id} has an empty {column}")
        try:
            numeric = float(raw)
        except ValueError as error:
            raise MatrixError(
                f"feature-valid window {window_id} has a malformed {column}: {raw!r}"
            ) from error
        if not math.isfinite(numeric):
            raise MatrixError(
                f"feature-valid window {window_id} has a non-finite {column}: {raw!r}"
            )
        values.append(raw)
    return values


def _reconcile_feature_artifact(
    manifest: dict[str, Any], rows: dict[str, dict[str, str]]
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    validity: dict[str, bool] = {}
    values: dict[str, list[str]] = {}
    exclusions = {reason: 0 for reason in features.EXCLUSION_PRIORITY}
    valid_count = 0

    for window_id, row in rows.items():
        _require_nonempty(
            row,
            ("session_id", "bag_id", "sensor_id"),
            f"features.csv window {window_id}",
        )
        is_valid = _strict_bool(
            row["feature_valid"], "feature_valid", f"features.csv window {window_id}"
        )
        validity[window_id] = is_valid
        reason = row["exclusion_reason"]
        if is_valid:
            if reason:
                raise MatrixError(
                    f"feature-valid window {window_id} still has an exclusion_reason"
                )
            values[window_id] = _validate_feature_values(window_id, row)
            valid_count += 1
        else:
            if reason not in exclusions:
                raise MatrixError(
                    f"feature-invalid window {window_id} has unknown "
                    f"exclusion_reason {reason!r}"
                )
            if any(row[column] != "" for column in FEATURE_COLUMNS):
                raise MatrixError(
                    f"feature-invalid window {window_id} still carries feature values"
                )
            exclusions[reason] += 1

    _reconcile_base_counts(
        manifest, len(rows), valid_count, "feature_manifest.json"
    )
    if manifest.get("exclusion_reason_counts") != exclusions:
        raise MatrixError(
            "feature_manifest exclusion_reason_counts do not reconcile with features.csv"
        )
    return validity, values


def _reconcile_label_artifact(
    manifest: dict[str, Any],
    rows: dict[str, dict[str, str]],
    boundary_version: str,
) -> dict[str, bool]:
    validity: dict[str, bool] = {}
    class_counts = {name: 0 for name in CLASS_NAMES}
    exclusions = {reason: 0 for reason in labeling.ALL_EXCLUSION_REASONS}
    valid_count = 0

    for window_id, row in rows.items():
        _require_nonempty(row, ("session_id",), f"labels.csv window {window_id}")
        if row["rulebook_version"] != RULEBOOK_VERSION:
            raise MatrixError(
                f"labels.csv window {window_id} rulebook_version must be "
                f"{RULEBOOK_VERSION}"
            )
        if row["boundary_config_version"] != boundary_version:
            raise MatrixError(
                f"labels.csv window {window_id} boundary_config_version "
                "disagrees with label_manifest"
            )
        partition = row["dataset_partition"]
        if partition not in labeling.PARTITION_VALUES:
            raise MatrixError(
                f"labels.csv window {window_id} has invalid "
                f"dataset_partition {partition!r}"
            )

        protocol_deviation = _strict_bool(
            row["protocol_deviation"],
            "protocol_deviation",
            f"labels.csv window {window_id}",
        )
        deviation_reason = row["protocol_deviation_reason"]
        if protocol_deviation and deviation_reason != "UNPLANNED_PHYSICAL_LEAK":
            raise MatrixError(
                f"labels.csv window {window_id} has invalid "
                "protocol_deviation_reason"
            )
        if not protocol_deviation and deviation_reason:
            raise MatrixError(
                f"labels.csv window {window_id} has a deviation reason while "
                "protocol_deviation is false"
            )

        is_valid = _strict_bool(
            row["label_valid"], "label_valid", f"labels.csv window {window_id}"
        )
        validity[window_id] = is_valid
        risk_label = row["risk_label"]
        risk_index = row["risk_label_index"]
        reason = row["exclusion_reason"]
        if is_valid:
            if reason:
                raise MatrixError(
                    f"label-valid window {window_id} still has an exclusion_reason"
                )
            if risk_label not in CLASS_NAME_TO_INDEX:
                raise MatrixError(
                    f"window {window_id} has an unknown risk_label {risk_label!r}"
                )
            if risk_index != str(CLASS_NAME_TO_INDEX[risk_label]):
                raise MatrixError(f"window {window_id} risk_label/index inconsistency")
            class_counts[risk_label] += 1
            valid_count += 1
        else:
            if risk_label or risk_index:
                raise MatrixError(
                    f"invalid label window {window_id} still carries a label/index"
                )
            if reason not in exclusions:
                raise MatrixError(
                    f"unknown exclusion_reason for {window_id}: {reason!r}"
                )
            exclusions[reason] += 1

    _reconcile_base_counts(manifest, len(rows), valid_count, "label_manifest.json")
    if manifest.get("risk_class_counts") != class_counts:
        raise MatrixError(
            "label_manifest risk_class_counts do not reconcile with labels.csv"
        )
    if manifest.get("exclusion_reason_counts") != exclusions:
        raise MatrixError(
            "label_manifest exclusion_reason_counts do not reconcile with labels.csv"
        )
    return validity


# --------------------------------------------------------------------------- #
# Output (staged, refuse-by-default)
# --------------------------------------------------------------------------- #
def _check_output_targets(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    existing = [output_dir / name for name in _OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"output already contains matrix artifacts: {names}; "
            "pass overwrite=True or --overwrite to replace them"
        )


def _write_matrix_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MODEL_MATRIX_COLUMNS)
        writer.writerows(rows)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _write_output_artifacts(
    output_dir: Path, rows: list[list[str]], manifest: dict[str, Any], *, overwrite: bool
) -> dict[str, Any]:
    """Stage both artifacts completely before replacing any existing output.

    The CSV is written to the staging directory and hashed there; the resulting
    ``model_matrix_sha256`` (covering the exact final CSV bytes) is added to the
    manifest before the manifest is written, so downstream stages can detect any
    tampering with ``model_matrix.csv``. Returns the augmented manifest.

    The two ``replace`` calls are individually atomic but are not a single
    OS-level transaction across both files.
    """
    _check_output_targets(output_dir, overwrite=overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ostosense-matrix-", dir=output_dir.parent) as tmp:
        stage = Path(tmp)
        csv_stage = stage / "model_matrix.csv"
        _write_matrix_csv(csv_stage, rows)
        manifest = {**manifest, "model_matrix_sha256": _sha256_file(csv_stage)}
        _write_manifest(stage / "matrix_manifest.json", manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            (stage / name).replace(output_dir / name)
    return manifest


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def build_model_matrix(
    features_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Join validated features and labels into a leakage-safe matrix; return manifest."""
    features_dir = Path(features_dir)
    labels_dir = Path(labels_dir)
    feat_csv_path = features_dir / "features.csv"
    feat_manifest_path = features_dir / "feature_manifest.json"
    lbl_csv_path = labels_dir / "labels.csv"
    lbl_manifest_path = labels_dir / "label_manifest.json"

    # ---- validate & stage everything BEFORE mutating output ----
    feat_manifest = _load_json_object(feat_manifest_path, "feature_manifest.json")
    lbl_manifest = _load_json_object(lbl_manifest_path, "label_manifest.json")
    boundary_version = _validate_manifest_versions(feat_manifest, lbl_manifest)

    feat_csv_sha = _sha256_file(feat_csv_path)
    feat_manifest_sha = _sha256_file(feat_manifest_path)
    lbl_csv_sha = _sha256_file(lbl_csv_path)
    lbl_manifest_sha = _sha256_file(lbl_manifest_path)

    # (4) label manifest must carry non-empty feature-artifact hashes matching the files.
    feature_artifacts = lbl_manifest.get("feature_artifact_sha256")
    if not isinstance(feature_artifacts, dict) or not feature_artifacts:
        raise MatrixError("label_manifest is missing non-empty feature_artifact_sha256")
    if feature_artifacts.get("features_csv_sha256") != feat_csv_sha:
        raise MatrixError("features.csv hash does not match label_manifest reference")
    if feature_artifacts.get("feature_manifest_sha256") != feat_manifest_sha:
        raise MatrixError("feature_manifest.json hash does not match label_manifest reference")

    # (5) raw sessions/samples hashes must agree across both manifests.
    label_inputs = lbl_manifest.get("input_sha256")
    if not isinstance(label_inputs, dict):
        raise MatrixError("label_manifest is missing input_sha256")
    if feat_manifest.get("input_sessions_sha256") != label_inputs.get("sessions_csv"):
        raise MatrixError("raw sessions.csv hash disagrees between manifests")
    if feat_manifest.get("input_samples_sha256") != label_inputs.get("samples_csv"):
        raise MatrixError("raw samples.csv hash disagrees between manifests")

    # (6) dataset origin agrees and stays synthetic.
    feat_origin = feat_manifest.get("input_dataset_origin")
    lbl_origin = lbl_manifest.get("dataset_origin")
    if feat_origin != ALLOWED_ORIGIN or lbl_origin != ALLOWED_ORIGIN or feat_origin != lbl_origin:
        raise MatrixError("dataset origin must agree and be SYNTHETIC_PIPELINE_TEST_ONLY")

    # (7) window convention matches canonical on both sides.
    _canonical_convention(feat_manifest, "feature_manifest.json")
    _canonical_convention(lbl_manifest, "label_manifest.json")

    # (5 leakage) feature allowlist guard: manifest must declare exactly the five.
    if feat_manifest.get("feature_columns") != list(FEATURE_COLUMNS):
        raise MatrixError("feature_manifest feature_columns is not exactly the canonical five")

    feature_rows = _read_csv_rows(feat_csv_path, features.FEATURES_CSV_COLUMNS, "features.csv")
    label_rows = _read_csv_rows(lbl_csv_path, labeling.LABELS_CSV_COLUMNS, "labels.csv")

    # (8) all boolean, validity, class, exclusion, and manifest counts reconcile.
    feature_validity, feature_values = _reconcile_feature_artifact(
        feat_manifest, feature_rows
    )
    label_validity = _reconcile_label_artifact(
        lbl_manifest, label_rows, boundary_version
    )
    if feat_manifest["candidate_window_count"] != lbl_manifest["candidate_window_count"]:
        raise MatrixError("feature and label candidate-window counts disagree")

    # (10) feature and label window sets must match exactly.
    if set(feature_rows) != set(label_rows):
        missing = sorted(set(feature_rows) ^ set(label_rows))[:5]
        raise MatrixError(f"feature and label window sets differ (e.g. {missing})")

    matrix_rows: list[list[str]] = []
    partition_by_session: dict[str, str] = {}
    partition_by_bag: dict[str, str] = {}
    partition_by_sensor: dict[str, str] = {}
    class_counts = {name: 0 for name in CLASS_NAMES}
    class_counts_by_partition: dict[str, dict[str, int]] = {}
    exclusion_counts: dict[str, int] = {reason: 0 for reason in labeling.ALL_EXCLUSION_REASONS}
    source_partition_window_counts: dict[str, int] = {}
    partition_row_counts: dict[str, int] = {}
    eligible = 0

    for window_id, feature_row in feature_rows.items():  # features.csv order is deterministic
        label_row = label_rows[window_id]

        # (11) window identity metadata must agree.
        if feature_row["session_id"] != label_row["session_id"]:
            raise MatrixError(f"session_id mismatch for {window_id}")
        for field in _IDENTITY_INT_FIELDS:
            try:
                int(feature_row[field])
                int(label_row[field])
            except ValueError as error:
                raise MatrixError(
                    f"{field} for {window_id} must be an integer"
                ) from error
            if feature_row[field] != label_row[field]:
                raise MatrixError(f"{field} mismatch for {window_id}")

        partition = label_row["dataset_partition"]
        source_partition_window_counts[partition] = (
            source_partition_window_counts.get(partition, 0) + 1
        )
        partition_row_counts.setdefault(partition, 0)
        class_counts_by_partition.setdefault(
            partition, {name: 0 for name in CLASS_NAMES}
        )

        # (16) leakage guard across every window, not just eligible ones.
        _assign_group(partition_by_session, label_row["session_id"], partition, "session")
        _assign_group(partition_by_bag, feature_row["bag_id"], partition, "bag")
        _assign_group(partition_by_sensor, feature_row["sensor_id"], partition, "sensor")

        feature_valid = feature_validity[window_id]
        label_valid = label_validity[window_id]

        if label_valid and not feature_valid:  # (13)
            raise MatrixError(f"label-valid window {window_id} has an invalid feature")
        if not feature_valid:
            if label_row["exclusion_reason"] != feature_row["exclusion_reason"]:
                raise MatrixError(
                    f"structural exclusion mismatch for {window_id}"
                )
        elif (
            not label_valid
            and label_row["exclusion_reason"] in features.EXCLUSION_PRIORITY
        ):
            raise MatrixError(
                f"feature-valid window {window_id} has a structural label exclusion"
            )

        if not label_valid:
            # a legitimate exclusion: it must not carry a usable label/index.
            if label_row["risk_label"] != "" or label_row["risk_label_index"] != "":
                raise MatrixError(f"invalid label window {window_id} still carries a label/index")
            reason = label_row["exclusion_reason"]
            if reason not in exclusion_counts:
                raise MatrixError(f"unknown exclusion_reason for {window_id}: {reason!r}")
            exclusion_counts[reason] += 1
            continue

        # eligible row: label_valid and feature_valid
        risk_label = label_row["risk_label"]
        risk_index = label_row["risk_label_index"]
        if label_row["exclusion_reason"] != "":  # valid label must have empty exclusion
            raise MatrixError(f"valid label window {window_id} still has an exclusion_reason")
        if risk_label not in CLASS_NAME_TO_INDEX:
            raise MatrixError(f"window {window_id} has an unknown risk_label {risk_label!r}")
        if risk_index != str(CLASS_NAME_TO_INDEX[risk_label]):  # (14)
            raise MatrixError(f"window {window_id} risk_label/index inconsistency")

        matrix_rows.append(
            [
                window_id,
                feature_row["session_id"],
                feature_row["bag_id"],
                feature_row["sensor_id"],
                partition,
                *feature_values[window_id],
                risk_label,
                risk_index,
            ]
        )
        eligible += 1
        class_counts[risk_label] += 1
        class_counts_by_partition[partition][risk_label] += 1
        partition_row_counts[partition] += 1

    candidate = len(feature_rows)
    manifest = {
        "matrix_builder_version": MATRIX_BUILDER_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "rulebook_version": RULEBOOK_VERSION,
        "dataset_origin": ALLOWED_ORIGIN,
        "feature_input_sha256": {
            "features_csv": feat_csv_sha,
            "feature_manifest": feat_manifest_sha,
        },
        "label_input_sha256": {
            "labels_csv": lbl_csv_sha,
            "label_manifest": lbl_manifest_sha,
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "risk_label_index",
        "target_label_column": "risk_label",
        "class_mapping": dict(CLASS_NAME_TO_INDEX),
        "class_order": list(CLASS_NAMES),
        "audit_columns": list(AUDIT_COLUMNS),
        "grouping_columns": list(GROUPING_COLUMNS),
        "partition_column": PARTITION_COLUMN,
        "window_convention": dict(_CANONICAL_CONVENTION),
        "partition_values": sorted(source_partition_window_counts),
        "source_partition_window_counts": source_partition_window_counts,
        "partition_row_counts": partition_row_counts,
        "eligible_row_count": eligible,
        "excluded_row_count": candidate - eligible,
        "exclusion_counts": exclusion_counts,
        "class_counts": class_counts,
        "class_counts_by_partition": class_counts_by_partition,
        "source_candidate_window_count": candidate,
        "warning": NO_PERFORMANCE_WARNING,
    }

    return _write_output_artifacts(Path(output_dir), matrix_rows, manifest, overwrite=overwrite)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.matrix",
        description="Deterministic leakage-safe feature-label matrix builder (pipeline test only).",
    )
    parser.add_argument("--features", required=True, help="Feature artifact directory.")
    parser.add_argument("--labels", required=True, help="Label artifact directory.")
    parser.add_argument("--output", required=True, help="Output directory for matrix artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing matrix artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_model_matrix(args.features, args.labels, args.output, overwrite=args.overwrite)
    print(
        f"matrix: {manifest['eligible_row_count']} eligible rows "
        f"(of {manifest['source_candidate_window_count']} candidate windows) to {args.output}"
    )
    print(f"class_counts: {manifest['class_counts']}")
    print(NO_PERFORMANCE_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
