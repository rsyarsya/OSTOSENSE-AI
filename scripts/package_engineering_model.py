"""Package a reproduced synthetic run, never a private/pilot model or raw data."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

from ostosense_ai import inference, runtime_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            "Output directory exists; rebuild into a new directory and compare first"
        )
    model_bytes = (args.run_dir / "model/ordinal_model.json").read_bytes()
    manifest_bytes = (args.run_dir / "model/training_manifest.json").read_bytes()
    model = json.loads(model_bytes)
    manifest = json.loads(manifest_bytes)
    inference.validate_model_artifact(model)
    sha = hashlib.sha256(model_bytes).hexdigest()
    if manifest["output_sha256"]["ordinal_model_json"] != sha:
        raise ValueError("Training provenance does not match model")
    cases = {}
    matrix = args.run_dir / "matrix/model_matrix.csv"
    if (
        hashlib.sha256(matrix.read_bytes()).hexdigest()
        != manifest["input_sha256"]["model_matrix_csv"]
    ):
        raise ValueError("Matrix provenance does not match training")
    matrix_manifest_path = args.run_dir / "matrix/matrix_manifest.json"
    if (
        hashlib.sha256(matrix_manifest_path.read_bytes()).hexdigest()
        != manifest["input_sha256"]["matrix_manifest_json"]
    ):
        raise ValueError("Matrix manifest provenance does not match training")
    matrix_manifest = json.loads(matrix_manifest_path.read_bytes())
    feature_manifest_path = args.run_dir / "features/feature_manifest.json"
    feature_path = args.run_dir / "features/features.csv"
    for path, key in (
        (feature_manifest_path, "feature_manifest"),
        (feature_path, "features_csv"),
    ):
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != matrix_manifest["feature_input_sha256"][key]
        ):
            raise ValueError("Feature provenance does not match the matrix")
    feature_manifest = json.loads(feature_manifest_path.read_bytes())
    sessions_path = args.run_dir / "raw/sessions.csv"
    if (
        hashlib.sha256(sessions_path.read_bytes()).hexdigest()
        != feature_manifest["input_sessions_sha256"]
    ):
        raise ValueError("Session provenance does not match the features")
    with sessions_path.open(newline="", encoding="utf-8") as handle:
        starts = {
            row["session_id"]: int(row["start_timestamp"])
            for row in csv.DictReader(handle)
        }
    with feature_path.open(newline="", encoding="utf-8") as handle:
        source_windows = {row["window_id"]: row for row in csv.DictReader(handle)}
    with matrix.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset_partition"] != "validation":
                continue
            values = [float(row[name]) for name in inference.FEATURE_COLUMNS]
            source = source_windows[row["window_id"]]
            if source["session_id"] != row["session_id"]:
                raise ValueError("Source window session differs from matrix session")
            window_end = int(source["window_end"]) - starts[row["session_id"]]
            prediction = runtime_output.predict_test_output_v2(
                model,
                values,
                source_window_end_ms=window_end,
                model_artifact_sha256=sha,
            )
            name = prediction["risk_class"]
            if name not in cases:
                cases[name] = {
                    "source_window_id": row["window_id"],
                    "source_session_id": row["session_id"],
                    "input": {
                        "feature_input_version": "0.1.0",
                        "data_source": "SYNTHETIC_FIXTURE",
                        "model_input_channel": "SYNTHETIC_CAPACITIVE",
                        "source_window_end_ms": window_end,
                        "feature_basis": "RAW_MINUS_SESSION_BASELINE",
                        "feature_order": list(inference.FEATURE_COLUMNS),
                        "features": values,
                    },
                    "expected": prediction,
                }
    if set(cases) != set(inference.CLASS_NAMES):
        raise ValueError("Validation vectors do not cover the four predicted classes")
    examples = (
        json.dumps(
            [cases[n] for n in inference.CLASS_NAMES],
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    files = {
        "ordinal_model.json": model_bytes,
        "training_manifest.json": manifest_bytes,
        "examples.json": examples,
    }
    bundle = {
        "bundle_version": "0.1.0",
        "model_id": "engineering-v0.3",
        "dataset_origin": inference.ALLOWED_ORIGIN,
        "sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output_dir.parent, prefix=".model-package-"
    ) as stage:
        staged = Path(stage) / "bundle"
        staged.mkdir()
        for name, data in files.items():
            (staged / name).write_bytes(data)
        (staged / "bundle.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.output_dir.exists():
            raise FileExistsError("Output directory appeared during packaging")
        staged.rename(args.output_dir)
    print(f"Packaged engineering-v0.3: {sha}; four synthetic examples")


if __name__ == "__main__":
    main()
