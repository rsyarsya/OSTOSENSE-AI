import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ostosense_ai import edge_export, matrix, training
from ostosense_ai.edge_export import EdgeExportError, export_edge_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
FIRMWARE_INCLUDE = PROJECT_ROOT / "firmware" / "include"
EVAL_FIX = Path(__file__).resolve().parent / "fixtures" / "ostosense-evaluation-v0.1"
LABEL_FIX = Path(__file__).resolve().parent / "fixtures" / "ostosense-labeling-v0.1"
BOUNDARY = LABEL_FIX / "boundary-engineering-test-only-v0.1.json"
FEATURE_COLUMNS = matrix.FEATURE_COLUMNS
CLASS_NAMES = matrix.CLASS_NAMES
CLASS_NAME_TO_INDEX = matrix.CLASS_NAME_TO_INDEX
MODEL_MATRIX_COLUMNS = matrix.MODEL_MATRIX_COLUMNS

PIPELINE_AVAILABLE = all(
    importlib.util.find_spec(m) is not None for m in ("numpy", "sklearn", "mord", "scipy")
)
GPP = shutil.which("g++")
BANNED_KEYS = {
    "macro_f1", "kappa", "confusion_matrix", "accuracy", "meets_target", "pass", "fail",
    "passed", "failed", "target_met", "notification", "notification_accuracy", "lead_time",
    "false_alarm", "event_level", "clinical", "predictions", "metrics",
}
EXPECTED_MANIFEST_KEYS = {
    "exporter_version", "dataset_origin", "input_sha256", "output_sha256", "feature_order",
    "class_order", "vector_count", "float_representation", "probability_tolerance", "warning",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mrow(i, partition, name, feats):
    tag = f"{partition}-{i}"
    return {
        "window_id": f"{partition}-win-{i:04d}", "session": f"s-{tag}", "bag": f"b-{tag}",
        "sensor": f"sen-{tag}", "partition": partition, "risk": name,
        "feats": [f"{v:.6f}" for v in feats],
    }


def _default_rows():
    return [
        _mrow(0, "development", "Safe", [0.0] * 5),
        _mrow(1, "development", "Monitor", [1.0] * 5),
        _mrow(2, "validation", "Safe", [0.1, 0.2, 0.3, 0.4, 0.5]),
        _mrow(3, "validation", "Caution", [2.0, -1.0, 0.5, 1.5, -0.5]),
    ]


def _write_matrix(matrix_dir: Path, rows):
    matrix_dir.mkdir(parents=True, exist_ok=True)
    csv_path = matrix_dir / "model_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MODEL_MATRIX_COLUMNS)
        for r in rows:
            writer.writerow([
                r["window_id"], r["session"], r["bag"], r["sensor"], r["partition"],
                *r["feats"], r["risk"], str(CLASS_NAME_TO_INDEX[r["risk"]]),
            ])
    class_counts = {n: 0 for n in CLASS_NAMES}
    partition_counts: dict[str, int] = {}
    ccbp: dict[str, dict[str, int]] = {}
    for r in rows:
        class_counts[r["risk"]] += 1
        partition_counts[r["partition"]] = partition_counts.get(r["partition"], 0) + 1
        ccbp.setdefault(r["partition"], {n: 0 for n in CLASS_NAMES})[r["risk"]] += 1
    source = dict(partition_counts)
    source[sorted(source)[0]] += 10
    manifest = {
        "matrix_builder_version": matrix.MATRIX_BUILDER_VERSION,
        "data_contract_version": matrix.DATA_CONTRACT_VERSION,
        "rulebook_version": matrix.RULEBOOK_VERSION,
        "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
        "feature_columns": list(FEATURE_COLUMNS), "target_column": "risk_label_index",
        "target_label_column": "risk_label", "class_mapping": dict(CLASS_NAME_TO_INDEX),
        "class_order": list(CLASS_NAMES), "audit_columns": list(matrix.AUDIT_COLUMNS),
        "grouping_columns": list(matrix.GROUPING_COLUMNS), "partition_column": matrix.PARTITION_COLUMN,
        "window_convention": dict(matrix._CANONICAL_CONVENTION), "partition_values": sorted(source),
        "source_partition_window_counts": source, "partition_row_counts": partition_counts,
        "eligible_row_count": len(rows), "excluded_row_count": 10,
        "exclusion_counts": {"PARTIAL_WINDOW": 10}, "class_counts": class_counts,
        "class_counts_by_partition": ccbp, "source_candidate_window_count": len(rows) + 10,
        "model_matrix_sha256": _sha(csv_path), "warning": "w",
    }
    (matrix_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return matrix_dir


def _model_artifact(**overrides):
    model = {
        "model_artifact_version": "0.1.0", "model_family": "mord.LogisticAT",
        "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
        "feature_order": list(FEATURE_COLUMNS), "class_order": list(CLASS_NAMES),
        "class_mapping": dict(CLASS_NAME_TO_INDEX),
        "scaler": {"mean": [0.0] * 5, "scale": [1.0] * 5},
        "beta": [1.0, 0.0, 0.0, 0.0, 0.0], "theta": [-1.0, 0.0, 1.0],
        "warning": "ENGINEERING_TEST_ONLY",
    }
    model.update(overrides)
    return model


def _valid_model_sanity():
    return {
        "scaler_mean_finite": True, "scaler_scale_finite": True, "scaler_scale_positive": True,
        "beta_finite_count": 5, "theta_finite_count": 3, "theta_strictly_increasing": True,
        "reference_probability_shape_valid": True, "reference_probabilities_finite": True,
        "reference_probabilities_within_unit_interval": True,
        "reference_max_probability_sum_error": 0.0, "exported_probability_shape_valid": True,
        "exported_probabilities_finite": True, "exported_probabilities_within_unit_interval": True,
        "exported_max_probability_sum_error": 0.0,
        "probability_sum_tolerance": training._PROBABILITY_SUM_TOLERANCE,
        "forward_max_probability_difference": 0.0,
        "forward_parity_tolerance": training._FORWARD_PARITY_TOLERANCE,
        "forward_label_parity": True,
    }


def _write_model(model_dir: Path, matrix_dir: Path, model_artifact=None):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ordinal_model.json"
    model_path.write_text(
        json.dumps(model_artifact or _model_artifact(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = training._read_matrix_rows(matrix_dir / "model_matrix.csv")
    development = [r for r in rows if r["dataset_partition"] == "development"]
    ignored = [r for r in rows if r["dataset_partition"] == "validation"]
    class_counts = {name: 0 for name in CLASS_NAMES}
    for r in development:
        class_counts[r["risk_label"]] += 1
    training_manifest = {
        "trainer_version": training.TRAINER_VERSION, "config_id": "training-v0.1",
        "model_artifact_version": training.MODEL_ARTIFACT_VERSION,
        "data_contract_version": matrix.DATA_CONTRACT_VERSION, "rulebook_version": matrix.RULEBOOK_VERSION,
        "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
        "input_sha256": {
            "training_config_json": "0" * 64,
            "model_matrix_csv": _sha(matrix_dir / "model_matrix.csv"),
            "matrix_manifest_json": _sha(matrix_dir / "matrix_manifest.json"),
        },
        "output_sha256": {"ordinal_model_json": _sha(model_path)},
        "dependency_versions": {"numpy": "x", "scikit-learn": "x", "mord": "x", "scipy": "x"},
        "fit_partition_policy": {"fit": "development", "ignored": "validation", "forbidden": "final_test"},
        "source_row_count": len(rows), "fitted_row_count": len(development),
        "ignored_row_count": len(ignored), "fitted_class_counts": class_counts,
        "fitted_session_count": len({r["session_id"] for r in development}),
        "fitted_bag_count": len({r["bag_id"] for r in development}),
        "fitted_sensor_count": len({r["sensor_id"] for r in development}),
        "preprocessing": "sklearn.preprocessing.StandardScaler", "sample_weighting": "uniform_window",
        "model_sanity": _valid_model_sanity(),
        "optimizer_convergence_status": training._OPTIMIZER_CONVERGENCE_STATUS,
        "warning": training.MANIFEST_WARNING,
    }
    (model_dir / "training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model_dir


def _edit_json(path, mutate):
    data = json.loads(Path(path).read_text("utf-8"))
    mutate(data)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _all_keys(item)


class EdgeExportTests(unittest.TestCase):
    def _setup(self, root, rows=None, model_artifact=None):
        rows = rows if rows is not None else _default_rows()
        m = _write_matrix(root / "matrix", rows)
        d = _write_model(root / "model", m, model_artifact)
        return m, d

    def test_schema_and_deterministic_hex_literals(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = _model_artifact(scaler={"mean": [0.5, -0.25, 2.0, -8.0, 0.125], "scale": [1.0] * 5})
            m, d = self._setup(root, model_artifact=model)
            manifest = export_edge_bundle(d, m, root / "edge")
            params = (root / "edge" / "ordinal_model_params.hpp").read_text("utf-8")
        self.assertEqual(set(manifest), EXPECTED_MANIFEST_KEYS)
        self.assertEqual(set(manifest["input_sha256"]),
                         {"ordinal_model_json", "training_manifest_json", "model_matrix_csv", "matrix_manifest_json"})
        self.assertEqual(set(manifest["output_sha256"]), {"ordinal_model_params_hpp", "golden_vectors_hpp"})
        self.assertEqual(manifest["float_representation"], "float32_hexadecimal_literals")
        self.assertEqual(manifest["probability_tolerance"], 1e-5)
        self.assertEqual(manifest["vector_count"], 2)
        # exact float32 hex literals present
        self.assertEqual(edge_export._f32_literal(0.5), "0x1.0000000000000p-1f")
        for value in (0.5, -0.25, 2.0, -8.0, 0.125):
            self.assertIn(edge_export._f32_literal(value), params)

    def test_output_hashes_reconcile_with_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            manifest = export_edge_bundle(d, m, root / "edge")
            self.assertEqual(manifest["output_sha256"]["ordinal_model_params_hpp"],
                             _sha(root / "edge" / "ordinal_model_params.hpp"))
            self.assertEqual(manifest["output_sha256"]["golden_vectors_hpp"],
                             _sha(root / "edge" / "golden_vectors.hpp"))

    def test_no_prohibited_fields(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            manifest = export_edge_bundle(d, m, root / "edge")
        offending = {k for k in _all_keys(manifest) if k.lower() in BANNED_KEYS}
        self.assertEqual(offending, set())

    def test_byte_identical_repeated_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            export_edge_bundle(d, m, root / "a")
            export_edge_bundle(d, m, root / "b")
            for f in ("ordinal_model_params.hpp", "golden_vectors.hpp", "edge_export_manifest.json"):
                self.assertEqual((root / "a" / f).read_bytes(), (root / "b" / f).read_bytes())

    def test_overwrite_refusal_and_preserves_unrelated(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            out = root / "edge"
            export_edge_bundle(d, m, out)
            keep = out / "keep.txt"
            keep.write_text("k", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_edge_bundle(d, m, out)
            export_edge_bundle(d, m, out, overwrite=True)
            self.assertEqual(keep.read_text("utf-8"), "k")

    def test_invalid_model_and_manifest_rejected(self):
        model_mutations = [
            {"model_artifact_version": "9.9.9"},
            {"dataset_origin": "REAL"},
            {"feature_order": ["a", "b", "c", "d", "e"]},
            {"class_order": ["Urgent", "Caution", "Monitor", "Safe"]},
            {"scaler": {"mean": [0, 0, 0, 0], "scale": [1, 1, 1, 1, 1]}},  # bad dimensions
            {"theta": [1.0, 0.0, -1.0]},  # not increasing
        ]
        for overrides in model_mutations:
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                with self.assertRaises(EdgeExportError):
                    m, d = self._setup(root, model_artifact=_model_artifact(**overrides))
                    export_edge_bundle(d, m, root / "edge")

    def test_float32_quantization_rejects_unusable_model(self):
        cases = [
            _model_artifact(
                scaler={"mean": [0.0] * 5, "scale": [1e-50, 1.0, 1.0, 1.0, 1.0]}
            ),
            _model_artifact(
                scaler={"mean": [0.0] * 5, "scale": [1e-45, 1.0, 1.0, 1.0, 1.0]}
            ),
            _model_artifact(theta=[0.0, 1e-50, 2e-50]),
            _model_artifact(beta=[1e100, 0.0, 0.0, 0.0, 0.0]),
        ]
        for model_artifact in cases:
            with self.subTest(model=model_artifact), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                m, d = self._setup(root, model_artifact=model_artifact)
                with self.assertRaisesRegex(EdgeExportError, "float32"):
                    export_edge_bundle(d, m, root / "edge")
                self.assertFalse((root / "edge" / "ordinal_model_params.hpp").exists())

    def test_float32_quantization_rejects_unrepresentable_feature(self):
        rows = _default_rows()
        rows[2]["feats"][0] = "1e100"
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root, rows)
            with self.assertRaisesRegex(EdgeExportError, "float32"):
                export_edge_bundle(d, m, root / "edge")
            self.assertFalse((root / "edge" / "golden_vectors.hpp").exists())

    def test_broken_hash_chain_rejected(self):
        cases = [
            ("model", lambda m, d: _edit_json(d / "ordinal_model.json", lambda x: x.__setitem__("beta", [2, 0, 0, 0, 0]))),
            ("training-input", lambda m, d: _edit_json(d / "training_manifest.json", lambda x: x["input_sha256"].__setitem__("model_matrix_csv", "0" * 64))),
            ("matrix-self", lambda m, d: _edit_json(m / "matrix_manifest.json", lambda x: x.__setitem__("model_matrix_sha256", "0" * 64))),
        ]
        for _label, tamper in cases:
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                m, d = self._setup(root)
                tamper(m, d)
                with self.assertRaises(EdgeExportError):
                    export_edge_bundle(d, m, root / "edge")

    def test_missing_validation_rows_rejected(self):
        rows = [_mrow(0, "development", "Safe", [0.0] * 5), _mrow(1, "development", "Monitor", [1.0] * 5)]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root, rows)
            with self.assertRaises(EdgeExportError):
                export_edge_bundle(d, m, root / "edge")

    def test_group_leakage_and_final_test_rejected(self):
        leak = _default_rows()
        leak[2]["bag"] = leak[0]["bag"]  # a bag spans development + validation
        final = _default_rows() + [_mrow(9, "final_test", "Safe", [0.0] * 5)]
        for rows in (leak, final):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                m, d = self._setup(root, rows)
                with self.assertRaises(EdgeExportError):
                    export_edge_bundle(d, m, root / "edge")

    def test_staging_failure_preserves_existing(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            out = root / "edge"
            out.mkdir()
            (out / "ordinal_model_params.hpp").write_bytes(b"PARAMS-SENTINEL")
            (out / "golden_vectors.hpp").write_bytes(b"SENTINEL")
            (out / "edge_export_manifest.json").write_bytes(b"MANIFEST-SENTINEL")
            with mock.patch.object(
                edge_export.training,
                "_dump_json",
                side_effect=RuntimeError("injected staging failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected staging failure"):
                    export_edge_bundle(d, m, out, overwrite=True)
            self.assertEqual(
                (out / "ordinal_model_params.hpp").read_bytes(), b"PARAMS-SENTINEL"
            )
            self.assertEqual((out / "golden_vectors.hpp").read_bytes(), b"SENTINEL")
            self.assertEqual(
                (out / "edge_export_manifest.json").read_bytes(), b"MANIFEST-SENTINEL"
            )


_PARITY_MAIN = """
#include <array>
#include <cmath>
#include <cstdio>
#include "ostosense/ordinal_inference.hpp"
#include "ordinal_model_params.hpp"
#include "golden_vectors.hpp"
using namespace ostosense;
int main() {
  double max_diff = 0.0;
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < generated::kGoldenVectorCount; ++i) {
    const auto& v = generated::kGoldenVectors[i];
    OrdinalPrediction out;
    if (PredictOrdinal(generated::kOrdinalModel, v.features, &out) != InferenceStatus::kOk) {
      std::fprintf(stderr, "status failure at %zu\\n", i);
      return 2;
    }
    if (static_cast<std::uint8_t>(out.predicted_class) != v.expected_class_index) {
      ++mismatches;
    }
    for (int k = 0; k < 4; ++k) {
      const double d = std::fabs((double)out.probabilities[k] - (double)v.reference_probabilities[k]);
      if (d > max_diff) {
        max_diff = d;
      }
    }
  }
  std::printf("count=%zu mismatches=%zu max_diff=%.9g\\n",
              generated::kGoldenVectorCount, mismatches, max_diff);
  if (mismatches != 0) {
    return 3;
  }
  if (max_diff > 1e-5) {
    return 4;
  }
  return 0;
}
"""


@unittest.skipUnless(PIPELINE_AVAILABLE and GPP, "requires the optional [pipeline] dependencies and g++")
class EdgeExportCrossLanguageParityTests(unittest.TestCase):
    def test_v03_generated_headers_match_python(self):
        from ostosense_ai import features, labeling, synthetic

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw, feat, lbl, mtx, model, edge = (root / d for d in ("raw", "feat", "lbl", "mtx", "model", "edge"))
            synthetic.generate(synthetic.load_config(REPO_ROOT / "configs" / "synthetic-v0.3.json"), 20260722, raw)
            features.extract(raw, features.load_config(REPO_ROOT / "configs" / "features-v0.1.json"), feat)
            labeling.label_dataset(
                raw, EVAL_FIX / "protocol_manifest.csv", EVAL_FIX / "partition_manifest.csv",
                BOUNDARY, lbl, features_dir=feat)
            matrix.build_model_matrix(feat, lbl, mtx)
            training.train_ordinal_model(mtx, REPO_ROOT / "configs" / "training-v0.1.json", model)
            manifest = export_edge_bundle(model, mtx, edge)

            self.assertEqual(manifest["vector_count"], 38)
            main = root / "parity_main.cpp"
            main.write_text(_PARITY_MAIN, encoding="utf-8")
            binary = root / "parity"
            compile_cmd = [
                GPP, "-std=c++17", "-Wall", "-Wextra", "-pedantic",
                "-I", str(FIRMWARE_INCLUDE), "-I", str(edge), str(main), "-o", str(binary),
            ]
            compiled = subprocess.run(compile_cmd, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            output = run.stdout.strip()
            self.assertIn("count=38", output)
            self.assertIn("mismatches=0", output)
            # max_diff must be <= 1e-5
            max_diff = float(output.split("max_diff=")[1])
            self.assertLessEqual(max_diff, 1e-5)


if __name__ == "__main__":
    unittest.main()
