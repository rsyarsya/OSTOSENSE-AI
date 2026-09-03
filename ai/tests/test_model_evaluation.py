import csv
import io
import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ostosense_ai import inference, matrix, model_evaluation, training
from ostosense_ai.model_evaluation import ModelEvaluationError, evaluate_validation_partition

REPO_ROOT = Path(__file__).resolve().parents[1]
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
BANNED_KEYS = {
    "meets_target", "pass", "fail", "passed", "failed", "target_met", "notification_accuracy",
    "notification", "lead_time", "false_alarm", "false_alarms", "event_level", "firmware", "clinical",
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
        _mrow(2, "validation", "Safe", [0.1] * 5),
        _mrow(3, "validation", "Caution", [2.0] * 5),
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


def _model_artifact():
    return {
        "model_artifact_version": "0.1.0", "model_family": "mord.LogisticAT",
        "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
        "feature_order": list(FEATURE_COLUMNS), "class_order": list(CLASS_NAMES),
        "class_mapping": dict(CLASS_NAME_TO_INDEX),
        "scaler": {"mean": [0.0] * 5, "scale": [1.0] * 5},
        "beta": [1.0, 0.0, 0.0, 0.0, 0.0], "theta": [-1.0, 0.0, 1.0],
        "warning": "ENGINEERING_TEST_ONLY",
    }


def _valid_model_sanity():
    return {
        "scaler_mean_finite": True,
        "scaler_scale_finite": True,
        "scaler_scale_positive": True,
        "beta_finite_count": 5,
        "theta_finite_count": 3,
        "theta_strictly_increasing": True,
        "reference_probability_shape_valid": True,
        "reference_probabilities_finite": True,
        "reference_probabilities_within_unit_interval": True,
        "reference_max_probability_sum_error": 0.0,
        "exported_probability_shape_valid": True,
        "exported_probabilities_finite": True,
        "exported_probabilities_within_unit_interval": True,
        "exported_max_probability_sum_error": 0.0,
        "probability_sum_tolerance": training._PROBABILITY_SUM_TOLERANCE,
        "forward_max_probability_difference": 0.0,
        "forward_parity_tolerance": training._FORWARD_PARITY_TOLERANCE,
        "forward_label_parity": True,
    }


def _write_model(model_dir: Path, matrix_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ordinal_model.json"
    model_path.write_text(json.dumps(_model_artifact(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = training._read_matrix_rows(matrix_dir / "model_matrix.csv")
    development = [row for row in rows if row["dataset_partition"] == "development"]
    ignored = [row for row in rows if row["dataset_partition"] == "validation"]
    class_counts = {name: 0 for name in CLASS_NAMES}
    for row in development:
        class_counts[row["risk_label"]] += 1
    training_manifest = {
        "trainer_version": training.TRAINER_VERSION,
        "config_id": "training-v0.1",
        "model_artifact_version": training.MODEL_ARTIFACT_VERSION,
        "data_contract_version": matrix.DATA_CONTRACT_VERSION,
        "rulebook_version": matrix.RULEBOOK_VERSION,
        "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
        "input_sha256": {
            "training_config_json": "0" * 64,
            "model_matrix_csv": _sha(matrix_dir / "model_matrix.csv"),
            "matrix_manifest_json": _sha(matrix_dir / "matrix_manifest.json"),
        },
        "output_sha256": {"ordinal_model_json": _sha(model_path)},
        "dependency_versions": {
            "numpy": "fixture", "scikit-learn": "fixture",
            "mord": "fixture", "scipy": "fixture",
        },
        "fit_partition_policy": {
            "fit": "development", "ignored": "validation", "forbidden": "final_test",
        },
        "source_row_count": len(rows),
        "fitted_row_count": len(development),
        "ignored_row_count": len(ignored),
        "fitted_class_counts": class_counts,
        "fitted_session_count": len({row["session_id"] for row in development}),
        "fitted_bag_count": len({row["bag_id"] for row in development}),
        "fitted_sensor_count": len({row["sensor_id"] for row in development}),
        "preprocessing": "sklearn.preprocessing.StandardScaler",
        "sample_weighting": "uniform_window",
        "model_sanity": _valid_model_sanity(),
        "optimizer_convergence_status": training._OPTIMIZER_CONVERGENCE_STATUS,
        "warning": training.MANIFEST_WARNING,
    }
    (model_dir / "training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model_dir


def _edit_json(path: Path, mutate):
    data = json.loads(Path(path).read_text("utf-8"))
    mutate(data)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ModelEvaluationValidationTests(unittest.TestCase):
    """Rejections that fire before prediction/metrics — dependency-free."""

    def _setup(self, root, rows=None):
        rows = rows if rows is not None else _default_rows()
        m = _write_matrix(root / "matrix", rows)
        d = _write_model(root / "model", m)
        return m, d

    def test_model_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            _edit_json(d / "ordinal_model.json", lambda x: x.__setitem__("beta", [2.0, 0, 0, 0, 0]))
            with self.assertRaises(ModelEvaluationError):
                evaluate_validation_partition(m, d, root / "out")

    def test_training_matrix_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            _edit_json(d / "training_manifest.json",
                       lambda x: x["input_sha256"].__setitem__("model_matrix_csv", "deadbeef"))
            with self.assertRaises(ModelEvaluationError):
                evaluate_validation_partition(m, d, root / "out")

    def test_matrix_self_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            _edit_json(m / "matrix_manifest.json",
                       lambda x: x.__setitem__("model_matrix_sha256", "deadbeef"))
            with self.assertRaises(ModelEvaluationError):
                evaluate_validation_partition(m, d, root / "out")

    def test_origin_and_version_mismatch_rejected(self):
        for mutate in (
            lambda x: x.__setitem__("dataset_origin", "REAL_DATA"),
            lambda x: x.__setitem__("data_contract_version", "v9.9"),
            lambda x: x.__setitem__("rulebook_version", "v0.1"),
        ):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                m, d = self._setup(root)
                _edit_json(m / "matrix_manifest.json", mutate)
                # matrix_manifest hash changed -> model must be re-tied; keep it consistent for
                # dataset_origin/version tests by re-pointing the training input hash.
                _edit_json(d / "training_manifest.json",
                           lambda x: x["input_sha256"].__setitem__(
                               "matrix_manifest_json", _sha(m / "matrix_manifest.json")))
                with self.assertRaises(ModelEvaluationError):
                    evaluate_validation_partition(m, d, root / "out")

    def test_training_manifest_origin_and_policy_rejected(self):
        for mutate in (
            lambda x: x.__setitem__("dataset_origin", "REAL_DATA"),
            lambda x: x.__setitem__("fit_partition_policy", {"fit": "validation", "ignored": "development", "forbidden": "final_test"}),
        ):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                m, d = self._setup(root)
                _edit_json(d / "training_manifest.json", mutate)
                _edit_json(d / "training_manifest.json",
                           lambda x: x["output_sha256"].__setitem__(
                               "ordinal_model_json", _sha(d / "ordinal_model.json")))
                with self.assertRaises(ModelEvaluationError):
                    evaluate_validation_partition(m, d, root / "out")

    def test_training_manifest_provenance_and_counts_rejected(self):
        mutations = (
            lambda x: x.__setitem__("trainer_version", "tampered"),
            lambda x: x.pop("data_contract_version"),
            lambda x: x.__setitem__("source_row_count", 999),
            lambda x: x["fitted_class_counts"].__setitem__("Safe", 999),
            lambda x: x["model_sanity"].__setitem__("forward_label_parity", False),
            lambda x: x["input_sha256"].__setitem__("training_config_json", "not-a-sha"),
        )
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                m, d = self._setup(root)
                _edit_json(d / "training_manifest.json", mutate)
                with self.assertRaises(ModelEvaluationError):
                    evaluate_validation_partition(m, d, root / "out")

    def test_invalid_model_error_is_wrapped(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            _edit_json(d / "ordinal_model.json", lambda x: x.__setitem__("theta", [1, 0, -1]))
            _edit_json(
                d / "training_manifest.json",
                lambda x: x["output_sha256"].__setitem__(
                    "ordinal_model_json", _sha(d / "ordinal_model.json")
                ),
            )
            with self.assertRaises(ModelEvaluationError) as context:
                evaluate_validation_partition(m, d, root / "out")
            self.assertIsInstance(context.exception.__cause__, inference.InferenceError)

    def test_metric_value_error_is_wrapped(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            with mock.patch.object(
                model_evaluation.evaluation,
                "evaluate_predictions",
                side_effect=ValueError("undefined fixture"),
            ):
                with self.assertRaises(ModelEvaluationError) as context:
                    evaluate_validation_partition(m, d, root / "out")
            self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_missing_validation_partition_rejected(self):
        rows = [_mrow(0, "development", "Safe", [0.0] * 5), _mrow(1, "development", "Monitor", [1.0] * 5)]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root, rows)
            with self.assertRaises(ModelEvaluationError):
                evaluate_validation_partition(m, d, root / "out")

    def test_group_leakage_rejected(self):
        rows = _default_rows()
        rows[2]["bag"] = rows[0]["bag"]  # a bag now spans development and validation
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root, rows)
            with self.assertRaises(ModelEvaluationError):
                evaluate_validation_partition(m, d, root / "out")

    def test_final_test_presence_rejected(self):
        rows = _default_rows() + [_mrow(9, "final_test", "Safe", [0.0] * 5)]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root, rows)
            with self.assertRaises(ModelEvaluationError):
                evaluate_validation_partition(m, d, root / "out")

    def test_cli_does_not_print_synthetic_metric_values(self):
        fake_manifest = {
            "validation_row_count": 2,
            "validation_session_count": 1,
            "validation_ground_truth_support": {
                "Safe": 1, "Monitor": 1, "Caution": 0, "Urgent": 0,
            },
            "pipeline_mechanics_metrics": {
                "macro_f1": 0.99, "quadratic_weighted_kappa": 0.99,
            },
        }
        output = io.StringIO()
        with mock.patch.object(
            model_evaluation,
            "evaluate_validation_partition",
            return_value=fake_manifest,
        ), redirect_stdout(output):
            result = model_evaluation.main(
                ["--matrix", "matrix", "--model", "model", "--output", "out"]
            )
        rendered = output.getvalue()
        self.assertEqual(result, 0)
        self.assertNotIn("macro_f1=", rendered)
        self.assertNotIn("quadratic_weighted_kappa=", rendered)
        self.assertIn(model_evaluation.NO_PERFORMANCE_WARNING, rendered)

    def test_staging_failure_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m, d = self._setup(root)
            out = root / "out"
            out.mkdir()
            predictions = out / "validation_predictions.csv"
            report = out / "validation_evaluation.json"
            predictions.write_bytes(b"PREDICTIONS_SENTINEL")
            report.write_bytes(b"REPORT_SENTINEL")
            with mock.patch.object(
                model_evaluation.evaluation,
                "evaluate_predictions",
                return_value={"fixture": True},
            ), mock.patch.object(
                model_evaluation,
                "_write_predictions_csv",
                side_effect=OSError("simulated staging failure"),
            ):
                with self.assertRaises(OSError):
                    evaluate_validation_partition(m, d, out, overwrite=True)
            self.assertEqual(predictions.read_bytes(), b"PREDICTIONS_SENTINEL")
            self.assertEqual(report.read_bytes(), b"REPORT_SENTINEL")


def _all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _all_keys(item)


@unittest.skipUnless(PIPELINE_AVAILABLE, "requires the optional [pipeline] dependencies")
class ModelEvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def _pipeline(cls, root, config_name):
        from ostosense_ai import features, labeling, synthetic

        raw, feat, lbl, mtx, model = (root / d for d in ("raw", "feat", "lbl", "mtx", "model"))
        synthetic.generate(synthetic.load_config(REPO_ROOT / "configs" / config_name), 20260722, raw)
        features.extract(raw, features.load_config(REPO_ROOT / "configs" / "features-v0.1.json"), feat)
        fixture = EVAL_FIX if config_name == "synthetic-v0.3.json" else LABEL_FIX
        labeling.label_dataset(
            raw, fixture / "protocol_manifest.csv", fixture / "partition_manifest.csv",
            BOUNDARY, lbl, features_dir=feat)
        matrix.build_model_matrix(feat, lbl, mtx)
        training.train_ordinal_model(mtx, REPO_ROOT / "configs" / "training-v0.1.json", model)
        return mtx, model

    def test_v03_golden_integration(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            mtx, model = self._pipeline(root, "synthetic-v0.3.json")
            manifest = evaluate_validation_partition(mtx, model, root / "val")
            with (root / "val" / "validation_predictions.csv").open(newline="", encoding="utf-8") as h:
                pred_rows = list(csv.reader(h))
        self.assertEqual(manifest["validation_row_count"], 38)
        self.assertEqual(manifest["validation_ground_truth_support"],
                         {"Safe": 21, "Monitor": 8, "Caution": 6, "Urgent": 3})
        metrics = manifest["pipeline_mechanics_metrics"]
        self.assertEqual(metrics["confusion_matrix"], [[21, 0, 0, 0], [2, 6, 0, 0], [0, 0, 6, 0], [0, 0, 0, 3]])
        self.assertAlmostEqual(metrics["macro_f1"], 0.952922077922078)
        self.assertAlmostEqual(metrics["quadratic_weighted_kappa"], 0.9732582688247713)
        # canonical metric delegation
        self.assertEqual(metrics["evaluation_scope"], __import__("ostosense_ai.evaluation", fromlist=["EVALUATION_SCOPE"]).EVALUATION_SCOPE)
        # schema + row order
        self.assertEqual(tuple(pred_rows[0]), model_evaluation.PREDICTIONS_COLUMNS)
        self.assertEqual([r[0] for r in pred_rows[1:]][:1], ["syn-safe-validation-001-win-0001"])
        self.assertEqual(len(pred_rows) - 1, 38)
        # no target/pass-fail/notification keys anywhere
        offending = {k for k in _all_keys(manifest) if k.lower() in BANNED_KEYS}
        self.assertEqual(offending, set())

    def test_determinism_and_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            mtx, model = self._pipeline(root, "synthetic-v0.3.json")
            evaluate_validation_partition(mtx, model, root / "a")
            evaluate_validation_partition(mtx, model, root / "b")
            for f in ("validation_predictions.csv", "validation_evaluation.json"):
                self.assertEqual((root / "a" / f).read_bytes(), (root / "b" / f).read_bytes())
            keep = (root / "a" / "keep.txt")
            keep.write_text("k", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                evaluate_validation_partition(mtx, model, root / "a")
            evaluate_validation_partition(mtx, model, root / "a", overwrite=True)
            self.assertEqual(keep.read_text("utf-8"), "k")

    def test_v03_preserves_original_nine_records_not_whole_files(self):
        from ostosense_ai import synthetic

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            v2 = root / "v2"
            v3 = root / "v3"
            synthetic.generate(
                synthetic.load_config(REPO_ROOT / "configs" / "synthetic-v0.2.json"),
                20260722,
                v2,
            )
            synthetic.generate(
                synthetic.load_config(REPO_ROOT / "configs" / "synthetic-v0.3.json"),
                20260722,
                v3,
            )
            with (v2 / "sessions.csv").open(newline="", encoding="utf-8") as handle:
                original_ids = {row["session_id"] for row in csv.DictReader(handle)}
            for filename in ("sessions.csv", "samples.csv", "events.csv"):
                with (v2 / filename).open(newline="", encoding="utf-8") as handle:
                    original_rows = list(csv.DictReader(handle))
                with (v3 / filename).open(newline="", encoding="utf-8") as handle:
                    extended_rows = list(csv.DictReader(handle))
                preserved_rows = [
                    row for row in extended_rows if row["session_id"] in original_ids
                ]
                self.assertEqual(original_rows, preserved_rows)
                self.assertNotEqual((v2 / filename).read_bytes(), (v3 / filename).read_bytes())

    def test_v02_dev_model_unchanged_under_v03(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, model2 = self._pipeline(root / "v2", "synthetic-v0.2.json")
            _, model3 = self._pipeline(root / "v3", "synthetic-v0.3.json")
            a = json.loads((model2 / "ordinal_model.json").read_text("utf-8"))
            b = json.loads((model3 / "ordinal_model.json").read_text("utf-8"))
        self.assertEqual(a["scaler"], b["scaler"])
        self.assertEqual(a["beta"], b["beta"])
        self.assertEqual(a["theta"], b["theta"])


if __name__ == "__main__":
    unittest.main()
