import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ostosense_ai import training
from ostosense_ai.training import TrainingError, train_ordinal_model

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "training-v0.1.json"
LABELING_FIX = Path(__file__).resolve().parent / "fixtures" / "ostosense-labeling-v0.1"
FEATURE_COLUMNS = training.FEATURE_COLUMNS
CLASS_NAMES = training.CLASS_NAMES
CLASS_NAME_TO_INDEX = training.CLASS_NAME_TO_INDEX
MODEL_MATRIX_COLUMNS = training.MODEL_MATRIX_COLUMNS

PIPELINE_AVAILABLE = all(
    importlib.util.find_spec(m) is not None for m in ("numpy", "sklearn", "mord", "scipy")
)
BANNED_KEYS = {
    "macro_f1", "f1", "precision", "recall", "kappa", "cohen_kappa",
    "quadratic_weighted_kappa", "confusion_matrix", "accuracy", "notification_accuracy",
    "predictions", "pass", "fail", "passed", "failed", "target_met", "score",
}


def _row(index, partition, name, feats, *, unique_group=True):
    tag = f"{partition}-{index}" if unique_group else partition
    return {
        "window_id": f"{partition}-win-{index:04d}",
        "session": f"s-{tag}", "bag": f"b-{tag}", "sensor": f"sen-{tag}",
        "partition": partition, "risk": name, "feats": [f"{v:.6f}" for v in feats],
    }


def _dev_rows():
    rows = []
    i = 0
    for cls_index, name in enumerate(CLASS_NAMES):
        for j in range(5):
            base = cls_index + 0.1 * j
            rows.append(_row(i, "development", name, [base] * 5))
            i += 1
    return rows


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
    class_counts_by_partition: dict[str, dict[str, int]] = {}
    for r in rows:
        partition = r["partition"]
        class_counts[r["risk"]] += 1
        partition_counts[partition] = partition_counts.get(partition, 0) + 1
        class_counts_by_partition.setdefault(
            partition, {name: 0 for name in CLASS_NAMES}
        )
        class_counts_by_partition[partition][r["risk"]] += 1
    source_partition_counts = dict(partition_counts)
    first_partition = sorted(source_partition_counts)[0]
    source_partition_counts[first_partition] += 10
    manifest = {
        "matrix_builder_version": training.matrix.MATRIX_BUILDER_VERSION,
        "data_contract_version": training.DATA_CONTRACT_VERSION,
        "rulebook_version": training.RULEBOOK_VERSION,
        "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "risk_label_index",
        "target_label_column": "risk_label",
        "class_mapping": dict(CLASS_NAME_TO_INDEX),
        "class_order": list(CLASS_NAMES),
        "audit_columns": list(training.matrix.AUDIT_COLUMNS),
        "grouping_columns": list(training.matrix.GROUPING_COLUMNS),
        "partition_column": training.matrix.PARTITION_COLUMN,
        "window_convention": dict(training.matrix._CANONICAL_CONVENTION),
        "partition_values": sorted(source_partition_counts),
        "source_partition_window_counts": source_partition_counts,
        "partition_row_counts": partition_counts,
        "eligible_row_count": len(rows),
        "excluded_row_count": 10,
        "exclusion_counts": {"PARTIAL_WINDOW": 10},
        "class_counts": class_counts,
        "class_counts_by_partition": class_counts_by_partition,
        "source_candidate_window_count": len(rows) + 10,
        "model_matrix_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "warning": "w",
    }
    (matrix_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return matrix_dir


def _rewrite_manifest_sha(matrix_dir: Path):
    """Recompute model_matrix_sha256 after a legitimate CSV rewrite."""
    manifest = json.loads((matrix_dir / "matrix_manifest.json").read_text("utf-8"))
    manifest["model_matrix_sha256"] = hashlib.sha256(
        (matrix_dir / "model_matrix.csv").read_bytes()).hexdigest()
    (matrix_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _edit_manifest(matrix_dir: Path, mutator):
    path = matrix_dir / "matrix_manifest.json"
    data = json.loads(path.read_text("utf-8"))
    mutator(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _all_keys(item)


class TrainingConfigTests(unittest.TestCase):
    def test_valid_config_passes(self):
        training.validate_training_config(training.load_config(CONFIG_PATH))

    def test_rejects_malformed_configs(self):
        base = training.load_config(CONFIG_PATH)
        mutations = [
            lambda c: c.pop("alpha"),
            lambda c: c.__setitem__("surprise", 1),
            lambda c: c.__setitem__("alpha", "1.0"),
            lambda c: c.__setitem__("alpha", 0),
            lambda c: c.__setitem__("alpha", -1.0),
            lambda c: c.__setitem__("max_iter", 0),
            lambda c: c.__setitem__("max_iter", 1.5),
            lambda c: c.__setitem__("max_iter", True),
            lambda c: c.__setitem__("fit_partition", "final_test"),
            lambda c: c.__setitem__("model", "sklearn.LogisticRegression"),
            lambda c: c.__setitem__("accepted_dataset_origin", "REAL"),
        ]
        for mutate in mutations:
            config = json.loads(json.dumps(base))
            mutate(config)
            with self.assertRaises(TrainingError):
                training.validate_training_config(config)


class TrainingValidationTests(unittest.TestCase):
    """Rejections that occur before any fit — dependency-free."""

    def _train(self, matrix_dir, output):
        return train_ordinal_model(matrix_dir, CONFIG_PATH, output)

    def test_missing_dependencies_raise_runtime_error(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write_matrix(root / "matrix", _dev_rows())
            with mock.patch("importlib.util.find_spec", return_value=None):
                with self.assertRaises(RuntimeError) as ctx:
                    self._train(root / "matrix", root / "out")
        self.assertIn("[pipeline]", str(ctx.exception))

    def test_matrix_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m = _write_matrix(root / "matrix", _dev_rows())
            csv_path = m / "model_matrix.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                table = list(csv.reader(handle))
            table[1][5] = "9.999999"  # tamper one feature cell; keep the CSV well-formed
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(table)
            # manifest model_matrix_sha256 is now stale -> the SHA guard must fire
            with self.assertRaises(TrainingError) as ctx:
                self._train(m, root / "out")
        self.assertIn("model_matrix_sha256", str(ctx.exception))

    def test_manifest_field_rejections(self):
        mutations = [
            lambda d: d.__setitem__("matrix_builder_version", "0.0.0"),
            lambda d: d.__setitem__("data_contract_version", "v999"),
            lambda d: d.__setitem__("rulebook_version", "v999"),
            lambda d: d.__setitem__("dataset_origin", "REAL_DATA"),
            lambda d: d.__setitem__("feature_columns", ["a", "b", "c", "d", "e"]),
            lambda d: d.__setitem__("target_column", "risk_label"),
            lambda d: d.__setitem__("target_label_column", "risk_label_index"),
            lambda d: d.__setitem__("class_mapping", {"Safe": 3, "Monitor": 2, "Caution": 1, "Urgent": 0}),
            lambda d: d.__setitem__("class_order", ["Urgent", "Caution", "Monitor", "Safe"]),
            lambda d: d.__setitem__("eligible_row_count", 999),
            lambda d: d.__setitem__("class_counts", {"Safe": 1, "Monitor": 1, "Caution": 1, "Urgent": 1}),
            lambda d: d.__setitem__("class_counts_by_partition", {"development": {"Safe": 999}}),
            lambda d: d.__setitem__("partition_values", ["validation"]),
            lambda d: d.__setitem__("source_candidate_window_count", 999),
            lambda d: d.__setitem__("excluded_row_count", 999),
            lambda d: d.__setitem__("exclusion_counts", {"PARTIAL_WINDOW": 999}),
        ]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                m = _write_matrix(root / "matrix", _dev_rows())
                _edit_manifest(m, mutate)
                with self.assertRaises(TrainingError):
                    self._train(m, root / "out")

    def test_bad_header_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m = _write_matrix(root / "matrix", _dev_rows())
            (m / "model_matrix.csv").write_text("wrong,header\n", encoding="utf-8")
            _rewrite_manifest_sha(m)
            with self.assertRaises(TrainingError):
                self._train(m, root / "out")

    def test_row_level_rejections(self):
        def with_rows(mutate_rows):
            rows = _dev_rows()
            mutate_rows(rows)
            return rows

        def nan_feature(rows):
            rows[0]["feats"] = ["nan", "1", "2", "3", "4"]

        def inf_feature(rows):
            rows[0]["feats"] = ["inf", "1", "2", "3", "4"]

        def dup_window(rows):
            rows.append(dict(rows[0]))

        def group_leak(rows):
            rows.append(_row(999, "validation", "Safe", [0.0] * 5))
            rows[-1]["bag"] = rows[0]["bag"]  # bag now spans development + validation

        for mutate in (nan_feature, inf_feature, dup_window, group_leak):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                m = _write_matrix(root / "matrix", with_rows(mutate))
                with self.assertRaises(TrainingError):
                    self._train(m, root / "out")

    def test_label_index_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m = _write_matrix(root / "matrix", _dev_rows())
            # break the risk_label/index pairing directly in the CSV, refresh the hash
            text = (m / "model_matrix.csv").read_text("utf-8").splitlines()
            parts = text[1].split(",")
            parts[-1] = "3" if parts[-1] != "3" else "0"
            text[1] = ",".join(parts)
            (m / "model_matrix.csv").write_text("\r\n".join(text) + "\r\n", encoding="utf-8")
            _rewrite_manifest_sha(m)
            with self.assertRaises(TrainingError):
                self._train(m, root / "out")

    def test_final_test_present_rejected(self):
        rows = _dev_rows() + [_row(500, "final_test", "Safe", [0.0] * 5)]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m = _write_matrix(root / "matrix", rows)
            with self.assertRaises(TrainingError):
                self._train(m, root / "out")

    def test_final_test_source_candidates_rejected_even_when_all_are_excluded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m = _write_matrix(root / "matrix", _dev_rows())

            def add_excluded_final_test_candidate(manifest):
                manifest["source_candidate_window_count"] += 1
                manifest["excluded_row_count"] += 1
                manifest["exclusion_counts"]["PARTIAL_WINDOW"] += 1
                manifest["source_partition_window_counts"]["final_test"] = 1
                manifest["partition_values"] = sorted(
                    manifest["source_partition_window_counts"]
                )

            _edit_manifest(m, add_excluded_final_test_candidate)
            with self.assertRaises(TrainingError):
                self._train(m, root / "out")

    def test_missing_development_class_rejected(self):
        rows = [r for r in _dev_rows() if r["risk"] != "Urgent"]  # no Urgent in development
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m = _write_matrix(root / "matrix", rows)
            with self.assertRaises(TrainingError):
                self._train(m, root / "out")

    def test_no_development_rows_rejected(self):
        rows = [dict(r, partition="validation",
                     window_id=r["window_id"].replace("development", "validation"),
                     session="v" + r["session"]) for r in _dev_rows()]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m = _write_matrix(root / "matrix", rows)
            with self.assertRaises(TrainingError):
                self._train(m, root / "out")

    def test_validation_failure_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            m = _write_matrix(root / "matrix", _dev_rows())
            _edit_manifest(m, lambda d: d.__setitem__("dataset_origin", "REAL_DATA"))
            out = root / "out"
            out.mkdir()
            (out / "ordinal_model.json").write_bytes(b"SENTINEL")
            with self.assertRaises(TrainingError):
                train_ordinal_model(m, CONFIG_PATH, out, overwrite=True)
            self.assertEqual((out / "ordinal_model.json").read_bytes(), b"SENTINEL")


@unittest.skipUnless(PIPELINE_AVAILABLE, "requires the optional [pipeline] dependencies")
class TrainingFitTests(unittest.TestCase):
    def _fit(self, root, rows=None, output="out", overwrite=False):
        m = _write_matrix(root / "matrix", rows if rows is not None else _dev_rows())
        return train_ordinal_model(m, CONFIG_PATH, root / output, overwrite=overwrite), root / output

    def test_export_shapes_and_ordering(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, out = self._fit(root)
            model = json.loads((out / "ordinal_model.json").read_text("utf-8"))
        self.assertEqual(len(model["beta"]), 5)
        self.assertEqual(len(model["theta"]), 3)
        self.assertTrue(all(model["theta"][i] < model["theta"][i + 1] for i in range(2)))
        self.assertEqual(model["feature_order"], list(FEATURE_COLUMNS))
        self.assertEqual(model["class_mapping"], dict(CLASS_NAME_TO_INDEX))

    def test_forward_parity_and_sanity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, _ = self._fit(root)
        sanity = manifest["model_sanity"]
        self.assertTrue(sanity["forward_label_parity"])
        self.assertLessEqual(sanity["forward_max_probability_difference"], 1e-6)
        self.assertLessEqual(sanity["reference_max_probability_sum_error"], 1e-9)
        self.assertLessEqual(sanity["exported_max_probability_sum_error"], 1e-9)
        self.assertTrue(sanity["reference_probabilities_finite"])
        self.assertTrue(sanity["exported_probabilities_finite"])
        self.assertTrue(sanity["theta_strictly_increasing"])

    def test_exported_probability_sanity_rejects_nan_and_bad_sum(self):
        import numpy as np

        reference = np.array([[0.25, 0.25, 0.25, 0.25]])
        pred = np.array([0])
        common = ([0.0] * 5, [1.0] * 5, [0.0] * 5, [0.0, 1.0, 2.0])
        invalid_forward = (
            np.array([[np.nan, np.nan, np.nan, np.nan]]),
            np.array([[0.2, 0.2, 0.2, 0.2]]),
            np.array([[0.5, 0.5, 0.0]]),
        )
        for forward in invalid_forward:
            with self.assertRaises(TrainingError):
                training._model_sanity(
                    np, *common, reference, forward, pred, pred
                )

    def test_output_hash_is_classified_as_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, out = self._fit(root)
            model_sha = hashlib.sha256(
                (out / "ordinal_model.json").read_bytes()
            ).hexdigest()
        self.assertNotIn("ordinal_model_json", manifest["input_sha256"])
        self.assertEqual(
            manifest["output_sha256"], {"ordinal_model_json": model_sha}
        )

    def test_scaler_fitted_on_development_only(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, out = self._fit(root)
            model = json.loads((out / "ordinal_model.json").read_text("utf-8"))
        # development feature mean is 1.5 + 0.1*mean(0..4) = 1.7 for every feature
        for value in model["scaler"]["mean"]:
            self.assertAlmostEqual(value, 1.7, places=6)

    def test_validation_rows_do_not_influence_model(self):
        dev = _dev_rows()
        calm = dev + [_row(900, "validation", "Safe", [0.0] * 5),
                      _row(901, "validation", "Urgent", [3.0] * 5)]
        wild = dev + [_row(900, "validation", "Safe", [1e6] * 5),
                      _row(901, "validation", "Urgent", [-1e6] * 5)]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write_matrix(root / "calm", calm)
            _write_matrix(root / "wild", wild)
            train_ordinal_model(root / "calm", CONFIG_PATH, root / "calm-out")
            train_ordinal_model(root / "wild", CONFIG_PATH, root / "wild-out")
            a = (root / "calm-out" / "ordinal_model.json").read_bytes()
            b = (root / "wild-out" / "ordinal_model.json").read_bytes()
        self.assertEqual(a, b)

    def test_byte_identical_repeated_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._fit(root, output="a")
            self._fit(root, output="b")
            for f in ("ordinal_model.json", "training_manifest.json"):
                self.assertEqual((root / "a" / f).read_bytes(), (root / "b" / f).read_bytes())

    def test_overwrite_refusal_preserves_unrelated(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, out = self._fit(root)
            keep = out / "keep.txt"
            keep.write_text("k", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train_ordinal_model(root / "matrix", CONFIG_PATH, out)
            train_ordinal_model(root / "matrix", CONFIG_PATH, out, overwrite=True)
            self.assertEqual(keep.read_text("utf-8"), "k")

    def test_no_performance_or_pass_fail_fields(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, out = self._fit(root)
            model = json.loads((out / "ordinal_model.json").read_text("utf-8"))
        for artifact in (manifest, model):
            offending = {k for k in _all_keys(artifact) if k.lower() in BANNED_KEYS}
            self.assertEqual(offending, set())
        self.assertIn("does not expose optimizer convergence", manifest["optimizer_convergence_status"])


@unittest.skipUnless(PIPELINE_AVAILABLE, "requires the optional [pipeline] dependencies")
class TrainingSyntheticIntegrationTests(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        from ostosense_ai import features, labeling, matrix, synthetic

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw, feat, lbl, mtx, model = (root / d for d in ("raw", "feat", "lbl", "mtx", "model"))
            synthetic.generate(
                synthetic.load_config(REPO_ROOT / "configs" / "synthetic-v0.2.json"), 20260722, raw)
            features.extract(raw, features.load_config(REPO_ROOT / "configs" / "features-v0.1.json"), feat)
            labeling.label_dataset(
                raw, LABELING_FIX / "protocol_manifest.csv", LABELING_FIX / "partition_manifest.csv",
                LABELING_FIX / "boundary-engineering-test-only-v0.1.json", lbl, features_dir=feat)
            matrix.build_model_matrix(feat, lbl, mtx)
            manifest = train_ordinal_model(mtx, CONFIG_PATH, model)
        self.assertEqual(manifest["fitted_row_count"], 76)
        self.assertEqual(manifest["ignored_row_count"], 0)
        self.assertEqual(
            manifest["fitted_class_counts"], {"Safe": 42, "Monitor": 16, "Caution": 12, "Urgent": 6})
        self.assertTrue(manifest["model_sanity"]["forward_label_parity"])


if __name__ == "__main__":
    unittest.main()
