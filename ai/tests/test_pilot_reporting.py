import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ostosense_ai import pilot_data, pilot_reporting, training


REPORTING_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "scipy", "sklearn", "mord", "matplotlib", "PIL")
)
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = REPO_ROOT / "configs" / "training-v0.1.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix_row(index, partition, class_index, offset):
    class_name = training.CLASS_NAMES[class_index]
    base = class_index * 2.0 + offset
    return {
        "window_id": f"{partition}-w{index:04d}",
        "session_id": f"{partition}-session-{index}",
        "bag_id": f"{partition}-bag-{index}",
        "sensor_id": f"{partition}-sensor-{index}",
        "dataset_partition": partition,
        **{
            column: f"{base + feature_index * 0.03:.6f}"
            for feature_index, column in enumerate(training.FEATURE_COLUMNS)
        },
        "risk_label": class_name,
        "risk_label_index": str(class_index),
    }


def _write_matrix(root: Path):
    matrix_dir = root / "matrix"
    matrix_dir.mkdir()
    rows = []
    index = 0
    for class_index in range(4):
        for repeat in range(6):
            rows.append(_matrix_row(index, "development", class_index, repeat * 0.12))
            index += 1
    for class_index in range(4):
        for repeat in range(2):
            rows.append(_matrix_row(index, "validation", class_index, 0.05 + repeat * 0.08))
            index += 1

    csv_path = matrix_dir / "model_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=training.MODEL_MATRIX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    class_counts = {name: 0 for name in training.CLASS_NAMES}
    partition_counts = {}
    class_counts_by_partition = {}
    for row in rows:
        partition = row["dataset_partition"]
        class_counts[row["risk_label"]] += 1
        partition_counts[partition] = partition_counts.get(partition, 0) + 1
        class_counts_by_partition.setdefault(
            partition, {name: 0 for name in training.CLASS_NAMES}
        )
        class_counts_by_partition[partition][row["risk_label"]] += 1
    manifest = {
        "matrix_builder_version": training.matrix.MATRIX_BUILDER_VERSION,
        "data_contract_version": training.DATA_CONTRACT_VERSION,
        "rulebook_version": training.RULEBOOK_VERSION,
        "dataset_origin": training.ALLOWED_ORIGIN,
        "feature_columns": list(training.FEATURE_COLUMNS),
        "target_column": "risk_label_index",
        "target_label_column": "risk_label",
        "class_mapping": dict(training.CLASS_NAME_TO_INDEX),
        "class_order": list(training.CLASS_NAMES),
        "audit_columns": list(training.matrix.AUDIT_COLUMNS),
        "grouping_columns": list(training.matrix.GROUPING_COLUMNS),
        "partition_column": training.matrix.PARTITION_COLUMN,
        "window_convention": dict(training.matrix._CANONICAL_CONVENTION),
        "partition_values": sorted(partition_counts),
        "source_partition_window_counts": dict(partition_counts),
        "partition_row_counts": dict(partition_counts),
        "eligible_row_count": len(rows),
        "excluded_row_count": 0,
        "exclusion_counts": {},
        "class_counts": class_counts,
        "class_counts_by_partition": class_counts_by_partition,
        "source_candidate_window_count": len(rows),
        "model_matrix_sha256": _sha(csv_path),
        "warning": "test",
    }
    (matrix_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return matrix_dir


def _write_pilot(root: Path):
    pilot_dir = root / "pilot"
    pilot_dir.mkdir()
    matrix_path = pilot_dir / "sensor_correlation_median.csv"
    matrix = [
        [1.0, 0.2, 0.1, 0.0, -0.1],
        [0.2, 1.0, 0.0, 0.1, 0.05],
        [0.1, 0.0, 1.0, 0.58, 0.15],
        [0.0, 0.1, 0.58, 1.0, 0.25],
        [-0.1, 0.05, 0.15, 0.25, 1.0],
    ]
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(pilot_data.CORRELATION_COLUMNS)
        for channel, row in zip(pilot_data.SENSOR_CHANNELS, matrix):
            writer.writerow([channel, *row])
    manifest = {
        "pilot_preparer_version": pilot_data.PILOT_PREPARER_VERSION,
        "dataset_origin": pilot_data.DATASET_ORIGIN,
        "output_sha256": {"sensor_correlation_median.csv": _sha(matrix_path)},
        "correlation": {
            "included_sessions": [f"S{index:02d}" for index in range(10)],
            "excluded_sessions": ["P006"],
        },
    }
    (pilot_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return pilot_dir


@unittest.skipUnless(REPORTING_AVAILABLE, "optional reporting stack is unavailable")
class OptimizerTraceTests(unittest.TestCase):
    def test_trace_is_finite_nonincreasing_and_matches_canonical_mord(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            matrix_dir = _write_matrix(root)
            trace, parity = pilot_reporting.build_optimizer_trace(
                matrix_dir, TRAINING_CONFIG
            )
        self.assertGreater(len(trace), 1)
        objectives = [float(row["objective"]) for row in trace]
        self.assertTrue(all(value == value for value in objectives))
        self.assertTrue(
            all(
                later <= earlier + 1e-8 * max(1.0, abs(earlier))
                for earlier, later in zip(objectives, objectives[1:])
            )
        )
        self.assertEqual(float(trace[0]["objective_normalized"]), 1.0)
        self.assertTrue(all(0.0 <= float(row["validation_macro_f1"]) <= 1.0 for row in trace))
        self.assertLessEqual(parity["beta_max_absolute_difference"], 1e-6)
        self.assertLessEqual(parity["theta_max_absolute_difference"], 1e-6)
        self.assertLessEqual(parity["probability_max_absolute_difference"], 1e-6)
        self.assertTrue(parity["class_prediction_parity"])


@unittest.skipUnless(REPORTING_AVAILABLE, "optional reporting stack is unavailable")
class ProgressFigureTests(unittest.TestCase):
    def test_four_pngs_are_nonblank_300_dpi_and_outputs_are_deterministic(self):
        from PIL import Image, ImageStat

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pilot_dir = _write_pilot(root)
            matrix_dir = _write_matrix(root)
            first = root / "first"
            second = root / "second"
            manifest = pilot_reporting.generate_progress_report(
                pilot_dir, matrix_dir, TRAINING_CONFIG, first
            )
            pilot_reporting.generate_progress_report(
                pilot_dir, matrix_dir, TRAINING_CONFIG, second
            )

            png_names = [name for name in pilot_reporting.REPORT_FILES if name.endswith(".png")]
            self.assertEqual(len(png_names), 4)
            for file_name in png_names:
                self.assertEqual((first / file_name).read_bytes(), (second / file_name).read_bytes())
                with Image.open(first / file_name) as image:
                    self.assertGreater(image.width, 1000)
                    self.assertGreater(image.height, 700)
                    dpi = image.info.get("dpi")
                    self.assertIsNotNone(dpi)
                    assert dpi is not None
                    self.assertAlmostEqual(dpi[0], 300, delta=1)
                    extrema = ImageStat.Stat(image.convert("L")).extrema[0]
                    self.assertLess(extrema[0], extrema[1])
            self.assertEqual(manifest["real_correlation_session_count"], 10)
            self.assertEqual(manifest["real_correlation_excluded_sessions"], ["P006"])
            self.assertTrue(manifest["optimizer_parity"]["class_prediction_parity"])
            with self.assertRaises(FileExistsError):
                pilot_reporting.generate_progress_report(
                    pilot_dir, matrix_dir, TRAINING_CONFIG, first
                )

    def test_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pilot_dir = _write_pilot(root)
            matrix_dir = _write_matrix(root)
            with (pilot_dir / "sensor_correlation_median.csv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            with self.assertRaises(pilot_reporting.PilotReportingError):
                pilot_reporting.generate_progress_report(
                    pilot_dir, matrix_dir, TRAINING_CONFIG, root / "out"
                )


if __name__ == "__main__":
    unittest.main()
