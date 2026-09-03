import csv
import json
import tempfile
import unittest
from pathlib import Path

from ostosense_ai import pilot_data
from ostosense_ai.pilot_data import PilotDataError, prepare_pilot_dataset


def _config(sessions):
    return {
        "config_id": "real-pilot-v0.1",
        "status": "REAL_PILOT_UNLABELED",
        "expected_interval_ms": 100,
        "samples_per_second": 10,
        "baseline_seconds": 20,
        "window_seconds": 120,
        "stride_seconds": 10,
        "window_interval": "(t-W,t]",
        "allowed_statuses": ["NOT_EVALUATED", "OK", "DISCONNECTED", "SATURATED"],
        "model_channels": ["Kap_4", "Kap_5", "Kap_7"],
        "audit_only_channels": ["Res_15", "Res_16"],
        "sessions": sessions,
        "warning": "test warning",
    }


def _session(session_id="P001", file_name="P001.csv", include=True):
    return {
        "session_id": session_id,
        "file_name": file_name,
        "scenario": "test",
        "position": "test",
        "include_in_correlation": include,
    }


def _write_raw(path: Path, count=1255, *, offset=100, reverse=False):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(pilot_data.RAW_COLUMNS)
        for index in range(count):
            direction = -index if reverse else index
            writer.writerow(
                [
                    index,
                    offset + index * 100,
                    10 + direction,
                    "NOT_EVALUATED",
                    20 + direction * 2,
                    "NOT_EVALUATED",
                    1000 + direction * 3,
                    "NOT_EVALUATED",
                    2000 + direction * 4,
                    "NOT_EVALUATED",
                    3000 + direction * 5,
                    "NOT_EVALUATED",
                ]
            )


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class PilotConfigTests(unittest.TestCase):
    def test_strict_config_validation(self):
        valid = _config([_session()])
        pilot_data.validate_config(valid)
        mutations = [
            lambda c: c.pop("baseline_seconds"),
            lambda c: c.__setitem__("extra", 1),
            lambda c: c.__setitem__("model_channels", ["Res_15"]),
            lambda c: c.__setitem__("expected_interval_ms", 125),
            lambda c: c.__setitem__("window_interval", "[t-W,t]"),
        ]
        for mutate in mutations:
            candidate = json.loads(json.dumps(valid))
            mutate(candidate)
            with self.subTest(candidate=candidate):
                with self.assertRaises(PilotDataError):
                    pilot_data.validate_config(candidate)


class PilotPreparationTests(unittest.TestCase):
    def _prepare(self, root, config=None, count=1255, offset=100):
        input_dir = root / "input"
        input_dir.mkdir()
        _write_raw(input_dir / "P001.csv", count=count, offset=offset)
        config = config or _config([_session()])
        output = root / "output"
        manifest = prepare_pilot_dataset(input_dir, config, output)
        return input_dir, output, manifest

    def test_golden_aggregation_baseline_features_and_counts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _input, output, manifest = self._prepare(root)
            self.assertEqual(manifest["raw_row_count"], 1255)
            self.assertEqual(manifest["full_1hz_bin_count"], 125)
            self.assertEqual(manifest["partial_final_raw_sample_count"], 5)
            self.assertEqual(manifest["unlabeled_window_count"], 1)

            seconds = _read_rows(output / "samples_1hz.csv")
            self.assertEqual(float(seconds[0]["Kap_4"]), 1013.5)
            self.assertEqual(float(seconds[0]["Kap_4_baseline"]), 1298.5)
            expected_delta = (1013.5 - 1298.5) / 1298.5
            self.assertAlmostEqual(float(seconds[0]["Kap_4_delta_norm"]), expected_delta)

            windows = _read_rows(output / "window_features_unlabeled.csv")
            self.assertEqual(len(windows), 1)
            self.assertEqual(int(windows[0]["sample_count"]), 120)
            self.assertEqual(len(pilot_data.MODEL_FEATURE_COLUMNS), 15)
            self.assertTrue(set(pilot_data.MODEL_FEATURE_COLUMNS).issubset(windows[0]))
            self.assertNotIn("res_15_delta_mean", windows[0])
            self.assertNotIn("res_16_delta_mean", windows[0])

    def test_zero_and_hundred_ms_start_conventions_are_valid(self):
        for offset in (0, 100):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                _input, output, _manifest = self._prepare(root, offset=offset)
                rows = _read_rows(output / "samples_1hz.csv")
                self.assertEqual(float(rows[0]["elapsed_s"]), 1.0)
                self.assertEqual(int(rows[0]["elapsed_ms"]), offset + 900)

    def test_schema_timing_sample_number_and_status_fail_before_output(self):
        mutations = []

        def bad_header(table):
            table[0][2] = "wrong"

        def bad_time(table):
            table[5][1] = "999"

        def bad_sample(table):
            table[5][0] = "99"

        def bad_status(table):
            table[5][3] = "UNKNOWN"

        def empty_value(table):
            table[5][2] = ""

        mutations.extend((bad_header, bad_time, bad_sample, bad_status, empty_value))
        for mutate in mutations:
            with self.subTest(mutate=mutate.__name__), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                input_dir = root / "input"
                input_dir.mkdir()
                raw = input_dir / "P001.csv"
                _write_raw(raw)
                with raw.open(newline="", encoding="utf-8") as handle:
                    table = list(csv.reader(handle))
                mutate(table)
                with raw.open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerows(table)
                output = root / "output"
                with self.assertRaises(PilotDataError):
                    prepare_pilot_dataset(input_dir, _config([_session()]), output)
                self.assertFalse(output.exists())

    def test_correlation_is_symmetric_five_by_five_with_unit_diagonal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _input, output, manifest = self._prepare(root)
            rows = _read_rows(output / "sensor_correlation_median.csv")
            self.assertEqual(len(rows), 5)
            matrix = [
                [float(row[channel]) for channel in pilot_data.SENSOR_CHANNELS]
                for row in rows
            ]
            for index in range(5):
                self.assertEqual(matrix[index][index], 1.0)
                for other in range(5):
                    self.assertAlmostEqual(matrix[index][other], matrix[other][index])
            self.assertEqual(manifest["correlation"]["included_sessions"], ["P001"])

    def test_fault_session_is_excluded_from_equal_weight_correlation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_dir = root / "input"
            input_dir.mkdir()
            _write_raw(input_dir / "P001.csv")
            _write_raw(input_dir / "P006.csv", reverse=True)
            config = _config(
                [_session(), _session("P006", "P006.csv", include=False)]
            )
            manifest = prepare_pilot_dataset(input_dir, config, root / "output")
            self.assertEqual(manifest["correlation"]["included_sessions"], ["P001"])
            self.assertEqual(manifest["correlation"]["excluded_sessions"], ["P006"])
            self.assertTrue(
                all(count == 1 for count in manifest["correlation"]["valid_session_pair_counts"].values())
            )

    def test_determinism_overwrite_protection_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_dir = root / "input"
            input_dir.mkdir()
            source = input_dir / "P001.csv"
            _write_raw(source)
            source_before = source.read_bytes()
            config = _config([_session()])
            first = root / "first"
            second = root / "second"
            prepare_pilot_dataset(input_dir, config, first)
            prepare_pilot_dataset(input_dir, config, second)
            for file_name in pilot_data.OUTPUT_FILES:
                self.assertEqual((first / file_name).read_bytes(), (second / file_name).read_bytes())
            with self.assertRaises(FileExistsError):
                prepare_pilot_dataset(input_dir, config, first)
            self.assertEqual(source.read_bytes(), source_before)


if __name__ == "__main__":
    unittest.main()
