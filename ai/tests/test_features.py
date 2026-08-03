import csv
import json
import tempfile
import unittest
from pathlib import Path

from ostosense_ai import features
from ostosense_contract import (
    Arm,
    CapQuality,
    EndReason,
    LigQuality,
    SampleRecord,
    SessionRecord,
    Tier1CsvLogger,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "features-v0.1.json"
TS0 = 1_700_000_000_000


def _config() -> dict:
    return features.load_config(CONFIG_PATH)


def _write_dataset(input_dir: Path, sessions: list[dict]) -> None:
    logger = Tier1CsvLogger(input_dir)
    for session in sessions:
        for timestamp, cap, cap_quality in session["samples"]:
            logger.append_sample(
                SampleRecord.create(
                    timestamp=timestamp,
                    session_id=session["session_id"],
                    capacitance_raw=cap,
                    lig_raw=400.0,
                    cap_quality=cap_quality,
                    lig_quality=LigQuality.OK,
                )
            )
    for session in sessions:
        logger.append_session(
            SessionRecord(
                session_id=session["session_id"],
                arm=session.get("arm", Arm.SAFE),
                bag_id=session.get("bag_id", "bag-1"),
                sensor_id=session.get("sensor_id", "sensor-1"),
                device_id="dev-1",
                fluid_type="synthetic",
                operator_id="op-1",
                baseline_value=session["baseline"],
                baseline_std=0.0,
                start_timestamp=session["start"],
                end_timestamp=session["end"],
                end_reason=EndReason.CEILING_REACHED,
                model_version="",
                firmware_version="fw-1",
            )
        )


def _read_features(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "features.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _replace_csv_value(
    path: Path, row_index: int, column: str, value: str
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    rows[row_index][column] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _linear_session(session_id: str = "golden") -> dict:
    # cap_i = 12 + i for i in 0..120; i=0 sits at t_ref and is excluded from window 0.
    samples = [(TS0 + i * 1000, 12.0 + i, CapQuality.OK) for i in range(121)]
    return {
        "session_id": session_id,
        "baseline": 12.0,
        "start": TS0,
        "end": TS0 + 120_000,
        "samples": samples,
    }


class FeatureExtractorTests(unittest.TestCase):
    def _run(self, sessions: list[dict], tmp: Path, *, overwrite: bool = False) -> dict:
        input_dir = tmp / "input"
        output_dir = tmp / "output"
        _write_dataset(input_dir, sessions)
        manifest = features.extract(
            input_dir, _config(), output_dir, overwrite=overwrite
        )
        return {
            "manifest": manifest,
            "rows": _read_features(output_dir),
            "output": output_dir,
        }

    # 1 + 3: exact (t-W,t] membership and a valid window has exactly 120 unique samples.
    def test_membership_and_full_window(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            result = self._run([_linear_session()], Path(name))
        rows = result["rows"]
        self.assertEqual(len(rows), 1)  # only k=0 fits before session end
        row = rows[0]
        self.assertEqual(row["feature_valid"], "true")
        # The t_ref sample is excluded while the sample at t is included.
        self.assertEqual(int(row["sample_count"]), 120)
        self.assertEqual(int(row["window_start"]), TS0)
        self.assertEqual(int(row["window_end"]), TS0 + 120_000)

    # 2: first endpoint and 10-second stride follow the protocol formula.
    def test_stride_and_first_endpoint(self) -> None:
        session = {
            "session_id": "stride",
            "baseline": 12.0,
            "start": TS0,
            "end": TS0 + 199_000,
            "samples": [(TS0 + i * 1000, 12.0, CapQuality.OK) for i in range(200)],
        }
        with tempfile.TemporaryDirectory() as name:
            rows = self._run([session], Path(name))["rows"]
        self.assertEqual(len(rows), 8)
        for k, row in enumerate(rows):
            self.assertEqual(int(row["window_index"]), k)
            self.assertEqual(int(row["window_end"]), TS0 + (120 + 10 * k) * 1000)
            self.assertEqual(int(row["window_start"]), int(row["window_end"]) - 120_000)
            self.assertEqual(row["feature_valid"], "true")

    # 4: windows never cross session boundaries.
    def test_windows_never_cross_session_boundaries(self) -> None:
        bounds = {"a": (TS0, TS0 + 199_000), "b": (TS0 + 10_000_000, TS0 + 10_199_000)}
        sessions = [
            {
                "session_id": sid,
                "baseline": 12.0,
                "start": start,
                "end": end,
                "samples": [
                    (start + i * 1000, 12.0, CapQuality.OK)
                    for i in range(200)
                ],
            }
            for sid, (start, end) in bounds.items()
        ]
        with tempfile.TemporaryDirectory() as name:
            rows = self._run(sessions, Path(name))["rows"]
        self.assertEqual({r["session_id"] for r in rows}, set(bounds))
        for row in rows:
            start, end = bounds[row["session_id"]]
            self.assertGreaterEqual(int(row["window_start"]), start)
            self.assertLessEqual(int(row["window_end"]), end)

    # 5: any non-OK capacitive sample invalidates the window.
    def test_non_ok_capacitive_sample_invalidates(self) -> None:
        session = _linear_session("invalid")
        session["samples"][60] = (TS0 + 60 * 1000, 42.0, CapQuality.DISCONNECTED)
        with tempfile.TemporaryDirectory() as name:
            rows = self._run([session], Path(name))["rows"]
        self.assertEqual(rows[0]["feature_valid"], "false")
        self.assertEqual(rows[0]["exclusion_reason"], "INVALID_CAP_QUALITY")
        for column in features.FEATURE_COLUMNS:
            self.assertEqual(rows[0][column], "")

    # 6: missing samples produce PARTIAL_WINDOW.
    def test_missing_samples_produce_partial_window(self) -> None:
        session = _linear_session("partial")
        del session["samples"][60]  # drop one interior member -> 119 in window 0
        with tempfile.TemporaryDirectory() as name:
            rows = self._run([session], Path(name))["rows"]
        self.assertEqual(int(rows[0]["sample_count"]), 119)
        self.assertEqual(rows[0]["exclusion_reason"], "PARTIAL_WINDOW")

    # 7: duplicate timestamps are not silently deduplicated.
    def test_duplicate_timestamps_flagged_not_deduplicated(self) -> None:
        session = _linear_session("dup")
        session["samples"].insert(61, (TS0 + 60 * 1000, 99.0, CapQuality.OK))
        with tempfile.TemporaryDirectory() as name:
            rows = self._run([session], Path(name))["rows"]
        self.assertEqual(rows[0]["feature_valid"], "false")
        self.assertEqual(rows[0]["exclusion_reason"], "DUPLICATE_TIMESTAMP")
        self.assertEqual(int(rows[0]["sample_count"]), 121)  # both rows kept

    # 8: adjacent timing outside 800-1200 ms is rejected.
    def test_timing_out_of_tolerance(self) -> None:
        gaps = [1000] * 58 + [1300] + [1000] * 59 + [700]  # 119 gaps, sum 119000
        offsets = [1000]
        for gap in gaps:
            offsets.append(offsets[-1] + gap)
        samples = [(TS0, 12.0, CapQuality.OK)]
        samples += [(TS0 + offset, 12.0, CapQuality.OK) for offset in offsets]
        session = {
            "session_id": "timing",
            "baseline": 12.0,
            "start": TS0,
            "end": TS0 + 120_000,
            "samples": samples,
        }
        with tempfile.TemporaryDirectory() as name:
            rows = self._run([session], Path(name))["rows"]
        self.assertEqual(int(rows[0]["sample_count"]), 120)
        self.assertEqual(rows[0]["exclusion_reason"], "TIMING_OUT_OF_TOLERANCE")

    # 9: feature values match a hand-calculated golden fixture.
    def test_features_match_golden_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            row = self._run([_linear_session()], Path(name))["rows"][0]
        # members are deltas 1..120 at elapsed seconds 0..119.
        self.assertEqual(row["cap_delta_mean"], "60.5")
        self.assertEqual(row["cap_delta_last"], "120.0")
        self.assertEqual(row["cap_delta_slope_per_s"], "1.0")
        self.assertEqual(row["cap_delta_variance"], "1199.916667")
        self.assertEqual(row["cap_delta_range"], "119.0")

    # 10: forbidden metadata is absent from FEATURE_COLUMNS (and its header slice).
    def test_forbidden_names_absent_from_feature_columns(self) -> None:
        self.assertTrue(
            features.FORBIDDEN_FEATURE_NAMES.isdisjoint(features.FEATURE_COLUMNS)
        )
        features._assert_feature_columns_safe()
        with tempfile.TemporaryDirectory() as name:
            output = self._run([_linear_session()], Path(name))["output"]
            header = (
                (output / "features.csv")
                .read_text("utf-8")
                .splitlines()[0]
                .split(",")
            )
        feature_slice = header[len(features.AUDIT_COLUMNS):]
        self.assertEqual(tuple(feature_slice), features.FEATURE_COLUMNS)
        self.assertTrue(features.FORBIDDEN_FEATURE_NAMES.isdisjoint(feature_slice))

    # 11: same input and config produce byte-identical outputs.
    def test_byte_identical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            input_dir = tmp / "input"
            _write_dataset(input_dir, [_linear_session(), _linear_session("second")])
            out_a, out_b = tmp / "a", tmp / "b"
            features.extract(input_dir, _config(), out_a)
            features.extract(input_dir, _config(), out_b)
            for name_ in ("features.csv", "feature_manifest.json"):
                self.assertEqual(
                    (out_a / name_).read_bytes(),
                    (out_b / name_).read_bytes(),
                )

    # 12: existing outputs are protected unless --overwrite is explicit.
    def test_existing_outputs_protected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            first = self._run([_linear_session()], tmp)
            output = first["output"]
            sentinel = output / "unrelated.txt"
            sentinel.write_text("keep me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                features.extract(tmp / "input", _config(), output)
            # overwrite succeeds and leaves unrelated files intact
            features.extract(tmp / "input", _config(), output, overwrite=True)
            self.assertEqual(sentinel.read_text("utf-8"), "keep me")

    # 13: invalid config or input fails before existing outputs are mutated.
    def test_invalid_inputs_do_not_mutate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            _write_dataset(tmp / "input", [_linear_session()])
            output = tmp / "output"
            output.mkdir()
            (output / "features.csv").write_bytes(b"SENTINEL")

            broken_config = _config()
            broken_config["window_interval"] = "[t-W,t]"
            with self.assertRaises(ValueError):
                features.extract(tmp / "input", broken_config, output)
            self.assertEqual((output / "features.csv").read_bytes(), b"SENTINEL")

            # orphan sample: valid config, malformed input, still no mutation
            orphan_input = tmp / "orphan"
            logger = Tier1CsvLogger(orphan_input)
            logger.append_sample(
                SampleRecord.create(
                    timestamp=TS0,
                    session_id="ghost",
                    capacitance_raw=12.0,
                    lig_raw=400.0,
                    cap_quality=CapQuality.OK,
                    lig_quality=LigQuality.OK,
                )
            )
            logger.append_session(
                SessionRecord(
                    session_id="real",
                    arm=Arm.SAFE,
                    bag_id="bag-1",
                    sensor_id="sensor-1",
                    device_id="dev-1",
                    fluid_type="synthetic",
                    operator_id="op-1",
                    baseline_value=12.0,
                    baseline_std=0.0,
                    start_timestamp=TS0,
                    end_timestamp=TS0 + 1000,
                    end_reason=EndReason.CEILING_REACHED,
                    model_version="",
                    firmware_version="fw-1",
                )
            )
            with self.assertRaises(ValueError):
                features.extract(orphan_input, _config(), output, overwrite=True)
            self.assertEqual((output / "features.csv").read_bytes(), b"SENTINEL")

    # 14: a synthetic-v0.2 style input manifest propagates its dataset origin.
    def test_input_origin_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            input_dir = tmp / "input"
            _write_dataset(input_dir, [_linear_session()])
            (input_dir / "manifest.json").write_text(
                json.dumps({"dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY"}),
                encoding="utf-8",
            )
            with_origin = features.extract(input_dir, _config(), tmp / "with")
            self.assertEqual(
                with_origin["input_dataset_origin"],
                "SYNTHETIC_PIPELINE_TEST_ONLY",
            )

            (input_dir / "manifest.json").unlink()
            without_origin = features.extract(input_dir, _config(), tmp / "without")
            self.assertEqual(
                without_origin["input_dataset_origin"],
                "UNDECLARED_INPUT_ORIGIN",
            )

    # 15: manifest counts reconcile exactly with features.csv.
    def test_manifest_counts_reconcile(self) -> None:
        session = _linear_session("recon")
        session["samples"][10] = (TS0 + 10 * 1000, 5.0, CapQuality.DISCONNECTED)
        with tempfile.TemporaryDirectory() as name:
            result = self._run([session], Path(name))
        manifest, rows = result["manifest"], result["rows"]
        valid = sum(1 for r in rows if r["feature_valid"] == "true")
        excluded = sum(1 for r in rows if r["feature_valid"] == "false")
        self.assertEqual(manifest["candidate_window_count"], len(rows))
        self.assertEqual(manifest["valid_window_count"], valid)
        self.assertEqual(manifest["excluded_window_count"], excluded)
        self.assertEqual(valid + excluded, len(rows))
        self.assertEqual(sum(manifest["exclusion_reason_counts"].values()), excluded)

    def test_rejects_contract_invalid_session_rows(self) -> None:
        cases = (
            ("baseline_std", "nan"),
            ("bag_id", ""),
        )
        for column, value in cases:
            with self.subTest(column=column):
                with tempfile.TemporaryDirectory() as name:
                    root = Path(name)
                    input_dir = root / "input"
                    output_dir = root / "output"
                    _write_dataset(input_dir, [_linear_session()])
                    _replace_csv_value(
                        input_dir / "sessions.csv", 0, column, value
                    )
                    with self.assertRaisesRegex(
                        ValueError, "invalid sessions.csv row"
                    ):
                        features.extract(input_dir, _config(), output_dir)
                    self.assertFalse((output_dir / "features.csv").exists())

    def test_rejects_contract_invalid_sample_rows(self) -> None:
        cases = (
            ("lig_raw", "nan"),
            ("system_quality", "UNSAFE"),
        )
        for column, value in cases:
            with self.subTest(column=column):
                with tempfile.TemporaryDirectory() as name:
                    root = Path(name)
                    input_dir = root / "input"
                    output_dir = root / "output"
                    _write_dataset(input_dir, [_linear_session()])
                    _replace_csv_value(
                        input_dir / "samples.csv", 1, column, value
                    )
                    with self.assertRaisesRegex(
                        ValueError, "invalid samples.csv row"
                    ):
                        features.extract(input_dir, _config(), output_dir)
                    self.assertFalse((output_dir / "features.csv").exists())

    def test_malformed_manifest_does_not_mutate_output(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_dir = root / "input"
            output_dir = root / "output"
            _write_dataset(input_dir, [_linear_session()])
            (input_dir / "manifest.json").write_text(
                "{broken", encoding="utf-8"
            )
            output_dir.mkdir()
            sentinel = output_dir / "features.csv"
            sentinel.write_bytes(b"SENTINEL")

            with self.assertRaisesRegex(ValueError, "invalid input manifest"):
                features.extract(
                    input_dir, _config(), output_dir, overwrite=True
                )
            self.assertEqual(sentinel.read_bytes(), b"SENTINEL")


class FeatureConfigValidationTests(unittest.TestCase):
    def test_rejects_malformed_configs(self) -> None:
        unknown = _config()
        unknown["surprise"] = 1
        with self.assertRaises(ValueError):
            features.validate_features_config(unknown)

        wrong_id = _config()
        wrong_id["config_id"] = "features-v9"
        with self.assertRaises(ValueError):
            features.validate_features_config(wrong_id)

        wrong_status = _config()
        wrong_status["status"] = "NOT LOCKED"
        with self.assertRaises(ValueError):
            features.validate_features_config(wrong_status)

        wrong_source = _config()
        wrong_source["working_source"] = "unverified.md"
        with self.assertRaises(ValueError):
            features.validate_features_config(wrong_source)

        missing = _config()
        del missing["stride_seconds"]
        with self.assertRaises(ValueError):
            features.validate_features_config(missing)

        bad_rate = _config()
        bad_rate["sampling_rate_hz"] = 2
        with self.assertRaises(ValueError):
            features.validate_features_config(bad_rate)

        bad_window = _config()
        bad_window["window_seconds"] = 1
        with self.assertRaises(ValueError):
            features.validate_features_config(bad_window)

        bad_stride = _config()
        bad_stride["stride_seconds"] = 1
        with self.assertRaises(ValueError):
            features.validate_features_config(bad_stride)

        bad_jitter = _config()
        bad_jitter["jitter_tolerance_ms"] = 201
        with self.assertRaises(ValueError):
            features.validate_features_config(bad_jitter)

        reordered = _config()
        reordered["features"] = list(reversed(reordered["features"]))
        with self.assertRaises(ValueError):
            features.validate_features_config(reordered)

        bad_interval = _config()
        bad_interval["window_interval"] = "[t-W,t]"
        with self.assertRaises(ValueError):
            features.validate_features_config(bad_interval)

        inconsistent = _config()
        inconsistent["stride_seconds"] = inconsistent["window_seconds"] + 1
        with self.assertRaises(ValueError):
            features.validate_features_config(inconsistent)


if __name__ == "__main__":
    unittest.main()
