import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ostosense_ai import features, labeling
from ostosense_ai.labeling import LabelingError, MalformedRequiredEvents, label_dataset
from ostosense_contract import (
    Arm,
    CapQuality,
    EndReason,
    EventRecord,
    EventType,
    LigQuality,
    SampleRecord,
    SessionRecord,
    Tier1CsvLogger,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LABELING_FIX = FIXTURES / "ostosense-labeling-v0.1"
GOOD_BOUNDARY = LABELING_FIX / "boundary-engineering-test-only-v0.1.json"
FEATURES_CONFIG = REPO_ROOT / "configs" / "features-v0.1.json"
SKLEARN_AVAILABLE = importlib.util.find_spec("numpy") is not None

TS0 = 1_700_000_000_000


def _base(index: int) -> int:
    return TS0 + index * 10_000_000


def _ok_samples(base: int, duration: int, *, warmup: int = 0, lig_ok: bool = True):
    rows = []
    for i in range(duration):
        if i < warmup:
            capq, ligq = CapQuality.WARMING_UP, LigQuality.WARMING_UP
        else:
            capq = CapQuality.OK
            ligq = LigQuality.OK if lig_ok else LigQuality.BASELINE_INVALID
        rows.append((base + i * 1000, 12.0, capq, ligq))
    return rows


def _write_dataset(
    root: Path,
    sessions: list[dict],
    *,
    origin: str | None = "SYNTHETIC_PIPELINE_TEST_ONLY",
):
    input_dir = root / "input"
    logger = Tier1CsvLogger(input_dir)
    for spec in sessions:
        for ts, cap, capq, ligq in spec["samples"]:
            logger.append_sample(
                SampleRecord.create(
                    timestamp=ts,
                    session_id=spec["id"],
                    capacitance_raw=cap,
                    lig_raw=400.0,
                    cap_quality=capq,
                    lig_quality=ligq,
                )
            )
        for order, (ts, etype) in enumerate(spec.get("events", [])):
            logger.append_event(
                EventRecord(
                    event_id=f"{spec['id']}-e{order:03d}",
                    session_id=spec["id"],
                    timestamp=ts,
                    event_type=etype,
                    event_metadata={},
                )
            )
    for spec in sessions:
        last_ts = spec["samples"][-1][0]
        logger.append_session(
            SessionRecord(
                session_id=spec["id"],
                arm=spec["arm"],
                bag_id=spec["bag"],
                sensor_id=spec["sensor"],
                device_id=spec.get("device", "synthetic-device-001"),
                fluid_type="synthetic",
                operator_id="synthetic-operator-001",
                baseline_value=12.0,
                baseline_std=0.0,
                start_timestamp=spec["samples"][0][0],
                end_timestamp=last_ts,
                end_reason=spec["end_reason"],
                model_version="",
                firmware_version="fw-1",
            )
        )
    (input_dir / "manifest.json").write_text(
        json.dumps({"dataset_origin": origin}) if origin is not None else json.dumps({}),
        encoding="utf-8",
    )

    protocol_path = root / "protocol_manifest.csv"
    with protocol_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(labeling.PROTOCOL_MANIFEST_FIELDS)
        for spec in sessions:
            writer.writerow(
                [
                    spec["id"],
                    "v0.1-SYNTHETIC_TEST_ONLY",
                    spec.get("planned_arm", spec["arm"].value),
                    spec.get("horizon", ""),
                    "SYNTHETIC_TEST_ONLY",
                    "",
                    "",
                    "",
                    "",
                    "synthetic-operator-001",
                    spec.get("protocol_bag", spec["bag"]),
                    spec["sensor"],
                    spec.get("device", "synthetic-device-001"),
                ]
            )

    partition_path = root / "partition_manifest.csv"
    with partition_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(labeling.PARTITION_MANIFEST_FIELDS)
        for spec in sessions:
            writer.writerow(
                [
                    spec["id"],
                    spec.get("partition", "development"),
                    "p-SYNTHETIC_TEST_ONLY",
                    spec["bag"],
                    spec["sensor"],
                ]
            )
    return input_dir, protocol_path, partition_path


def _run(
    root,
    sessions,
    *,
    boundary=GOOD_BOUNDARY,
    features_dir=None,
    overwrite=False,
    origin: str | None = "SYNTHETIC_PIPELINE_TEST_ONLY",
):
    input_dir, protocol_path, partition_path = _write_dataset(root, sessions, origin=origin)
    return label_dataset(
        input_dir,
        protocol_path,
        partition_path,
        boundary,
        root / "output",
        features_dir=features_dir,
        overwrite=overwrite,
    )


def _read_labels(output_dir: Path) -> dict[str, dict[str, str]]:
    with (output_dir / "labels.csv").open(newline="", encoding="utf-8") as handle:
        return {row["window_id"]: row for row in csv.DictReader(handle)}


def _gradual_leak(session_id, index, *, horizon="", t_leak_s=310, duration=331):
    base = _base(index)
    return {
        "id": session_id,
        "arm": Arm.LEAK_GRADUAL,
        "bag": f"bag-{session_id}",
        "sensor": f"sensor-{session_id}",
        "end_reason": EndReason.LEAK_CONFIRMED,
        "horizon": horizon,
        "samples": _ok_samples(base, duration),
        "events": [
            (base + 5 * 1000, EventType.INJECTION_START),
            (base + 6 * 1000, EventType.INJECTION_END),
            (base + t_leak_s * 1000, EventType.PHYSICAL_LEAK_OBSERVED),
            (base + (t_leak_s + 3) * 1000, EventType.LEAK_FLAG_CONFIRMED),
        ],
    }


class LabelingLogicTests(unittest.TestCase):
    def test_boundary_inclusivity_and_all_four_classes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sid = "syn-grad"
            _run(root, [_gradual_leak(sid, 0)])
            rows = _read_labels(root / "output")
        # T_leak=310s, window_end=120+10k s; tau=310-window_end.
        self.assertEqual(rows[f"{sid}-win-0000"]["risk_label"], "Safe")      # tau=190 > B3
        self.assertEqual(rows[f"{sid}-win-0001"]["risk_label"], "Monitor")   # tau=180 == B3
        self.assertEqual(rows[f"{sid}-win-0010"]["risk_label"], "Caution")   # tau=90  == B2
        self.assertEqual(rows[f"{sid}-win-0016"]["risk_label"], "Urgent")    # tau=30  == B1
        self.assertEqual(rows[f"{sid}-win-0019"]["exclusion_reason"], "POST_LEAK")  # tau=0
        labels = {r["risk_label"] for r in rows.values() if r["label_valid"] == "true"}
        self.assertEqual(labels, {"Safe", "Monitor", "Caution", "Urgent"})
        for row in rows.values():
            if row["risk_label"]:
                self.assertEqual(
                    row["risk_label_index"], str(labeling.CLASS_NAME_TO_INDEX[row["risk_label"]])
                )
            else:
                self.assertEqual(row["risk_label_index"], "")

    def test_safe_horizon_completed_missing_and_too_short(self):
        cases = {
            "completed": (200, "180", "Safe", ""),
            "missing": (200, "", "", "CENSORED_NO_SAFE_HORIZON"),
            "too_short": (150, "180", "", "CENSORED_NO_SAFE_HORIZON"),
        }
        for name, (duration, horizon, expect_label, expect_reason) in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                base = _base(1)
                session = {
                    "id": "syn-safe",
                    "arm": Arm.SAFE,
                    "bag": "bag-safe",
                    "sensor": "sensor-safe",
                    "end_reason": EndReason.CEILING_REACHED,
                    "horizon": horizon,
                    "samples": _ok_samples(base, duration),
                    "events": [],
                }
                _run(root, [session])
                rows = _read_labels(root / "output")
            valid_labels = {r["risk_label"] for r in rows.values() if r["label_valid"] == "true"}
            reasons = {r["exclusion_reason"] for r in rows.values() if r["label_valid"] == "false"}
            if expect_label:
                self.assertIn(expect_label, valid_labels, name)
            if expect_reason:
                self.assertIn(expect_reason, reasons, name)

    def test_non_leaking_fill_completed_and_censored(self):
        base = _base(2)
        # Gradual non-leak: keep real pre-injection candidate windows, then fill.
        events = [
            (base + 180 * 1000, EventType.INJECTION_START),
            (base + 200 * 1000, EventType.INJECTION_END),
        ]
        for horizon, duration, expect in (("60", 400, "Safe"), ("", 400, "CENSORED_NO_SAFE_HORIZON")):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                session = {
                    "id": "syn-fill",
                    "arm": Arm.LEAK_GRADUAL,
                    "bag": "bag-fill",
                    "sensor": "sensor-fill",
                    "end_reason": EndReason.CEILING_REACHED,
                    "horizon": horizon,
                    "samples": _ok_samples(base, duration),
                    "events": events,
                }
                _run(root, [session])
                rows = _read_labels(root / "output")
            post = [r for r in rows.values() if int(r["window_end"]) >= base + 210 * 1000]
            values = {(r["risk_label"] or r["exclusion_reason"]) for r in post}
            self.assertIn(expect, values, horizon)
            # Dry windows before injection must exist and remain Safe.
            dry = [r for r in rows.values() if int(r["window_end"]) < base + 180 * 1000]
            self.assertTrue(dry, "fixture must exercise pre-injection dry windows")
            for r in dry:
                self.assertEqual(r["risk_label"], "Safe")

    def test_unplanned_physical_leak_sets_protocol_deviation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            session = _gradual_leak("syn-dev", 3, horizon="180")
            manifest = _run(root, [session])
            rows = _read_labels(root / "output")
        self.assertEqual(manifest["protocol_deviation_session_count"], 1)
        self.assertTrue(all(r["protocol_deviation"] == "true" for r in rows.values()))
        self.assertTrue(all(r["protocol_deviation_reason"] == "UNPLANNED_PHYSICAL_LEAK" for r in rows.values()))
        self.assertIn("Urgent", {r["risk_label"] for r in rows.values()})  # still labeled

    def test_sudden_and_field_arms_excluded(self):
        base_s, base_f = _base(4), _base(5)
        sudden = {
            "id": "syn-sud", "arm": Arm.LEAK_SUDDEN, "bag": "bag-sud", "sensor": "sensor-sud",
            "end_reason": EndReason.LEAK_CONFIRMED, "samples": _ok_samples(base_s, 200),
            "events": [(base_s + 190 * 1000, EventType.PHYSICAL_LEAK_OBSERVED)],
        }
        field = {
            "id": "syn-field", "arm": Arm.FIELD, "bag": "bag-field", "sensor": "sensor-field",
            "end_reason": EndReason.MANUAL_STOP, "samples": _ok_samples(base_f, 200), "events": [],
        }
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _run(root, [sudden, field])
            rows = _read_labels(root / "output")
        sud_reasons = {r["exclusion_reason"] for r in rows.values() if r["session_id"] == "syn-sud"}
        fld_reasons = {r["exclusion_reason"] for r in rows.values() if r["session_id"] == "syn-field"}
        self.assertEqual(sud_reasons, {"SUDDEN_ARM"})
        self.assertEqual(fld_reasons, {"FIELD_ARM_EXCLUDED"})

    def test_structural_exclusions_propagate(self):
        base = _base(6)
        # 131 OK samples; mutate to hit each structural reason inside window 0.
        def safe_session(samples):
            return {
                "id": "syn-struct", "arm": Arm.SAFE, "bag": "bag-st", "sensor": "sensor-st",
                "end_reason": EndReason.CEILING_REACHED, "horizon": "60", "samples": samples,
                "events": [],
            }
        base_samples = _ok_samples(base, 131, warmup=5)  # window 0 has warmup -> INVALID_CAP
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _run(root, [safe_session(base_samples)])
            rows = _read_labels(root / "output")
        self.assertEqual(rows["syn-struct-win-0000"]["exclusion_reason"], "INVALID_CAP_QUALITY")

        # PARTIAL: drop an interior sample of window 0
        partial = [s for s in _ok_samples(base, 131) if s[0] != base + 60 * 1000]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _run(root, [safe_session(partial)])
            rows = _read_labels(root / "output")
        self.assertEqual(rows["syn-struct-win-0000"]["exclusion_reason"], "PARTIAL_WINDOW")

        # DUPLICATE: insert a duplicate timestamp into window 0
        dup = list(_ok_samples(base, 131))
        dup.insert(61, (base + 60 * 1000, 99.0, CapQuality.OK, LigQuality.OK))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _run(root, [safe_session(dup)])
            rows = _read_labels(root / "output")
        self.assertEqual(rows["syn-struct-win-0000"]["exclusion_reason"], "DUPLICATE_TIMESTAMP")

        # TIMING: retain 120 unique members but create 1300/700 ms gaps.
        timing = list(_ok_samples(base, 131))
        ts, cap, capq, ligq = timing[60]
        timing[60] = (ts + 300, cap, capq, ligq)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _run(root, [safe_session(timing)])
            rows = _read_labels(root / "output")
        self.assertEqual(
            rows["syn-struct-win-0000"]["exclusion_reason"],
            "TIMING_OUT_OF_TOLERANCE",
        )


class LabelingValidationTests(unittest.TestCase):
    def _safe_session(self, index=0):
        base = _base(index)
        return {
            "id": "syn-x", "arm": Arm.SAFE, "bag": "bag-x", "sensor": "sensor-x",
            "end_reason": EndReason.CEILING_REACHED, "horizon": "180",
            "samples": _ok_samples(base, 200), "events": [],
        }

    def test_malformed_events_fail_atomically(self):
        base = _base(7)
        session = {
            "id": "syn-mal", "arm": Arm.LEAK_GRADUAL, "bag": "bag-m", "sensor": "sensor-m",
            "end_reason": EndReason.LEAK_CONFIRMED, "horizon": "",
            "samples": _ok_samples(base, 200),
            # end_reason LEAK_CONFIRMED but NO physical leak -> malformed
            "events": [(base + 5 * 1000, EventType.INJECTION_START), (base + 6 * 1000, EventType.INJECTION_END)],
        }
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output = root / "output"
            output.mkdir()
            (output / "labels.csv").write_bytes(b"SENTINEL")
            input_dir, pp, qp = _write_dataset(root, [session])
            with self.assertRaises(MalformedRequiredEvents):
                label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, output)
            self.assertEqual((output / "labels.csv").read_bytes(), b"SENTINEL")

    def test_safe_with_physical_leak_is_malformed(self):
        base = _base(8)
        session = {
            "id": "syn-safeleak", "arm": Arm.SAFE, "bag": "b", "sensor": "s",
            "end_reason": EndReason.MANUAL_STOP, "horizon": "180",
            "samples": _ok_samples(base, 200),
            "events": [(base + 100 * 1000, EventType.PHYSICAL_LEAK_OBSERVED)],
        }
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(MalformedRequiredEvents):
                _run(Path(name), [session])

    def test_overlapping_injection_is_malformed(self):
        base = _base(9)
        session = {
            "id": "syn-ov", "arm": Arm.LEAK_GRADUAL, "bag": "b", "sensor": "s",
            "end_reason": EndReason.CEILING_REACHED, "horizon": "",
            "samples": _ok_samples(base, 200),
            "events": [
                (base + 5 * 1000, EventType.INJECTION_START),
                (base + 6 * 1000, EventType.INJECTION_START),  # second start, no end -> overlap
            ],
        }
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(MalformedRequiredEvents):
                _run(Path(name), [session])

    def test_unclosed_injection_start_is_malformed(self):
        base = _base(12)
        session = {
            "id": "syn-unclosed", "arm": Arm.LEAK_GRADUAL,
            "bag": "bag-unclosed", "sensor": "sensor-unclosed",
            "end_reason": EndReason.CEILING_REACHED, "horizon": "60",
            "samples": _ok_samples(base, 240),
            "events": [(base + 150 * 1000, EventType.INJECTION_START)],
        }
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(MalformedRequiredEvents):
                _run(Path(name), [session])

    def test_gradual_leak_requires_injection_events(self):
        base = _base(13)
        session = {
            "id": "syn-no-injection", "arm": Arm.LEAK_GRADUAL,
            "bag": "bag-no-injection", "sensor": "sensor-no-injection",
            "end_reason": EndReason.LEAK_CONFIRMED, "horizon": "",
            "samples": _ok_samples(base, 331),
            "events": [(base + 310 * 1000, EventType.PHYSICAL_LEAK_OBSERVED)],
        }
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(MalformedRequiredEvents):
                _run(Path(name), [session])

    def test_physical_leak_before_gradual_injection_is_malformed(self):
        base = _base(14)
        session = {
            "id": "syn-causal", "arm": Arm.LEAK_GRADUAL,
            "bag": "bag-causal", "sensor": "sensor-causal",
            "end_reason": EndReason.LEAK_CONFIRMED, "horizon": "",
            "samples": _ok_samples(base, 240),
            "events": [
                (base + 150 * 1000, EventType.PHYSICAL_LEAK_OBSERVED),
                (base + 160 * 1000, EventType.INJECTION_START),
                (base + 170 * 1000, EventType.INJECTION_END),
            ],
        }
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(MalformedRequiredEvents):
                _run(Path(name), [session])

    def test_safe_arm_rejects_injection_events(self):
        session = self._safe_session(index=15)
        base = session["samples"][0][0]
        session["events"] = [
            (base + 100 * 1000, EventType.INJECTION_START),
            (base + 110 * 1000, EventType.INJECTION_END),
        ]
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(MalformedRequiredEvents):
                _run(Path(name), [session])

    def test_false_positive_lig_flag_does_not_invalidate_safe_labels(self):
        session = self._safe_session(index=16)
        base = session["samples"][0][0]
        session["events"] = [
            (base + 190 * 1000, EventType.LEAK_FLAG_CONFIRMED),
        ]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _run(root, [session])
            rows = _read_labels(root / "output")
        self.assertTrue(rows)
        self.assertTrue(all(row["risk_label"] == "Safe" for row in rows.values()))

    def test_duplicate_event_id_is_malformed(self):
        session = self._safe_session(index=17)
        base = session["samples"][0][0]
        session["events"] = [
            (base + 185 * 1000, EventType.LEAK_FLAG_FIRST),
            (base + 190 * 1000, EventType.LEAK_FLAG_CONFIRMED),
        ]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_dir, protocol_path, partition_path = _write_dataset(root, [session])
            events_path = input_dir / "events.csv"
            with events_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[1]["event_id"] = rows[0]["event_id"]
            with events_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(MalformedRequiredEvents):
                label_dataset(
                    input_dir, protocol_path, partition_path,
                    GOOD_BOUNDARY, root / "output",
                )

    def test_non_synthetic_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(LabelingError):
                _run(Path(name), [self._safe_session()], origin="REAL_PILOT_DATA")

    def test_missing_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(LabelingError):
                _run(Path(name), [self._safe_session()], origin=None)

    def test_boundary_outside_fixtures_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            bad = Path(name) / "boundary.json"
            bad.write_text(GOOD_BOUNDARY.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(LabelingError):
                _run(Path(name), [self._safe_session()], boundary=bad)

    def test_invalid_boundary_values_rejected(self):
        good = json.loads(GOOD_BOUNDARY.read_text(encoding="utf-8"))
        variants = [
            {**good, "b1_s": 90, "b2_s": 30},   # unordered
            {**good, "b1_s": 0},                # non-positive
            {**good, "b2_s": 90.0},             # non-integral
            {**good, "status": "PRODUCTION"},   # not ENGINEERING_TEST_ONLY
            {**good, "allowed_dataset_origin": "REAL"},  # wrong origin
            {k: v for k, v in good.items() if k != "b3_s"},  # missing key
        ]
        with tempfile.TemporaryDirectory() as fixture_name:
            tmp_fixture = Path(fixture_name)
            with mock.patch.object(labeling, "_FIXTURES_DIR", tmp_fixture.resolve()):
                for i, variant in enumerate(variants):
                    path = tmp_fixture / f"boundary-{i}.json"
                    path.write_text(json.dumps(variant), encoding="utf-8")
                    with tempfile.TemporaryDirectory() as name:
                        with self.assertRaises(LabelingError):
                            _run(Path(name), [self._safe_session()], boundary=path)

    def test_identity_mismatch_rejected(self):
        session = self._safe_session()
        session["protocol_bag"] = "WRONG-BAG"
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(LabelingError):
                _run(Path(name), [session])

    def test_partition_leakage_rejected(self):
        base0, base1 = _base(10), _base(11)
        shared = "shared-bag"
        s0 = {"id": "syn-p0", "arm": Arm.SAFE, "bag": shared, "sensor": "sen0",
              "end_reason": EndReason.CEILING_REACHED, "horizon": "180",
              "samples": _ok_samples(base0, 200), "events": [], "partition": "development"}
        s1 = {"id": "syn-p1", "arm": Arm.SAFE, "bag": shared, "sensor": "sen1",
              "end_reason": EndReason.CEILING_REACHED, "horizon": "180",
              "samples": _ok_samples(base1, 200), "events": [], "partition": "final_test"}
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(LabelingError):
                _run(Path(name), [s0, s1])


class LabelingIntegrityTests(unittest.TestCase):
    def _dataset_and_features(self, root):
        session = _gradual_leak("syn-i", 0)
        input_dir, pp, qp = _write_dataset(root, [session])
        feat_dir = root / "features"
        features.extract(input_dir, features.load_config(FEATURES_CONFIG), feat_dir)
        return input_dir, pp, qp, feat_dir

    def test_features_omitted_matches_supplied_identities(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_dir, pp, qp, feat_dir = self._dataset_and_features(root)
            out_plain, out_feat = root / "plain", root / "feat"
            label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, out_plain)
            label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, out_feat, features_dir=feat_dir)
            self.assertEqual((out_plain / "labels.csv").read_bytes(), (out_feat / "labels.csv").read_bytes())

    def test_supplied_features_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_dir, pp, qp, feat_dir = self._dataset_and_features(root)
            manifest_path = feat_dir / "feature_manifest.json"
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["input_samples_sha256"] = "deadbeef"
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(LabelingError):
                label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, root / "out", features_dir=feat_dir)

    def test_duplicate_feature_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_dir, pp, qp, feat_dir = self._dataset_and_features(root)
            path = feat_dir / "features.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            rows.append(rows[1])
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            with self.assertRaises(LabelingError):
                label_dataset(
                    input_dir, pp, qp, GOOD_BOUNDARY, root / "out",
                    features_dir=feat_dir,
                )

    def test_capacitance_and_lig_mutation_do_not_change_labels(self):
        for column in ("capacitance_raw", "lig_raw"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                session = _gradual_leak("syn-mut", 0)
                _run(root, [session])
                baseline = (root / "output" / "labels.csv").read_bytes()
                # mutate the raw column, keep timestamps/qualities identical
                samples_csv = root / "input" / "samples.csv"
                with samples_csv.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                    fields = list(rows[0].keys())
                for row in rows:
                    row[column] = str(float(row[column]) + 7.5)
                with samples_csv.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                label_dataset(
                    root / "input", root / "protocol_manifest.csv", root / "partition_manifest.csv",
                    GOOD_BOUNDARY, root / "output", overwrite=True,
                )
                self.assertEqual((root / "output" / "labels.csv").read_bytes(), baseline, column)

    def test_byte_identical_repeated_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            session = _gradual_leak("syn-det", 0)
            input_dir, pp, qp = _write_dataset(root, [session])
            label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, root / "a")
            label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, root / "b")
            for f in ("labels.csv", "label_manifest.json"):
                self.assertEqual((root / "a" / f).read_bytes(), (root / "b" / f).read_bytes())

    def test_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            session = _gradual_leak("syn-ow", 0)
            input_dir, pp, qp = _write_dataset(root, [session])
            output = root / "output"
            label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, output)
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, output)
            label_dataset(input_dir, pp, qp, GOOD_BOUNDARY, output, overwrite=True)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_staging_failure_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            session = _gradual_leak("syn-stage", 0)
            input_dir, pp, qp = _write_dataset(root, [session])
            output = root / "output"
            output.mkdir()
            old_labels = b"OLD-LABELS"
            old_manifest = b"OLD-MANIFEST"
            (output / "labels.csv").write_bytes(old_labels)
            (output / "label_manifest.json").write_bytes(old_manifest)
            with mock.patch.object(
                labeling, "_write_manifest", side_effect=OSError("simulated write failure")
            ):
                with self.assertRaises(OSError):
                    label_dataset(
                        input_dir, pp, qp, GOOD_BOUNDARY, output, overwrite=True
                    )
            self.assertEqual((output / "labels.csv").read_bytes(), old_labels)
            self.assertEqual(
                (output / "label_manifest.json").read_bytes(), old_manifest
            )

    def test_manifest_counts_reconcile(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = _run(root, [_gradual_leak("syn-rec", 0)])
            rows = _read_labels(root / "output")
        valid = sum(1 for r in rows.values() if r["label_valid"] == "true")
        excluded = sum(1 for r in rows.values() if r["label_valid"] == "false")
        self.assertEqual(manifest["candidate_window_count"], len(rows))
        self.assertEqual(manifest["valid_window_count"], valid)
        self.assertEqual(manifest["excluded_window_count"], excluded)
        self.assertEqual(sum(manifest["risk_class_counts"].values()), valid)
        self.assertEqual(sum(manifest["exclusion_reason_counts"].values()), excluded)


@unittest.skipUnless(SKLEARN_AVAILABLE, "requires numpy for the synthetic generator")
class LabelingSyntheticIntegrationTests(unittest.TestCase):
    def test_full_pipeline_covers_all_classes(self):
        from ostosense_ai import synthetic

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw = root / "raw"
            synthetic.generate(
                synthetic.load_config(REPO_ROOT / "configs" / "synthetic-v0.2.json"),
                20260722,
                raw,
            )
            feat = root / "features"
            features.extract(raw, features.load_config(FEATURES_CONFIG), feat)
            manifest = label_dataset(
                raw,
                LABELING_FIX / "protocol_manifest.csv",
                LABELING_FIX / "partition_manifest.csv",
                GOOD_BOUNDARY,
                root / "labels",
                features_dir=feat,
            )
        self.assertTrue(all(manifest["risk_class_counts"][c] > 0 for c in ("Safe", "Monitor", "Caution", "Urgent")))
        for reason in ("POST_LEAK", "SUDDEN_ARM", "CENSORED_NO_SAFE_HORIZON", "INVALID_CAP_QUALITY", "PARTIAL_WINDOW"):
            self.assertGreater(manifest["exclusion_reason_counts"][reason], 0, reason)


if __name__ == "__main__":
    unittest.main()
