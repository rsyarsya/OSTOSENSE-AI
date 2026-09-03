"""Focused tests for the deterministic real-data intake and QC gate.

SYNTHETIC/ENGINEERING_TEST_ONLY. These exercise QC mechanics only; they are not
OSTOSENSE AI accuracy, notification, sensor, firmware, or clinical evidence. The
module and all non-pipeline tests run under the standard library alone.
"""

import csv
import importlib.util
import json
import statistics
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from ostosense_ai import raw_qc
from ostosense_ai.labeling import PROTOCOL_MANIFEST_FIELDS
from ostosense_contract import CapQuality, LigQuality, aggregate_system_quality
from ostosense_contract.schema import EventRecord, SampleRecord, SessionRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "raw-qc-v0.1.json"
NUMPY = importlib.util.find_spec("numpy") is not None

T0 = 1_700_000_000_000


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def session_row(session_id="s1", arm="SAFE", bag_id="bag-1", sensor_id="sen-1",
                device_id="dev-1", fluid_type="saline", operator_id="op-1",
                baseline_value=12.0, baseline_std=0.05, start=T0, end=None,
                end_reason="MANUAL_STOP", model_version="", firmware_version="fw-1"):
    if end is None:
        end = start + 300_000
    return [session_id, arm, bag_id, sensor_id, device_id, fluid_type, operator_id,
            baseline_value, baseline_std, start, end, end_reason, model_version, firmware_version]


def sample_row(
    ts,
    session_id="s1",
    cap: Any = 12.0,
    capq="OK",
    ligq="OK",
    lig=400.0,
    sysq=None,
):
    if sysq is None:
        sysq = aggregate_system_quality(CapQuality(capq), LigQuality(ligq)).value
    return [ts, session_id, cap, lig, capq, ligq, sysq, "", ""]


def linear_samples(session_id, start, count, interval=1000, cap=12.0, capq="OK", ligq="OK"):
    return [sample_row(start + i * interval, session_id, cap=cap, capq=capq, ligq=ligq)
            for i in range(count)]


def event_row(event_id, session_id, ts, event_type, metadata="{}"):
    return [event_id, session_id, ts, event_type, metadata]


def manifest_row(session_id, planned_arm="SAFE", operator_id="op-1", bag_id="bag-1",
                 sensor_id="sen-1", device_id="dev-1", *, protocol_version="v0.1-shakedown",
                 planned_safe_horizon_s=None, target="TARGET-1", injection_profile=None,
                 injection_method=None, planned_flow_ml_min="", physical_leak_observation_method=None):
    """Build a strict-valid protocol_manifest row (arm-aware defaults)."""
    if planned_arm == "SAFE":
        horizon = "180" if planned_safe_horizon_s is None else planned_safe_horizon_s
        profile, method, flow, observation = "", "", "", ""
    elif planned_arm == "LEAK_GRADUAL":
        horizon = "" if planned_safe_horizon_s is None else planned_safe_horizon_s
        profile = "stepwise" if injection_profile is None else injection_profile
        method = "manual_syringe" if injection_method is None else injection_method
        flow = planned_flow_ml_min
        observation = "video" if physical_leak_observation_method is None else physical_leak_observation_method
    elif planned_arm == "LEAK_SUDDEN":
        horizon = "" if planned_safe_horizon_s is None else planned_safe_horizon_s
        profile, method, flow = "", "", ""
        observation = "video" if physical_leak_observation_method is None else physical_leak_observation_method
    else:  # FIELD or an intentionally wrong planned_arm
        horizon = "" if planned_safe_horizon_s is None else planned_safe_horizon_s
        profile, method, flow, observation = "", "", "", ""
    return [session_id, protocol_version, planned_arm, horizon, target, profile, method,
            flow, observation, operator_id, bag_id, sensor_id, device_id]


def build_input(
    root,
    sessions,
    samples,
    events,
    *,
    origin: str | None = "SYNTHETIC_PIPELINE_TEST_ONLY",
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "sessions.csv", SessionRecord.FIELDS, sessions)
    _write_csv(root / "samples.csv", SampleRecord.FIELDS, samples)
    _write_csv(root / "events.csv", EventRecord.FIELDS, events)
    if origin is not None:
        (root / "manifest.json").write_text(
            json.dumps({"dataset_origin": origin}), encoding="utf-8")
    return root


def write_manifest(path, rows):
    _write_csv(path, PROTOCOL_MANIFEST_FIELDS, rows)
    return Path(path)


def read_sessions(output_dir):
    with (Path(output_dir) / "qc_sessions.csv").open(newline="", encoding="utf-8") as handle:
        return {row["session_id"]: row for row in csv.DictReader(handle)}


def read_issues(output_dir):
    with (Path(output_dir) / "qc_issues.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def codes_for(issues, session_id):
    return [row["code"] for row in issues if row["session_id"] == session_id]


class RawQcBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_qc(
        self,
        sessions,
        samples,
        events,
        *,
        manifest_rows=None,
        origin: str | None = "SYNTHETIC_PIPELINE_TEST_ONLY",
        name="case",
        overwrite=False,
    ):
        inp = build_input(self.root / f"{name}-in", sessions, samples, events, origin=origin)
        manifest_path = None
        if manifest_rows is not None:
            manifest_path = write_manifest(self.root / f"{name}-protocol.csv", manifest_rows)
        out = self.root / f"{name}-out"
        report = raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=manifest_path,
                                   overwrite=overwrite)
        return report, out


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
class ConfigTests(unittest.TestCase):
    def _base(self):
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_reference_config_is_valid(self):
        raw_qc.validate_qc_config(self._base())

    def test_rejects_missing_unknown_and_bad_values(self):
        bad = self._base(); del bad["baseline_window_s"]
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)
        bad = self._base(); bad["extra"] = 1
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)
        bad = self._base(); bad["expected_interval_ms"] = 0
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)
        bad = self._base(); bad["expected_interval_ms"] = "1000"
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)
        bad = self._base(); bad["protocol_version"] = "v9"
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)

    def test_rejects_inconsistent_timing(self):
        bad = self._base(); bad["jitter_tolerance_ms"] = 1000  # >= expected_interval
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)
        bad = self._base(); bad["unmarked_gap_threshold_ms"] = 1100  # <= 1000+200
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)
        bad = self._base(); bad["minimum_pre_injection_dry_s"] = 30  # < baseline_window_s
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)

    def test_rejects_bad_provenance(self):
        bad = self._base(); bad["provenance"] = {"expected_interval_ms": "SOMETHING"}
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.validate_qc_config(bad)


# --------------------------------------------------------------------------- #
# Golden + status composition
# --------------------------------------------------------------------------- #
class GoldenTests(RawQcBase):
    def test_golden_safe_and_gradual_pass(self):
        sessions = [
            session_row("safe-1", arm="SAFE", start=T0, end=T0 + 70_000),
            session_row("grad-1", arm="LEAK_GRADUAL", bag_id="bag-2", sensor_id="sen-2",
                        start=T0 + 1_000_000, end=T0 + 1_200_000),
        ]
        samples = linear_samples("safe-1", T0, 65)
        samples += linear_samples("grad-1", T0 + 1_000_000, 180)
        g0 = T0 + 1_000_000
        events = [
            event_row("e1", "grad-1", g0 + 130_000, "INJECTION_START"),
            event_row("e2", "grad-1", g0 + 140_000, "INJECTION_END"),
            event_row("e3", "grad-1", g0 + 145_000, "LEAK_FLAG_FIRST"),
            event_row("e4", "grad-1", g0 + 148_000, "LEAK_FLAG_CONFIRMED"),
            event_row("e5", "grad-1", g0 + 150_000, "PHYSICAL_LEAK_OBSERVED"),
        ]
        manifest = [manifest_row("safe-1", planned_arm="SAFE"),
                    manifest_row("grad-1", planned_arm="LEAK_GRADUAL", bag_id="bag-2", sensor_id="sen-2")]
        report, out = self.run_qc(sessions, samples, events, manifest_rows=manifest, name="golden")
        rows = read_sessions(out)
        for sid in ("safe-1", "grad-1"):
            self.assertEqual(rows[sid]["contract_status"], "PASS")
            self.assertEqual(rows[sid]["protocol_status"], "PASS")
            self.assertEqual(rows[sid]["overall_status"], "PASS")
        self.assertEqual(report["status_counts"], {"PASS": 2, "FAIL": 0, "PARTIAL": 0})
        self.assertEqual(report["dataset_origin"], "SYNTHETIC_PIPELINE_TEST_ONLY")

    def test_missing_manifest_is_partial_and_undeclared(self):
        sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 65)
        report, out = self.run_qc(sessions, samples, [], origin=None, name="partial")
        rows = read_sessions(out)
        self.assertEqual(rows["s1"]["protocol_status"], "NOT_EVALUATED")
        self.assertEqual(rows["s1"]["overall_status"], "PARTIAL")
        self.assertEqual(report["dataset_origin"], "UNDECLARED")
        self.assertIn("PROTOCOL_MANIFEST_NOT_PROVIDED", codes_for(read_issues(out), ""))
        self.assertEqual(report["status_counts"], {"PASS": 0, "FAIL": 0, "PARTIAL": 1})


# --------------------------------------------------------------------------- #
# Fatal input failures (before output mutation)
# --------------------------------------------------------------------------- #
class FatalInputTests(RawQcBase):
    def _assert_no_output(self, out):
        for name in raw_qc._OUTPUT_FILES:
            self.assertFalse((Path(out) / name).exists())

    def test_wrong_header_fails_before_output(self):
        inp = self.root / "in"; inp.mkdir()
        _write_csv(inp / "sessions.csv", ("wrong", "header"), [])
        _write_csv(inp / "samples.csv", SampleRecord.FIELDS, [])
        _write_csv(inp / "events.csv", EventRecord.FIELDS, [])
        out = self.root / "out"
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, out)
        self._assert_no_output(out)

    def test_malformed_row_fails_before_output(self):
        inp = build_input(self.root / "in", [session_row("s1")], [], [])
        # append a short (malformed) samples row by hand
        with (inp / "samples.csv").open("a", encoding="utf-8") as handle:
            handle.write("1,2,3\n")
        out = self.root / "out"
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, out)
        self._assert_no_output(out)

    def test_missing_mandatory_file(self):
        inp = self.root / "in"; inp.mkdir()
        _write_csv(inp / "sessions.csv", SessionRecord.FIELDS, [session_row("s1")])
        _write_csv(inp / "samples.csv", SampleRecord.FIELDS, [])
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "out")

    def test_unparseable_and_unknown_enum(self):
        inp = build_input(self.root / "in1", [session_row("s1")],
                          [sample_row(T0, "s1", cap="not-a-number")], [])
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "o1")
        bad_enum_row = [T0, "s1", 12.0, 400.0, "MYSTERY", "OK", "NORMAL", "", ""]
        inp = build_input(self.root / "in2", [session_row("s1")], [bad_enum_row], [])
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "o2")
        inp = build_input(
            self.root / "in3",
            [session_row("s1", end_reason="UNKNOWN")],
            linear_samples("s1", T0, 65),
            [],
        )
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "o3")
        for index, metadata in enumerate(('{"value":NaN}', ""), start=4):
            inp = build_input(
                self.root / f"in{index}",
                [session_row("s1")],
                linear_samples("s1", T0, 65),
                [event_row("e1", "s1", T0 + 1_000, "ALERT_RAISED", metadata)],
            )
            with self.assertRaises(raw_qc.RawQcError):
                raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / f"o{index}")

    def test_orphan_session_references_rejected(self):
        inp = build_input(self.root / "in-s", [session_row("s1")],
                          [sample_row(T0, "ghost")], [])
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "o-s")
        inp = build_input(self.root / "in-e", [session_row("s1")],
                          linear_samples("s1", T0, 3),
                          [event_row("e1", "ghost", T0, "INJECTION_START")])
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "o-e")

    def test_duplicate_session_id_rejected(self):
        inp = build_input(self.root / "in", [session_row("s1"), session_row("s1")],
                          linear_samples("s1", T0, 3), [])
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "out")

    def test_extra_protocol_manifest_session_is_rejected_before_output(self):
        inp = build_input(
            self.root / "in-extra-manifest",
            [session_row("s1", start=T0, end=T0 + 70_000)],
            linear_samples("s1", T0, 65),
            [],
        )
        manifest = write_manifest(
            self.root / "extra-protocol.csv",
            [manifest_row("s1"), manifest_row("ghost")],
        )
        out = self.root / "out-extra-manifest"
        with self.assertRaisesRegex(raw_qc.RawQcError, "unknown session_id"):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=manifest)
        self._assert_no_output(out)


# --------------------------------------------------------------------------- #
# Contract findings
# --------------------------------------------------------------------------- #
class ContractTests(RawQcBase):
    def test_sample_and_event_outside_bounds(self):
        sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 65) + [sample_row(T0 - 5_000, "s1")]
        events = [event_row("e1", "s1", T0 + 500_000, "MANUAL_SESSION_RESET")]
        _, out = self.run_qc(sessions, samples, events, manifest_rows=[manifest_row("s1")], name="oob")
        rows = read_sessions(out)
        self.assertEqual(rows["s1"]["contract_status"], "FAIL")
        self.assertEqual(rows["s1"]["overall_status"], "FAIL")
        codes = codes_for(read_issues(out), "s1")
        self.assertIn("SAMPLE_OUTSIDE_SESSION", codes)
        self.assertIn("EVENT_OUTSIDE_SESSION", codes)

    def test_duplicate_event_id(self):
        sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 65)
        events = [event_row("dup", "s1", T0 + 1_000, "ALERT_RAISED"),
                  event_row("dup", "s1", T0 + 2_000, "ALERT_ACKNOWLEDGED")]
        _, out = self.run_qc(sessions, samples, events, manifest_rows=[manifest_row("s1")], name="dupe")
        rows = read_sessions(out)
        self.assertEqual(rows["s1"]["contract_status"], "FAIL")
        self.assertIn("DUPLICATE_EVENT_ID", codes_for(read_issues(out), "s1"))

    def test_system_quality_mismatch(self):
        sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 64)
        samples.append(sample_row(T0 + 64_000, "s1", capq="OK", ligq="OK", sysq="UNSAFE"))
        _, out = self.run_qc(sessions, samples, [], manifest_rows=[manifest_row("s1")], name="sqm")
        rows = read_sessions(out)
        self.assertEqual(rows["s1"]["contract_status"], "FAIL")
        self.assertIn("SYSTEM_QUALITY_MISMATCH", codes_for(read_issues(out), "s1"))

    def test_empty_required_id(self):
        cases: tuple[tuple[dict[str, Any], str], ...] = (
            ({"bag_id": ""}, "bag"),
            ({"fluid_type": ""}, "fluid"),
        )
        for kwargs, name in cases:
            sessions = [
                session_row("s1", start=T0, end=T0 + 70_000, **kwargs)
            ]
            samples = linear_samples("s1", T0, 65)
            _, out = self.run_qc(
                sessions,
                samples,
                [],
                manifest_rows=[manifest_row("s1")],
                name=f"emptyid-{name}",
            )
            rows = read_sessions(out)
            self.assertEqual(rows["s1"]["contract_status"], "FAIL")
            self.assertIn("INVALID_RECORD", codes_for(read_issues(out), "s1"))


# --------------------------------------------------------------------------- #
# Protocol timing
# --------------------------------------------------------------------------- #
class TimingTests(RawQcBase):
    def test_duplicate_and_descending_timestamps(self):
        sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
        samples = [sample_row(T0, "s1"), sample_row(T0, "s1"),          # duplicate
                   sample_row(T0 + 1_000, "s1"), sample_row(T0 + 500, "s1")]  # descending
        _, out = self.run_qc(sessions, samples, [], manifest_rows=[manifest_row("s1")], name="tsorder")
        rows = read_sessions(out)
        codes = codes_for(read_issues(out), "s1")
        self.assertIn("DUPLICATE_SAMPLE_TIMESTAMP", codes)
        self.assertIn("TIMESTAMP_OUT_OF_ORDER", codes)
        self.assertEqual(rows["s1"]["protocol_status"], "FAIL")
        self.assertEqual(int(rows["s1"]["duplicate_timestamp_count"]), 1)
        self.assertEqual(int(rows["s1"]["out_of_order_interval_count"]), 1)

    def test_tolerance_boundaries_accepted(self):
        for interval, name in ((800, "lo"), (1200, "hi")):
            sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
            samples = linear_samples("s1", T0, 6, interval=interval)
            _, out = self.run_qc(sessions, samples, [], manifest_rows=[manifest_row("s1")], name=f"tol-{name}")
            rows = read_sessions(out)
            self.assertEqual(int(rows["s1"]["out_of_tolerance_interval_count"]), 0)
            self.assertNotIn("INTERVAL_OUT_OF_TOLERANCE", codes_for(read_issues(out), "s1"))

    def test_tolerance_boundaries_rejected(self):
        for interval, name in ((799, "lo"), (1201, "hi")):
            sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
            samples = linear_samples("s1", T0, 6, interval=interval)
            _, out = self.run_qc(sessions, samples, [], manifest_rows=[manifest_row("s1")], name=f"rej-{name}")
            rows = read_sessions(out)
            self.assertGreaterEqual(int(rows["s1"]["out_of_tolerance_interval_count"]), 1)
            self.assertIn("INTERVAL_OUT_OF_TOLERANCE", codes_for(read_issues(out), "s1"))
            self.assertEqual(rows["s1"]["protocol_status"], "FAIL")

    def test_gap_without_and_with_data_gap(self):
        # A gap always violates the sampling tolerance; an unmarked gap adds a
        # logger-specific UNMARKED_DATA_GAP finding.
        sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
        samples = [sample_row(T0, "s1"), sample_row(T0 + 1_000, "s1"),
                   sample_row(T0 + 3_500, "s1")]  # 2500 ms gap, next sample OK
        _, out = self.run_qc(sessions, samples, [], manifest_rows=[manifest_row("s1")], name="gap-unmarked")
        rows = read_sessions(out)
        codes = codes_for(read_issues(out), "s1")
        self.assertEqual(int(rows["s1"]["out_of_tolerance_interval_count"]), 1)
        self.assertEqual(int(rows["s1"]["unmarked_gap_count"]), 1)
        self.assertIn("INTERVAL_OUT_OF_TOLERANCE", codes)
        self.assertIn("UNMARKED_DATA_GAP", codes)
        self.assertEqual(rows["s1"]["protocol_status"], "FAIL")

        # DATA_GAP acknowledges the logger gap, but cannot make its timing valid.
        sessions = [session_row("s2", start=T0, end=T0 + 70_000)]
        samples = [sample_row(T0, "s2"), sample_row(T0 + 1_000, "s2"),
                   sample_row(T0 + 3_500, "s2", capq="DATA_GAP")]
        _, out = self.run_qc(sessions, samples, [], manifest_rows=[manifest_row("s2")], name="gap-marked")
        rows = read_sessions(out)
        codes = codes_for(read_issues(out), "s2")
        self.assertEqual(int(rows["s2"]["out_of_tolerance_interval_count"]), 1)
        self.assertEqual(int(rows["s2"]["unmarked_gap_count"]), 0)
        self.assertIn("INTERVAL_OUT_OF_TOLERANCE", codes)
        self.assertNotIn("UNMARKED_DATA_GAP", codes)
        self.assertEqual(rows["s2"]["protocol_status"], "FAIL")


# --------------------------------------------------------------------------- #
# Baseline + dry phase
# --------------------------------------------------------------------------- #
class BaselineTests(RawQcBase):
    def test_baseline_median_and_pstdev_golden(self):
        values = [float(i) for i in range(60)]
        sessions = [session_row("s1", baseline_value=99.0, baseline_std=0.0, start=T0, end=T0 + 60_000)]
        samples = [sample_row(T0 + i * 1000, "s1", cap=values[i]) for i in range(60)]
        _, out = self.run_qc(sessions, samples, [], manifest_rows=[manifest_row("s1")], name="baseline")
        row = read_sessions(out)["s1"]
        self.assertEqual(int(row["baseline_sample_count"]), 60)
        self.assertEqual(float(row["baseline_recomputed_median"]),
                         round(statistics.median(values), 6))
        self.assertEqual(float(row["baseline_recomputed_std"]),
                         round(statistics.pstdev(values), 6))
        # Differences are reported but never thresholded: still PASS.
        self.assertEqual(float(row["baseline_value_abs_diff"]),
                         round(abs(statistics.median(values) - 99.0), 6))
        self.assertEqual(row["overall_status"], "PASS")

    def test_single_sample_does_not_complete_60_second_baseline(self):
        sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
        samples = [sample_row(T0, "s1")]
        _, out = self.run_qc(
            sessions,
            samples,
            [],
            manifest_rows=[manifest_row("s1")],
            name="baseline-one-sample",
        )
        row = read_sessions(out)["s1"]
        self.assertEqual(int(row["baseline_sample_count"]), 1)
        self.assertEqual(row["protocol_status"], "FAIL")
        self.assertIn(
            "BASELINE_WINDOW_INCOMPLETE",
            codes_for(read_issues(out), "s1"),
        )

    def test_dry_phase_exactly_120s_accepted_and_119s_rejected(self):
        def gradual(dry_ms, name):
            start = T0
            sessions = [session_row("g", arm="LEAK_GRADUAL", start=start, end=start + 200_000)]
            samples = linear_samples("g", start, 125)
            inj = start + dry_ms
            events = [event_row("e1", "g", inj, "INJECTION_START"),
                      event_row("e2", "g", inj + 5_000, "INJECTION_END"),
                      event_row("e3", "g", inj + 8_000, "PHYSICAL_LEAK_OBSERVED")]
            return self.run_qc(sessions, samples, events,
                               manifest_rows=[manifest_row("g", planned_arm="LEAK_GRADUAL")], name=name)

        _, out = gradual(120_000, "dry120")
        self.assertNotIn("PREINJECTION_DRY_TOO_SHORT", codes_for(read_issues(out), "g"))
        _, out = gradual(119_000, "dry119")
        self.assertIn("PREINJECTION_DRY_TOO_SHORT", codes_for(read_issues(out), "g"))
        self.assertEqual(read_sessions(out)["g"]["protocol_status"], "FAIL")


# --------------------------------------------------------------------------- #
# Arm event rules + manifest
# --------------------------------------------------------------------------- #
class ArmRuleTests(RawQcBase):
    def test_required_events_per_arm(self):
        start = T0
        sessions = [
            session_row("safe", arm="SAFE", start=start, end=start + 300_000),
            session_row("grad", arm="LEAK_GRADUAL", bag_id="b2", sensor_id="n2", start=start, end=start + 300_000),
            session_row("sudden", arm="LEAK_SUDDEN", bag_id="b3", sensor_id="n3", start=start, end=start + 300_000),
            session_row("field", arm="FIELD", bag_id="b4", sensor_id="n4", start=start, end=start + 300_000),
        ]
        samples = (linear_samples("safe", start, 5) + linear_samples("grad", start, 5)
                   + linear_samples("sudden", start, 5) + linear_samples("field", start, 5))
        events = [
            event_row("s1", "safe", start + 1000, "PHYSICAL_LEAK_OBSERVED"),   # illegal for SAFE
            event_row("g1", "grad", start + 130_000, "INJECTION_START"),        # missing INJECTION_END
            event_row("g2", "grad", start + 150_000, "PHYSICAL_LEAK_OBSERVED"),
            # sudden: missing PHYSICAL_LEAK_OBSERVED
        ]
        _, out = self.run_qc(sessions, samples, events, name="arms")
        issues = read_issues(out)
        self.assertIn("UNEXPECTED_PHYSICAL_LEAK_SAFE", codes_for(issues, "safe"))
        self.assertIn("MISSING_REQUIRED_EVENT", codes_for(issues, "grad"))
        self.assertIn("MISSING_REQUIRED_EVENT", codes_for(issues, "sudden"))
        self.assertIn("FIELD_ARM_NOT_SUPPORTED", codes_for(issues, "field"))

    def test_missing_lig_flags_are_warnings_only(self):
        start = T0
        sessions = [session_row("grad", arm="LEAK_GRADUAL", start=start, end=start + 300_000)]
        samples = linear_samples("grad", start, 180)
        events = [event_row("e1", "grad", start + 130_000, "INJECTION_START"),
                  event_row("e2", "grad", start + 140_000, "INJECTION_END"),
                  event_row("e3", "grad", start + 150_000, "PHYSICAL_LEAK_OBSERVED")]
        _, out = self.run_qc(sessions, samples, events,
                             manifest_rows=[manifest_row("grad", planned_arm="LEAK_GRADUAL")], name="ligwarn")
        row = read_sessions(out)["grad"]
        issues = [i for i in read_issues(out) if i["session_id"] == "grad"]
        lig = [i for i in issues if i["code"] == "MISSING_LIG_FLAG_EVENT"]
        self.assertEqual(len(lig), 1)
        self.assertEqual(lig[0]["severity"], "WARNING")
        self.assertEqual(row["protocol_status"], "PASS")
        self.assertEqual(row["overall_status"], "PASS")

    def test_manifest_identity_and_arm_mismatch(self):
        sessions = [session_row("s1", arm="SAFE", bag_id="bag-1", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 65)
        manifest = [manifest_row("s1", planned_arm="LEAK_GRADUAL", bag_id="WRONG")]
        _, out = self.run_qc(sessions, samples, [], manifest_rows=manifest, name="mm")
        row = read_sessions(out)["s1"]
        codes = codes_for(read_issues(out), "s1")
        self.assertIn("MANIFEST_ARM_MISMATCH", codes)
        self.assertIn("MANIFEST_ID_MISMATCH", codes)
        self.assertEqual(row["protocol_status"], "FAIL")


# --------------------------------------------------------------------------- #
# Strict protocol-manifest semantics
# --------------------------------------------------------------------------- #
class ManifestValidationTests(RawQcBase):
    def _run_safe(self, mrow, name):
        sessions = [session_row("s1", arm="SAFE", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 65)
        return self.run_qc(sessions, samples, [], manifest_rows=[mrow], name=name)

    def _run_gradual(self, mrow, name):
        start = T0
        sessions = [session_row("g", arm="LEAK_GRADUAL", start=start, end=start + 400_000)]
        samples = linear_samples("g", start, 200)
        events = [event_row("g1", "g", start + 130_000, "INJECTION_START"),
                  event_row("g2", "g", start + 140_000, "INJECTION_END"),
                  event_row("g3", "g", start + 150_000, "PHYSICAL_LEAK_OBSERVED"),
                  event_row("g4", "g", start + 145_000, "LEAK_FLAG_FIRST"),
                  event_row("g5", "g", start + 148_000, "LEAK_FLAG_CONFIRMED")]
        return self.run_qc(sessions, samples, events, manifest_rows=[mrow], name=name)

    def test_valid_rows_for_each_arm_pass(self):
        start = T0
        sessions = [
            session_row("safe", arm="SAFE", start=start, end=start + 300_000),
            session_row("grad", arm="LEAK_GRADUAL", bag_id="b2", sensor_id="n2", start=start, end=start + 300_000),
            session_row("fill", arm="LEAK_GRADUAL", bag_id="b3", sensor_id="n3", start=start, end=start + 300_000),
            session_row("sudden", arm="LEAK_SUDDEN", bag_id="b4", sensor_id="n4", start=start, end=start + 300_000),
        ]
        samples = (linear_samples("safe", start, 65) + linear_samples("grad", start, 200)
                   + linear_samples("fill", start, 200) + linear_samples("sudden", start, 65))
        events = [
            event_row("g1", "grad", start + 130_000, "INJECTION_START"),
            event_row("g2", "grad", start + 140_000, "INJECTION_END"),
            event_row("g3", "grad", start + 150_000, "PHYSICAL_LEAK_OBSERVED"),
            event_row("g4", "grad", start + 145_000, "LEAK_FLAG_FIRST"),
            event_row("g5", "grad", start + 148_000, "LEAK_FLAG_CONFIRMED"),
            event_row("f1", "fill", start + 130_000, "INJECTION_START"),
            event_row("f2", "fill", start + 140_000, "INJECTION_END"),
            event_row("f3", "fill", start + 145_000, "LEAK_FLAG_FIRST"),
            event_row("f4", "fill", start + 148_000, "LEAK_FLAG_CONFIRMED"),
            event_row("u1", "sudden", start + 10_000, "PHYSICAL_LEAK_OBSERVED"),
            event_row("u2", "sudden", start + 9_000, "LEAK_FLAG_FIRST"),
            event_row("u3", "sudden", start + 11_000, "LEAK_FLAG_CONFIRMED"),
        ]
        manifest = [
            manifest_row("safe", planned_arm="SAFE"),
            manifest_row("grad", planned_arm="LEAK_GRADUAL", bag_id="b2", sensor_id="n2"),
            manifest_row("fill", planned_arm="LEAK_GRADUAL", bag_id="b3", sensor_id="n3", planned_safe_horizon_s="600"),
            manifest_row("sudden", planned_arm="LEAK_SUDDEN", bag_id="b4", sensor_id="n4"),
        ]
        report, out = self.run_qc(sessions, samples, events, manifest_rows=manifest, name="valid-arms")
        rows = read_sessions(out)
        for sid in ("safe", "grad", "fill", "sudden"):
            self.assertEqual(rows[sid]["overall_status"], "PASS", sid)
        self.assertEqual(report["status_counts"], {"PASS": 4, "FAIL": 0, "PARTIAL": 0})

    def test_safe_manifest_semantic_failures(self):
        cases = {
            "version": manifest_row("s1", protocol_version="v0.1"),
            "empty_id": manifest_row("s1", operator_id=""),
            "unsafe_id": manifest_row("s1", bag_id="bad,comma"),
            "missing_target": manifest_row("s1", target=""),
            "horizon_missing": manifest_row("s1", planned_safe_horizon_s=""),
            "horizon_nonpositive": manifest_row("s1", planned_safe_horizon_s="0"),
        }
        for label, mrow in cases.items():
            _, out = self._run_safe(mrow, f"sem-{label}")
            self.assertIn("INVALID_PROTOCOL_MANIFEST", codes_for(read_issues(out), "s1"), label)
            self.assertEqual(read_sessions(out)["s1"]["protocol_status"], "FAIL", label)

    def test_gradual_manifest_semantic_failures(self):
        cases = {
            "profile": manifest_row("g", planned_arm="LEAK_GRADUAL", injection_profile="spray"),
            "method": manifest_row("g", planned_arm="LEAK_GRADUAL", injection_method="gravity"),
            "flow": manifest_row("g", planned_arm="LEAK_GRADUAL", planned_flow_ml_min="-5"),
            "observation": manifest_row("g", planned_arm="LEAK_GRADUAL", physical_leak_observation_method=""),
            "horizon_bad": manifest_row("g", planned_arm="LEAK_GRADUAL", planned_safe_horizon_s="-1"),
        }
        for label, mrow in cases.items():
            _, out = self._run_gradual(mrow, f"gm-{label}")
            self.assertIn("INVALID_PROTOCOL_MANIFEST", codes_for(read_issues(out), "g"), label)

    def test_sudden_forbidden_horizon(self):
        sessions = [session_row("u", arm="LEAK_SUDDEN", start=T0, end=T0 + 70_000)]
        samples = linear_samples("u", T0, 65)
        events = [event_row("u1", "u", T0 + 10_000, "PHYSICAL_LEAK_OBSERVED")]
        mrow = manifest_row("u", planned_arm="LEAK_SUDDEN", planned_safe_horizon_s="120")
        _, out = self.run_qc(sessions, samples, events, manifest_rows=[mrow], name="sudden-hz")
        self.assertIn("INVALID_PROTOCOL_MANIFEST", codes_for(read_issues(out), "u"))

    def test_missing_manifest_row_is_id_mismatch(self):
        sessions = [session_row("s1", start=T0, end=T0 + 70_000),
                    session_row("s2", bag_id="b2", sensor_id="n2", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 65) + linear_samples("s2", T0, 65)
        _, out = self.run_qc(sessions, samples, [], manifest_rows=[manifest_row("s1")], name="misscover")
        self.assertIn("MANIFEST_ID_MISMATCH", codes_for(read_issues(out), "s2"))
        self.assertEqual(read_sessions(out)["s2"]["protocol_status"], "FAIL")

    def test_duplicate_manifest_session_is_fatal_before_output(self):
        inp = build_input(self.root / "dup-in", [session_row("s1", start=T0, end=T0 + 70_000)],
                          linear_samples("s1", T0, 65), [])
        man = write_manifest(self.root / "dup-p.csv", [manifest_row("s1"), manifest_row("s1")])
        out = self.root / "dup-out"
        with self.assertRaisesRegex(raw_qc.RawQcError, "duplicate session_id"):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=man)
        for name in raw_qc._OUTPUT_FILES:
            self.assertFalse((out / name).exists())


# --------------------------------------------------------------------------- #
# Event rules reconciled with the locked scenarios
# --------------------------------------------------------------------------- #
class EventRuleTests(RawQcBase):
    def _gradual(self, events, name, *, horizon=None):
        start = T0
        sessions = [session_row("g", arm="LEAK_GRADUAL", start=start, end=start + 400_000)]
        samples = linear_samples("g", start, 200)
        mrow = manifest_row("g", planned_arm="LEAK_GRADUAL", planned_safe_horizon_s=horizon)
        return self.run_qc(sessions, samples, events, manifest_rows=[mrow], name=name)

    def test_valid_multi_step_gradual_injection(self):
        s = T0
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 135_000, "INJECTION_END"),
                  event_row("c", "g", s + 140_000, "INJECTION_START"),
                  event_row("d", "g", s + 145_000, "INJECTION_END"),
                  event_row("e", "g", s + 150_000, "PHYSICAL_LEAK_OBSERVED"),
                  event_row("f", "g", s + 146_000, "LEAK_FLAG_FIRST"),
                  event_row("h", "g", s + 148_000, "LEAK_FLAG_CONFIRMED")]
        _, out = self._gradual(events, "multi-step")
        codes = codes_for(read_issues(out), "g")
        self.assertNotIn("MALFORMED_REQUIRED_EVENTS", codes)
        self.assertEqual(read_sessions(out)["g"]["protocol_status"], "PASS")

    def test_injection_pairing_faults(self):
        s = T0
        base_leak = event_row("p", "g", s + 160_000, "PHYSICAL_LEAK_OBSERVED")
        cases = {
            "end_before_start": [event_row("a", "g", s + 130_000, "INJECTION_END"),
                                 event_row("b", "g", s + 140_000, "INJECTION_START"),
                                 event_row("c", "g", s + 150_000, "INJECTION_END"), base_leak],
            "overlapping_start": [event_row("a", "g", s + 130_000, "INJECTION_START"),
                                  event_row("b", "g", s + 135_000, "INJECTION_START"),
                                  event_row("c", "g", s + 140_000, "INJECTION_END"), base_leak],
            "unclosed_start": [event_row("a", "g", s + 130_000, "INJECTION_START"), base_leak],
        }
        for label, events in cases.items():
            _, out = self._gradual(events, f"pair-{label}")
            self.assertIn("MALFORMED_REQUIRED_EVENTS", codes_for(read_issues(out), "g"), label)

    def test_multiple_physical_leaks_is_malformed(self):
        s = T0
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END"),
                  event_row("c", "g", s + 150_000, "PHYSICAL_LEAK_OBSERVED"),
                  event_row("d", "g", s + 160_000, "PHYSICAL_LEAK_OBSERVED")]
        _, out = self._gradual(events, "two-leaks")
        self.assertIn("MALFORMED_REQUIRED_EVENTS", codes_for(read_issues(out), "g"))

    def test_planned_gradual_leak_without_physical_leak_fails(self):
        s = T0
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END")]
        _, out = self._gradual(events, "grad-noleak")  # horizon empty -> leak required
        self.assertIn("MISSING_REQUIRED_EVENT", codes_for(read_issues(out), "g"))
        self.assertEqual(read_sessions(out)["g"]["protocol_status"], "FAIL")

    def test_non_leaking_fill_without_leak_passes(self):
        s = T0
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END")]
        _, out = self._gradual(events, "fill-noleak", horizon="600")
        row = read_sessions(out)["g"]
        codes = codes_for(read_issues(out), "g")
        self.assertEqual(row["protocol_status"], "PASS")
        self.assertNotIn("MISSING_REQUIRED_EVENT", codes)
        self.assertNotIn("MISSING_LIG_FLAG_EVENT", codes)

    def test_gradual_without_manifest_does_not_infer_planned_leak(self):
        s = T0
        sessions = [session_row("g", arm="LEAK_GRADUAL", start=s, end=s + 400_000)]
        samples = linear_samples("g", s, 200)
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END")]
        _, out = self.run_qc(sessions, samples, events, name="gradual-no-manifest")
        row = read_sessions(out)["g"]
        codes = codes_for(read_issues(out), "g")
        self.assertEqual(row["protocol_status"], "NOT_EVALUATED")
        self.assertEqual(row["overall_status"], "PARTIAL")
        self.assertNotIn("MISSING_REQUIRED_EVENT", codes)
        self.assertNotIn("MISSING_LIG_FLAG_EVENT", codes)

    def test_invalid_manifest_does_not_infer_planned_leak(self):
        s = T0
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END")]
        _, out = self._gradual(events, "invalid-horizon-context", horizon="-1")
        codes = codes_for(read_issues(out), "g")
        self.assertIn("INVALID_PROTOCOL_MANIFEST", codes)
        self.assertNotIn("MISSING_REQUIRED_EVENT", codes)
        self.assertNotIn("MISSING_LIG_FLAG_EVENT", codes)

    def test_physical_leak_before_injection_is_malformed(self):
        s = T0
        events = [event_row("p", "g", s + 125_000, "PHYSICAL_LEAK_OBSERVED"),
                  event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END"),
                  event_row("f", "g", s + 126_000, "LEAK_FLAG_FIRST"),
                  event_row("c", "g", s + 127_000, "LEAK_FLAG_CONFIRMED")]
        _, out = self._gradual(events, "leak-before-injection")
        self.assertIn("MALFORMED_REQUIRED_EVENTS", codes_for(read_issues(out), "g"))
        self.assertEqual(read_sessions(out)["g"]["protocol_status"], "FAIL")

    def test_leak_confirmed_without_physical_leak_is_malformed(self):
        s = T0
        sessions = [session_row(
            "g", arm="LEAK_GRADUAL", start=s, end=s + 400_000,
            end_reason="LEAK_CONFIRMED",
        )]
        samples = linear_samples("g", s, 200)
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END")]
        manifest = [manifest_row(
            "g", planned_arm="LEAK_GRADUAL", planned_safe_horizon_s="600",
        )]
        _, out = self.run_qc(
            sessions, samples, events, manifest_rows=manifest, name="confirmed-no-leak",
        )
        codes = codes_for(read_issues(out), "g")
        self.assertIn("MALFORMED_REQUIRED_EVENTS", codes)
        self.assertNotIn("MISSING_REQUIRED_EVENT", codes)
        self.assertEqual(read_sessions(out)["g"]["protocol_status"], "FAIL")

    def test_actual_leak_without_lig_flags_warns(self):
        s = T0
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END"),
                  event_row("p", "g", s + 150_000, "PHYSICAL_LEAK_OBSERVED")]
        _, out = self._gradual(events, "leak-no-lig-flags")
        issues = [i for i in read_issues(out) if i["session_id"] == "g"]
        warnings = [i for i in issues if i["code"] == "MISSING_LIG_FLAG_EVENT"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["severity"], "WARNING")
        self.assertEqual(read_sessions(out)["g"]["protocol_status"], "PASS")

    def test_unexpected_leak_in_fill_is_warning_only(self):
        s = T0
        events = [event_row("a", "g", s + 130_000, "INJECTION_START"),
                  event_row("b", "g", s + 140_000, "INJECTION_END"),
                  event_row("c", "g", s + 150_000, "PHYSICAL_LEAK_OBSERVED"),
                  event_row("d", "g", s + 145_000, "LEAK_FLAG_FIRST"),
                  event_row("e", "g", s + 148_000, "LEAK_FLAG_CONFIRMED")]
        _, out = self._gradual(events, "fill-leak", horizon="600")
        issues = [i for i in read_issues(out) if i["session_id"] == "g"]
        unplanned = [i for i in issues if i["code"] == "UNPLANNED_PHYSICAL_LEAK"]
        self.assertEqual(len(unplanned), 1)
        self.assertEqual(unplanned[0]["severity"], "WARNING")
        self.assertEqual(read_sessions(out)["g"]["overall_status"], "PASS")

    def test_safe_injection_events_fail(self):
        sessions = [session_row("s1", arm="SAFE", start=T0, end=T0 + 300_000)]
        samples = linear_samples("s1", T0, 65)
        events = [event_row("a", "s1", T0 + 130_000, "INJECTION_START"),
                  event_row("b", "s1", T0 + 140_000, "INJECTION_END")]
        _, out = self.run_qc(sessions, samples, events, manifest_rows=[manifest_row("s1")], name="safe-inject")
        codes = codes_for(read_issues(out), "s1")
        self.assertIn("MALFORMED_REQUIRED_EVENTS", codes)
        self.assertEqual(read_sessions(out)["s1"]["protocol_status"], "FAIL")

    def test_device_restart_during_session_fails(self):
        sessions = [session_row("s1", arm="SAFE", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 65)
        events = [event_row("a", "s1", T0 + 10_000, "DEVICE_RESTART")]
        _, out = self.run_qc(sessions, samples, events, manifest_rows=[manifest_row("s1")], name="restart")
        self.assertIn("DEVICE_RESTART_DURING_SESSION", codes_for(read_issues(out), "s1"))
        self.assertEqual(read_sessions(out)["s1"]["protocol_status"], "FAIL")

    def test_sudden_leak_rules(self):
        def sudden(events, name):
            sessions = [session_row("u", arm="LEAK_SUDDEN", start=T0, end=T0 + 70_000)]
            samples = linear_samples("u", T0, 65)
            mrow = manifest_row("u", planned_arm="LEAK_SUDDEN")
            return self.run_qc(sessions, samples, events, manifest_rows=[mrow], name=name)

        # Exactly one physical leak + LIG flags -> PASS.
        ok = [event_row("a", "u", T0 + 10_000, "PHYSICAL_LEAK_OBSERVED"),
              event_row("b", "u", T0 + 9_000, "LEAK_FLAG_FIRST"),
              event_row("c", "u", T0 + 11_000, "LEAK_FLAG_CONFIRMED")]
        _, out = sudden(ok, "sudden-ok")
        self.assertEqual(read_sessions(out)["u"]["protocol_status"], "PASS")
        # Zero physical leaks -> MISSING_REQUIRED_EVENT.
        _, out = sudden([], "sudden-none")
        self.assertIn("MISSING_REQUIRED_EVENT", codes_for(read_issues(out), "u"))
        # Two physical leaks -> MALFORMED_REQUIRED_EVENTS.
        two = [event_row("a", "u", T0 + 10_000, "PHYSICAL_LEAK_OBSERVED"),
               event_row("b", "u", T0 + 20_000, "PHYSICAL_LEAK_OBSERVED")]
        _, out = sudden(two, "sudden-two")
        self.assertIn("MALFORMED_REQUIRED_EVENTS", codes_for(read_issues(out), "u"))


# --------------------------------------------------------------------------- #
# Determinism / output safety / input immutability
# --------------------------------------------------------------------------- #
class OutputSafetyTests(RawQcBase):
    def _clean_input(self, name="c"):
        sessions = [session_row("s1", start=T0, end=T0 + 70_000)]
        samples = linear_samples("s1", T0, 65)
        inp = build_input(self.root / f"{name}-in", sessions, samples, [])
        man = write_manifest(self.root / f"{name}-p.csv", [manifest_row("s1")])
        return inp, man

    def test_byte_identical_outputs(self):
        inp, man = self._clean_input("det")
        raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "a", protocol_manifest=man)
        raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "b", protocol_manifest=man)
        for name in raw_qc._OUTPUT_FILES:
            self.assertEqual((self.root / "a" / name).read_bytes(),
                             (self.root / "b" / name).read_bytes())

    def test_refuse_overwrite_preserves_unrelated(self):
        inp, man = self._clean_input("ov")
        out = self.root / "out"
        raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=man)
        keep = out / "keep.txt"; keep.write_text("k", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=man)
        raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=man, overwrite=True)
        self.assertEqual(keep.read_text(encoding="utf-8"), "k")

    def test_failed_validation_preserves_previous_outputs(self):
        inp, man = self._clean_input("fv")
        out = self.root / "out"
        raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=man)
        before = {n: (out / n).read_bytes() for n in raw_qc._OUTPUT_FILES}
        # Break the config so validation fails during the second run.
        bad_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        bad_config["expected_interval_ms"] = -1
        with self.assertRaises(raw_qc.RawQcError):
            raw_qc.run_raw_qc(inp, bad_config, out, protocol_manifest=man, overwrite=True)
        for name, data in before.items():
            self.assertEqual((out / name).read_bytes(), data)

    def test_staged_write_failure_preserves_previous_outputs(self):
        inp, man = self._clean_input("sw")
        out = self.root / "out"
        raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=man)
        before = {n: (out / n).read_bytes() for n in raw_qc._OUTPUT_FILES}
        with (
            mock.patch.object(raw_qc, "_dump_json", side_effect=RuntimeError("injected")),
            self.assertRaisesRegex(RuntimeError, "injected"),
        ):
            raw_qc.run_raw_qc(inp, CONFIG_PATH, out, protocol_manifest=man, overwrite=True)
        for name, data in before.items():
            self.assertEqual((out / name).read_bytes(), data)

    def test_inputs_are_not_mutated(self):
        inp, man = self._clean_input("im")
        before = {p.name: p.read_bytes() for p in inp.iterdir()}
        before_manifest = man.read_bytes()
        raw_qc.run_raw_qc(inp, CONFIG_PATH, self.root / "out", protocol_manifest=man)
        after = {p.name: p.read_bytes() for p in inp.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(man.read_bytes(), before_manifest)


# --------------------------------------------------------------------------- #
# Synthetic integration smoke (pipeline mechanics only)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(NUMPY, "synthetic generator requires the optional [pipeline] numpy dependency")
class SyntheticSmokeTests(unittest.TestCase):
    def test_synthetic_v03_reports_intentional_gap_fault(self):
        from ostosense_ai import synthetic

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw = root / "raw"
            synthetic.generate(
                synthetic.load_config(REPO_ROOT / "configs" / "synthetic-v0.3.json"), 20260722, raw)
            manifest = REPO_ROOT / "tests" / "fixtures" / "ostosense-evaluation-v0.1" / "protocol_manifest.csv"
            report = raw_qc.run_raw_qc(raw, CONFIG_PATH, root / "qc", protocol_manifest=manifest)
            # Pipeline mechanics only, explicitly NOT an OSTOSENSE performance result.
            self.assertEqual(report["dataset_origin"], "SYNTHETIC_PIPELINE_TEST_ONLY")
            self.assertEqual(report["status_counts"], {"PASS": 10, "FAIL": 1, "PARTIAL": 0})
            rows = read_sessions(root / "qc")
            self.assertEqual(rows["syn-fault-gap-001"]["overall_status"], "FAIL")
            self.assertIn(
                "INTERVAL_OUT_OF_TOLERANCE",
                codes_for(read_issues(root / "qc"), "syn-fault-gap-001"),
            )

            report_partial = raw_qc.run_raw_qc(raw, CONFIG_PATH, root / "qc2")
            self.assertEqual(
                report_partial["status_counts"],
                {"PASS": 0, "FAIL": 1, "PARTIAL": 10},
            )


if __name__ == "__main__":
    unittest.main()
