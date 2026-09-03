import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
    from ostosense_ai import synthetic
    from ostosense_contract import (
        Arm,
        CapQuality,
        EndReason,
        EventType,
        LigQuality,
        SystemQuality,
        aggregate_system_quality,
    )
else:
    try:
        import numpy as np
    except ModuleNotFoundError:
        DEPENDENCIES_AVAILABLE = False
    else:
        DEPENDENCIES_AVAILABLE = True
        from ostosense_ai import synthetic
        from ostosense_contract import (
            Arm,
            CapQuality,
            EndReason,
            EventType,
            LigQuality,
            SystemQuality,
            aggregate_system_quality,
        )


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "synthetic-v0.2.json"
PRIMARY_SEED = 20260722
VARIANT_SEED = 999


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_by_session(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["session_id"], []).append(row)
    return grouped


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    'install the optional "pipeline" dependencies to run these tests',
)
class SyntheticGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.config = synthetic.load_config(CONFIG_PATH)

        cls.primary_dir = root / "primary"
        cls.repeat_dir = root / "repeat"
        cls.variant_dir = root / "variant"
        cls.primary_manifest = synthetic.generate(
            cls.config, PRIMARY_SEED, cls.primary_dir
        )
        synthetic.generate(cls.config, PRIMARY_SEED, cls.repeat_dir)
        synthetic.generate(cls.config, VARIANT_SEED, cls.variant_dir)

        cls.sessions = _read_rows(cls.primary_dir / "sessions.csv")
        cls.samples = _read_rows(cls.primary_dir / "samples.csv")
        cls.events = _read_rows(cls.primary_dir / "events.csv")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # 1. Same seed and config -> byte-identical artifacts.
    def test_same_seed_produces_byte_identical_artifacts(self) -> None:
        for name in ("sessions.csv", "samples.csv", "events.csv", "manifest.json"):
            self.assertEqual(
                (self.primary_dir / name).read_bytes(),
                (self.repeat_dir / name).read_bytes(),
                f"{name} is not byte-identical for the same seed",
            )

    # 2. Different seeds change at least the stochastic sensor values.
    def test_different_seed_changes_sensor_values_but_not_structure(self) -> None:
        variant_samples = _read_rows(self.variant_dir / "samples.csv")
        primary_cap = [
            row["capacitance_raw"]
            for row in self.samples
            if row["session_id"] == "syn-safe-001"
        ]
        variant_cap = [
            row["capacitance_raw"]
            for row in variant_samples
            if row["session_id"] == "syn-safe-001"
        ]
        self.assertNotEqual(primary_cap, variant_cap)

        variant_manifest = json.loads(
            (self.variant_dir / "manifest.json").read_text("utf-8")
        )
        self.assertEqual(
            variant_manifest["session_count"],
            self.primary_manifest["session_count"],
        )
        self.assertEqual(
            variant_manifest["scenario_session_counts"],
            self.primary_manifest["scenario_session_counts"],
        )
        self.assertEqual(
            (self.primary_dir / "samples.csv").read_text("utf-8").splitlines()[0],
            (self.variant_dir / "samples.csv").read_text("utf-8").splitlines()[0],
        )

    # 3. All generated CSVs pass the existing contract validation (header check).
    def test_generated_csvs_reopen_under_contract_logger(self) -> None:
        from ostosense_contract import Tier1CsvLogger

        Tier1CsvLogger(self.primary_dir)  # re-validates the three contract headers

    # 4. Generated enum values match Data Contract v1.1 exactly.
    def test_enum_values_match_contract(self) -> None:
        arms = {a.value for a in Arm}
        end_reasons = {e.value for e in EndReason}
        cap_q = {c.value for c in CapQuality}
        lig_q = {lig.value for lig in LigQuality}
        system_q = {s.value for s in SystemQuality}
        event_types = {e.value for e in EventType}
        for row in self.sessions:
            self.assertIn(row["arm"], arms)
            self.assertIn(row["end_reason"], end_reasons)
        for row in self.samples:
            self.assertIn(row["cap_quality"], cap_q)
            self.assertIn(row["lig_quality"], lig_q)
            self.assertIn(row["system_quality"], system_q)
        for row in self.events:
            self.assertIn(row["event_type"], event_types)

    # 5. session_id references remain consistent across all three tables.
    def test_session_ids_consistent_across_tables(self) -> None:
        session_ids = {row["session_id"] for row in self.sessions}
        sample_ids = {row["session_id"] for row in self.samples}
        event_ids = {row["session_id"] for row in self.events}
        self.assertEqual(sample_ids, session_ids)
        self.assertTrue(event_ids.issubset(session_ids))

    # 6. Timestamps are ordered within each session.
    def test_timestamps_ordered_within_session(self) -> None:
        for rows in _group_by_session(self.samples).values():
            stamps = [int(r["timestamp"]) for r in rows]
            self.assertEqual(stamps, sorted(stamps))
            self.assertEqual(len(stamps), len(set(stamps)))  # strictly increasing
        for rows in _group_by_session(self.events).values():
            stamps = [int(r["timestamp"]) for r in rows]
            self.assertEqual(stamps, sorted(stamps))

    # 7. Nominal 1 Hz rows, except for intentional contract-valid gaps.
    def test_nominal_1hz_except_intentional_gaps(self) -> None:
        for session_id, rows in _group_by_session(self.samples).items():
            stamps = [int(r["timestamp"]) for r in rows]
            deltas = [b - a for a, b in zip(stamps, stamps[1:])]
            if "fault-gap" in session_id:
                irregular = [d for d in deltas if d != 1000]
                self.assertEqual(len(irregular), 1)
                self.assertGreater(irregular[0], 1000)
            else:
                self.assertTrue(all(d == 1000 for d in deltas), session_id)

    # 8. SAFE sessions contain no physical-leak event.
    def test_safe_sessions_have_no_physical_leak(self) -> None:
        safe_ids = {
            row["session_id"]
            for row in self.sessions
            if row["arm"] == Arm.SAFE.value
        }
        leaked = {
            r["session_id"]
            for r in self.events
            if r["event_type"] == EventType.PHYSICAL_LEAK_OBSERVED.value
        }
        self.assertTrue(safe_ids.isdisjoint(leaked))

    # 9. Gradual and sudden scenarios have coherent synthetic event sequences.
    def test_leak_scenarios_have_coherent_event_sequences(self) -> None:
        events_by_session = _group_by_session(self.events)
        signed_latencies = []

        for gradual_id in (
            "syn-gradual-observed-first-001",
            "syn-gradual-flag-first-001",
        ):
            ordered = events_by_session[gradual_id]
            positions = {r["event_type"]: int(r["timestamp"]) for r in ordered}
            for required in (
                EventType.INJECTION_START,
                EventType.INJECTION_END,
                EventType.PHYSICAL_LEAK_OBSERVED,
                EventType.LEAK_FLAG_FIRST,
                EventType.LEAK_FLAG_CONFIRMED,
            ):
                self.assertIn(required.value, positions)
            self.assertLess(
                positions[EventType.INJECTION_START.value],
                positions[EventType.INJECTION_END.value],
            )
            self.assertLess(
                positions[EventType.INJECTION_END.value],
                positions[EventType.PHYSICAL_LEAK_OBSERVED.value],
            )
            self.assertLess(
                positions[EventType.LEAK_FLAG_FIRST.value],
                positions[EventType.LEAK_FLAG_CONFIRMED.value],
            )
            signed_latencies.append(
                positions[EventType.LEAK_FLAG_FIRST.value]
                - positions[EventType.PHYSICAL_LEAK_OBSERVED.value]
            )

        self.assertTrue(any(value > 0 for value in signed_latencies))
        self.assertTrue(any(value < 0 for value in signed_latencies))

        sudden = events_by_session["syn-sudden-001"]
        sudden_types = [r["event_type"] for r in sudden]
        self.assertNotIn(EventType.INJECTION_START.value, sudden_types)
        sudden_positions = {r["event_type"]: int(r["timestamp"]) for r in sudden}
        self.assertIn(EventType.PHYSICAL_LEAK_OBSERVED.value, sudden_positions)
        self.assertLess(
            sudden_positions[EventType.LEAK_FLAG_FIRST.value],
            sudden_positions[EventType.LEAK_FLAG_CONFIRMED.value],
        )

    # 10. Quality states and system_quality remain consistent.
    def test_system_quality_derived_from_channel_quality(self) -> None:
        seen = set()
        for row in self.samples:
            expected = aggregate_system_quality(
                CapQuality(row["cap_quality"]), LigQuality(row["lig_quality"])
            )
            self.assertEqual(row["system_quality"], expected.value)
            seen.add(row["system_quality"])
        # scenario coverage exercises more than the nominal state
        self.assertIn(SystemQuality.NORMAL.value, seen)
        self.assertIn(SystemQuality.INITIALIZING.value, seen)
        self.assertIn(SystemQuality.ML_UNAVAILABLE.value, seen)
        self.assertIn(SystemQuality.FAILSAFE_DEGRADED.value, seen)
        self.assertIn(SystemQuality.UNSAFE.value, seen)

    # 11. Manifest contains the synthetic-only warning and provenance.
    def test_manifest_declares_synthetic_origin_and_provenance(self) -> None:
        manifest = self.primary_manifest
        self.assertEqual(manifest["dataset_origin"], "SYNTHETIC_PIPELINE_TEST_ONLY")
        self.assertIn("performance", manifest["warning"].lower())
        self.assertEqual(manifest["seed"], PRIMARY_SEED)
        self.assertEqual(len(manifest["config_sha256"]), 64)
        self.assertEqual(manifest["session_count"], len(self.sessions))
        self.assertIn("nominal_sampling_rate_hz", manifest)
        provenance = manifest["numeric_parameter_provenance"]
        self.assertTrue(provenance)
        self.assertTrue(
            all(label in synthetic.ALLOWED_PROVENANCE for label in provenance.values())
        )
        # persisted manifest matches the returned dict
        persisted = json.loads((self.primary_dir / "manifest.json").read_text("utf-8"))
        self.assertEqual(persisted, manifest)

    # Provenance guard: every numeric config parameter is classified.
    def test_config_provenance_is_complete_and_enforced(self) -> None:
        synthetic.validate_config_provenance(self.config)

        missing = json.loads(json.dumps(self.config))
        missing["provenance"].pop("signal.cap_baseline")
        with self.assertRaises(ValueError):
            synthetic.validate_config_provenance(missing)

        invalid = json.loads(json.dumps(self.config))
        invalid["provenance"]["signal.cap_baseline"] = "MADE_UP_LABEL"
        with self.assertRaises(ValueError):
            synthetic.validate_config_provenance(invalid)

    def test_existing_artifacts_require_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sessions.csv"
            target.write_text("USER_DATA\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                synthetic.generate(
                    self.config, PRIMARY_SEED, tmp
                )
            self.assertEqual(
                target.read_text(encoding="utf-8"), "USER_DATA\n"
            )
            synthetic.generate(
                self.config, PRIMARY_SEED, tmp, overwrite=True
            )
            self.assertNotEqual(target.read_text(encoding="utf-8"), "USER_DATA\n")

    def test_invalid_config_fails_before_output_mutation(self) -> None:
        bad_configs = []

        bad = json.loads(json.dumps(self.config))
        bad["sampling_rate_hz"] = 2
        bad_configs.append(("unsupported rate", bad))

        bad = json.loads(json.dumps(self.config))
        bad["warmup_s"] = 300
        bad_configs.append(("warmup consumes the usable session", bad))

        bad = json.loads(json.dumps(self.config))
        bad["phases"]["gradual_dry_fraction"] = 0.9
        bad_configs.append(("reversed gradual phases", bad))

        bad = json.loads(json.dumps(self.config))
        bad["phases"]["leak_confirm_delay_s"] = 2
        bad_configs.append(("confirmation before first flag", bad))

        bad = json.loads(json.dumps(self.config))
        bad["identity"]["operator_typo"] = "silently-ignored-before-v0.2"
        bad_configs.append(("unknown nested key", bad))

        bad = json.loads(json.dumps(self.config))
        bad["scenarios"][0]["scenario_typo"] = "silently-ignored-before-v0.2"
        bad_configs.append(("unknown scenario key", bad))

        for name, bad_config in bad_configs:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    marker = Path(tmp) / "sessions.csv"
                    marker.write_text("USER_DATA\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        synthetic.generate(
                            bad_config, PRIMARY_SEED, tmp, overwrite=True
                        )
                    self.assertEqual(
                        marker.read_text(encoding="utf-8"), "USER_DATA\n"
                    )

    def test_all_events_stay_within_session_bounds(self) -> None:
        bounds = {
            row["session_id"]: (
                int(row["start_timestamp"]), int(row["end_timestamp"])
            )
            for row in self.sessions
        }
        for event in self.events:
            start, end = bounds[event["session_id"]]
            self.assertLessEqual(start, int(event["timestamp"]))
            self.assertLessEqual(int(event["timestamp"]), end)

    def test_session_baseline_matches_first_valid_window(self) -> None:
        samples_by_session = _group_by_session(self.samples)
        baseline_count = self.config["baseline_window_s"]
        for session in self.sessions:
            valid = [
                float(row["capacitance_raw"])
                for row in samples_by_session[session["session_id"]]
                if row["cap_quality"] == CapQuality.OK.value
            ][:baseline_count]
            self.assertEqual(len(valid), baseline_count)
            expected_median = round(float(np.median(valid)), 6)
            expected_std = round(float(np.std(valid)), 6)
            self.assertAlmostEqual(float(session["baseline_value"]), expected_median)
            self.assertAlmostEqual(float(session["baseline_std"]), expected_std)

    def test_hardware_ids_include_repeated_independent_groups(self) -> None:
        bag_ids = [row["bag_id"] for row in self.sessions]
        sensor_ids = [row["sensor_id"] for row in self.sessions]
        self.assertGreater(len(set(bag_ids)), 1)
        self.assertGreater(len(set(sensor_ids)), 1)
        self.assertLess(len(set(bag_ids)), len(bag_ids))
        self.assertLess(len(set(sensor_ids)), len(sensor_ids))
        self.assertEqual(len(bag_ids), self.primary_manifest["session_count"])
        self.assertEqual(len(sensor_ids), self.primary_manifest["session_count"])

    def test_confirmation_count_is_derived_from_delays(self) -> None:
        phases = self.config["phases"]
        expected = phases["leak_confirm_delay_s"] - phases["leak_flag_delay_s"] + 1
        confirmed = [
            event
            for event in self.events
            if event["event_type"] == EventType.LEAK_FLAG_CONFIRMED.value
        ]
        self.assertTrue(confirmed)
        for event in confirmed:
            self.assertEqual(
                json.loads(event["event_metadata"])["consecutive_samples"],
                expected,
            )


if __name__ == "__main__":
    unittest.main()
