import csv
import json
import tempfile
import unittest
from pathlib import Path

from ostosense_contract import (
    Arm,
    CapQuality,
    EndReason,
    EventRecord,
    EventType,
    LigQuality,
    SampleRecord,
    SessionRecord,
    SystemQuality,
    Tier1CsvLogger,
    aggregate_system_quality,
)


class QualityMatrixTests(unittest.TestCase):
    def test_quality_matrix(self) -> None:
        self.assertEqual(
            aggregate_system_quality(CapQuality.WARMING_UP, LigQuality.WARMING_UP),
            SystemQuality.INITIALIZING,
        )
        self.assertEqual(
            aggregate_system_quality(CapQuality.OK, LigQuality.WARMING_UP),
            SystemQuality.INITIALIZING,
        )
        self.assertEqual(
            aggregate_system_quality(CapQuality.WARMING_UP, LigQuality.OK),
            SystemQuality.INITIALIZING,
        )
        self.assertEqual(
            aggregate_system_quality(CapQuality.OK, LigQuality.OK),
            SystemQuality.NORMAL,
        )
        self.assertEqual(
            aggregate_system_quality(CapQuality.DATA_GAP, LigQuality.OK),
            SystemQuality.ML_UNAVAILABLE,
        )
        self.assertEqual(
            aggregate_system_quality(CapQuality.OK, LigQuality.DISCONNECTED),
            SystemQuality.FAILSAFE_DEGRADED,
        )
        self.assertEqual(
            aggregate_system_quality(CapQuality.OK, LigQuality.BASELINE_INVALID),
            SystemQuality.FAILSAFE_DEGRADED,
        )
        self.assertEqual(
            aggregate_system_quality(
                CapQuality.BASELINE_INVALID, LigQuality.DISCONNECTED
            ),
            SystemQuality.UNSAFE,
        )

    def test_sample_rejects_inconsistent_system_quality(self) -> None:
        with self.assertRaises(ValueError):
            SampleRecord(
                timestamp=1_750_000_000_000,
                session_id="session-1",
                capacitance_raw=12.0,
                lig_raw=400.0,
                cap_quality=CapQuality.OK,
                lig_quality=LigQuality.DISCONNECTED,
                system_quality=SystemQuality.NORMAL,
            )


class LoggerTests(unittest.TestCase):
    def test_records_reject_untyped_enum_strings(self) -> None:
        with self.assertRaises(ValueError):
            EventRecord(
                event_id="event-1",
                session_id="session-1",
                timestamp=1_750_000_000_000,
                event_type="INJECTION_START",  # type: ignore[arg-type]
                event_metadata={},
            )

    def test_writes_three_contract_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            logger = Tier1CsvLogger(temporary_directory)
            start = 1_750_000_000_000

            logger.append_sample(
                SampleRecord.create(
                    timestamp=start,
                    session_id="session-1",
                    capacitance_raw=12.5,
                    lig_raw=410.0,
                    cap_quality=CapQuality.OK,
                    lig_quality=LigQuality.OK,
                    activity_state="STATIC",
                    orientation_position="UPRIGHT",
                )
            )
            logger.append_event(
                EventRecord(
                    event_id="event-1",
                    session_id="session-1",
                    timestamp=start + 1_000,
                    event_type=EventType.INJECTION_END,
                    event_metadata={
                        "delivered_volume_ml": 20.0,
                        "cumulative_volume_ml": 20.0,
                    },
                )
            )
            logger.append_event(
                EventRecord(
                    event_id="event-2",
                    session_id="session-1",
                    timestamp=start + 2_000,
                    event_type=EventType.LIG_CALIBRATION_FAILED,
                    event_metadata={
                        "duration_ms": 2_000,
                        "sample_count": 2,
                        "failure_reason": "PENDING_HARDWARE_VALIDITY_RULE",
                    },
                )
            )
            logger.append_session(
                SessionRecord(
                    session_id="session-1",
                    arm=Arm.LEAK_GRADUAL,
                    bag_id="bag-1",
                    sensor_id="sensor-1",
                    device_id="device-1",
                    fluid_type="saline-glycerin-v1",
                    operator_id="operator-1",
                    baseline_value=12.0,
                    baseline_std=0.1,
                    start_timestamp=start,
                    end_timestamp=start + 60_000,
                    end_reason=EndReason.MANUAL_STOP,
                    model_version="",
                    firmware_version="fw-0.1.0",
                )
            )

            root = Path(temporary_directory)
            self.assertTrue((root / "sessions.csv").exists())
            self.assertTrue((root / "samples.csv").exists())
            self.assertTrue((root / "events.csv").exists())

            with (root / "events.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["event_type"], "INJECTION_END")
            self.assertEqual(
                json.loads(rows[0]["event_metadata"])["delivered_volume_ml"],
                20.0,
            )
            self.assertEqual(
                rows[1]["event_type"], "LIG_CALIBRATION_FAILED"
            )

    def test_existing_wrong_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sessions.csv"
            path.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                Tier1CsvLogger(temporary_directory)


if __name__ == "__main__":
    unittest.main()
