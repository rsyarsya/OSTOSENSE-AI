import json
import re
import subprocess
import unittest
from pathlib import Path


AI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_ROOT.parent
INVENTORY_PATH = AI_ROOT / "data-manifests" / "real-pilot-v0.1.inventory.json"
RAW_PILOT_NAME = re.compile(r"^P00[1-7](?:_[0-9]+)?\.csv$")


class RealPilotInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

    def test_inventory_is_unlabeled_and_raw_files_are_not_published(self):
        self.assertEqual(self.inventory["dataset_origin"], "REAL_PILOT_UNLABELED")
        self.assertEqual(self.inventory["label_status"], "UNLABELED")
        self.assertEqual(
            self.inventory["raw_data_publication"],
            "NOT_INCLUDED_IN_PUBLIC_REPOSITORY",
        )
        self.assertEqual(self.inventory["runtime_experimental_channel"], "Kap_7")
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertFalse(
            [path for path in tracked if RAW_PILOT_NAME.fullmatch(Path(path).name)]
        )

    def test_session_identity_hashes_and_totals_reconcile(self):
        sessions = self.inventory["sessions"]
        self.assertEqual(len(sessions), 11)
        self.assertEqual(len({row["session_id"] for row in sessions}), 11)
        self.assertEqual(len({row["file_name"] for row in sessions}), 11)
        for row in sessions:
            digest = row["sha256"]
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(character in "0123456789abcdef" for character in digest))

        totals = self.inventory["totals"]
        fields = (
            ("raw_row_count", "raw_row_count"),
            ("complete_1hz_bin_count", "complete_1hz_bin_count"),
            ("partial_final_raw_sample_count", "partial_final_raw_sample_count"),
            ("unlabeled_window_count", "unlabeled_window_count"),
        )
        self.assertEqual(totals["session_count"], len(sessions))
        for total_field, session_field in fields:
            self.assertEqual(
                totals[total_field],
                sum(row[session_field] for row in sessions),
            )

    def test_sensor_fault_session_is_excluded_from_descriptive_analysis(self):
        sessions = {row["session_id"]: row for row in self.inventory["sessions"]}
        self.assertFalse(sessions["P006"]["descriptive_analysis_included"])
        self.assertEqual(sessions["P006"]["unlabeled_window_count"], 0)
        self.assertTrue(all(sessions[name]["descriptive_analysis_included"] for name in sessions if name != "P006"))

    def test_inventory_contains_no_performance_claims(self):
        forbidden = {"accuracy", "macro_f1", "quadratic_weighted_kappa", "meets_target"}

        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        self.assertTrue(forbidden.isdisjoint(keys(self.inventory)))


if __name__ == "__main__":
    unittest.main()
