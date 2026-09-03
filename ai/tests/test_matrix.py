import csv
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from ostosense_ai import features, labeling, matrix
from ostosense_ai.matrix import MatrixError, build_model_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELING_FIX = Path(__file__).resolve().parent / "fixtures" / "ostosense-labeling-v0.1"
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

RAW_SESSIONS_SHA = "aaaa"
RAW_SAMPLES_SHA = "bbbb"
FEATURE_CONVENTION = {**matrix._CANONICAL_CONVENTION, "working_source": "protocol"}
LABEL_CONVENTION = dict(matrix._CANONICAL_CONVENTION)


def _win(i, *, session="s1", bag="b1", sensor="sen1", partition="development",
         fv=True, lv=True, risk="", reason="", struct_reason="", feats=None):
    return {
        "window_id": f"{session}-win-{i:04d}",
        "session": session, "bag": bag, "sensor": sensor, "partition": partition,
        "windex": i, "wstart": i * 10_000, "wend": i * 10_000 + 120_000,
        "fv": fv, "lv": lv, "risk": risk, "reason": reason, "struct_reason": struct_reason,
        "feats": feats if feats is not None else ["1.5", "2.5", "3.5", "4.5", "5.5"],
    }


def _default_windows():
    return [
        _win(0, risk="Safe", feats=["0.1", "0.2", "0.3", "0.4", "0.5"]),
        _win(1, risk="Monitor", feats=["1.1", "1.2", "1.3", "1.4", "1.5"]),
        _win(2, risk="Caution", feats=["2.1", "2.2", "2.3", "2.4", "2.5"]),
        _win(3, risk="Urgent", feats=["3.1", "3.2", "3.3", "3.4", "3.5"]),
        _win(4, fv=False, lv=False, reason="INVALID_CAP_QUALITY",
             struct_reason="INVALID_CAP_QUALITY", feats=["", "", "", "", ""]),
        _win(5, fv=True, lv=False, reason="SUDDEN_ARM"),
        _win(6, fv=True, lv=False, reason="POST_LEAK"),
    ]


def _write_features(feat_dir: Path, windows, *, origin, convention):
    feat_dir.mkdir(parents=True, exist_ok=True)
    with (feat_dir / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(features.FEATURES_CSV_COLUMNS)
        for w in windows:
            writer.writerow([
                w["window_id"], w["session"], w["bag"], w["sensor"], w["windex"],
                w["wstart"], w["wend"], 120, "true" if w["fv"] else "false",
                w["struct_reason"] if not w["fv"] else "", *w["feats"],
            ])
    valid = sum(1 for w in windows if w["fv"])
    reason_counts = {r: 0 for r in features.EXCLUSION_PRIORITY}
    for w in windows:
        if not w["fv"]:
            reason_counts[w["struct_reason"]] += 1
    manifest = {
        "extractor_version": "0.1.1", "config_id": "features-v0.1", "config_sha256": "cfg",
        "data_contract_version": "v1.1", "input_sessions_sha256": RAW_SESSIONS_SHA,
        "input_samples_sha256": RAW_SAMPLES_SHA, "input_dataset_origin": origin,
        "window_convention": convention, "feature_columns": list(matrix.FEATURE_COLUMNS),
        "candidate_window_count": len(windows), "valid_window_count": valid,
        "excluded_window_count": len(windows) - valid, "exclusion_reason_counts": reason_counts,
        "session_count": 1, "warning": "w",
    }
    (feat_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return matrix._sha256_file(feat_dir / "features.csv"), matrix._sha256_file(feat_dir / "feature_manifest.json")


def _write_labels(lbl_dir: Path, windows, *, origin, convention, feat_csv_sha, feat_manifest_sha):
    lbl_dir.mkdir(parents=True, exist_ok=True)
    with (lbl_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(labeling.LABELS_CSV_COLUMNS)
        for w in windows:
            idx = str(matrix.CLASS_NAME_TO_INDEX[w["risk"]]) if w["lv"] else ""
            writer.writerow([
                w["window_id"], w["session"], w["windex"], w["wstart"], w["wend"],
                w["risk"] if w["lv"] else "", idx, "true" if w["lv"] else "false",
                "" if w["lv"] else w["reason"], "v0.3", "b-ENGINEERING_TEST_ONLY",
                w["partition"], "false", "",
            ])
    valid = sum(1 for w in windows if w["lv"])
    risk_counts = {name: 0 for name in matrix.CLASS_NAMES}
    reason_counts = {r: 0 for r in labeling.ALL_EXCLUSION_REASONS}
    for w in windows:
        if w["lv"]:
            risk_counts[w["risk"]] += 1
        else:
            reason_counts[w["reason"]] += 1
    manifest = {
        "labeler_version": "0.1.1", "rulebook_version": "v0.3", "data_contract_version": "v1.1",
        "boundary_config_version": "b-ENGINEERING_TEST_ONLY", "boundary_config_sha256": "bs",
        "dataset_origin": origin,
        "input_sha256": {"sessions_csv": RAW_SESSIONS_SHA, "samples_csv": RAW_SAMPLES_SHA,
                         "events_csv": "e", "protocol_manifest_csv": "p",
                         "partition_manifest_csv": "q", "boundary_config_json": "b"},
        "feature_artifact_sha256": {"features_csv_sha256": feat_csv_sha,
                                    "feature_manifest_sha256": feat_manifest_sha},
        "window_convention": convention, "session_count": 1,
        "candidate_window_count": len(windows), "valid_window_count": valid,
        "excluded_window_count": len(windows) - valid, "risk_class_counts": risk_counts,
        "exclusion_reason_counts": reason_counts, "protocol_deviation_session_count": 0,
        "warning": "w",
    }
    (lbl_dir / "label_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build(root: Path, windows=None, *, feat_origin="SYNTHETIC_PIPELINE_TEST_ONLY",
           lbl_origin="SYNTHETIC_PIPELINE_TEST_ONLY", feat_conv=None, lbl_conv=None):
    windows = windows if windows is not None else _default_windows()
    feat_dir, lbl_dir = root / "features", root / "labels"
    csha, msha = _write_features(feat_dir, windows, origin=feat_origin,
                                 convention=feat_conv or FEATURE_CONVENTION)
    _write_labels(lbl_dir, windows, origin=lbl_origin, convention=lbl_conv or LABEL_CONVENTION,
                  feat_csv_sha=csha, feat_manifest_sha=msha)
    return feat_dir, lbl_dir


def _edit_json(path: Path, mutator):
    data = json.loads(path.read_text(encoding="utf-8"))
    mutator(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _edit_csv(path: Path, mutator):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError("test CSV must contain a header")
        fields = list(reader.fieldnames)
        rows = list(reader)
    mutator(rows, fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_matrix(output_dir: Path):
    with (output_dir / "model_matrix.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class MatrixBuildTests(unittest.TestCase):
    def test_schema_and_column_order(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            build_model_matrix(feat, lbl, root / "out")
            header = (root / "out" / "model_matrix.csv").read_text("utf-8").splitlines()[0].split(",")
        self.assertEqual(tuple(header), matrix.MODEL_MATRIX_COLUMNS)
        self.assertEqual(header[5:10], list(matrix.FEATURE_COLUMNS))

    def test_golden_join(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            manifest = build_model_matrix(feat, lbl, root / "out")
            rows = {r["window_id"]: r for r in _read_matrix(root / "out")}
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows["s1-win-0000"]["risk_label"], "Safe")
        self.assertEqual(rows["s1-win-0000"]["risk_label_index"], "0")
        self.assertEqual(rows["s1-win-0000"]["cap_delta_mean"], "0.1")
        self.assertEqual(rows["s1-win-0003"]["risk_label"], "Urgent")
        self.assertEqual(rows["s1-win-0003"]["risk_label_index"], "3")
        self.assertEqual(manifest["eligible_row_count"], 4)
        self.assertEqual(manifest["class_counts"], {"Safe": 1, "Monitor": 1, "Caution": 1, "Urgent": 1})
        self.assertEqual(manifest["source_candidate_window_count"], 7)
        self.assertEqual(manifest["target_column"], "risk_label_index")
        self.assertEqual(manifest["target_label_column"], "risk_label")
        self.assertEqual(
            manifest["grouping_columns"], ["session_id", "bag_id", "sensor_id"]
        )
        self.assertEqual(manifest["partition_column"], "dataset_partition")

    def test_model_matrix_sha256_covers_final_csv_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            manifest = build_model_matrix(feat, lbl, root / "out")
            actual = matrix._sha256_file(root / "out" / "model_matrix.csv")
        self.assertEqual(len(manifest["model_matrix_sha256"]), 64)
        self.assertEqual(manifest["model_matrix_sha256"], actual)
        # tampering with the CSV after generation is detectable against the manifest hash
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            out = root / "out"
            manifest = build_model_matrix(feat, lbl, out)
            (out / "model_matrix.csv").write_text("tampered\n", encoding="utf-8")
            self.assertNotEqual(manifest["model_matrix_sha256"], matrix._sha256_file(out / "model_matrix.csv"))

    def test_five_feature_allowlist_and_no_forbidden_metadata(self):
        matrix._assert_matrix_columns_safe()
        self.assertEqual(len(matrix.FEATURE_COLUMNS), 5)
        self.assertTrue(set(matrix.FEATURE_COLUMNS).isdisjoint(matrix._NON_FEATURE_NAMES))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            manifest = build_model_matrix(feat, lbl, root / "out")
        self.assertEqual(manifest["feature_columns"], list(matrix.FEATURE_COLUMNS))
        for forbidden in ("session_id", "bag_id", "dataset_partition", "risk_label", "arm"):
            self.assertNotIn(forbidden, manifest["feature_columns"])

    def test_byte_identical_repeated_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            build_model_matrix(feat, lbl, root / "a")
            build_model_matrix(feat, lbl, root / "b")
            for f in ("model_matrix.csv", "matrix_manifest.json"):
                self.assertEqual((root / "a" / f).read_bytes(), (root / "b" / f).read_bytes())

    def test_overwrite_refusal_and_preserves_unrelated(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            out = root / "out"
            build_model_matrix(feat, lbl, out)
            keep = out / "keep.txt"
            keep.write_text("k", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_model_matrix(feat, lbl, out)
            build_model_matrix(feat, lbl, out, overwrite=True)
            self.assertEqual(keep.read_text("utf-8"), "k")

    def test_input_files_unmodified(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            before = {p: matrix._sha256_file(p) for p in (
                feat / "features.csv", feat / "feature_manifest.json",
                lbl / "labels.csv", lbl / "label_manifest.json")}
            build_model_matrix(feat, lbl, root / "out")
            after = {p: matrix._sha256_file(p) for p in before}
        self.assertEqual(before, after)


class MatrixValidationTests(unittest.TestCase):
    def _expect_error(self, mutate):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            mutate(root, feat, lbl)
            with self.assertRaises((MatrixError, ValueError)):
                build_model_matrix(feat, lbl, root / "out")

    def test_duplicate_feature_window_id(self):
        def mutate(root, feat, lbl):
            _edit_csv(feat / "features.csv", lambda rows, f: rows.append(dict(rows[0])))
            _edit_json(feat / "feature_manifest.json",
                       lambda m: m.__setitem__("candidate_window_count", m["candidate_window_count"] + 1))
        self._expect_error(mutate)

    def test_missing_or_extra_windows(self):
        def mutate(root, feat, lbl):
            _edit_csv(lbl / "labels.csv", lambda rows, f: rows[0].__setitem__("window_id", "s1-win-9999"))
        self._expect_error(mutate)

    def test_identity_mismatch(self):
        def mutate(root, feat, lbl):
            _edit_csv(lbl / "labels.csv", lambda rows, f: rows[0].__setitem__("window_start", "77"))
        self._expect_error(mutate)

    def test_manifest_hash_mismatch(self):
        def mutate(root, feat, lbl):
            _edit_json(lbl / "label_manifest.json",
                       lambda m: m["feature_artifact_sha256"].__setitem__("features_csv_sha256", "deadbeef"))
        self._expect_error(mutate)

    def test_raw_hash_disagreement(self):
        def mutate(root, feat, lbl):
            _edit_json(feat / "feature_manifest.json",
                       lambda m: m.__setitem__("input_sessions_sha256", "zzzz"))
        self._expect_error(mutate)

    def test_dataset_origin_mismatch(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root, feat_origin="REAL_DATA")
            with self.assertRaises(MatrixError):
                build_model_matrix(feat, lbl, root / "out")

    def test_window_convention_mismatch(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bad = {**FEATURE_CONVENTION, "window_seconds": 999}
            feat, lbl = _build(root, feat_conv=bad)
            with self.assertRaises(MatrixError):
                build_model_matrix(feat, lbl, root / "out")

    def test_count_mismatch(self):
        def mutate(root, feat, lbl):
            _edit_json(feat / "feature_manifest.json",
                       lambda m: m.__setitem__("valid_window_count", 999))
        self._expect_error(mutate)

    def test_feature_valid_vs_label_valid_inconsistency(self):
        windows = _default_windows()
        windows[4]["lv"] = True   # structurally invalid feature, but label claims valid
        windows[4]["risk"] = "Safe"
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root, windows)
            with self.assertRaises(MatrixError):
                build_model_matrix(feat, lbl, root / "out")

    def test_label_name_index_inconsistency(self):
        def mutate(root, feat, lbl):
            _edit_csv(lbl / "labels.csv",
                      lambda rows, f: rows[0].__setitem__("risk_label_index", "3"))  # Safe but 3
        self._expect_error(mutate)

    def test_invalid_label_carries_index_is_rejected(self):
        def mutate(root, feat, lbl):
            _edit_csv(lbl / "labels.csv",
                      lambda rows, f: rows[4].__setitem__("risk_label_index", "0"))  # excluded row
        self._expect_error(mutate)

    def test_non_finite_and_malformed_features(self):
        for bad in ("nan", "inf", "-inf", "", "abc"):
            windows = _default_windows()
            windows[0]["feats"] = [bad, "1.0", "2.0", "3.0", "4.0"]
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                feat, lbl = _build(root, windows)
                with self.assertRaises(MatrixError):
                    build_model_matrix(feat, lbl, root / "out")

    def test_strict_boolean_fields(self):
        cases = (
            ("feature_valid", "features.csv", 4),
            ("label_valid", "labels.csv", 4),
            ("protocol_deviation", "labels.csv", 0),
        )
        for field, filename, row_index in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as name:
                    root = Path(name)
                    feat, lbl = _build(root)
                    target = feat if filename == "features.csv" else lbl
                    _edit_csv(
                        target / filename,
                        lambda rows, fields, i=row_index, f=field: rows[i].__setitem__(
                            f, "garbage"
                        ),
                    )
                    if filename == "features.csv":
                        _edit_json(
                            lbl / "label_manifest.json",
                            lambda data: data["feature_artifact_sha256"].__setitem__(
                                "features_csv_sha256",
                                matrix._sha256_file(feat / "features.csv"),
                            ),
                        )
                    with self.assertRaises(MatrixError):
                        build_model_matrix(feat, lbl, root / "out")

    def test_manifest_and_row_versions_are_enforced(self):
        mutations = (
            lambda feat, lbl: _edit_json(
                feat / "feature_manifest.json",
                lambda data: data.__setitem__("data_contract_version", "v9"),
            ),
            lambda feat, lbl: _edit_json(
                lbl / "label_manifest.json",
                lambda data: data.__setitem__("rulebook_version", "v9"),
            ),
            lambda feat, lbl: _edit_csv(
                lbl / "labels.csv",
                lambda rows, fields: rows[0].__setitem__("rulebook_version", "v9"),
            ),
            lambda feat, lbl: _edit_csv(
                lbl / "labels.csv",
                lambda rows, fields: rows[0].__setitem__(
                    "boundary_config_version", "wrong"
                ),
            ),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                with tempfile.TemporaryDirectory() as name:
                    root = Path(name)
                    feat, lbl = _build(root)
                    mutate(feat, lbl)
                    with self.assertRaises(MatrixError):
                        build_model_matrix(feat, lbl, root / "out")

    def test_manifest_class_and_exclusion_counts_reconcile(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            _edit_csv(
                lbl / "labels.csv",
                lambda rows, fields: (
                    rows[0].__setitem__("risk_label", "Urgent"),
                    rows[0].__setitem__("risk_label_index", "3"),
                ),
            )
            with self.assertRaises(MatrixError):
                build_model_matrix(feat, lbl, root / "out")

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            _edit_json(
                feat / "feature_manifest.json",
                lambda data: data["exclusion_reason_counts"].__setitem__(
                    "INVALID_CAP_QUALITY", 999
                ),
            )
            with self.assertRaises(MatrixError):
                build_model_matrix(feat, lbl, root / "out")

    def test_invalid_partition_and_empty_group_ids(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            _edit_csv(
                lbl / "labels.csv",
                lambda rows, fields: [
                    row.__setitem__("dataset_partition", "typo") for row in rows
                ],
            )
            with self.assertRaises(MatrixError):
                build_model_matrix(feat, lbl, root / "out")

        for field in ("bag_id", "sensor_id"):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as name:
                    root = Path(name)
                    feat, lbl = _build(root)
                    _edit_csv(
                        feat / "features.csv",
                        lambda rows, fields, f=field: [
                            row.__setitem__(f, "") for row in rows
                        ],
                    )
                    _edit_json(
                        lbl / "label_manifest.json",
                        lambda data: data["feature_artifact_sha256"].__setitem__(
                            "features_csv_sha256",
                            matrix._sha256_file(feat / "features.csv"),
                        ),
                    )
                    with self.assertRaises(MatrixError):
                        build_model_matrix(feat, lbl, root / "out")

    def test_structural_exclusion_must_match(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            _edit_csv(
                feat / "features.csv",
                lambda rows, fields: rows[4].__setitem__(
                    "exclusion_reason", "PARTIAL_WINDOW"
                ),
            )
            _edit_json(
                feat / "feature_manifest.json",
                lambda data: (
                    data["exclusion_reason_counts"].__setitem__(
                        "INVALID_CAP_QUALITY", 0
                    ),
                    data["exclusion_reason_counts"].__setitem__("PARTIAL_WINDOW", 1),
                ),
            )
            _edit_json(
                lbl / "label_manifest.json",
                lambda data: (
                    data["feature_artifact_sha256"].__setitem__(
                        "features_csv_sha256",
                        matrix._sha256_file(feat / "features.csv"),
                    ),
                    data["feature_artifact_sha256"].__setitem__(
                        "feature_manifest_sha256",
                        matrix._sha256_file(feat / "feature_manifest.json"),
                    ),
                ),
            )
            with self.assertRaises(MatrixError):
                build_model_matrix(feat, lbl, root / "out")

    def test_zero_eligible_partition_remains_visible_in_manifest(self):
        windows = [
            _win(
                0,
                session="s1",
                bag="b1",
                sensor="sen1",
                partition="development",
                risk="Safe",
            ),
            _win(
                0,
                session="s2",
                bag="b2",
                sensor="sen2",
                partition="final_test",
                fv=True,
                lv=False,
                reason="SUDDEN_ARM",
            ),
        ]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root, windows)
            manifest = build_model_matrix(feat, lbl, root / "out")
        self.assertEqual(manifest["partition_values"], ["development", "final_test"])
        self.assertEqual(
            manifest["source_partition_window_counts"],
            {"development": 1, "final_test": 1},
        )
        self.assertEqual(
            manifest["partition_row_counts"],
            {"development": 1, "final_test": 0},
        )

    def test_partition_leakage_session_bag_sensor(self):
        cases = (
            [
                _win(0, session="same", bag="b1", sensor="a", partition="development", risk="Safe"),
                _win(1, session="same", bag="b1", sensor="a", partition="final_test", risk="Safe"),
            ],
            [
                _win(0, session="s1", bag="shared", sensor="a", partition="development", risk="Safe"),
                _win(0, session="s2", bag="shared", sensor="b", partition="final_test", risk="Safe"),
            ],
            [
                _win(0, session="s1", bag="a", sensor="shared", partition="development", risk="Safe"),
                _win(0, session="s2", bag="b", sensor="shared", partition="final_test", risk="Safe"),
            ],
        )
        for index, windows in enumerate(cases):
            with self.subTest(group=index):
                with tempfile.TemporaryDirectory() as name:
                    root = Path(name)
                    feat, lbl = _build(root, windows)
                    with self.assertRaises(MatrixError):
                        build_model_matrix(feat, lbl, root / "out")

    def test_staging_failure_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            feat, lbl = _build(root)
            out = root / "out"
            out.mkdir()
            matrix_path = out / "model_matrix.csv"
            manifest_path = out / "matrix_manifest.json"
            matrix_path.write_bytes(b"MATRIX_SENTINEL")
            manifest_path.write_bytes(b"MANIFEST_SENTINEL")
            with mock.patch.object(
                matrix, "_write_manifest", side_effect=OSError("staging failed")
            ):
                with self.assertRaises(OSError):
                    build_model_matrix(feat, lbl, out, overwrite=True)
            self.assertEqual(matrix_path.read_bytes(), b"MATRIX_SENTINEL")
            self.assertEqual(manifest_path.read_bytes(), b"MANIFEST_SENTINEL")


@unittest.skipUnless(NUMPY_AVAILABLE, "requires numpy for the synthetic generator")
class MatrixSyntheticIntegrationTests(unittest.TestCase):
    def test_end_to_end_counts(self):
        from ostosense_ai import synthetic

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw, feat, lbl, mtx = root / "raw", root / "feat", root / "lbl", root / "mtx"
            synthetic.generate(
                synthetic.load_config(REPO_ROOT / "configs" / "synthetic-v0.2.json"), 20260722, raw)
            features.extract(raw, features.load_config(REPO_ROOT / "configs" / "features-v0.1.json"), feat)
            labeling.label_dataset(
                raw,
                LABELING_FIX / "protocol_manifest.csv",
                LABELING_FIX / "partition_manifest.csv",
                LABELING_FIX / "boundary-engineering-test-only-v0.1.json",
                lbl,
                features_dir=feat,
            )
            manifest = build_model_matrix(feat, lbl, mtx)
            rows = _read_matrix(mtx)
        self.assertEqual(manifest["source_candidate_window_count"], 162)
        self.assertEqual(manifest["eligible_row_count"], 76)
        self.assertEqual(len(rows), 76)
        self.assertEqual(
            manifest["class_counts"], {"Safe": 42, "Monitor": 16, "Caution": 12, "Urgent": 6})


if __name__ == "__main__":
    unittest.main()
