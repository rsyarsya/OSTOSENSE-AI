import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ostosense_ai import inference, runtime_output

AI_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = AI_ROOT / "contracts"
EXAMPLE_DIR = CONTRACT_DIR / "examples"


def _model(**overrides):
    model = {
        "model_artifact_version": "0.1.0",
        "model_family": "mord.LogisticAT",
        "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
        "feature_order": list(inference.FEATURE_COLUMNS),
        "class_order": list(inference.CLASS_NAMES),
        "class_mapping": dict(inference.CLASS_NAME_TO_INDEX),
        "scaler": {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "beta": [1.0, 0.0, 0.0, 0.0, 0.0],
        "theta": [-1.0, 0.0, 1.0],
    }
    model.update(overrides)
    return model


class RuntimeOutputCoreTests(unittest.TestCase):
    def test_live_unavailable_has_exact_contract_and_no_risk(self):
        payload = runtime_output.unavailable_output()

        self.assertEqual(tuple(payload), runtime_output.OUTPUT_FIELDS)
        self.assertEqual(payload["runtime_output_version"], "0.1.0")
        self.assertEqual(payload["mode"], "LIVE")
        self.assertEqual(payload["data_source"], "NONE")
        self.assertEqual(payload["model_status"], "UNAVAILABLE")
        self.assertFalse(payload["prediction_available"])
        self.assertIsNone(payload["risk_class"])
        self.assertIsNone(payload["risk_class_index"])
        self.assertIsNone(payload["model_artifact_version"])
        self.assertEqual(payload["evidence_scope"], "NO_PREDICTION")

    def test_engineering_prediction_returns_class_not_probability(self):
        payload = runtime_output.predict_test_output(
            _model(), [5.0, 0.0, 0.0, 0.0, 0.0]
        )

        self.assertEqual(payload["mode"], "ENGINEERING_TEST")
        self.assertEqual(payload["data_source"], "SYNTHETIC_FIXTURE")
        self.assertEqual(payload["model_status"], "TEST_ONLY")
        self.assertTrue(payload["prediction_available"])
        self.assertEqual(payload["risk_class"], "Urgent")
        self.assertEqual(payload["risk_class_index"], 3)
        self.assertEqual(payload["model_artifact_version"], "0.1.0")
        self.assertEqual(payload["evidence_scope"], "PIPELINE_MECHANICS_ONLY")
        self.assertNotIn("probabilities", payload)
        self.assertNotIn("risk_percentage", payload)
        self.assertNotIn("countdown", payload)

    def test_synthetic_model_is_rejected_in_live_mode(self):
        with self.assertRaisesRegex(runtime_output.RuntimeOutputError, "LIVE"):
            runtime_output.predict_test_output(
                _model(), [0.0] * 5, mode="LIVE"
            )

    def test_wrong_test_data_source_is_rejected(self):
        with self.assertRaisesRegex(runtime_output.RuntimeOutputError, "SYNTHETIC_FIXTURE"):
            runtime_output.predict_test_output(
                _model(), [0.0] * 5, data_source="REAL_SENSOR"
            )

    def test_unavailable_output_rejects_non_live_state(self):
        invalid = (
            {"mode": "ENGINEERING_TEST"},
            {"data_source": "SYNTHETIC_FIXTURE"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(runtime_output.RuntimeOutputError):
                    runtime_output.unavailable_output(**kwargs)

    def test_bad_feature_rows_are_wrapped_as_runtime_errors(self):
        rows: tuple[Any, ...] = (
            [0.0] * 4,
            [0.0, 0.0, math.nan, 0.0, 0.0],
            None,
        )
        for row in rows:
            with self.subTest(row=row):
                with self.assertRaises(runtime_output.RuntimeOutputError):
                    runtime_output.predict_test_output(_model(), row)

    def test_inputs_are_not_mutated(self):
        model = _model()
        original_model = copy.deepcopy(model)
        features = [5.0, 0.0, 0.0, 0.0, 0.0]

        runtime_output.predict_test_output(model, features)

        self.assertEqual(model, original_model)
        self.assertEqual(features, [5.0, 0.0, 0.0, 0.0, 0.0])

    def test_validator_rejects_cross_domain_and_inconsistent_payloads(self):
        valid = runtime_output.unavailable_output()
        mutations = (
            lambda d: d.__setitem__("risk_percentage", 50),
            lambda d: d.__setitem__("prediction_available", True),
            lambda d: d.__setitem__("risk_class", "Safe"),
            lambda d: d.__setitem__("risk_class_index", 0),
            lambda d: d.__setitem__("model_status", "READY"),
        )
        for mutate in mutations:
            payload = copy.deepcopy(valid)
            mutate(payload)
            with self.assertRaises(runtime_output.RuntimeOutputError):
                runtime_output.validate_runtime_output(payload)

    def test_validator_rejects_non_integer_class_index(self):
        payload = runtime_output.predict_test_output(
            _model(), [5.0, 0.0, 0.0, 0.0, 0.0]
        )
        payload["risk_class_index"] = 3.0

        with self.assertRaisesRegex(runtime_output.RuntimeOutputError, "integer"):
            runtime_output.validate_runtime_output(payload)


class RuntimeOutputFileTests(unittest.TestCase):
    def test_writer_is_byte_identical_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            payload = runtime_output.unavailable_output()

            runtime_output.write_runtime_output(first, payload)
            runtime_output.write_runtime_output(second, payload)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            original = first.read_bytes()
            with self.assertRaises(FileExistsError):
                runtime_output.write_runtime_output(first, payload)
            self.assertEqual(first.read_bytes(), original)

    def test_validation_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "runtime.json"
            output.write_bytes(b"SENTINEL")
            invalid = runtime_output.unavailable_output()
            invalid["risk_percentage"] = 50

            with self.assertRaises(runtime_output.RuntimeOutputError):
                runtime_output.write_runtime_output(output, invalid, overwrite=True)

            self.assertEqual(output.read_bytes(), b"SENTINEL")

    def test_unavailable_cli_writes_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "runtime.json"

            exit_code = runtime_output.main(
                ["unavailable", "--output", str(output)]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            runtime_output.validate_runtime_output(payload)
            self.assertFalse(payload["prediction_available"])

    def test_predict_test_cli_uses_exact_feature_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.json"
            feature_path = root / "features.json"
            output = root / "runtime.json"
            model_path.write_text(json.dumps(_model()), encoding="utf-8")
            feature_path.write_text(
                json.dumps(
                    {
                        "data_source": "SYNTHETIC_FIXTURE",
                        "features": [5.0, 0.0, 0.0, 0.0, 0.0],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = runtime_output.main(
                [
                    "predict-test",
                    "--model",
                    str(model_path),
                    "--features",
                    str(feature_path),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["risk_class"], "Urgent")
            self.assertEqual(payload["risk_class_index"], 3)

    def test_predict_test_cli_rejects_malformed_feature_documents(self):
        malformed = (
            [5.0, 0.0, 0.0, 0.0, 0.0],
            {"features": [5.0, 0.0, 0.0, 0.0, 0.0]},
            {
                "data_source": "REAL_SENSOR",
                "features": [5.0, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "data_source": "SYNTHETIC_FIXTURE",
                "features": [5.0, 0.0, 0.0, 0.0, 0.0],
                "extra": True,
            },
        )
        for document in malformed:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model_path = root / "model.json"
                feature_path = root / "features.json"
                output = root / "runtime.json"
                model_path.write_text(json.dumps(_model()), encoding="utf-8")
                feature_path.write_text(json.dumps(document), encoding="utf-8")

                with self.assertRaises(runtime_output.RuntimeOutputError):
                    runtime_output.main(
                        [
                            "predict-test",
                            "--model",
                            str(model_path),
                            "--features",
                            str(feature_path),
                            "--output",
                            str(output),
                        ]
                    )

                self.assertFalse(output.exists())


class RuntimeOutputArtifactTests(unittest.TestCase):
    def test_json_schema_matches_runtime_contract(self):
        schema = json.loads(
            (CONTRACT_DIR / "ai-runtime-output-v0.1.schema.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["title"], "OSTOSENSE AI Runtime Output v0.1")
        self.assertEqual(schema["required"], list(runtime_output.OUTPUT_FIELDS))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(len(schema["oneOf"]), 5)
        self.assertEqual(
            set(schema["properties"]), set(runtime_output.OUTPUT_FIELDS)
        )

    def test_checked_examples_are_valid_and_keep_domains_separate(self):
        forbidden = {
            "probabilities",
            "risk_percentage",
            "countdown",
            "lig_raw",
            "direct_leak",
            "bag_fill",
            "humidity",
            "notification",
            "clinical_action",
        }
        examples = {
            "live-unavailable.json": (False, None, None),
            "engineering-test-monitor.json": (True, "Monitor", 1),
        }
        for filename, expected in examples.items():
            with self.subTest(filename=filename):
                payload = json.loads(
                    (EXAMPLE_DIR / filename).read_text(encoding="utf-8")
                )
                runtime_output.validate_runtime_output(payload)
                self.assertEqual(
                    (
                        payload["prediction_available"],
                        payload["risk_class"],
                        payload["risk_class_index"],
                    ),
                    expected,
                )
                self.assertTrue(forbidden.isdisjoint(payload))


if __name__ == "__main__":
    unittest.main()
