import copy
import importlib.util
import json
import unittest
from pathlib import Path

from ostosense_ai import runtime_output

AI_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = AI_ROOT / "contracts"
JSONSCHEMA_AVAILABLE = importlib.util.find_spec("jsonschema") is not None


@unittest.skipUnless(JSONSCHEMA_AVAILABLE, "jsonschema is in the optional quality extra")
class RuntimeJsonSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jsonschema import Draft202012Validator

        cls.validator_type = Draft202012Validator

    def _schema(self, version: str):
        return json.loads(
            (CONTRACT_DIR / f"ai-runtime-output-v{version}.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def _feature_input_schema(self):
        return json.loads(
            (CONTRACT_DIR / "ai-feature-input-v0.1.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_schemas_are_valid_draft_2020_12_documents(self):
        for version in ("0.1", "0.2"):
            with self.subTest(version=version):
                self.validator_type.check_schema(self._schema(version))
        self.validator_type.check_schema(self._feature_input_schema())

    def test_feature_input_examples_pass_schema_and_runtime_reader(self):
        validator = self.validator_type(self._feature_input_schema())
        examples = sorted(
            (CONTRACT_DIR / "examples" / "feature-input-v0.1").glob("*.json")
        )
        self.assertEqual(len(examples), 2)
        for path in examples:
            with self.subTest(path=path.name):
                document = json.loads(path.read_text(encoding="utf-8"))
                validator.validate(document)
                runtime_output._read_v2_feature_document(
                    path,
                    expected_data_source=document["data_source"],
                    expected_channel=document["model_input_channel"],
                )

    def test_feature_input_schema_rejects_scaled_pilot_features(self):
        document = json.loads(
            (
                CONTRACT_DIR
                / "examples"
                / "feature-input-v0.1"
                / "real-kap7.json"
            ).read_text(encoding="utf-8")
        )
        document["feature_basis"] = "PILOT_DELTA_NORM"

        errors = list(
            self.validator_type(self._feature_input_schema()).iter_errors(document)
        )
        self.assertTrue(errors)

    def test_all_examples_pass_external_and_internal_validation(self):
        examples = {
            "0.1": sorted((CONTRACT_DIR / "examples").glob("*.json")),
            "0.2": sorted((CONTRACT_DIR / "examples" / "v0.2").glob("*.json")),
        }
        for version, paths in examples.items():
            validator = self.validator_type(self._schema(version))
            for path in paths:
                with self.subTest(version=version, path=path.name):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    validator.validate(payload)
                    runtime_output.validate_runtime_output(payload)

    def test_schema_rejects_cross_state_payloads(self):
        payload = runtime_output.predict_live_experimental_output(
            {
                "model_artifact_version": "0.1.0",
                "model_family": "mord.LogisticAT",
                "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
                "feature_order": list(runtime_output.inference.FEATURE_COLUMNS),
                "class_order": list(runtime_output.inference.CLASS_NAMES),
                "class_mapping": dict(runtime_output.inference.CLASS_NAME_TO_INDEX),
                "scaler": {"mean": [0.0] * 5, "scale": [1.0] * 5},
                "beta": [1.0, 0.0, 0.0, 0.0, 0.0],
                "theta": [-1.0, 0.0, 1.0],
            },
            [0.0] * 5,
            source_window_end_ms=120_000,
            model_artifact_sha256="a" * 64,
        )
        invalid = copy.deepcopy(payload)
        invalid["model_input_channel"] = "Kap_4"

        errors = list(self.validator_type(self._schema("0.2")).iter_errors(invalid))
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
