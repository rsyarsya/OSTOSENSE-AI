import copy
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ostosense_ai import bundled_model, integration_cli, live_window
from test_live_service import raw_request

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


class IntegrationCliTests(unittest.TestCase):
    def test_usage_errors_are_distinct_from_waiting(self):
        self.assertEqual(self.run_cli(["evaluate-window"]), 1)
        self.assertEqual(self.run_cli(["--help"]), 0)

    def run_cli(self, args):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return integration_cli.main(args)

    def test_examples_determinism_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.json", Path(folder) / "b.json"
            for path in (first, second):
                self.assertEqual(self.run_cli(["examples", "--output", str(path)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(self.run_cli(["examples", "--output", str(first)]), 1)
            self.assertEqual(self.run_cli(["examples", "--output", str(first), "--overwrite"]), 0)

    def test_evaluate_exit_codes_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as folder:
            source, result = Path(folder) / "input.json", Path(folder) / "result.json"
            for end, code in ((30000, 2), (120000, 0)):
                source.write_text(json.dumps(raw_request(end)), encoding="utf-8")
                before = source.read_bytes()
                args = [
                    "evaluate-window",
                    "--input",
                    str(source),
                    "--output",
                    str(result),
                    "--overwrite",
                ]
                self.assertEqual(self.run_cli(args), code)
                self.assertEqual(before, source.read_bytes())
                previous = result.read_bytes()
                self.assertEqual(self.run_cli(args), code)
                self.assertEqual(previous, result.read_bytes())

    def test_validation_and_staging_failures_preserve_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, result = Path(folder) / "input.json", Path(folder) / "result.json"
            result.write_text("previous", encoding="utf-8")
            source.write_text("{}", encoding="utf-8")
            args = [
                "evaluate-window",
                "--input",
                str(source),
                "--output",
                str(result),
                "--overwrite",
            ]
            self.assertEqual(self.run_cli(args), 1)
            self.assertEqual(result.read_text(), "previous")
            with patch(
                "ostosense_ai.integration_cli.Path.replace", side_effect=OSError("staging failure")
            ):
                self.assertEqual(
                    self.run_cli(["examples", "--output", str(result), "--overwrite"]), 1
                )
            self.assertEqual(result.read_text(), "previous")

    def test_refuses_input_output_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.json"
            source.write_text(json.dumps(raw_request()), encoding="utf-8")
            before = source.read_bytes()
            self.assertEqual(
                self.run_cli(
                    [
                        "evaluate-window",
                        "--input",
                        str(source),
                        "--output",
                        str(source),
                        "--overwrite",
                    ]
                ),
                1,
            )
            self.assertEqual(source.read_bytes(), before)


@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "optional quality dependency")
class ServiceSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource

        cls.schemas = {}
        resources = []
        for path in CONTRACTS.glob("*.schema.json"):
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            cls.schemas[path.name] = schema
            resources.append((schema["$id"], Resource.from_contents(schema)))
        cls.registry = Registry().with_resources(resources)
        cls.validator_type = Draft202012Validator

    def validator(self, name):
        return self.validator_type(self.schemas[name], registry=self.registry)

    def test_raw_schema_and_response_use_offline_refs(self):
        request = raw_request()
        bundle = bundled_model.load_bundled_model()
        self.validator("ai-raw-window-input-v0.1.schema.json").validate(request)
        result = live_window.evaluate_window(request, bundle["model"], bundle["model_sha256"])
        self.validator("ai-service-response-v0.1.schema.json").validate(result)
        for path in (CONTRACTS / "examples/service-v0.1").glob("*-input.json"):
            data = json.loads(path.read_text())
            self.validator("ai-raw-window-input-v0.1.schema.json").validate(data)
            result = live_window.evaluate_window(data, bundle["model"], bundle["model_sha256"])
            self.validator("ai-service-response-v0.1.schema.json").validate(result)

    def test_sample_and_request_extras_rejected(self):
        schema = self.validator("ai-raw-window-input-v0.1.schema.json")
        for location in ("request", "sample"):
            data = raw_request()
            target = data if location == "request" else data["window_samples"][0]
            target["extra"] = True
            self.assertTrue(list(schema.iter_errors(data)))

    def test_runtime_contract_cannot_be_bypassed_by_service_wrapper(self):
        result = live_window.unavailable("COLLECTING_BASELINE")
        validator = self.validator("ai-service-response-v0.1.schema.json")
        validator.validate(result)
        changed = copy.deepcopy(result)
        changed["ai"]["risk_class"] = "Safe"
        self.assertTrue(list(validator.iter_errors(changed)))
        changed = copy.deepcopy(result)
        changed["diagnostics"]["availability_reason"] = "READY"
        self.assertTrue(list(validator.iter_errors(changed)))


if __name__ == "__main__":
    unittest.main()
