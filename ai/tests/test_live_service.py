import copy
import importlib.util
import math
import os
import unittest
from unittest.mock import patch

from ostosense_ai import bundled_model, features, live_window, runtime_output


def raw_request(end=120000, status="NOT_EVALUATED"):
    rows = [
        {
            "sample_no": i,
            "elapsed_ms": i * 100,
            "kap_7_raw": 30000 + i / 100,
            "kap_7_status": status,
        }
        for i in range(end // 100)
    ]
    return {
        "request_version": "0.1.0",
        "device_id": "test-device",
        "session_id": "test-session",
        "acquisition_id": "test-run",
        "operator_confirmed": True,
        "source_window_end_ms": end,
        "baseline_samples": copy.deepcopy(rows[:600]),
        "window_samples": copy.deepcopy(rows[-1200:]),
    }


class BundledModelTests(unittest.TestCase):
    def test_four_examples_are_actual_inference(self):
        loaded = bundled_model.load_bundled_model()
        self.assertEqual(
            [row["expected"]["risk_class"] for row in loaded["examples"]],
            ["Safe", "Monitor", "Caution", "Urgent"],
        )
        for row in loaded["examples"]:
            runtime_output.validate_runtime_output(row["expected"])
            self.assertEqual(row["expected"]["model_artifact_sha256"], loaded["model_sha256"])

    def test_corrupt_bundle_is_rejected(self):
        with patch.object(bundled_model, "MODEL_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                bundled_model.load_bundled_model()


class LiveWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = bundled_model.load_bundled_model()

    def evaluate(self, data):
        return live_window.evaluate_window(data, self.bundle["model"], self.bundle["model_sha256"])

    def test_raw_features_equal_canonical_formula(self):
        request = raw_request()
        before = copy.deepcopy(request)
        output = self.evaluate(request)
        members = live_window._reduce(request["window_samples"], 0, 120)
        expected = features._compute_features(members, output["diagnostics"]["baseline_value"])
        self.assertEqual(
            output["feature_input"]["features"],
            [expected[name] for name in features.FEATURE_COLUMNS],
        )
        self.assertEqual(output["ai"]["model_status"], "UNVALIDATED")
        self.assertEqual(output["diagnostics"]["quality_verification"], "OPERATOR_ONLY")
        runtime_output.validate_runtime_output(output["ai"])
        self.assertEqual(request, before)

    def test_hand_calculated_median_baseline_and_features(self):
        request = raw_request()
        for row in request["window_samples"]:
            second = row["elapsed_ms"] // 1000
            row["kap_7_raw"] = 999999 if row["sample_no"] % 10 == 9 else 1000 + second
        request["baseline_samples"] = copy.deepcopy(request["window_samples"][:600])
        result = self.evaluate(request)
        self.assertEqual(result["diagnostics"]["baseline_value"], 1029.5)
        self.assertEqual(result["feature_input"]["features"], [30.0, 89.5, 1.0, 1199.916667, 119.0])

    def test_automatic_quality_and_determinism(self):
        request = raw_request(status="OK")
        self.assertEqual(self.evaluate(request), self.evaluate(request))
        self.assertEqual(self.evaluate(request)["diagnostics"]["quality_verification"], "AUTOMATIC")

    def test_confirmation_and_warmup(self):
        request = raw_request()
        request["operator_confirmed"] = False
        self.assertEqual(
            self.evaluate(request)["diagnostics"]["availability_reason"],
            "OPERATOR_CONFIRMATION_REQUIRED",
        )
        for end, reason in ((30000, "COLLECTING_BASELINE"), (60000, "COLLECTING_WINDOW")):
            with self.subTest(end=end):
                result = self.evaluate(raw_request(end))
                self.assertFalse(result["ai"]["prediction_available"])
                self.assertEqual(result["diagnostics"]["availability_reason"], reason)

    def test_frozen_baseline_in_later_window(self):
        result = self.evaluate(raw_request(240000))
        first = self.evaluate(raw_request())
        self.assertEqual(
            result["diagnostics"]["baseline_value"], first["diagnostics"]["baseline_value"]
        )
        self.assertEqual(result["ai"]["source_window_end_ms"], 240000)

    def test_missing_duplicate_unordered_gap_and_timing(self):
        mutations = [
            (lambda rows: rows.pop(), "INCOMPLETE_DATA"),
            (lambda rows: rows.__setitem__(900, dict(rows[899])), "DUPLICATE_SAMPLE"),
            (lambda rows: rows.__setitem__(slice(900, 902), rows[900:902][::-1]), "OUT_OF_ORDER"),
            (lambda rows: rows[900].__setitem__("sample_no", 90000), "DATA_GAP"),
            (lambda rows: rows[900].__setitem__("elapsed_ms", 90050), "TIMING_OUT_OF_TOLERANCE"),
        ]
        for mutate, reason in mutations:
            request = raw_request()
            mutate(request["window_samples"])
            with self.subTest(reason=reason):
                output = self.evaluate(request)
                self.assertEqual(output["diagnostics"]["availability_reason"], reason)
                self.assertIsNone(output["ai"]["risk_class"])

    def test_known_bad_quality_and_unknown_status_fail_closed(self):
        for status in ("DISCONNECTED", "SATURATED", "ADC_SATURATED", "DATA_GAP", "NONSENSE"):
            request = raw_request()
            request["window_samples"][900]["kap_7_status"] = status
            with self.subTest(status=status):
                self.assertEqual(
                    self.evaluate(request)["diagnostics"]["availability_reason"], "SENSOR_PROBLEM"
                )

    def test_baseline_gap_requires_new_dry_measurement(self):
        request = raw_request()
        request["baseline_samples"].pop()
        self.assertEqual(
            self.evaluate(request)["diagnostics"]["availability_reason"], "BASELINE_INCOMPLETE_DATA"
        )

    def test_overlap_conflict_rejected(self):
        request = raw_request()
        request["baseline_samples"][20]["kap_7_raw"] += 1
        with self.assertRaisesRegex(live_window.LiveWindowError, "conflicting"):
            self.evaluate(request)

    def test_overlap_timestamp_shift_is_not_silently_accepted(self):
        request = raw_request()
        request["window_samples"][20]["elapsed_ms"] += 5
        with self.assertRaisesRegex(live_window.LiveWindowError, "conflicting"):
            self.evaluate(request)

    def test_requests_do_not_share_session_state(self):
        first = raw_request()
        second = raw_request()
        second["acquisition_id"] = "another-run"
        second["device_id"] = "another-device"
        for key in ("baseline_samples", "window_samples"):
            for row in second[key]:
                row["kap_7_raw"] += 1000
        before = self.evaluate(first)
        other = self.evaluate(second)
        self.assertEqual(before, self.evaluate(first))
        self.assertAlmostEqual(
            other["diagnostics"]["baseline_value"] - before["diagnostics"]["baseline_value"], 1000
        )

    def test_malformed_identity_stride_sample_and_nonfinite(self):
        for field, value in (
            ("device_id", ""),
            ("operator_confirmed", 1),
            ("source_window_end_ms", True),
            ("source_window_end_ms", 120100),
        ):
            request = raw_request()
            request[field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(live_window.LiveWindowError),
            ):
                self.evaluate(request)
        for value in (math.nan, math.inf, -1, True, 2**40):
            request = raw_request()
            request["window_samples"][900]["kap_7_raw"] = value
            with self.subTest(value=value), self.assertRaises(live_window.LiveWindowError):
                self.evaluate(request)


HAS_SERVER = all(importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx"))


@unittest.skipUnless(HAS_SERVER, "optional [serve,serve-test] dependencies required")
class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        from ostosense_ai.service import create_app

        self.settings = patch.dict(
            os.environ,
            {
                "OSTOSENSE_SERVICE_TOKEN": "x" * 32,
                "OSTOSENSE_ALLOW_TEST": "true",
                "OSTOSENSE_ENABLE_EXPERIMENTAL": "true",
            },
        )
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.headers = {"Authorization": "Bearer " + "x" * 32}

    async def test_health_and_four_http_classes(self):
        self.assertTrue((await self.client.get("/health")).json()["ready"])
        examples = (await self.client.get("/v1/examples", headers=self.headers)).json()
        for case in examples:
            response = await self.client.post(
                "/v1/predict-features", json=case["input"], headers=self.headers
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["ai"], case["expected"])

    async def test_live_http_and_rejection(self):
        response = await self.client.post(
            "/v1/evaluate-window", json=raw_request(), headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ai"]["model_status"], "UNVALIDATED")
        response = await self.client.post("/v1/evaluate-window", json={}, headers=self.headers)
        self.assertEqual(response.status_code, 422)

    async def test_authentication_and_body_limit(self):
        self.assertEqual((await self.client.post("/v1/evaluate-window", json={})).status_code, 401)
        self.assertEqual((await self.client.get("/v1/examples")).status_code, 401)
        self.assertEqual(
            (
                await self.client.post(
                    "/v1/evaluate-window", content="{" * 530000, headers=self.headers
                )
            ).status_code,
            413,
        )

    async def test_no_real_feature_bypass(self):
        data = bundled_model.load_bundled_model()["examples"][0]["input"]
        data["data_source"] = "REAL_SENSOR"
        self.assertEqual(
            (
                await self.client.post("/v1/predict-features", json=data, headers=self.headers)
            ).status_code,
            422,
        )

    async def test_default_modes_disabled(self):
        import httpx
        from ostosense_ai.service import create_app

        with patch.dict(
            os.environ, {"OSTOSENSE_ALLOW_TEST": "false", "OSTOSENSE_ENABLE_EXPERIMENTAL": "false"}
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
            ) as client:
                self.assertEqual(
                    (await client.get("/v1/examples", headers=self.headers)).status_code, 403
                )
                self.assertEqual(
                    (
                        await client.post(
                            "/v1/evaluate-window", json=raw_request(), headers=self.headers
                        )
                    ).status_code,
                    403,
                )

    def test_missing_token_refuses_start(self):
        from ostosense_ai.service import create_app

        with patch.dict(os.environ, {"OSTOSENSE_SERVICE_TOKEN": ""}):
            with self.assertRaises(RuntimeError):
                create_app()

    async def test_broken_model_is_not_ready_and_does_not_predict(self):
        import httpx
        from ostosense_ai.service import create_app

        with patch.object(bundled_model, "load_bundled_model", side_effect=ValueError("tamper")):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
            ) as client:
                self.assertEqual((await client.get("/health")).status_code, 503)
                result = await client.post(
                    "/v1/evaluate-window", json=raw_request(), headers=self.headers
                )
                self.assertEqual(result.status_code, 503)
                self.assertIsNone(result.json()["ai"]["risk_class"])


if __name__ == "__main__":
    unittest.main()
