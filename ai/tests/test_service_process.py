"""Loopback-only process test: a real Uvicorn listener, no external upload."""

import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from test_live_service import raw_request


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("fastapi", "uvicorn")),
    "optional [serve] dependencies required",
)
class ServiceProcessTests(unittest.TestCase):
    def test_real_listener_model_examples_and_raw_request(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        token = secrets.token_hex(32)
        env = {
            **os.environ,
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "OSTOSENSE_SERVICE_TOKEN": token,
            "OSTOSENSE_ALLOW_TEST": "true",
            "OSTOSENSE_ENABLE_EXPERIMENTAL": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "ostosense_ai.service"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{port}"
        opener = build_opener(ProxyHandler({}))

        def request(path, data=None, auth=True):
            headers = {"Content-Type": "application/json"}
            if auth:
                headers["Authorization"] = "Bearer " + token
            encoded = json.dumps(data).encode() if data is not None else None
            with opener.open(
                Request(url + path, data=encoded, headers=headers), timeout=5
            ) as response:
                return json.load(response)

        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    health = request("/health", auth=False)
                    break
                except (URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail("AI process did not become ready on loopback")
                    time.sleep(0.1)
            self.assertTrue(health["ready"])
            examples = request("/v1/examples")
            classes = []
            for case in examples:
                actual = request("/v1/predict-features", case["input"])["ai"]
                self.assertEqual(actual, case["expected"])
                classes.append(actual["risk_class"])
            self.assertEqual(classes, ["Safe", "Monitor", "Caution", "Urgent"])
            probe = Path(__file__).resolve().parents[2] / "scripts/probe_ai_service.py"
            checked = subprocess.run(
                [sys.executable, str(probe), "--url", url],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(checked.stdout.count("matches bundled model"), 4)
            result = request("/v1/evaluate-window", raw_request())
            self.assertEqual(result["ai"]["model_status"], "UNVALIDATED")
            self.assertEqual(result["diagnostics"]["quality_verification"], "OPERATOR_ONLY")
            with self.assertRaises(HTTPError) as error:
                request("/v1/examples", auth=False)
            self.assertEqual(error.exception.code, 401)
            error.exception.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
