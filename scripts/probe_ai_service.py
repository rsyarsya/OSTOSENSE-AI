"""Check the running engineering service against the model assets in this clone."""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if (
        (
            parsed.scheme != "https"
            and not (
                parsed.scheme == "http"
                and parsed.hostname in ("127.0.0.1", "localhost", "::1")
            )
        )
        or parsed.username
        or parsed.password
    ):
        parser.error("Use HTTPS or loopback HTTP, without credentials in the URL")
    token = os.environ.get("OSTOSENSE_SERVICE_TOKEN", "")
    if len(token) < 32:
        parser.error("Set OSTOSENSE_SERVICE_TOKEN in the environment")
    root = (
        Path(__file__).resolve().parents[1]
        / "ai/src/ostosense_ai/model_assets/engineering_v03"
    )
    cases = json.loads((root / "examples.json").read_text(encoding="utf-8"))
    opener = build_opener(NoRedirect)
    try:
        for case in cases:
            request = Request(
                args.url.rstrip("/") + "/v1/predict-features",
                data=json.dumps(case["input"]).encode(),
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                },
            )
            with opener.open(request, timeout=5) as response:
                result = json.load(response)
            if result["ai"] != case["expected"]:
                raise ValueError("Model response differs from the bundled reference")
            print(f"Simulasi AI: {result['ai']['risk_class']} - sesuai model bawaan")
    except (URLError, ValueError, KeyError) as error:
        if isinstance(error, HTTPError):
            error.close()
        print(
            f"Service probe failed ({type(error).__name__}); inspect service readiness/mode",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
