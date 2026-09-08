"""Internal HTTP adapter; run with python -m ostosense_ai.service.

Dependencies are optional: pip install './ai[serve]'. The model and numerical
implementation remain standard-library-only. Never expose the backend token in a UI.
"""

import hmac
import json
import os
from typing import Any

from ostosense_ai import bundled_model, live_window, runtime_output

MAX_REQUEST_BYTES = 524288


def create_app():
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ModuleNotFoundError as error:
        if error.name not in ("fastapi", "starlette"):
            raise
        raise RuntimeError(
            "Install the optional server dependencies: pip install './ai[serve]'"
        ) from error

    token = os.environ.get("OSTOSENSE_SERVICE_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError(
            "Set OSTOSENSE_SERVICE_TOKEN to a private random value of at least 32 characters"
        )
    testing = os.environ.get("OSTOSENSE_ALLOW_TEST", "false") == "true"
    experimental = os.environ.get("OSTOSENSE_ENABLE_EXPERIMENTAL", "false") == "true"
    try:
        bundle = bundled_model.load_bundled_model()
    except (OSError, ValueError, KeyError, TypeError):
        bundle = None
    app = FastAPI(
        title="OSTOSENSE AI - experimental integration",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def authorize(request: Request) -> None:
        if not hmac.compare_digest(
            request.headers.get("authorization", "").encode(), ("Bearer " + token).encode()
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")

    async def body(request: Request) -> Any:
        authorize(request)
        chunks = bytearray()
        async for chunk in request.stream():
            if len(chunks) + len(chunk) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="Request too large")
            chunks.extend(chunk)
        try:
            return json.loads(chunks)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise HTTPException(status_code=422, detail="Invalid JSON") from error

    @app.get("/health")
    def health():
        return JSONResponse(
            status_code=200 if bundle else 503,
            content={
                "ready": bundle is not None,
                "model_id": bundled_model.MODEL_ID,
                "model_sha256": bundled_model.MODEL_SHA256 if bundle else None,
                "test_enabled": testing,
                "experimental_enabled": experimental,
                "deployment_scope": "EXPERIMENTAL_INTEGRATION_ONLY",
            },
        )

    @app.get("/v1/examples")
    def examples(request: Request):
        authorize(request)
        if not testing:
            raise HTTPException(status_code=403, detail="Internal test mode is disabled")
        if not bundle:
            raise HTTPException(status_code=503, detail="MODEL_UNAVAILABLE")
        return bundle["examples"]

    @app.post("/v1/predict-features")
    async def predict_features(request: Request):
        data = await body(request)
        if not testing:
            raise HTTPException(status_code=403, detail="Internal test mode is disabled")
        if not bundle:
            return JSONResponse(
                status_code=503, content=live_window.unavailable("MODEL_UNAVAILABLE")
            )
        try:
            if not isinstance(data, dict):
                raise ValueError("Feature input must be an object")
            runtime_output._require_exact_fields(data, runtime_output.FEATURE_INPUT_FIELDS)
            expected = {
                "feature_input_version": "0.1.0",
                "data_source": "SYNTHETIC_FIXTURE",
                "model_input_channel": "SYNTHETIC_CAPACITIVE",
                "feature_basis": "RAW_MINUS_SESSION_BASELINE",
                "feature_order": bundle["model"]["feature_order"],
            }
            if any(data[key] != value for key, value in expected.items()):
                raise ValueError("Only canonical synthetic test features are accepted here")
            result = runtime_output.predict_test_output_v2(
                bundle["model"],
                data["features"],
                source_window_end_ms=data["source_window_end_ms"],
                model_artifact_sha256=bundle["model_sha256"],
            )
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise HTTPException(status_code=422, detail="Invalid feature input") from error
        return {
            "service_response_version": "0.1.0",
            "ai": result,
            "diagnostics": {
                "availability_reason": "READY",
                "quality_verification": "TEST_ONLY",
                "baseline_value": None,
            },
            "feature_input": data,
        }

    @app.post("/v1/evaluate-window")
    async def evaluate(request: Request):
        data = await body(request)
        if not experimental:
            raise HTTPException(status_code=403, detail="Experimental mode is disabled")
        if not bundle:
            return JSONResponse(
                status_code=503, content=live_window.unavailable("MODEL_UNAVAILABLE")
            )
        try:
            return live_window.evaluate_window(data, bundle["model"], bundle["model_sha256"])
        except live_window.LiveWindowError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (runtime_output.RuntimeOutputError, OverflowError, ArithmeticError):
            return live_window.unavailable("NUMERICAL_ERROR")

    return app


def main():
    try:
        import uvicorn
    except ModuleNotFoundError as error:
        if error.name != "uvicorn":
            raise
        raise RuntimeError("Install server dependencies: pip install './ai[serve]'") from error
    uvicorn.run(
        create_app(),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
