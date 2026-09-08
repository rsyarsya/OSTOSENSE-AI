"""Load integrity-checked engineering assets from a clone or an installed wheel."""

import hashlib
import json
from importlib.resources import files
from typing import Any

from ostosense_ai import inference, runtime_output

MODEL_ID = "engineering-v0.3"
MODEL_SHA256 = "ceb2922841431a38c274e0ebd5c350873a28a95f40a6f6aab2194a1603b93cef"


def load_bundled_model() -> dict[str, Any]:
    root = files("ostosense_ai").joinpath("model_assets", "engineering_v03")
    bundle = json.loads(root.joinpath("bundle.json").read_bytes())
    if (
        bundle["bundle_version"] != "0.1.0"
        or bundle["dataset_origin"] != inference.ALLOWED_ORIGIN
        or bundle["model_id"] != MODEL_ID
        or bundle["sha256"]["ordinal_model.json"] != MODEL_SHA256
    ):
        raise ValueError("Bundled model identity mismatch")
    documents = {}
    for name in ("ordinal_model.json", "training_manifest.json", "examples.json"):
        raw = root.joinpath(name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != bundle["sha256"][name]:
            raise ValueError(f"Bundled asset SHA-256 mismatch: {name}")
        documents[name] = json.loads(raw)
    model = documents["ordinal_model.json"]
    inference.validate_model_artifact(model)
    manifest = documents["training_manifest.json"]
    if (
        manifest["output_sha256"]["ordinal_model_json"] != MODEL_SHA256
        or manifest["dataset_origin"] != inference.ALLOWED_ORIGIN
        or manifest["fit_partition_policy"]
        != {"fit": "development", "ignored": "validation", "forbidden": "final_test"}
    ):
        raise ValueError("Bundled training provenance mismatch")
    if [case["expected"]["risk_class"] for case in documents["examples.json"]] != list(
        inference.CLASS_NAMES
    ):
        raise ValueError("Bundled examples must cover the four classes in canonical order")
    for case in documents["examples.json"]:
        prediction = runtime_output.predict_test_output_v2(
            model,
            case["input"]["features"],
            source_window_end_ms=case["input"]["source_window_end_ms"],
            model_artifact_sha256=MODEL_SHA256,
        )
        if prediction != case["expected"]:
            raise ValueError("Bundled example differs from actual inference")
    return {
        "model": model,
        "manifest": manifest,
        "examples": documents["examples.json"],
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
    }


def main() -> None:
    bundle = load_bundled_model()
    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_sha256": MODEL_SHA256,
                "examples": [case["expected"] for case in bundle["examples"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
