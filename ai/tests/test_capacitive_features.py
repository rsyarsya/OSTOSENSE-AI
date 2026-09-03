"""Synthetic cross-language parity for the portable capacitive-feature kernel.

SYNTHETIC_PIPELINE_TEST_ONLY. This reconstructs each eligible validation window
from the synthetic ``samples.csv`` / ``sessions.csv`` (the exact ``(t-W, t]``
convention used by ``features.py``), feeds the raw capacitance / timestamps /
quality / session baseline into the C++ ``ComputeCapacitiveFeatures`` kernel,
and checks the completed portable path:

    120 raw capacitive samples -> five capacitive features -> ordinal inference

against the Python reference. Expected features come from ``features.csv`` (not
a re-implementation) and expected classes/probabilities come from the existing
Python inference exported as golden vectors. All generated headers are written
into a temporary directory and removed with it; none are committed into the
repository. A passing run proves host-side synthetic feature + inference parity
mechanics only -- not ESP32 deployment, firmware parity on hardware, sensor
validation, or OSTOSENSE performance.
"""

import csv
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ostosense_ai import edge_export, matrix
from ostosense_ai.edge_export import export_edge_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
FIRMWARE_INCLUDE = PROJECT_ROOT / "firmware" / "include"
EVAL_FIX = Path(__file__).resolve().parent / "fixtures" / "ostosense-evaluation-v0.1"
LABEL_FIX = Path(__file__).resolve().parent / "fixtures" / "ostosense-labeling-v0.1"
BOUNDARY = LABEL_FIX / "boundary-engineering-test-only-v0.1.json"
FEATURE_COLUMNS = matrix.FEATURE_COLUMNS

PIPELINE_AVAILABLE = all(
    importlib.util.find_spec(m) is not None for m in ("numpy", "sklearn", "mord", "scipy")
)
GPP = shutil.which("g++")

# Python data-contract cap_quality strings -> C++ ostosense::CapQuality members.
CAP_QUALITY_CPP = {
    "OK": "kOk",
    "DISCONNECTED": "kDisconnected",
    "ADC_SATURATED": "kAdcSaturated",
    "BASELINE_INVALID": "kBaselineInvalid",
    "DATA_GAP": "kDataGap",
    "WARMING_UP": "kWarmingUp",
}

_GENERATED_WARNING = (
    "AUTO-GENERATED test fixture (SYNTHETIC_PIPELINE_TEST_ONLY, "
    "ENGINEERING_TEST_ONLY). Reconstructed raw validation windows; do NOT commit "
    "into the production firmware include path."
)

# The harness exercises the full portable path per validation window:
#   (a) raw window -> C++ ComputeCapacitiveFeatures, compared to features.csv,
#   (b) Python features -> C++ inference (inference-only parity vs Python), and
#   (c) C++ features -> C++ inference (full-path class parity vs Python).
_HARNESS_MAIN = r"""
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "ostosense/capacitive_features.hpp"
#include "ostosense/ordinal_inference.hpp"
#include "ordinal_model_params.hpp"
#include "golden_vectors.hpp"
#include "reconstructed_windows.hpp"

using namespace ostosense;

int main() {
  const std::size_t n = kReconstructedWindows.size();
  if (n != generated::kGoldenVectorCount) {
    std::fprintf(stderr, "reconstructed/golden count mismatch\n");
    return 5;
  }

  std::size_t feat_status_fail = 0, feat_mismatch = 0;
  std::size_t infer_status_fail = 0;
  std::size_t inference_only_class_mismatch = 0, fullpath_class_mismatch = 0;
  double max_feat_diff = 0.0, io_prob_diff = 0.0, fp_prob_diff = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const ReconstructedWindow& w = kReconstructedWindows[i];
    const generated::GoldenVector& g = generated::kGoldenVectors[i];

    // (a) Raw window -> C++ features, compared to the Python features.csv values.
    CapacitiveFeatureVector fv;
    if (ComputeCapacitiveFeatures(w.samples.data(), w.samples.size(), w.baseline, &fv) !=
        FeatureStatus::kOk) {
      ++feat_status_fail;
      continue;
    }
    for (int k = 0; k < 5; ++k) {
      const double d = std::fabs(static_cast<double>(fv.values[k]) - w.expected_features[k]);
      if (d > max_feat_diff) max_feat_diff = d;
      if (d > 1e-5) ++feat_mismatch;
    }

    // (b) Inference-only parity: Python features -> C++ inference == Python inference.
    OrdinalPrediction io;
    if (PredictOrdinal(generated::kOrdinalModel, g.features, &io) != InferenceStatus::kOk) {
      ++infer_status_fail;
      continue;
    }
    if (static_cast<std::uint8_t>(io.predicted_class) != g.expected_class_index) {
      ++inference_only_class_mismatch;
    }
    for (int k = 0; k < 4; ++k) {
      const double d = std::fabs(static_cast<double>(io.probabilities[k]) -
                                 static_cast<double>(g.reference_probabilities[k]));
      if (d > io_prob_diff) io_prob_diff = d;
    }

    // (c) Full path: C++ features -> C++ inference; class must match Python inference.
    OrdinalPrediction fp;
    if (PredictOrdinal(generated::kOrdinalModel, fv.values, &fp) != InferenceStatus::kOk) {
      ++infer_status_fail;
      continue;
    }
    if (static_cast<std::uint8_t>(fp.predicted_class) != g.expected_class_index) {
      ++fullpath_class_mismatch;
    }
    for (int k = 0; k < 4; ++k) {
      const double d = std::fabs(static_cast<double>(fp.probabilities[k]) -
                                 static_cast<double>(g.reference_probabilities[k]));
      if (d > fp_prob_diff) fp_prob_diff = d;
    }
  }

  std::printf(
      "count=%zu feat_status_fail=%zu feat_mismatch=%zu infer_status_fail=%zu "
      "inference_only_class_mismatch=%zu fullpath_class_mismatch=%zu "
      "max_feat_diff=%.9g io_prob_diff=%.9g fp_prob_diff=%.9g\n",
      n, feat_status_fail, feat_mismatch, infer_status_fail,
      inference_only_class_mismatch, fullpath_class_mismatch, max_feat_diff,
      io_prob_diff, fp_prob_diff);

  if (feat_status_fail || infer_status_fail) return 2;
  if (feat_mismatch) return 3;
  if (inference_only_class_mismatch || fullpath_class_mismatch) return 4;
  if (max_feat_diff > 1e-5 || io_prob_diff > 1e-5) return 6;
  return 0;
}
"""


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reconstruct_windows(raw_dir, feat_dir, matrix_dir):
    """Reconstruct the eligible validation windows in matrix (== golden) order."""
    samples_by_session: dict[str, list[dict]] = {}
    for row in _read_csv(raw_dir / "samples.csv"):
        samples_by_session.setdefault(row["session_id"], []).append(row)
    sessions = {row["session_id"]: row for row in _read_csv(raw_dir / "sessions.csv")}
    features_by_id = {row["window_id"]: row for row in _read_csv(feat_dir / "features.csv")}

    windows = []
    for mrow in _read_csv(matrix_dir / "model_matrix.csv"):
        if mrow["dataset_partition"] != "validation":
            continue
        window_id = mrow["window_id"]
        feat = features_by_id[window_id]
        # The matrix must carry the very features.csv values it was built from.
        for column in FEATURE_COLUMNS:
            assert float(mrow[column]) == float(feat[column]), (window_id, column)
        session_id = feat["session_id"]
        window_start, window_end = int(feat["window_start"]), int(feat["window_end"])
        members = [
            s
            for s in samples_by_session[session_id]
            if window_start < int(s["timestamp"]) <= window_end
        ]
        members.sort(key=lambda s: int(s["timestamp"]))
        assert len(members) == 120, (window_id, len(members))
        windows.append(
            {
                "samples": [
                    (int(s["timestamp"]), float(s["capacitance_raw"]), s["cap_quality"])
                    for s in members
                ],
                "baseline": float(sessions[session_id]["baseline_value"]),
                "expected": [float(feat[column]) for column in FEATURE_COLUMNS],
            }
        )
    return windows


def _reconstructed_header(windows) -> str:
    lines = [
        f"// {_GENERATED_WARNING}\n",
        "#pragma once\n\n",
        "#include <array>\n#include <cstddef>\n#include <cstdint>\n\n",
        '#include "ostosense/capacitive_features.hpp"\n\n',
        "namespace ostosense {\n\n",
        "struct ReconstructedWindow {\n",
        "  std::array<CapacitiveWindowSample, 120> samples;\n",
        "  float baseline;\n",
        "  std::array<double, 5> expected_features;  // from features.csv (Python reference)\n",
        "};\n\n",
        f"inline const std::array<ReconstructedWindow, {len(windows)}> "
        "kReconstructedWindows = {{\n",
    ]
    for window in windows:
        lines.append("  ReconstructedWindow{\n")
        lines.append("    std::array<CapacitiveWindowSample, 120>{{\n")
        for timestamp, capacitance, quality in window["samples"]:
            quality_cpp = CAP_QUALITY_CPP[quality]
            lines.append(
                f"      CapacitiveWindowSample{{ {timestamp}ULL, "
                f"{edge_export._f32_literal(capacitance)}, CapQuality::{quality_cpp} }},\n"
            )
        lines.append("    }},\n")
        lines.append(f"    {edge_export._f32_literal(window['baseline'])},\n")
        expected = ", ".join(repr(value) for value in window["expected"])
        lines.append(f"    std::array<double, 5>{{ {expected} }},\n")
        lines.append("  },\n")
    lines.append("}};\n\n")
    lines.append("}  // namespace ostosense\n")
    return "".join(lines)


class CapacitiveFeatureSourceGuardTests(unittest.TestCase):
    def test_kernel_avoids_string_locale_and_heap_dependencies(self):
        header = (FIRMWARE_INCLUDE / "ostosense" / "capacitive_features.hpp").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "snprintf",
            "strtod",
            "setlocale",
            "<cstdio>",
            "<cstdlib>",
            "<locale>",
            "std::vector",
            "operator new",
            "malloc(",
            "calloc(",
            "realloc(",
        ):
            self.assertNotIn(forbidden, header)


@unittest.skipUnless(
    PIPELINE_AVAILABLE and GPP, "requires the optional [pipeline] dependencies and g++"
)
class CapacitiveFeatureCrossLanguageParityTests(unittest.TestCase):
    def test_v03_raw_to_features_to_inference_parity(self):
        from ostosense_ai import features, labeling, synthetic, training

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw, feat, lbl, mtx, model, edge = (
                root / d for d in ("raw", "feat", "lbl", "mtx", "model", "edge")
            )
            synthetic.generate(
                synthetic.load_config(REPO_ROOT / "configs" / "synthetic-v0.3.json"),
                20260722,
                raw,
            )
            features.extract(
                raw, features.load_config(REPO_ROOT / "configs" / "features-v0.1.json"), feat
            )
            labeling.label_dataset(
                raw,
                EVAL_FIX / "protocol_manifest.csv",
                EVAL_FIX / "partition_manifest.csv",
                BOUNDARY,
                lbl,
                features_dir=feat,
            )
            matrix.build_model_matrix(feat, lbl, mtx)
            training.train_ordinal_model(
                mtx, REPO_ROOT / "configs" / "training-v0.1.json", model
            )
            manifest = export_edge_bundle(model, mtx, edge)
            self.assertEqual(manifest["vector_count"], 38)

            windows = _reconstruct_windows(raw, feat, mtx)
            self.assertEqual(len(windows), 38)

            # Generated headers live only in the temporary edge directory.
            (edge / "reconstructed_windows.hpp").write_text(
                _reconstructed_header(windows), encoding="utf-8"
            )
            main = root / "capacitive_parity_main.cpp"
            main.write_text(_HARNESS_MAIN, encoding="utf-8")

            binary = root / "capacitive_parity"
            compile_cmd = [
                GPP, "-std=c++17", "-Wall", "-Wextra", "-pedantic",
                "-I", str(FIRMWARE_INCLUDE), "-I", str(edge), str(main), "-o", str(binary),
            ]
            compiled = subprocess.run(compile_cmd, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)

            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)

            output = run.stdout.strip()
            metrics = dict(token.split("=") for token in output.split())
            self.assertEqual(int(metrics["count"]), 38)
            self.assertEqual(int(metrics["feat_status_fail"]), 0)
            self.assertEqual(int(metrics["feat_mismatch"]), 0)
            self.assertEqual(int(metrics["infer_status_fail"]), 0)
            self.assertEqual(int(metrics["inference_only_class_mismatch"]), 0)
            self.assertEqual(int(metrics["fullpath_class_mismatch"]), 0)
            self.assertLessEqual(float(metrics["max_feat_diff"]), 1e-5)
            self.assertLessEqual(float(metrics["io_prob_diff"]), 1e-5)
            self.assertLessEqual(float(metrics["fp_prob_diff"]), 1e-5)

    def test_no_generated_headers_remain_in_repository(self):
        # The firmware include path may carry hand-written kernels, never generated fixtures.
        generated_names = {
            "reconstructed_windows.hpp",
            "golden_vectors.hpp",
            "ordinal_model_params.hpp",
        }
        present = {p.name for p in (FIRMWARE_INCLUDE / "ostosense").glob("*.hpp")}
        self.assertEqual(present & generated_names, set())


if __name__ == "__main__":
    unittest.main()
