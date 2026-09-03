// Standalone hand-written tests for the portable capacitive-feature kernel.
// SYNTHETIC_PIPELINE_TEST_ONLY; proves host-side feature-extraction mechanics
// only (matching ai/src/ostosense_ai/features.py numerics on one window).
//
// Build (single line): g++ -std=c++17 -Wall -Wextra -pedantic -I firmware/include
//   firmware/tests/capacitive_features_test.cpp -o /tmp/capacitive_features_test
// Returns 0 on success, non-zero when any check fails.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "ostosense/capacitive_features.hpp"

using ostosense::CapacitiveFeatureVector;
using ostosense::CapacitiveWindowSample;
using ostosense::CapQuality;
using ostosense::ComputeCapacitiveFeatures;
using ostosense::FeatureStatus;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++g_failures;
  }
}

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// Build a nominal 120-sample window at 1 Hz starting at 1000 ms, quality kOk,
// with capacitance_raw[i] = base + step * i.
std::array<CapacitiveWindowSample, 120> LinearWindow(float base, float step) {
  std::array<CapacitiveWindowSample, 120> w{};
  for (std::size_t i = 0; i < 120; ++i) {
    w[i].timestamp_ms = 1000 + 1000 * static_cast<std::uint64_t>(i);
    w[i].capacitance_raw = base + step * static_cast<float>(i);
    w[i].cap_quality = CapQuality::kOk;
  }
  return w;
}

}  // namespace

int main() {
  CapacitiveFeatureVector out;

  // 1. Golden linear window: delta_i = i + 1 (baseline 0, capacitance i+1, 1 Hz).
  //    deltas 1..120 -> mean 60.5, last 120, slope 1/s, population variance
  //    (120^2 - 1)/12 = 1199.916667, range 119.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) == FeatureStatus::kOk,
          "golden: status kOk");
    Check(out.values[0] == 60.5f, "golden: mean 60.5");
    Check(out.values[1] == 120.0f, "golden: last 120");
    Check(out.values[2] == 1.0f, "golden: slope 1/s");
    Check(out.values[3] == static_cast<float>(1199.916667), "golden: variance");
    Check(out.values[4] == 119.0f, "golden: range 119");
  }

  // 2. Constant window: every delta equal -> mean = last = delta, slope 0,
  //    variance 0, range 0. Baseline offset chosen so delta = 2.5.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(14.5f, 0.0f);
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 12.0f, &out) == FeatureStatus::kOk,
          "constant: status kOk");
    Check(out.values[0] == 2.5f, "constant: mean 2.5");
    Check(out.values[1] == 2.5f, "constant: last 2.5");
    Check(out.values[2] == 0.0f, "constant: slope 0");
    Check(out.values[3] == 0.0f, "constant: variance 0");
    Check(out.values[4] == 0.0f, "constant: range 0");
  }

  // 3. Jittered but in-tolerance timing (intervals alternate 900/1100 ms, both
  //    inside 800..1200) is accepted.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(12.0f, 0.01f);
    std::uint64_t t = 1000;
    for (std::size_t i = 0; i < 120; ++i) {
      w[i].timestamp_ms = t;
      t += (i % 2 == 0) ? 900 : 1100;
    }
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 12.0f, &out) == FeatureStatus::kOk,
          "jitter: in-tolerance timing accepted");
    for (float v : out.values) {
      Check(std::isfinite(v), "jitter: finite features");
    }
  }

  // 4. The kernel keeps direct finite float model inputs. Decimal six-place
  //    rounding remains an offline CSV-serialization concern.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(0.0078125f, 0.0f);
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) == FeatureStatus::kOk,
          "fractional: status kOk");
    Check(out.values[0] == 0.0078125f, "fractional: mean preserved");
    Check(out.values[1] == 0.0078125f, "fractional: last preserved");
    Check(out.values[2] == 0.0f, "fractional: slope zero");
    Check(out.values[3] == 0.0f, "fractional: variance zero");
    Check(out.values[4] == 0.0f, "fractional: range zero");
  }

  // 5. Null output -> kInvalidOutput.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, nullptr) ==
              FeatureStatus::kInvalidOutput,
          "null output -> kInvalidOutput");
  }

  // 6. Null samples -> kInvalidInput (and output reset to zeros).
  {
    out.values = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    Check(ComputeCapacitiveFeatures(nullptr, 120, 0.0f, &out) == FeatureStatus::kInvalidInput,
          "null samples -> kInvalidInput");
    for (float v : out.values) Check(v == 0.0f, "null samples: output cleared");
  }

  // 7. Non-finite baseline -> kInvalidInput.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), kNaN, &out) == FeatureStatus::kInvalidInput,
          "nan baseline -> kInvalidInput");
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), kInf, &out) == FeatureStatus::kInvalidInput,
          "inf baseline -> kInvalidInput");
  }

  // 8. Non-finite capacitance reading -> kInvalidInput.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    w[42].capacitance_raw = kInf;
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) == FeatureStatus::kInvalidInput,
          "inf reading -> kInvalidInput");
    w[42].capacitance_raw = kNaN;
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) == FeatureStatus::kInvalidInput,
          "nan reading -> kInvalidInput");
  }

  // 9. Duplicate timestamp -> kDuplicateTimestamp.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    w[10].timestamp_ms = w[11].timestamp_ms;  // collide two timestamps
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) ==
              FeatureStatus::kDuplicateTimestamp,
          "duplicate timestamp -> kDuplicateTimestamp");
  }

  // 10. Partial window (count != 120) -> kPartialWindow.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    Check(ComputeCapacitiveFeatures(w.data(), 119, 0.0f, &out) == FeatureStatus::kPartialWindow,
          "count 119 -> kPartialWindow");
    Check(ComputeCapacitiveFeatures(w.data(), 0, 0.0f, &out) == FeatureStatus::kPartialWindow,
          "count 0 -> kPartialWindow");
  }

  // 11. Precedence: duplicate is detected before partial (count 119 + duplicate).
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    w[5].timestamp_ms = w[6].timestamp_ms;
    Check(ComputeCapacitiveFeatures(w.data(), 119, 0.0f, &out) ==
              FeatureStatus::kDuplicateTimestamp,
          "precedence: duplicate before partial");
  }

  // 12. Descending timestamps (unique, count 120) -> kTimestampOutOfOrder.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    const std::uint64_t a = w[50].timestamp_ms;
    w[50].timestamp_ms = w[51].timestamp_ms;
    w[51].timestamp_ms = a;  // swap -> strictly descending pair, still unique
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) ==
              FeatureStatus::kTimestampOutOfOrder,
          "descending -> kTimestampOutOfOrder");
  }

  // 13. Timing outside 800..1200 ms -> kTimingOutOfTolerance (both edges).
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    for (std::size_t i = 1; i < 120; ++i) w[i].timestamp_ms = w[i - 1].timestamp_ms + 1300;
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) ==
              FeatureStatus::kTimingOutOfTolerance,
          "interval 1300 -> kTimingOutOfTolerance");
    for (std::size_t i = 1; i < 120; ++i) w[i].timestamp_ms = w[i - 1].timestamp_ms + 700;
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) ==
              FeatureStatus::kTimingOutOfTolerance,
          "interval 700 -> kTimingOutOfTolerance");
  }

  // 14. Inclusive tolerance edges 800 and 1200 ms are accepted.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(12.0f, 0.0f);
    for (std::size_t i = 1; i < 120; ++i) w[i].timestamp_ms = w[i - 1].timestamp_ms + 800;
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 12.0f, &out) == FeatureStatus::kOk,
          "interval 800 accepted");
    for (std::size_t i = 1; i < 120; ++i) w[i].timestamp_ms = w[i - 1].timestamp_ms + 1200;
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 12.0f, &out) == FeatureStatus::kOk,
          "interval 1200 accepted");
  }

  // 15. Invalid capacitive quality -> kInvalidQuality.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    w[7].cap_quality = CapQuality::kWarmingUp;
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) ==
              FeatureStatus::kInvalidQuality,
          "bad quality -> kInvalidQuality");
  }

  // 16. Precedence: timing is detected before quality.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    w[3].timestamp_ms = w[2].timestamp_ms + 1300;  // timing violation
    for (std::size_t i = 4; i < 120; ++i) w[i].timestamp_ms = w[i - 1].timestamp_ms + 1000;
    w[9].cap_quality = CapQuality::kDataGap;  // also a quality violation
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) ==
              FeatureStatus::kTimingOutOfTolerance,
          "precedence: timing before quality");
  }

  // 17. Numerical overflow: finite float readings whose feature magnitudes exceed
  //     float range (variance ~ (3e38)^2) -> kNumericalError.
  {
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(0.0f, 0.0f);
    for (std::size_t i = 0; i < 120; ++i) {
      w[i].capacitance_raw = (i % 2 == 0) ? 0.0f : 3.0e38f;
    }
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) == FeatureStatus::kNumericalError,
          "overflow -> kNumericalError");
  }

  // 18. A failed call resets the output to zeros (no stale results).
  {
    out.values = {7.0f, 7.0f, 7.0f, 7.0f, 7.0f};
    std::array<CapacitiveWindowSample, 120> w = LinearWindow(1.0f, 1.0f);
    w[0].cap_quality = CapQuality::kDisconnected;
    Check(ComputeCapacitiveFeatures(w.data(), w.size(), 0.0f, &out) ==
              FeatureStatus::kInvalidQuality,
          "stale-reset: status");
    for (float v : out.values) Check(v == 0.0f, "stale-reset: output cleared");
  }

  if (g_failures == 0) {
    std::printf("capacitive_features_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "capacitive_features_test: %d failure(s)\n", g_failures);
  return 1;
}
