// Portable host-side capacitive-feature kernel for OSTOSENSE (pipeline test only).
//
// SYNTHETIC_PIPELINE_TEST_ONLY. This header-only, dependency-free C++17 kernel
// reproduces the five canonical baseline-normalized capacitive features computed
// by ai/src/ostosense_ai/features.py over a single (t-W, t] window of 120 raw
// samples, so a portable "120 raw samples -> five features -> ordinal inference"
// path can be checked for parity with the Python reference. It is written to be
// reusable later on an ESP32-S3, but is NOT ESP32 deployment, NOT firmware parity
// on hardware, and NOT evidence of OSTOSENSE performance.
//
// No dynamic allocation, no exceptions, no JSON, no Arduino/PlatformIO/ESP-IDF,
// no embedded model. Only the capacitive channel is consumed: no LIG, labels,
// events, partitions, boundaries, or scenario metadata enter the feature vector.
// This is a pure feature kernel: no ring buffer, scheduler, session state, or
// sampling loop.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ostosense/data_contract.hpp"  // ostosense::CapQuality

namespace ostosense {

struct CapacitiveWindowSample {
  std::uint64_t timestamp_ms;
  float capacitance_raw;
  CapQuality cap_quality;
};

struct CapacitiveFeatureVector {
  // Canonical order: cap_delta_mean, cap_delta_last, cap_delta_slope_per_s,
  //                  cap_delta_variance, cap_delta_range.
  std::array<float, 5> values;
};

enum class FeatureStatus : std::uint8_t {
  kOk = 0,
  kInvalidOutput,
  kInvalidInput,
  kDuplicateTimestamp,
  kPartialWindow,
  kTimestampOutOfOrder,
  kTimingOutOfTolerance,
  kInvalidQuality,
  kNumericalError,
};

namespace detail {

constexpr std::size_t kExpectedWindow = 120;
constexpr std::uint64_t kNominalIntervalMs = 1000;
constexpr std::uint64_t kJitterMs = 200;  // inclusive tolerance 800..1200 ms

// Decimal six-place rounding belongs to the deterministic offline CSV
// serialization. The portable kernel emits direct finite float model inputs and
// is checked against those serialized Python features within the declared parity
// tolerance, avoiding locale-sensitive string conversion in embedded-facing code.

}  // namespace detail

// Compute the five canonical capacitive features for one window. Returns kOk and
// fills *output only on success; on any failure the output is reset to zeros so a
// failed call never exposes stale results. Delta = capacitance_raw - baseline;
// slope is OLS against elapsed seconds from the first timestamp; variance is the
// population variance (ddof=0); range is max(delta) - min(delta).
inline FeatureStatus ComputeCapacitiveFeatures(const CapacitiveWindowSample* samples,
                                               std::size_t count, float baseline,
                                               CapacitiveFeatureVector* output) {
  if (output == nullptr) {
    return FeatureStatus::kInvalidOutput;
  }
  output->values = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  if (samples == nullptr) {
    return FeatureStatus::kInvalidInput;
  }
  if (!std::isfinite(baseline)) {
    return FeatureStatus::kInvalidInput;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(samples[i].capacitance_raw)) {
      return FeatureStatus::kInvalidInput;
    }
  }

  // Structural checks, in the exact Python (_classify) precedence.
  // (1) duplicate timestamp (any equal pair; matches len(set) != len).
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = i + 1; j < count; ++j) {
      if (samples[i].timestamp_ms == samples[j].timestamp_ms) {
        return FeatureStatus::kDuplicateTimestamp;
      }
    }
  }
  // (2) partial window (count must be exactly 120).
  if (count != detail::kExpectedWindow) {
    return FeatureStatus::kPartialWindow;
  }
  // (3) descending timestamps (equal handled above -> strictly ascending here).
  for (std::size_t i = 1; i < count; ++i) {
    if (samples[i].timestamp_ms < samples[i - 1].timestamp_ms) {
      return FeatureStatus::kTimestampOutOfOrder;
    }
  }
  // (4) timing tolerance 800..1200 ms inclusive.
  constexpr std::uint64_t kLow = detail::kNominalIntervalMs - detail::kJitterMs;   // 800
  constexpr std::uint64_t kHigh = detail::kNominalIntervalMs + detail::kJitterMs;  // 1200
  for (std::size_t i = 1; i < count; ++i) {
    const std::uint64_t interval = samples[i].timestamp_ms - samples[i - 1].timestamp_ms;
    if (interval < kLow || interval > kHigh) {
      return FeatureStatus::kTimingOutOfTolerance;
    }
  }
  // (5) capacitive quality.
  for (std::size_t i = 0; i < count; ++i) {
    if (samples[i].cap_quality != CapQuality::kOk) {
      return FeatureStatus::kInvalidQuality;
    }
  }

  // Feature computation with double accumulators.
  const double baseline_d = static_cast<double>(baseline);
  const std::uint64_t first_ts = samples[0].timestamp_ms;
  const double n = static_cast<double>(count);

  double sum_delta = 0.0;
  double sum_elapsed = 0.0;
  double delta_min = 0.0;
  double delta_max = 0.0;
  double last_delta = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double delta = static_cast<double>(samples[i].capacitance_raw) - baseline_d;
    const double elapsed = static_cast<double>(samples[i].timestamp_ms - first_ts) / 1000.0;
    sum_delta += delta;
    sum_elapsed += elapsed;
    if (i == 0) {
      delta_min = delta;
      delta_max = delta;
    } else {
      if (delta < delta_min) delta_min = delta;
      if (delta > delta_max) delta_max = delta;
    }
    last_delta = delta;
  }
  const double delta_mean = sum_delta / n;
  const double x_mean = sum_elapsed / n;

  double sxx = 0.0;
  double sxy = 0.0;
  double var_acc = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double delta = static_cast<double>(samples[i].capacitance_raw) - baseline_d;
    const double elapsed = static_cast<double>(samples[i].timestamp_ms - first_ts) / 1000.0;
    const double dx = elapsed - x_mean;
    const double dd = delta - delta_mean;
    sxx += dx * dx;
    sxy += dx * dd;
    var_acc += dd * dd;
  }
  if (!(sxx > 0.0)) {  // a valid 120-sample ascending window always has sxx > 0
    return FeatureStatus::kNumericalError;
  }

  const std::array<double, 5> raw = {
      delta_mean,
      last_delta,
      sxy / sxx,       // slope per second
      var_acc / n,     // population variance (ddof=0)
      delta_max - delta_min,
  };
  for (double value : raw) {
    if (!std::isfinite(value)) {
      return FeatureStatus::kNumericalError;
    }
  }
  for (std::size_t i = 0; i < 5; ++i) {
    const float as_float = static_cast<float>(raw[i]);
    if (!std::isfinite(as_float)) {
      return FeatureStatus::kNumericalError;
    }
    output->values[i] = (as_float == 0.0f) ? 0.0f : as_float;
  }
  return FeatureStatus::kOk;
}

}  // namespace ostosense
