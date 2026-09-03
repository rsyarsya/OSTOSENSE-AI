// Portable host-side ordinal-inference core for OSTOSENSE (pipeline test only).
//
// SYNTHETIC_PIPELINE_TEST_ONLY. This header-only, dependency-free C++17
// implementation reproduces the exported `mord.LogisticAT` cumulative
// all-threshold forward pass so a portable inference path can be checked for
// parity with the Python reference. It is written to be reusable later on an
// ESP32-S3, but this file is NOT ESP32 deployment, NOT firmware parity on
// physical hardware, and NOT evidence of OSTOSENSE performance.
//
// No dynamic allocation, no exceptions, no JSON, no embedded model — the model
// and golden vectors are provided separately (e.g. generated test-only headers).

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ostosense {

enum class RiskClass : std::uint8_t {
  kSafe = 0,
  kMonitor = 1,
  kCaution = 2,
  kUrgent = 3,
};

enum class InferenceStatus : std::uint8_t {
  kOk = 0,
  kInvalidModel,
  kInvalidFeatures,
  kNumericalError,
  kInvalidOutput,
};

struct OrdinalModel {
  std::array<float, 5> mean;
  std::array<float, 5> scale;
  std::array<float, 5> beta;
  std::array<float, 3> theta;
};

struct OrdinalPrediction {
  std::array<float, 4> probabilities;
  RiskClass predicted_class;
};

namespace detail {

// Overflow-safe logistic sigmoid for float32.
inline float StableSigmoid(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float exp_x = std::exp(x);
  return exp_x / (1.0f + exp_x);
}

template <std::size_t N>
inline bool AllFinite(const std::array<float, N>& values) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

}  // namespace detail

// Canonical forward pass:
//   z = (x - mean) / scale
//   eta = beta^T z
//   P(Y <= k) = sigmoid(theta[k] - eta)      for k = 0, 1, 2
//   P(Y = 0) = P(Y <= 0)
//   P(Y = j) = P(Y <= j) - P(Y <= j-1)       for j = 1, 2
//   P(Y = 3) = 1 - P(Y <= 2)
//   predicted class = argmax_j P(Y = j)      (strict '>' -> ties keep lowest index)
inline InferenceStatus PredictOrdinal(const OrdinalModel& model,
                                      const std::array<float, 5>& features,
                                      OrdinalPrediction* output) {
  if (output == nullptr) {
    return InferenceStatus::kInvalidOutput;
  }
  // Reset to a safe default so a failed call never exposes stale results.
  output->probabilities = {0.0f, 0.0f, 0.0f, 0.0f};
  output->predicted_class = RiskClass::kSafe;

  // Validate the model.
  if (!detail::AllFinite(model.mean) || !detail::AllFinite(model.scale) ||
      !detail::AllFinite(model.beta) || !detail::AllFinite(model.theta)) {
    return InferenceStatus::kInvalidModel;
  }
  for (float s : model.scale) {
    if (!(s > 0.0f)) {  // rejects zero, negative, and NaN
      return InferenceStatus::kInvalidModel;
    }
  }
  if (!(model.theta[0] < model.theta[1] && model.theta[1] < model.theta[2])) {
    return InferenceStatus::kInvalidModel;
  }

  // Validate the features.
  if (!detail::AllFinite(features)) {
    return InferenceStatus::kInvalidFeatures;
  }

  // z = (x - mean) / scale ; eta = beta^T z
  float eta = 0.0f;
  for (std::size_t i = 0; i < 5; ++i) {
    const float z = (features[i] - model.mean[i]) / model.scale[i];
    eta += model.beta[i] * z;
  }
  if (!std::isfinite(eta)) {
    return InferenceStatus::kNumericalError;
  }

  // Cumulative logits P(Y <= k) = sigmoid(theta[k] - eta).
  std::array<float, 3> cumulative{};
  for (std::size_t k = 0; k < 3; ++k) {
    cumulative[k] = detail::StableSigmoid(model.theta[k] - eta);
    if (!std::isfinite(cumulative[k])) {
      return InferenceStatus::kNumericalError;
    }
  }

  const std::array<float, 4> probabilities = {
      cumulative[0],
      cumulative[1] - cumulative[0],
      cumulative[2] - cumulative[1],
      1.0f - cumulative[2],
  };

  // Require finite in-range probabilities and a unit sum within tolerance;
  // never clamp malformed output into a seemingly valid prediction.
  constexpr float kProbabilityTolerance = 1.0e-5f;
  float sum = 0.0f;
  for (float p : probabilities) {
    if (!std::isfinite(p) || p < -kProbabilityTolerance ||
        p > 1.0f + kProbabilityTolerance) {
      return InferenceStatus::kInvalidOutput;
    }
    sum += p;
  }
  if (std::fabs(sum - 1.0f) > kProbabilityTolerance) {
    return InferenceStatus::kInvalidOutput;
  }

  // Strict '>' argmax so ties retain the lowest class index (matches NumPy/mord).
  std::size_t best = 0;
  for (std::size_t i = 1; i < 4; ++i) {
    if (probabilities[i] > probabilities[best]) {
      best = i;
    }
  }

  output->probabilities = probabilities;
  output->predicted_class = static_cast<RiskClass>(static_cast<std::uint8_t>(best));
  return InferenceStatus::kOk;
}

}  // namespace ostosense
