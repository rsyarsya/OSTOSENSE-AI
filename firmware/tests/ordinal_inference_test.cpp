// Standalone hand-written tests for the portable ordinal-inference core.
// SYNTHETIC_PIPELINE_TEST_ONLY; proves host-side inference mechanics only.
//
// Build (single line): g++ -std=c++17 -Wall -Wextra -pedantic -I firmware/include
//   firmware/tests/ordinal_inference_test.cpp -o /tmp/ordinal_inference_test
// Returns 0 on success, non-zero when any check fails.

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

#include "ostosense/ordinal_inference.hpp"

using ostosense::InferenceStatus;
using ostosense::OrdinalModel;
using ostosense::OrdinalPrediction;
using ostosense::PredictOrdinal;
using ostosense::RiskClass;

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

// A well-behaved reference model: identity scaler, first feature drives eta.
// Wide, symmetric theta so every one of the four classes can be the mode.
OrdinalModel BaseModel() {
  return OrdinalModel{
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},   // mean
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},   // scale
      {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},   // beta
      {-3.0f, 0.0f, 3.0f},              // theta
  };
}

bool UnitSum(const OrdinalPrediction& p) {
  float s = 0.0f;
  for (float v : p.probabilities) {
    if (!std::isfinite(v) || v < -1e-5f || v > 1.0f + 1e-5f) return false;
    s += v;
  }
  return std::fabs(s - 1.0f) <= 1e-5f;
}

}  // namespace

int main() {
  const OrdinalModel model = BaseModel();
  OrdinalPrediction out;

  // 1. Normal four-class prediction set: large -eta -> Safe, large +eta -> Urgent,
  //    and intermediate etas move the mode monotonically upward.
  {
    struct Case { float x0; RiskClass expected; };
    const std::array<Case, 4> cases = {{
        {-4.5f, RiskClass::kSafe},
        {-1.5f, RiskClass::kMonitor},
        {1.5f, RiskClass::kCaution},
        {4.5f, RiskClass::kUrgent},
    }};
    for (const Case& c : cases) {
      const std::array<float, 5> f = {c.x0, 0.0f, 0.0f, 0.0f, 0.0f};
      Check(PredictOrdinal(model, f, &out) == InferenceStatus::kOk, "normal: status ok");
      Check(out.predicted_class == c.expected, "normal: expected class");
      Check(UnitSum(out), "normal: unit sum");
    }
  }

  // 2. Exact tie (eta = 0, symmetric theta) ties Monitor and Caution; the lower
  //    index (Monitor) must win.
  {
    const std::array<float, 5> f = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Check(PredictOrdinal(model, f, &out) == InferenceStatus::kOk, "tie: status ok");
    Check(out.predicted_class == RiskClass::kMonitor, "tie: lowest index");
    Check(std::fabs(out.probabilities[1] - out.probabilities[2]) < 1e-6f, "tie: symmetric middle");
    Check(std::fabs(out.probabilities[0] - out.probabilities[3]) < 1e-6f, "tie: symmetric tails");
  }

  // 3. Large positive/negative eta without overflow.
  {
    const std::array<float, 5> big = {1e30f, 0.0f, 0.0f, 0.0f, 0.0f};
    Check(PredictOrdinal(model, big, &out) == InferenceStatus::kOk, "bigpos: ok");
    Check(out.predicted_class == RiskClass::kUrgent, "bigpos: urgent");
    Check(UnitSum(out), "bigpos: unit sum");
    const std::array<float, 5> small = {-1e30f, 0.0f, 0.0f, 0.0f, 0.0f};
    Check(PredictOrdinal(model, small, &out) == InferenceStatus::kOk, "bigneg: ok");
    Check(out.predicted_class == RiskClass::kSafe, "bigneg: safe");
    Check(UnitSum(out), "bigneg: unit sum");
  }

  const std::array<float, 5> zero_features = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // 4. Zero / negative / non-finite scale -> kInvalidModel.
  {
    for (float bad : {0.0f, -1.0f, kNaN, kInf}) {
      OrdinalModel m = BaseModel();
      m.scale[2] = bad;
      Check(PredictOrdinal(m, zero_features, &out) == InferenceStatus::kInvalidModel,
            "bad scale -> kInvalidModel");
    }
  }

  // 5. Non-increasing or non-finite theta -> kInvalidModel.
  {
    OrdinalModel m = BaseModel();
    m.theta = {1.0f, 0.0f, -1.0f};  // decreasing
    Check(PredictOrdinal(m, zero_features, &out) == InferenceStatus::kInvalidModel,
          "decreasing theta -> kInvalidModel");
    m = BaseModel();
    m.theta = {0.0f, 0.0f, 1.0f};   // equal (not strictly increasing)
    Check(PredictOrdinal(m, zero_features, &out) == InferenceStatus::kInvalidModel,
          "equal theta -> kInvalidModel");
    m = BaseModel();
    m.theta[1] = kInf;
    Check(PredictOrdinal(m, zero_features, &out) == InferenceStatus::kInvalidModel,
          "non-finite theta -> kInvalidModel");
  }

  // 6. Non-finite beta / mean -> kInvalidModel.
  {
    OrdinalModel m = BaseModel();
    m.beta[0] = kNaN;
    Check(PredictOrdinal(m, zero_features, &out) == InferenceStatus::kInvalidModel,
          "non-finite beta -> kInvalidModel");
    m = BaseModel();
    m.mean[4] = kInf;
    Check(PredictOrdinal(m, zero_features, &out) == InferenceStatus::kInvalidModel,
          "non-finite mean -> kInvalidModel");
  }

  // 7. Non-finite features -> kInvalidFeatures.
  {
    const std::array<float, 5> nan_f = {kNaN, 0.0f, 0.0f, 0.0f, 0.0f};
    Check(PredictOrdinal(model, nan_f, &out) == InferenceStatus::kInvalidFeatures,
          "non-finite feature -> kInvalidFeatures");
    const std::array<float, 5> inf_f = {0.0f, 0.0f, kInf, 0.0f, 0.0f};
    Check(PredictOrdinal(model, inf_f, &out) == InferenceStatus::kInvalidFeatures,
          "inf feature -> kInvalidFeatures");
  }

  // 8. Null output -> kInvalidOutput.
  {
    Check(PredictOrdinal(model, zero_features, nullptr) == InferenceStatus::kInvalidOutput,
          "null output -> kInvalidOutput");
  }

  // 9. A failed call must reset the output to a safe default (no stale results).
  {
    OrdinalPrediction stale;
    stale.probabilities = {9.0f, 9.0f, 9.0f, 9.0f};
    stale.predicted_class = RiskClass::kUrgent;
    OrdinalModel m = BaseModel();
    m.scale[0] = 0.0f;
    Check(PredictOrdinal(m, zero_features, &stale) == InferenceStatus::kInvalidModel,
          "stale-reset: status");
    Check(stale.predicted_class == RiskClass::kSafe, "stale-reset: class default");
    for (float p : stale.probabilities) {
      Check(p == 0.0f, "stale-reset: probabilities cleared");
    }
  }

  if (g_failures == 0) {
    std::printf("ordinal_inference_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "ordinal_inference_test: %d failure(s)\n", g_failures);
  return 1;
}
