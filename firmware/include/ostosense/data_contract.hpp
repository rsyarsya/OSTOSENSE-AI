#pragma once

#include <stdint.h>

namespace ostosense {

enum class CapQuality : uint8_t {
  kOk = 0,
  kDisconnected,
  kAdcSaturated,
  kBaselineInvalid,
  kDataGap,
  kWarmingUp,
};

enum class LigQuality : uint8_t {
  kOk = 0,
  kDisconnected,
  kAdcSaturated,
  kBaselineInvalid,
  kDataGap,
  kWarmingUp,
};

enum class SystemQuality : uint8_t {
  kInitializing = 0,
  kNormal,
  kMlUnavailable,
  kFailsafeDegraded,
  kUnsafe,
};

constexpr bool IsCapValid(CapQuality quality) {
  return quality == CapQuality::kOk;
}

constexpr bool IsLigValid(LigQuality quality) {
  return quality == LigQuality::kOk;
}

constexpr bool IsMlAvailable(CapQuality cap_quality, LigQuality lig_quality) {
  return IsCapValid(cap_quality) && lig_quality != LigQuality::kWarmingUp;
}

constexpr bool IsDirectLeakDetectionAvailable(LigQuality quality) {
  return IsLigValid(quality);
}

constexpr SystemQuality AggregateSystemQuality(CapQuality cap_quality,
                                               LigQuality lig_quality) {
  if (cap_quality == CapQuality::kWarmingUp ||
      lig_quality == LigQuality::kWarmingUp) {
    return SystemQuality::kInitializing;
  }
  const bool cap_valid = IsCapValid(cap_quality);
  const bool lig_valid = IsLigValid(lig_quality);
  if (cap_valid && lig_valid) {
    return SystemQuality::kNormal;
  }
  if (!cap_valid && lig_valid) {
    return SystemQuality::kMlUnavailable;
  }
  if (cap_valid && !lig_valid) {
    return SystemQuality::kFailsafeDegraded;
  }
  return SystemQuality::kUnsafe;
}

struct SensorSample {
  uint64_t timestamp_ms;
  float capacitance_raw;
  float lig_raw;
  CapQuality cap_quality;
  LigQuality lig_quality;
  SystemQuality system_quality;
};

constexpr SensorSample MakeSensorSample(uint64_t timestamp_ms,
                                        float capacitance_raw, float lig_raw,
                                        CapQuality cap_quality,
                                        LigQuality lig_quality) {
  return SensorSample{timestamp_ms,
                      capacitance_raw,
                      lig_raw,
                      cap_quality,
                      lig_quality,
                      AggregateSystemQuality(cap_quality, lig_quality)};
}

static_assert(AggregateSystemQuality(CapQuality::kOk, LigQuality::kOk) ==
              SystemQuality::kNormal);
static_assert(
    AggregateSystemQuality(CapQuality::kWarmingUp, LigQuality::kWarmingUp) ==
    SystemQuality::kInitializing);
static_assert(
    AggregateSystemQuality(CapQuality::kOk, LigQuality::kWarmingUp) ==
    SystemQuality::kInitializing);
static_assert(AggregateSystemQuality(CapQuality::kDataGap, LigQuality::kOk) ==
              SystemQuality::kMlUnavailable);
static_assert(
    AggregateSystemQuality(CapQuality::kOk, LigQuality::kDisconnected) ==
    SystemQuality::kFailsafeDegraded);
static_assert(
    AggregateSystemQuality(CapQuality::kOk, LigQuality::kBaselineInvalid) ==
    SystemQuality::kFailsafeDegraded);
static_assert(
    AggregateSystemQuality(CapQuality::kBaselineInvalid,
                           LigQuality::kDisconnected) ==
    SystemQuality::kUnsafe);

}  // namespace ostosense
