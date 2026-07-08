#include "ostosense/data_contract.hpp"

using ostosense::AggregateSystemQuality;
using ostosense::CapQuality;
using ostosense::IsDirectLeakDetectionAvailable;
using ostosense::IsMlAvailable;
using ostosense::LigQuality;
using ostosense::MakeSensorSample;
using ostosense::SystemQuality;

int main() {
  const auto initializing = MakeSensorSample(
      0, 0.0F, 0.0F, CapQuality::kWarmingUp, LigQuality::kWarmingUp);
  if (initializing.system_quality != SystemQuality::kInitializing ||
      IsMlAvailable(initializing.cap_quality, initializing.lig_quality) ||
      IsDirectLeakDetectionAvailable(initializing.lig_quality)) {
    return 1;
  }

  const auto normal = MakeSensorSample(1, 12.5F, 410.0F, CapQuality::kOk,
                                       LigQuality::kOk);
  if (normal.system_quality != SystemQuality::kNormal) {
    return 2;
  }

  const auto no_failsafe = MakeSensorSample(
      2, 12.5F, 0.0F, CapQuality::kOk, LigQuality::kDisconnected);
  if (!IsMlAvailable(no_failsafe.cap_quality, no_failsafe.lig_quality) ||
      IsDirectLeakDetectionAvailable(no_failsafe.lig_quality) ||
      no_failsafe.system_quality != SystemQuality::kFailsafeDegraded) {
    return 3;
  }

  const auto invalid_lig_baseline = MakeSensorSample(
      3, 12.5F, 405.0F, CapQuality::kOk, LigQuality::kBaselineInvalid);
  if (IsDirectLeakDetectionAvailable(invalid_lig_baseline.lig_quality) ||
      invalid_lig_baseline.system_quality !=
          SystemQuality::kFailsafeDegraded) {
    return 4;
  }

  const auto unsafe = AggregateSystemQuality(CapQuality::kDataGap,
                                              LigQuality::kDisconnected);
  return unsafe == SystemQuality::kUnsafe ? 0 : 5;
}
