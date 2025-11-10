#pragma once
#include "ImuParser.h"
#include <array>
#include <cmath>

namespace ov_sensors {

// Minimal JY901 (WitMotion) binary protocol parser
// 11-byte packets: 0x55, type, 8 data bytes, checksum
// types: 0x51 acc, 0x52 gyro, 0x53 angle, 0x54 mag
class JY901Parser final : public ImuParser {
public:
  bool feed(uint8_t b, ImuSample &out) override;

private:
  std::array<uint8_t, 11> buf_{};
  int idx_ = 0;
  // aggregation
  double acc_[3]{};
  bool has_acc_ = false;
  double gyr_[3]{};
  bool has_gyr_ = false;
  double rpy_deg_[3]{};
  bool has_rpy_ = false;
  double mag_[3]{};
  bool has_mag_ = false;
};

} // namespace ov_sensors
