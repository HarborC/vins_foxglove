#include "JY901Parser.h"

namespace ov_sensors {
bool JY901Parser::feed(uint8_t b, ImuSample &out) {
  buf_[idx_++] = b;
  if (idx_ < 11)
    return false;

  // Align to 0x55 header and 0x5x type
  if (buf_[0] != 0x55 || (buf_[1] & 0x50) != 0x50) {
    // shift left by 1
    for (int i = 1; i < 11; ++i)
      buf_[i - 1] = buf_[i];
    idx_ = 10;
    return false;
  }

  // copy data
  int16_t s[4]{};
  std::memcpy(s, &buf_[2], 8);

  switch (buf_[1]) {
  case 0x51: { // accel in g-range (16g)
    for (int i = 0; i < 3; ++i)
      acc_[i] = (double)s[i] / 32768.0 * 16.0 * 9.81; // m/s^2
    has_acc_ = true;
  } break;
  case 0x52: { // gyro in dps (2000dps)
    for (int i = 0; i < 3; ++i)
      gyr_[i] = (double)s[i] / 32768.0 * 2000.0 * M_PI / 180.0; // rad/s
    has_gyr_ = true;
  } break;
  case 0x53: { // angle in deg
    for (int i = 0; i < 3; ++i)
      rpy_deg_[i] = (double)s[i] / 32768.0 * 180.0;
    has_rpy_ = true;
  } break;
  case 0x54: { // mag raw
    for (int i = 0; i < 3; ++i)
      mag_[i] = (double)s[i];
    has_mag_ = true;
  } break;
  default:
    break;
  }

  idx_ = 0; // reset packet index

  if (has_acc_ && has_gyr_ && has_rpy_) {
    out.accel = Eigen::Vector3d(acc_[0], acc_[1], acc_[2]);
    out.gyro = Eigen::Vector3d(gyr_[0], gyr_[1], gyr_[2]);
    out.has_accel = true;
    out.has_gyro = true;
    if (has_mag_) {
      out.mag = Eigen::Vector3d(mag_[0], mag_[1], mag_[2]);
      out.has_mag = true;
    }

    // build rotation from rpy (ZYX yaw-pitch-roll order)
    const double roll = rpy_deg_[0] * M_PI / 180.0;
    const double pitch = rpy_deg_[1] * M_PI / 180.0;
    const double yaw = rpy_deg_[2] * M_PI / 180.0;
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    out.R = R;
    out.has_R = true;

    // reset aggregation flags (next frame)
    has_acc_ = has_gyr_ = has_rpy_ = has_mag_ = false;
    return true;
  }
  return false;
}

} // namespace ov_sensors