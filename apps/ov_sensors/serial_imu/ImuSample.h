#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <string>

namespace ov_sensors {

struct ImuSample {
  double timestamp = 0.0;                          // MONOTONIC_RAW seconds
  Eigen::Vector3d accel = Eigen::Vector3d::Zero(); // m/s^2
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();  // rad/s
  Eigen::Vector3d mag = Eigen::Vector3d::Zero();   // raw or uT
  Eigen::Matrix3d R =
      Eigen::Matrix3d::Identity(); // optional orientation (if sensor provides)
  bool has_accel = false;
  bool has_gyro = false;
  bool has_mag = false;
  bool has_R = false;
};

} // namespace ov_sensors
