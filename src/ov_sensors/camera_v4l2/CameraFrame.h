#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <vector>
#include <string>

namespace ov_sensors {

struct CameraFrame {
  double timestamp_raw = 0.0;          // CLOCK_MONOTONIC_RAW seconds
  std::vector<int> sensor_ids;         // camera indices (e.g. {0,1})
  std::vector<cv::Mat> images;         // grayscale images
};

} // namespace ov_sensors
