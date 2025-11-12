#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

#include <opencv2/opencv.hpp>

namespace ov_sensors {
struct CameraFrame;
struct ImuSample;
} 

class VIDataset {
public:
  using Ptr = std::shared_ptr<VIDataset>;
  VIDataset(const std::string& dataset_path, int use_type = 0);
  ~VIDataset() {};

  bool loadNextStereoData(ov_sensors::CameraFrame& f);

  bool loadNextIMUData(ov_sensors::ImuSample& imu_sample);

  void saveIMUData(const ov_sensors::ImuSample& imu_sample);

  void saveStereoData(const ov_sensors::CameraFrame& f);

private:
  std::string dataset_path_;
  std::string left_images_dir_;
  std::string right_images_dir_;
  std::string imu_data_file_;
  std::string stereo_data_file_;
  int use_type_; // 0: save, 1: load

  std::shared_ptr<std::ofstream> g_imu_csv_;
  std::shared_ptr<std::ofstream> g_stereo_csv_;
  std::shared_ptr<std::ifstream> g_imu_csv_in_;
  std::shared_ptr<std::ifstream> g_stereo_csv_in_;
};