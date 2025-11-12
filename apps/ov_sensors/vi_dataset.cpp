#include "vi_dataset.h"
#include "serial_imu/ImuSample.h"
#include "camera_v4l2/CameraFrame.h"

namespace fs = std::filesystem;

VIDataset::VIDataset(const std::string& dataset_path, int use_type)
    : dataset_path_(dataset_path), use_type_(use_type) {
  // Constructor implementation (if needed)
  left_images_dir_ = dataset_path_ + "/images/left/";
  right_images_dir_ = dataset_path_ + "/images/right/";
  imu_data_file_ = dataset_path_ + "/imu_data.csv";
  stereo_data_file_ = dataset_path_ + "/stereo_data.csv";

  if (use_type_ == 0) { // save
    g_imu_csv_ = std::make_shared<std::ofstream>(imu_data_file_);
    // Write CSV header
    if (g_imu_csv_ && g_imu_csv_->tellp() == 0) {
        (*g_imu_csv_) << "# timestamp acc_x acc_y acc_z gyro_x gyro_y gyro_z qw qx qy qz mag_x mag_y mag_z\n";
    }

    fs::create_directories(left_images_dir_);
    fs::create_directories(right_images_dir_);

    
    g_stereo_csv_ = std::make_shared<std::ofstream>(stereo_data_file_);
    if (g_stereo_csv_ && g_stereo_csv_->tellp() == 0) {
        (*g_stereo_csv_) << "# timestamp image_path\n";
    }
  }

  if (use_type_ == 1) { // load
    g_imu_csv_in_ = std::make_shared<std::ifstream>(imu_data_file_);
    // Skip header line
    std::string header_line;
    std::getline(*g_imu_csv_in_, header_line);
    g_stereo_csv_in_ = std::make_shared<std::ifstream>(stereo_data_file_);
    // Skip header line
    std::getline(*g_stereo_csv_in_, header_line);
  }
}

void VIDataset::saveIMUData(const ov_sensors::ImuSample& imu_sample) {
    if (!g_imu_csv_ || !g_imu_csv_->good()) return;
    Eigen::Quaterniond q(imu_sample.R);
    (*g_imu_csv_) << std::fixed << std::setprecision(6) << imu_sample.timestamp << " "
                 << imu_sample.accel(0) << " " << imu_sample.accel(1) << " " << imu_sample.accel(2) << " "
                 << imu_sample.gyro(0) << " " << imu_sample.gyro(1) << " " << imu_sample.gyro(2) << " "
                 << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                 << imu_sample.mag(0) << " " << imu_sample.mag(1) << " " << imu_sample.mag(2) << "\n";
    g_imu_csv_->flush();
}

void VIDataset::saveStereoData(const ov_sensors::CameraFrame& f) {
    // Implementation for saving stereo image data
    std::ostringstream oss; oss << std::fixed << std::setprecision(6) << f.timestamp_raw;
    const std::string ts = oss.str();
    cv::imwrite(left_images_dir_  + "/" + ts + ".png", f.images[0]);
    cv::imwrite(right_images_dir_ + "/" + ts + ".png", f.images[1]);
    if (!g_stereo_csv_ || !g_stereo_csv_->good()) return;
    (*g_stereo_csv_) << std::fixed << std::setprecision(6) << f.timestamp_raw << " " << ts + ".png" << "\n";
    g_stereo_csv_->flush();
}

bool VIDataset::loadNextStereoData(ov_sensors::CameraFrame& f) {
    if (!g_stereo_csv_in_ || !g_stereo_csv_in_->good()) return false;
    std::string line;
    if (!std::getline(*g_stereo_csv_in_, line)) return false;

    std::istringstream iss(line);
    double timestamp;
    std::string image_path;
    if (!(iss >> timestamp >> image_path)) return false;

    f.timestamp_raw = timestamp;
    f.sensor_ids = {0, 1}; // Assuming stereo cameras are indexed as 0 and 1

    // Load images
    cv::Mat left_image = cv::imread(left_images_dir_ + "/" + image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat right_image = cv::imread(right_images_dir_ + "/" + image_path, cv::IMREAD_GRAYSCALE);
    if (left_image.empty() || right_image.empty()) return false;

    f.images = {left_image, right_image};
    return true;
}

bool VIDataset::loadNextIMUData(ov_sensors::ImuSample& imu_sample) {
    if (!g_imu_csv_in_ || !g_imu_csv_in_->good()) return false;
    std::string line;
    if (!std::getline(*g_imu_csv_in_, line)) return false;

    std::istringstream iss(line);
    double timestamp;
    double acc_x, acc_y, acc_z;
    double gyro_x, gyro_y, gyro_z;
    double qw, qx, qy, qz;
    double mag_x, mag_y, mag_z;

    if (!(iss >> timestamp >> acc_x >> acc_y >> acc_z
              >> gyro_x >> gyro_y >> gyro_z
              >> qw >> qx >> qy >> qz
              >> mag_x >> mag_y >> mag_z)) {
        return false;
    }

    imu_sample.timestamp = timestamp;
    imu_sample.accel = Eigen::Vector3d(acc_x, acc_y, acc_z);
    imu_sample.gyro = Eigen::Vector3d(gyro_x, gyro_y, gyro_z);
    imu_sample.mag = Eigen::Vector3d(mag_x, mag_y, mag_z);
    Eigen::Quaterniond q(qw, qx, qy, qz);
    imu_sample.R = q.toRotationMatrix();
    imu_sample.has_accel = true;
    imu_sample.has_gyro = true;
    imu_sample.has_mag = true;
    imu_sample.has_R = true;

    return true;
}