// 标准库 & 第三方
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

// 使用 C++17 文件系统命名空间
namespace fs = std::filesystem;

// 包含项目中的头文件
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"
#include "foxglove/FGVisualizer.h"

#include "utils/sensor_data.h"
#include "utils/memory_utils.h"
#include "serial_imu/ImuSample.h"
#include "serial_imu/ImuDriver.h"
#include "camera_v4l2/CameraFrame.h"
#include "camera_v4l2/V4L2CameraDriver.h"
#include "calibration/apriltags.h"
#include "calibration/aprilgrid.h" 
#include "calibration/frame_quality_evaluator.h"

#include "vi_dataset.h"

using namespace std;
using namespace cv;
using namespace ov_msckf;

// 全局变量定义
int task = 0; // 任务标志：0-MSCKF，1-Only IMU Record，2-Only Stereo Record, 3-Stereo IMU Record, 4-Only Visualize, 5-MSCKF+DEBUG
VIDataset::Ptr g_vi_dataset;
std::shared_ptr<FGVisualizer> viz_;

// 新驱动对象
std::shared_ptr<ov_sensors::ImuDriver> imu_driver_;
std::shared_ptr<ov_sensors::V4L2CameraDriver> cam_driver_;

std::string imu_device_ = "/dev/ttyS3";
std::string cam_device_ = "/dev/video73";

std::atomic<bool> thread_update_running = false;
std::shared_ptr<CAMERA_CALIB::AprilGrid> april_grid;
std::shared_ptr<FrameQualityEvaluator> dqe_left;
std::shared_ptr<FrameQualityEvaluator> dqe_right;

void stereoCalibDataGet(const ov_sensors::CameraFrame& image_msg) {
    int64_t time_us = (image_msg.timestamp_raw * 1e6);

    if (task == 3) {
        cv::Mat image_combined;
        cv::hconcat(image_msg.images[0], image_msg.images[1], image_combined);
        viz_->show_image("dete_images", time_us, image_combined, "dete_images");
        thread_update_running = false;
        return;
    }

    static int image_num = 0;
    const int numTags = april_grid->getTagCols() * april_grid->getTagRows();
    CAMERA_CALIB::ApriltagDetector ad(numTags);

    CAMERA_CALIB::CalibCornerData ccd_good_left;
    CAMERA_CALIB::CalibCornerData ccd_bad_left;
    ad.detectTags(image_msg.images[0], ccd_good_left.corners, ccd_good_left.corner_ids, ccd_good_left.radii,
                    ccd_bad_left.corners, ccd_bad_left.corner_ids, ccd_bad_left.radii);

    vector<Point2f> current_corners_left;
    for (int i_p = 0; i_p < ccd_good_left.corner_ids.size(); i_p++) {
        const Eigen::Vector2d& pt = ccd_good_left.corners[i_p];
        current_corners_left.emplace_back(pt.x(), pt.y());
    }

    bool isFrameAcceptable_left = dqe_left->isFrameAcceptable(image_msg.images[0], current_corners_left);

    CAMERA_CALIB::CalibCornerData ccd_good_right;
    CAMERA_CALIB::CalibCornerData ccd_bad_right;
    ad.detectTags(image_msg.images[1], ccd_good_right.corners, ccd_good_right.corner_ids, ccd_good_right.radii,
                    ccd_bad_right.corners, ccd_bad_right.corner_ids, ccd_bad_right.radii);

    vector<Point2f> current_corners_right;
    for (int i_p = 0; i_p < ccd_good_right.corner_ids.size(); i_p++) {
        const Eigen::Vector2d& pt = ccd_good_right.corners[i_p];
        current_corners_right.emplace_back(pt.x(), pt.y());
    }

    bool isFrameAcceptable_right = dqe_right->isFrameAcceptable(image_msg.images[1], current_corners_right);

    if (isFrameAcceptable_left || isFrameAcceptable_right) {
        // g_vi_dataset->saveStereoData(image_msg);
        image_num++;
    }

    cv::Mat image_combined;
    cv::hconcat(dqe_left->global_coverage_.viz_mat, dqe_right->global_coverage_.viz_mat, image_combined);
    // 图像中间用一条线分割
    cv::line(image_combined, cv::Point(image_combined.cols / 2, 0), cv::Point(image_combined.cols / 2, image_combined.rows), cv::Scalar(255, 255, 255), 2);
    // 图片下方黑框显示各种状态
    cv::line(image_combined, cv::Point(0, image_combined.rows - 50), cv::Point(image_combined.cols, image_combined.rows - 50), cv::Scalar(0, 0, 0), 2);
    cv::Mat status_image = cv::Mat::zeros(cv::Size(image_combined.cols, 50), CV_8UC3);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Image Num: " << image_num << " " << " | Left Coverage Ratio: " << dqe_left->calcCoverageRatio() << " " << " | Right Coverage Ratio: " << dqe_right->calcCoverageRatio();
    cv::putText(status_image, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::vconcat(image_combined, status_image, image_combined);
    viz_->show_image("dete_images", time_us, image_combined, "dete_images");

    thread_update_running = false;
}

// 启动新的 IMU 驱动并记录 CSV + 可视化
void startIMUDriverRecord(const std::string& serial_device = "/dev/ttyS3") {
  using namespace ov_sensors;
  if (imu_driver_) return;
  imu_driver_ = std::make_shared<ImuDriver>(serial_device);
  imu_driver_->setCallback([&](const ImuSample& s){
    ov_core::ImuData msg;
    msg.timestamp = s.timestamp;
    if (s.has_gyro)  msg.wm = s.gyro;
    if (s.has_accel) msg.am = s.accel;
    if (s.has_mag)   msg.hm = s.mag;
    if (s.has_R)     msg.Rm = s.R; else msg.Rm.setIdentity();

    if (task == 0 || task == 5) {
        viz_->feedIMU(msg);
    }
    
    if (g_vi_dataset)
        g_vi_dataset->saveIMUData(s);
    viz_->publish_raw_imu(msg);
  });
  if (!imu_driver_->start()) {
    std::cerr << "Failed to start ImuDriver on " << serial_device << std::endl;
  } else {
    std::cout << "IMU driver started on " << serial_device << std::endl;
  }
}

void startCameraDriverRecord(const std::string& device_path = "/dev/video73", double track_frequency = 30.0) {
    using namespace ov_sensors;
    if (cam_driver_) return;

    if (task == 2) {
        april_grid = std::make_shared<CAMERA_CALIB::AprilGrid>(std::string(PROJ_DIR) + "/thirdparty/kalibrlib/apps/others/aprilgrid.yaml");
        dqe_left = std::make_shared<FrameQualityEvaluator>();
        dqe_right = std::make_shared<FrameQualityEvaluator>();
    }

    V4L2CameraDriver::Config cfg;
    cfg.device = device_path;
    cfg.track_frequency = track_frequency;
    cam_driver_ = std::make_shared<V4L2CameraDriver>(cfg);
    cam_driver_->setCallback([&](const ov_sensors::CameraFrame& f){
        if (task == 0 || task == 5) {
            ov_core::CameraData msg;
            msg.timestamp = f.timestamp_raw;
            msg.sensor_ids = f.sensor_ids;
            // clone 防止底层复用缓冲
            msg.images.resize(f.images.size());
            for (size_t i=0;i<f.images.size();++i) msg.images[i] = f.images[i].clone();
            viz_->feedStereo(msg);
        }
        if (g_vi_dataset)
            g_vi_dataset->saveStereoData(f);

        if (task == 2 || task == 3) {
            if (!thread_update_running) {
                thread_update_running = true;
                std::thread(&stereoCalibDataGet, f).detach();
            }
        }
    });
    if (!cam_driver_->start()) {
        std::cerr << "Camera driver start failed on " << device_path << std::endl;
    } else {
        std::cout << "Camera driver started on " << device_path << std::endl;
    }
}

// 主函数
int main(int argc, char **argv) {
    if (argc > 1) {
        task = atoi(argv[1]);
    }

    std::string config_path = std::string(PROJ_DIR) + "/config/ours/estimator_config.yaml";
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);

    // 初始化 VIO 管理器参数
    VioManagerOptions params;
    params.print_and_load(parser);
    std::shared_ptr<VioManager> sys = std::make_shared<VioManager>(params);
    viz_ = std::make_shared<FGVisualizer>(sys);

    std::string root_dir = std::string(PROJ_DIR) + "/calib_data/";

    // 创建时间戳目录
    if (task == 0 || task == 5) {
        root_dir = viz_->debug_dir;
        if (task == 5) {
            g_vi_dataset = std::make_shared<VIDataset>(root_dir, 0);
        }
    } else if (task >= 1 && task <=3) {
        if (task == 1) {
            root_dir += "/imu_calib/";
        } else if (task == 2) {
            root_dir += "/camera_calib/";
        } else if (task == 3) {
            root_dir += "/camera_imu_calib/";
        }

        if (fs::exists(root_dir)) 
            fs::remove_all(root_dir);
        fs::create_directories(root_dir);
        g_vi_dataset = std::make_shared<VIDataset>(root_dir, 0);
    } 

    startIMUDriverRecord(imu_device_);
    if (task != 1) {
        startCameraDriverRecord(cam_device_, sys->get_params().track_frequency);
    }

    viz_->run();

    return 0;
}
