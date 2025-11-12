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
#include <foxglove/visualizer.h>
#include "utils/sensor_data.h"
#include "utils/memory_utils.h"
#include "serial_imu/ImuSample.h"
#include "serial_imu/ImuDriver.h"
#include "camera_v4l2/CameraFrame.h"
#include "camera_v4l2/V4L2CameraDriver.h"
#include "calibration/apriltags.h"
#include "calibration/aprilgrid.h" 
#include "calibration/frame_quality_evaluator.h"

using namespace std;
using namespace cv;

// 全局变量定义
foxglove_viz::Visualizer::Ptr viz_; // 可视化工具指针
std::deque<ov_core::CameraData> camera_queue; // 相机数据队列
std::mutex camera_queue_mtx; // 互斥锁
std::atomic<bool> thread_update_running = false;
int is_record_camera = 0;
std::shared_ptr<CAMERA_CALIB::AprilGrid> april_grid;
std::shared_ptr<FrameQualityEvaluator> dqe_left;
std::shared_ptr<FrameQualityEvaluator> dqe_right;

std::string root_dir = std::string(PROJ_DIR) + "/calib_data/";
std::string left_images_dir, right_images_dir;

// 新驱动对象
std::shared_ptr<ov_sensors::ImuDriver> g_imu;
std::shared_ptr<ov_sensors::V4L2CameraDriver> g_cam;
std::shared_ptr<std::ofstream> g_imu_csv;

// 视频缓冲区结构
struct Buffer {
    void* start;
    size_t length;
};

inline double now_mono_raw() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 启动新的 IMU 驱动并记录 CSV + 可视化
void startIMUDriverRecord(const std::string& serial_device = "/dev/ttyS3") {
  using namespace ov_sensors;
  if (g_imu) return;
  g_imu_csv = std::make_shared<std::ofstream>(root_dir + "/imu_data.txt", std::ios::app);
  if (g_imu_csv && g_imu_csv->tellp() == 0) {
    (*g_imu_csv) << "# timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,qw,qx,qy,qz,mag_x,mag_y,mag_z\n";
  }
  auto writeIMU = [&](const ov_core::ImuData& msg){
    if (!g_imu_csv || !g_imu_csv->good()) return;
    Eigen::Quaterniond q(msg.Rm);
    (*g_imu_csv) << std::fixed << std::setprecision(6) << msg.timestamp << ","
                 << msg.am(0) << "," << msg.am(1) << "," << msg.am(2) << ","
                 << msg.wm(0) << "," << msg.wm(1) << "," << msg.wm(2) << ","
                 << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
                 << msg.hm(0) << "," << msg.hm(1) << "," << msg.hm(2) << "\n";
    g_imu_csv->flush();
  };
  g_imu = std::make_shared<ImuDriver>(serial_device);
  g_imu->setCallback([&](const ImuSample& s){
    ov_core::ImuData msg;
    msg.timestamp = s.timestamp;
    if (s.has_gyro)  msg.wm = s.gyro;
    if (s.has_accel) msg.am = s.accel;
    if (s.has_mag)   msg.hm = s.mag;
    if (s.has_R)     msg.Rm = s.R; else msg.Rm.setIdentity();
    writeIMU(msg);
    viz_->publishIMU("raw_imu", int64_t(msg.timestamp * 1e6), "IMU", msg.am, msg.wm, Eigen::Quaterniond(msg.Rm), msg.hm);
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity(); T.block(0,0,3,3) = msg.Rm.cast<float>();
    viz_->showPose("imu_angle", int64_t(msg.timestamp * 1e6), T, "LOCAL_WORLD", "IMU_R");
  });
  if (!g_imu->start()) {
    std::cerr << "Failed to start ImuDriver on " << serial_device << std::endl;
  } else {
    std::cout << "IMU driver started on " << serial_device << std::endl;
  }
}

void stereoCalibDataGet(const ov_core::CameraData& image_msg) {
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
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << image_msg.timestamp;
        std::string timestamp_str = oss.str();

        std::string left_path = left_images_dir + "/" + timestamp_str + ".png";
        std::string right_path = right_images_dir + "/" + timestamp_str + ".png";

        cv::imwrite(left_path, image_msg.images[0]);
        cv::imwrite(right_path, image_msg.images[1]);
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
    
    int64_t time_us = (image_msg.timestamp * 1e6);
    viz_->showImage("dete_images", time_us, image_combined, "dete_images", true);

    thread_update_running = false;
}

void startCameraDriverRecord(const std::string& device_path = "/dev/video73", double track_frequency = 20.0) {
    using namespace ov_sensors;
    if (g_cam) return;
    left_images_dir = root_dir + "/images/left/";
    right_images_dir = root_dir + "/images/right/";
    fs::create_directories(left_images_dir);
    fs::create_directories(right_images_dir);

    V4L2CameraDriver::Config cfg;
    cfg.device = device_path;
    cfg.track_frequency = track_frequency;
    g_cam = std::make_shared<V4L2CameraDriver>(cfg);
    g_cam->setCallback([&](const CameraFrame& f){
        ov_core::CameraData msg;
        msg.timestamp  = f.timestamp_raw;
        msg.sensor_ids = f.sensor_ids;
        msg.images.resize(f.images.size());
        for (size_t i=0;i<f.images.size();++i) msg.images[i] = f.images[i].clone();

        if (is_record_camera == 2) {
            std::ostringstream oss; oss << std::fixed << std::setprecision(6) << msg.timestamp;
            const std::string ts = oss.str();
            cv::imwrite(left_images_dir  + "/" + ts + ".png", msg.images[0]);
            cv::imwrite(right_images_dir + "/" + ts + ".png", msg.images[1]);
        }

        {
            std::lock_guard<std::mutex> lk(camera_queue_mtx);
            camera_queue.push_back(msg);
            if (camera_queue.size() > 240) camera_queue.pop_front();
        }

        if (is_record_camera == 1) {
            if (!thread_update_running) {
                thread_update_running = true;
                std::thread(&stereoCalibDataGet, msg).detach();
            }
        }
    });
    if (!g_cam->start()) {
        std::cerr << "Camera driver start failed on " << device_path << std::endl;
    } else {
        std::cout << "Camera driver started on " << device_path << std::endl;
    }
}

// 主函数
int main(int argc, char **argv) {
    if (argc > 1) {
        is_record_camera = atoi(argv[1]);
    }

    // 创建时间戳目录
    if (is_record_camera == 0) {
        root_dir += "/imu_calib/";
    } else if (is_record_camera == 1) {
        root_dir += "/camera_calib/";
    } else if (is_record_camera == 2) {
        root_dir += "/camera_imu_calib/";
    } else {
        std::cout << "Invalid argument" << std::endl;
        return -1;
    }

    if (fs::exists(root_dir)) 
        fs::remove_all(root_dir);
    fs::create_directories(root_dir);

    // 初始化可视化器
    viz_ = std::make_shared<foxglove_viz::Visualizer>(8088, 2);

    // 启动 IMU 驱动（异步回调）
    startIMUDriverRecord("/dev/ttyS3");

    // 启动相机线程（可选）
    if (is_record_camera) {
        if (is_record_camera == 1) {
            april_grid = std::make_shared<CAMERA_CALIB::AprilGrid>(std::string(PROJ_DIR) + "/thirdparty/kalibrlib/apps/others/aprilgrid.yaml");
            dqe_left = std::make_shared<FrameQualityEvaluator>();
            dqe_right = std::make_shared<FrameQualityEvaluator>();
        }

        // 启动相机驱动（异步回调）
        startCameraDriverRecord("/dev/video73", 20.0);
        std::cout << "Camera driver thread started." << std::endl;

        while (1)
        {
            if (!camera_queue.empty()) {
                ov_core::CameraData image;
                {
                    std::lock_guard<std::mutex> lock(camera_queue_mtx);
                    image = camera_queue.front();
                    camera_queue.pop_front();
                }

                cv::Mat image_combined;
                cv::hconcat(image.images[0], image.images[1], image_combined);
                int64_t time_us = (image.timestamp * 1e6);
                viz_->showImage("track_images", time_us, image_combined, "track_images", true);
            } else {
                usleep(10); // 队列为空时休眠
            }
        }
    }

    return 0;
}
