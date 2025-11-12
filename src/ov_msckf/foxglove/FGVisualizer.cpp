#include "FGVisualizer.h"

#include "core/VioManager.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/dataset_reader.h"
#include "utils/memory_utils.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>
#include <filesystem>
#include <signal.h>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;
using namespace std;
namespace fs = std::filesystem;

namespace {
  std::atomic<bool>* g_run_flag = nullptr;
  void rsio_on_signal(int) {
    if (g_run_flag) g_run_flag->store(false);
  }
}

FGVisualizer::FGVisualizer(std::shared_ptr<VioManager> app, bool is_viz)
  : _app(app), thread_update_running(false) {

  create_debug_dir();

  if (is_viz) {
    _viz = std::make_shared<foxglove_viz::Visualizer>(8088, 2);
  }
  _dash_board = std::make_shared<DashBoard>();
}

FGVisualizer::~FGVisualizer() {
  visualize_final();
}

void FGVisualizer::create_debug_dir() {
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  std::string timestamp_str = oss.str();

  debug_dir += "/" + timestamp_str + "/";
  fs::create_directories(debug_dir);

  initSpdlog("vins", debug_dir);

  odom_out_ptr_ = std::make_shared<std::ofstream>(debug_dir + "odom_imu.txt");
  odom_cam_ptr_ = std::make_shared<std::ofstream>(debug_dir + "odom_cam.txt");
}

void FGVisualizer::visualize_final() {

  // Final time offset value
  if (_app->get_state()->_options.do_calib_camera_timeoffset) {
    PRINT_INFO(REDPURPLE "camera-imu timeoffset = %.5f\n\n" RESET, _app->get_state()->_calib_dt_CAMtoIMU->value()(0));
  }

  // Final camera intrinsics
  if (_app->get_state()->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = _app->get_state()->_cam_intrinsics.at(i);
      PRINT_INFO(REDPURPLE "cam%d intrinsics:\n" RESET, (int)i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n" RESET, calib->value()(0), calib->value()(1), calib->value()(2), calib->value()(3));
      PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,%.5f\n\n" RESET, calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Final camera extrinsics
  if (_app->get_state()->_options.do_calib_camera_pose) {
    for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = _app->get_state()->_calib_IMUtoCAM.at(i);
      Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
      T_CtoI.block(0, 0, 3, 3) = quat_2_Rot(calib->quat()).transpose();
      T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * calib->pos();
      PRINT_INFO(REDPURPLE "T_C%dtoI:\n" RESET, i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(0, 0), T_CtoI(0, 1), T_CtoI(0, 2), T_CtoI(0, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(1, 0), T_CtoI(1, 1), T_CtoI(1, 2), T_CtoI(1, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(2, 0), T_CtoI(2, 1), T_CtoI(2, 2), T_CtoI(2, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n\n" RESET, T_CtoI(3, 0), T_CtoI(3, 1), T_CtoI(3, 2), T_CtoI(3, 3));
    }
  }

  // IMU intrinsics
  if (_app->get_state()->_options.do_calib_imu_intrinsics) {
    Eigen::Matrix3d Dw = State::Dm(_app->get_state()->_options.imu_model, _app->get_state()->_calib_imu_dw->value());
    Eigen::Matrix3d Da = State::Dm(_app->get_state()->_options.imu_model, _app->get_state()->_calib_imu_da->value());
    Eigen::Matrix3d Tw = Dw.colPivHouseholderQr().solve(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d Ta = Da.colPivHouseholderQr().solve(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d R_IMUtoACC = _app->get_state()->_calib_imu_ACCtoIMU->Rot().transpose();
    Eigen::Matrix3d R_IMUtoGYRO = _app->get_state()->_calib_imu_GYROtoIMU->Rot().transpose();
    PRINT_INFO(REDPURPLE "Tw:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Tw(0, 0), Tw(0, 1), Tw(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Tw(1, 0), Tw(1, 1), Tw(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, Tw(2, 0), Tw(2, 1), Tw(2, 2));
    PRINT_INFO(REDPURPLE "Ta:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Ta(0, 0), Ta(0, 1), Ta(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Ta(1, 0), Ta(1, 1), Ta(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, Ta(2, 0), Ta(2, 1), Ta(2, 2));
    PRINT_INFO(REDPURPLE "R_IMUtoACC:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoACC(0, 0), R_IMUtoACC(0, 1), R_IMUtoACC(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoACC(1, 0), R_IMUtoACC(1, 1), R_IMUtoACC(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, R_IMUtoACC(2, 0), R_IMUtoACC(2, 1), R_IMUtoACC(2, 2));
    PRINT_INFO(REDPURPLE "R_IMUtoGYRO:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoGYRO(0, 0), R_IMUtoGYRO(0, 1), R_IMUtoGYRO(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoGYRO(1, 0), R_IMUtoGYRO(1, 1), R_IMUtoGYRO(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, R_IMUtoGYRO(2, 0), R_IMUtoGYRO(2, 1), R_IMUtoGYRO(2, 2));
  }

  // IMU intrinsics gravity sensitivity
  if (_app->get_state()->_options.do_calib_imu_g_sensitivity) {
    Eigen::Matrix3d Tg = State::Tg(_app->get_state()->_calib_imu_tg->value());
    PRINT_INFO(REDPURPLE "Tg:\n" RESET);
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f,\n" RESET, Tg(0, 0), Tg(0, 1), Tg(0, 2));
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f,\n" RESET, Tg(1, 0), Tg(1, 1), Tg(1, 2));
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f\n\n" RESET, Tg(2, 0), Tg(2, 1), Tg(2, 2));
  }

  // Print the total time
  rT2 = boost::posix_time::microsec_clock::local_time();
  PRINT_INFO(REDPURPLE "TIME: %.3f seconds\n\n" RESET, (rT2 - rT1).total_microseconds() * 1e-6);
}

void FGVisualizer::setDevicesAndLatency(const std::string &imu_dev,
                                        const std::string &cam_dev,
                                        const std::string &pose_dev,
                                        double cam_latency){
  pose_serial_device_ = pose_dev;
  cam_fixed_latency_ = cam_latency;
}

void FGVisualizer::feedIMU(const ov_core::ImuData& msg) {
  _app->feed_measurement_imu(msg);
  visualize_odometry(msg.timestamp);

  // 触发消费相机队列（固定延迟落后条件）
  // 更新 IMU 时间 (转换到相机时基) 并唤醒消费者
  double timestamp_imu_inC = msg.timestamp - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
  {
    std::lock_guard<std::mutex> lk(imu_time_mtx_);
    last_imu_timestamp_inC_ = timestamp_imu_inC;
    imu_new_flag_ = true;
  }
  imu_cv_.notify_one();
}

void FGVisualizer::startCore() {
  // 启动统一消费者线程（只启动一次）
  if (!running_) {
    running_ = true;
    while (running_) {
      std::unique_lock<std::mutex> lk(imu_time_mtx_);
      imu_cv_.wait(lk, [&]{ return !running_ || imu_new_flag_; });
      if (!running_) break;
      imu_new_flag_ = false;
      double imu_ts_inC = last_imu_timestamp_inC_;
      lk.unlock();

      // 消费相机队列（固定延迟）
      while (true) {
        ov_core::CameraData front;
          {
            std::lock_guard<std::mutex> qlk(camera_queue_mtx);
            if (camera_queue.empty()) break;
            if ((camera_queue.front().timestamp + cam_fixed_latency_) >= imu_ts_inC) break;
            front = camera_queue.front();
            camera_queue.pop_front();
          }
          auto t0 = boost::posix_time::microsec_clock::local_time();
          double update_dt_ms = 1000.0 * (imu_ts_inC - front.timestamp);
          last_images_timestamp = front.timestamp;
          _app->feed_measurement_camera(front);
          auto t1 = boost::posix_time::microsec_clock::local_time();
          publish_cameras();
          visualize();
          auto t2 = boost::posix_time::microsec_clock::local_time();
          
          double time_slam = (t1 - t0).total_microseconds() * 1e-6;
          double time_total = (t2 - t0).total_microseconds() * 1e-6;
          _dash_board->setNameAndValue(0, "TIME", time_total);
          _dash_board->setNameAndValue(1, "TIME_SLAM", time_slam);
          _dash_board->setNameAndValue(2, "HZ", 1.0 / std::max(1e-6, time_total));
          _dash_board->setNameAndValue(3, "UPDATE_DT_MS", update_dt_ms);
          _dash_board->print();
      }
    }
  }
}

void FGVisualizer::visualize_odometry(double timestamp) {
  // Return if we have not inited
  if (!_app->initialized())
    return;

  int64_t time_us = int64_t(timestamp * 1e6);

  // Get fast propagate state at the desired timestamp
  std::shared_ptr<State> state = _app->get_state();
  Eigen::Matrix<double, 13, 1> state_plus = Eigen::Matrix<double, 13, 1>::Zero();
  Eigen::Matrix<double, 12, 12> cov_plus = Eigen::Matrix<double, 12, 12>::Zero();
  if (!_app->get_propagator()->fast_state_propagate(state, timestamp, state_plus, cov_plus))
    return;

  Eigen::Matrix4f T_w_i = Eigen::Matrix4f::Identity();
  T_w_i.block(0, 0, 3, 3) = Eigen::Quaternionf(state_plus(3), state_plus(0), state_plus(1), state_plus(2)).toRotationMatrix();
  T_w_i.block(0, 3, 3, 1) = Eigen::Vector3f(state_plus(4), state_plus(5), state_plus(6));

  (*odom_out_ptr_) << std::fixed << std::setprecision(6) << timestamp << " "
                   << state_plus(4) << " " << state_plus(5) << " " << state_plus(6) << " "
                    << state_plus(0) << " " << state_plus(1) << " " << state_plus(2) << " " << state_plus(3) << "\n";

  /***************************************六自由度数据输出（异步队列）start*************************************/
  Eigen::Vector3f pos = T_w_i.block<3,1>(0, 3);
  Eigen::Vector3f euler = T_w_i.block<3,3>(0, 0).eulerAngles(0, 1, 2); // RPY
  Pose6DFrame f;
  f.data = {pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]};
  f.timestamp = timestamp;
  {
    std::lock_guard<std::mutex> lk(pose_mtx_);
    pose_queue_.push_back(std::move(f));
  }
  pose_cv_.notify_one();
  /***************************************六自由度数据输出（异步队列）end*************************************/

  if (!_viz) return;
  poses_imu_odom.push_back({timestamp, T_w_i});
  _viz->showPose("pose_imu_odom", time_us, T_w_i, "LOCAL_WORLD", "IMU");

  const double window_duration = 10.0;               // 秒
  const double window_start = timestamp - window_duration;
  while (!poses_imu_odom.empty() && poses_imu_odom.front().first < window_start) {
    poses_imu_odom.pop_front();
  }

  std::vector<Eigen::Matrix4f> path_imu;
  path_imu.reserve(poses_imu_odom.size());

  const size_t max_pts = 200;
  const size_t N = poses_imu_odom.size();

  if (N <= max_pts) {
    path_imu.reserve(N);
    for (const auto& ps : poses_imu_odom) {
      path_imu.push_back(ps.second);
    }
  } else {
    const size_t stride = (N + max_pts - 1) / max_pts;
    path_imu.reserve(max_pts);
    
    // 直接按索引采样，避免遍历全部元素
    for (size_t i = 0; i < max_pts && (i * stride) < N; ++i) {
      const size_t idx = i * stride;
      path_imu.push_back(poses_imu_odom[idx].second);  // O(1) 随机访问
    }
  }

  _viz->showPath("path_imu_odom_win10", time_us, path_imu, "LOCAL_WORLD");
}

void FGVisualizer::feedStereo(ov_core::CameraData& msg) {
  if (_app->get_params().use_mask) {
    msg.masks.push_back(_app->get_params().masks.at(0));
    msg.masks.push_back(_app->get_params().masks.at(1));
  } else {
    msg.masks.emplace_back(cv::Mat::zeros(msg.images[0].rows, msg.images[0].cols, CV_8UC1));
    msg.masks.emplace_back(cv::Mat::zeros(msg.images[1].rows, msg.images[1].cols, CV_8UC1));
  }

  {
    std::lock_guard<std::mutex> lk(camera_queue_mtx);
    camera_queue.push_back(std::move(msg));
  }
}

void FGVisualizer::stopDrivers() {
  // 停止消费者
  if (running_) {
    running_ = false;
    imu_cv_.notify_all();
    if (consumer_thread_.joinable()) consumer_thread_.join();
  }
  // 停止姿态线程
  if (pose_thread_running_) {
    pose_thread_running_ = false;
    pose_cv_.notify_all();
    if (pose_thread_.joinable()) pose_thread_.join();
  }
  // 关闭姿态串口
  if (pose_serial_open_ && pose_serial_fd_>=0) { ::close(pose_serial_fd_); pose_serial_fd_=-1; pose_serial_open_=false; }
}

static int openPoseSerial(const std::string &dev){
  int fd = ::open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) { perror("open pose serial"); return -1; }
  termios tty{}; if (tcgetattr(fd,&tty)!=0){ perror("tcgetattr pose"); ::close(fd); return -1; }
  cfsetospeed(&tty,B115200); cfsetispeed(&tty,B115200);
  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; tty.c_iflag &= ~IGNBRK; tty.c_lflag=0; tty.c_oflag=0; tty.c_cc[VMIN]=0; tty.c_cc[VTIME]=0;
  tty.c_cflag |= (CLOCAL | CREAD); tty.c_cflag &= ~(PARENB | PARODD); tty.c_cflag &= ~CSTOPB; tty.c_cflag &= ~CRTSCTS;
  if (tcsetattr(fd,TCSANOW,&tty)!=0){ perror("tcsetattr pose"); ::close(fd); return -1; }
  return fd;
}

void FGVisualizer::startPoseThread() {
  if (pose_thread_running_) return;
  pose_thread_running_ = true;
  pose_thread_ = std::thread([&]{
    // 打开串口一次
    if (!pose_serial_open_) {
      pose_serial_fd_ = openPoseSerial(pose_serial_device_);
      pose_serial_open_ = pose_serial_fd_ >= 0;
      if (!pose_serial_open_) {
        PRINT_ERROR("Pose serial open failed: %s\n", pose_serial_device_.c_str());
      } else {
        PRINT_INFO("Pose serial opened: %s\n", pose_serial_device_.c_str());
      }
    }
    // 发送主循环
    while (pose_thread_running_) {
      Pose6DFrame item;
      {
        std::unique_lock<std::mutex> lk(pose_mtx_);
        pose_cv_.wait(lk, [&]{ return !pose_thread_running_ || !pose_queue_.empty(); });
        if (!pose_thread_running_) break;
        if (pose_queue_.empty()) continue;
        item = std::move(pose_queue_.front());
        pose_queue_.pop_front();
      }

      if (!pose_serial_open_ || pose_serial_fd_ < 0) continue;

      std::vector<uint8_t> frame;
      frame.reserve(2 + item.data.size()*sizeof(float) + 1);
      frame.push_back(0xAA);
      frame.push_back(0x55);
      uint8_t* data_ptr = reinterpret_cast<uint8_t*>(item.data.data());
      frame.insert(frame.end(), data_ptr, data_ptr + item.data.size()*sizeof(float));
      frame.push_back(0x0D);
      ssize_t written = ::write(pose_serial_fd_, frame.data(), frame.size());
      (void)written;
    }
  });
}

void FGVisualizer::run() {
  startPoseThread();
  startCore();
}

void FGVisualizer::visualize() {
  // Return if we have already visualized
  if (last_visualization_timestamp == _app->get_state()->_timestamp && _app->initialized())
    return;
  last_visualization_timestamp = _app->get_state()->_timestamp;

  publish_images();

  // Return if we have not inited
  if (!_app->initialized())
    return;

  if (!start_time_set) {
    rT1 = boost::posix_time::microsec_clock::local_time();
    start_time_set = true;
  }

  // publish state
  publish_state();

  // publish points
  publish_features();
}

void FGVisualizer::publish_raw_imu(const ov_core::ImuData& s) {
  if (!_viz) return;

  _viz->publishIMU("raw_imu", int64_t(s.timestamp * 1e6), "IMU",
                  s.am, s.wm, Eigen::Quaterniond(s.Rm), s.hm);

  Eigen::Matrix4f T_w_imu = Eigen::Matrix4f::Identity();
  T_w_imu.block(0, 0, 3, 3) = s.Rm.cast<float>();
  _viz->showPose("imu_angle", int64_t(s.timestamp * 1e6), T_w_imu, "LOCAL_WORLD", "IMU_R");
}

void FGVisualizer::publish_state() {
  int64_t time_us = (last_images_timestamp * 1e6);
  std::shared_ptr<State> state = _app->get_state();

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;

  Eigen::Matrix4d poseIinM = Eigen::Matrix4d::Identity();
  Eigen::Quaterniond q_IinM(state->_imu->quat()(3), state->_imu->quat()(0), state->_imu->quat()(1), state->_imu->quat()(2));
  Eigen::Vector3d t_IinM = state->_imu->pos();
  poseIinM.block(0, 0, 3, 3) = q_IinM.toRotationMatrix();
  poseIinM.block(0, 3, 3, 1) = t_IinM;
  Eigen::Matrix4f poseIinM_f = poseIinM.cast<float>();

  (*odom_cam_ptr_) << std::fixed << std::setprecision(6) << last_images_timestamp << " "
                   << t_IinM.x() << " " << t_IinM.y() << " " << t_IinM.z() << " "
                   << q_IinM.x() << " " << q_IinM.y() << " " << q_IinM.z() << " " << q_IinM.w() << "\n";

  if (!_viz) return;
  _viz->showPose("pose_imu", time_us, poseIinM_f, "LOCAL_WORLD", "IMU");

  poses_imu_dq.push_back(std::pair<double, Eigen::Matrix4f>(timestamp_inI, poseIinM_f));

  const double window_duration = 10.0;               // 秒
  const double window_start = timestamp_inI - window_duration;
  while (!poses_imu_dq.empty() && poses_imu_dq.front().first < window_start) {
    poses_imu_dq.pop_front();
  }

  std::vector<Eigen::Matrix4f> path_imu;
  path_imu.reserve(poses_imu_dq.size());

  const size_t max_pts = 200;
  const size_t N = poses_imu_dq.size();

  if (N <= max_pts) {
    path_imu.reserve(N);
    for (const auto& ps : poses_imu_dq) {
      path_imu.push_back(ps.second);
    }
  } else {
    const size_t stride = (N + max_pts - 1) / max_pts;
    path_imu.reserve(max_pts);
    
    // 直接按索引采样，避免遍历全部元素
    for (size_t i = 0; i < max_pts && (i * stride) < N; ++i) {
      const size_t idx = i * stride;
      path_imu.push_back(poses_imu_dq[idx].second);  // O(1) 随机访问
    }
  }

  _viz->showPath("path_imu_win10", time_us, path_imu, "LOCAL_WORLD");
}

void FGVisualizer::publish_features() {
  if (!_viz) return;

  int64_t time_us = (last_images_timestamp * 1e6);

  // Get our good MSCKF features
  std::vector<Eigen::Vector3d> feats_msckf = _app->get_good_features_MSCKF();
  std::vector<std::vector<float>> feats_msckf_pcd;
  std::vector<std::vector<uint8_t>> feats_msckf_color;
  for (const auto &feat : feats_msckf) {
    std::vector<float> pcd = {feat.x(), feat.y(), feat.z()};
    std::vector<uint8_t> color = {255, 0, 0, 255};
    feats_msckf_pcd.push_back(pcd);
    feats_msckf_color.push_back(color);
  }
  
  _viz->showPointCloud("points_msckf", time_us, feats_msckf_pcd, feats_msckf_color, "LOCAL_WORLD");

  // Get our good SLAM features
  std::vector<Eigen::Vector3d> feats_slam = _app->get_features_SLAM();
  std::vector<std::vector<float>> feats_slam_pcd;
  std::vector<std::vector<uint8_t>> feats_slam_color;
  for (auto &feat : feats_slam) {
    std::vector<float> pcd = {feat.x(), feat.y(), feat.z()};
    std::vector<uint8_t> color = {0, 255, 0, 255};
    feats_slam_pcd.push_back(pcd);
    feats_slam_color.push_back(color);
  }
  _viz->showPointCloud("points_slam", time_us, feats_slam_pcd, feats_slam_color, "LOCAL_WORLD");


  // Get our good ARUCO features
  std::vector<Eigen::Vector3d> feats_aruco = _app->get_features_ARUCO();
  std::vector<std::vector<float>> feats_aruco_pcd;
  std::vector<std::vector<uint8_t>> feats_aruco_color;
  for (auto &feat : feats_aruco) {
    std::vector<float> pcd = {feat.x(), feat.y(), feat.z()};
    std::vector<uint8_t> color = {0, 0, 255, 255};
    feats_aruco_pcd.push_back(pcd);
    feats_aruco_color.push_back(color);
  }
  _viz->showPointCloud("points_aruco", time_us, feats_aruco_pcd, feats_aruco_color, "LOCAL_WORLD");
}

void FGVisualizer::publish_cameras() {
  if (!_viz) return;

  int64_t time_us = (last_images_timestamp * 1e6);

  // publish extrinsics
  for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
    Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
    T_CtoI.block(0, 0, 3, 3) = _app->get_state()->_calib_IMUtoCAM.at(i)->Rot().transpose();
    T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * _app->get_state()->_calib_IMUtoCAM.at(i)->pos();
    _viz->showPose("T_C" + std::to_string(i) + "toI", time_us, T_CtoI.cast<float>(), "IMU", "CAM_" + std::to_string(i));
  }

  // publish intrinsics
  for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
    auto calib = _app->get_state()->_cam_intrinsics_cameras.at(i);
    if (calib) {
      cv::Matx33d Kcv = calib->get_K();
      Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
      K(0, 0) = Kcv(0, 0); // fx
      K(1, 1) = Kcv(1, 1); // fy
      K(0, 2) = Kcv(0, 2); // cx
      K(1, 2) = Kcv(1, 2); // cy
      int width = calib->w();
      int height = calib->h();
      Eigen::Matrix<float, 3, 4> P;
      P.setZero();
      P.block<3, 3>(0, 0) = K;
      _viz->showCameraCalibration("intrinsics_" + std::to_string(i), time_us, "CAM_" + std::to_string(i), K, width, height, P);
    } else {
      PRINT_ERROR("Camera intrinsics for camera %d are not set.\n", i);
    }
  }
}

void FGVisualizer::publish_images() {
  if (!_viz) return;

  int64_t time_us = (last_images_timestamp * 1e6);

  // Return if we have already visualized
  if (_app->get_state() == nullptr)
    return;
  if (last_visualization_timestamp_image == _app->get_state()->_timestamp && _app->initialized())
    return;
  last_visualization_timestamp_image = _app->get_state()->_timestamp;

  // Get our image of history tracks
  cv::Mat img_history = _app->get_historical_viz_image();
  if (img_history.empty())
    return;

  cv::Mat img_history_downsampled;
  cv::resize(img_history, img_history_downsampled, cv::Size(img_history.cols / 2, img_history.rows / 2));

  _viz->showImage("track_images", time_us, img_history_downsampled, "STEREO", true);
}

void FGVisualizer::show_image(const std::string &topic_nm, const int64_t &usec, const cv::Mat &viz_img, const std::string &parent_frm) {
  if (!_viz) return;
  _viz->showImage(topic_nm, usec, viz_img, parent_frm, true);
}