#include "FGVisualizer.h"

#include "core/VioManager.h"
#include "sim/Simulator.h"
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
#include <librealsense2/rs.hpp>

#include "serial_imu/ImuDriver.h"
#include "camera_v4l2/V4L2CameraDriver.h"
#include "calibration/apriltags.h"
#include "calibration/aprilgrid.h"

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

FGVisualizer::FGVisualizer(std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim)
  : _app(app), _sim(sim), thread_update_running(false) {

  create_debug_dir();
    
  _viz = std::make_shared<foxglove_viz::Visualizer>(8088, 2);
  _dash_board = std::make_shared<DashBoard>();
}


void FGVisualizer::runRealsenseIO() {
  // ===== 参数 =====
  constexpr int FE_W = 848, FE_H = 800, FE_FPS = 30; // T265 典型设置
  constexpr double CAM_FIXED_LATENCY = 0.030;         // 相机固定延迟（30ms）
  const bool SAVE_DEBUG_IMAGES = is_debug;            // 打开调试图保存
  const std::string left_dir  = debug_dir + "/images/left/";
  const std::string right_dir = debug_dir + "/images/right/";
  fs::create_directories(left_dir);
  fs::create_directories(right_dir);

  // ===== RealSense：单管道、统一回调 =====
  rs2::pipeline pipe;
  rs2::config   cfg;
  cfg.enable_stream(RS2_STREAM_FISHEYE, 1, FE_W, FE_H, RS2_FORMAT_Y8, FE_FPS);
  cfg.enable_stream(RS2_STREAM_FISHEYE, 2, FE_W, FE_H, RS2_FORMAT_Y8, FE_FPS);
  cfg.enable_stream(RS2_STREAM_GYRO,  -1, 0, 0, RS2_FORMAT_MOTION_XYZ32F, 200);
  cfg.enable_stream(RS2_STREAM_ACCEL, -1, 0, 0, RS2_FORMAT_MOTION_XYZ32F,  62);

  // 线程安全 frameset 队列（回调 → 图像线程）
  rs2::frame_queue img_q(60);

  // IMU 入队（回调 → 主循环）
  struct GyroS { double t; float x,y,z; };
  struct AccS  { double t; float x,y,z; };
  std::deque<GyroS> qg;  std::mutex mtxg;
  std::deque<AccS>  qa;  std::mutex mtxa;

  // 启动并注册回调（只入队，吞异常）
  rs2::pipeline_profile profile;
  try {
    profile = pipe.start(cfg, [&](const rs2::frame& f){
      try {
        if (auto mf = f.as<rs2::motion_frame>()) {
          const double t = mf.get_timestamp() * 1e-3; // 设备时钟: ms→s
          const rs2_vector v = mf.get_motion_data();
          const rs2_stream  st = mf.get_profile().stream_type();
          if (st == RS2_STREAM_GYRO) {
            std::lock_guard<std::mutex> lk(mtxg);
            qg.push_back({t, v.x, v.y, v.z});            // rad/s
            while (qg.size() > 6000) qg.pop_front();     // ~30s 缓冲
          } else if (st == RS2_STREAM_ACCEL) {
            std::lock_guard<std::mutex> lk(mtxa);
            qa.push_back({t, v.x, v.y, v.z});            // m/s^2
            while (qa.size() > 3000) qa.pop_front();     // ~48s 缓冲
          }
        } else if (auto fs = f.as<rs2::frameset>()) {
          img_q.enqueue(fs);
        }
      } catch (...) {}
    });
  } catch (const rs2::error& e) {
    PRINT_ERROR("RealSense start failed: %s @ %s\n", e.what(), e.get_failed_function());
    return;
  }

  PRINT_INFO("Started device: %s | FW %s\n",
             profile.get_device().get_info(RS2_CAMERA_INFO_NAME),
             profile.get_device().get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION));

  // ===== 工具：对 accel 线性插值到 tg（短队列，持锁扫描）=====
  auto interp_acc = [&](double tg, AccS& out)->bool {
    std::lock_guard<std::mutex> lk(mtxa);
    if (qa.empty()) return false;
    if (tg <= qa.front().t) { out = qa.front(); out.t = tg; return true; }
    if (tg >= qa.back().t)  { out = qa.back();  out.t = tg; return true; }
    for (size_t i=1;i<qa.size();++i) {
      if (qa[i-1].t <= tg && tg <= qa[i].t) {
        const auto& a0 = qa[i-1]; const auto& a1 = qa[i];
        const double dt = a1.t - a0.t; if (dt <= 0) { out = a0; out.t = tg; return true; }
        const double w = (tg - a0.t)/dt;
        out.t = tg;
        out.x = float((1-w)*a0.x + w*a1.x);
        out.y = float((1-w)*a0.y + w*a1.y);
        out.z = float((1-w)*a0.z + w*a1.z);
        return true;
      }
    }
    return false;
  };

  // ===== 保存线程（可选，无压缩 PNG）=====
  struct SaveJob { std::string path; cv::Mat img; };
  std::queue<SaveJob> save_q;
  std::mutex save_mtx; std::condition_variable save_cv;
  std::atomic<bool> saving{SAVE_DEBUG_IMAGES};

  std::thread th_save;
  if (SAVE_DEBUG_IMAGES) {
    th_save = std::thread([&]{
      while (saving) {
        SaveJob job;
        {
          std::unique_lock<std::mutex> lk(save_mtx);
          save_cv.wait(lk, [&]{ return !save_q.empty() || !saving; });
          if (!saving && save_q.empty()) break;
          job = std::move(save_q.front()); save_q.pop();
        }
        try {
          std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
          cv::imwrite(job.path, job.img, params);
        } catch (...) {}
      }
    });
  }

  // ===== 图像线程：从 img_q 取 → 节流 → clone → 入 camera_queue（有界）=====
  std::atomic<bool> running{true};

  // 简易信号退出（Ctrl-C）——可选
  g_run_flag = &running;
  ::signal(SIGINT,  rsio_on_signal);
  ::signal(SIGTERM, rsio_on_signal);

  double last_left_t = -1.0, ema_fps = -1.0; const double ema_alpha = 0.2;
  uint64_t fps_cnt = 0;
  double last_cam_ts_for_throttle = -1.0;

  auto th_img = std::thread([&]{
    const double time_delta = 1.0 / _app->get_params().track_frequency;
    static const size_t CAMERA_QUEUE_CAP = 120; // ~6s @20Hz

    while (running) {
      rs2::frameset fs;
      if (!img_q.poll_for_frame(&fs)) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

      auto fL = fs.get_fisheye_frame(1);
      auto fR = fs.get_fisheye_frame(2);
      if (!fL || !fR) continue;

      const double tL = fL.get_timestamp() * 1e-3;
      const double tR = fR.get_timestamp() * 1e-3;
      const double ts = std::min(tL, tR);

      // FPS（以左目为准，仅日志）
      if (last_left_t > 0.0) {
        const double dt = tL - last_left_t;
        if (dt > 0) {
          const double inst = 1.0/dt;
          ema_fps = (ema_fps<0)? inst : (ema_alpha*inst + (1-ema_alpha)*ema_fps);
          if ((++fps_cnt % 30) == 0)
            PRINT_INFO("[RS] L-Inst FPS: %.2f | EMA: %.2f\n", inst, ema_fps);
        }
      }
      last_left_t = tL;

      // 节流
      if (last_cam_ts_for_throttle >= 0.0 && ts < last_cam_ts_for_throttle + time_delta) continue;
      last_cam_ts_for_throttle = ts;

      // 组 CameraData（必须 clone）
      ov_core::CameraData cam;
      cam.timestamp  = ts;
      cam.sensor_ids = {0,1};
      cam.images.resize(2);

      cv::Mat imgL(cv::Size(fL.get_width(), fL.get_height()), CV_8UC1, (void*)fL.get_data(), cv::Mat::AUTO_STEP);
      cv::Mat imgR(cv::Size(fR.get_width(), fR.get_height()), CV_8UC1, (void*)fR.get_data(), cv::Mat::AUTO_STEP);
      cam.images[0] = imgL.clone();
      cam.images[1] = imgR.clone();

      if (_app->get_params().use_mask) {
        cam.masks.push_back(_app->get_params().masks.at(0));
        cam.masks.push_back(_app->get_params().masks.at(1));
      } else {
        cam.masks.emplace_back(cv::Mat::zeros(cam.images[0].rows, cam.images[0].cols, CV_8UC1));
        cam.masks.emplace_back(cv::Mat::zeros(cam.images[1].rows, cam.images[1].cols, CV_8UC1));
      }

      // （可选）异步保存
      if (SAVE_DEBUG_IMAGES) {
        std::ostringstream oss; oss << std::fixed << std::setprecision(6) << ts;
        const std::string ts_str = oss.str();
        {
          std::lock_guard<std::mutex> lk(save_mtx);
          save_q.push({left_dir  + "/" + ts_str + ".png", cam.images[0].clone()});
          save_q.push({right_dir + "/" + ts_str + ".png", cam.images[1].clone()});
        }
        save_cv.notify_one();
      }

      // 入业务队列（有界）
      {
        std::lock_guard<std::mutex> lk(camera_queue_mtx);
        if (camera_queue.size() > CAMERA_QUEUE_CAP) camera_queue.pop_front();
        camera_queue.push_back(std::move(cam));
      }
    }
  });

  // ===== 主循环：IMU 同步 + 固定延迟消费“最多 1 帧”相机 =====
  auto t_last_log = std::chrono::steady_clock::now();
  while (running) {
    // 取一个 gyro 样本
    GyroS g{}; bool got_g = false;
    {
      std::lock_guard<std::mutex> lk(mtxg);
      if (!qg.empty()) { g = qg.front(); qg.pop_front(); got_g = true; }
    }
    if (!got_g) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

    // 对齐 accel → g.t
    AccS aI{};
    if (!interp_acc(g.t, aI)) {
      std::lock_guard<std::mutex> lk(mtxa);
      if (!qa.empty()) {
        const auto& a0 = qa.front(); const auto& a1 = qa.back();
        aI = (std::abs(g.t - a0.t) < std::abs(g.t - a1.t)) ? a0 : a1;
        aI.t = g.t;
      } else {
        aI = {g.t, 0,0,0};
      }
    }

    // 同步 IMU（如果此处只采集，可改成推送到你的输出队列）
    ov_core::ImuData imu;
    imu.timestamp = g.t;
    imu.wm << double(g.x), double(g.y), double(g.z);
    imu.am << double(aI.x), double(aI.y), double(aI.z);
    imu.hm.setZero();
    imu.Rm.setIdentity();

    _app->feed_measurement_imu(imu);
    _viz->publishIMU("raw_imu", int64_t(imu.timestamp * 1e6), "IMU",
                     imu.am, imu.wm, Eigen::Quaterniond(imu.Rm), imu.hm);
    {
      Eigen::Matrix4f T_w_imu = Eigen::Matrix4f::Identity();
      T_w_imu.block(0,0,3,3) = imu.Rm.cast<float>();
      _viz->showPose("imu_angle", int64_t(imu.timestamp * 1e6), T_w_imu, "LOCAL_WORLD", "IMU_R");
    }

    // 固定延迟 + 快进丢旧帧（本 tick 最多消费 1 帧）
    const double dt_bias = _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
    const double ts_imu_inC = imu.timestamp - dt_bias;
    const double L = CAM_FIXED_LATENCY;
    const double MAX_BACKLOG = 0.20; // 允许最大落后 200ms

    {
      std::lock_guard<std::mutex> lk(camera_queue_mtx);
      while (camera_queue.size() > 1 &&
             camera_queue.front().timestamp + L < ts_imu_inC - MAX_BACKLOG) {
        camera_queue.pop_front();
      }
    }

    ov_core::CameraData cam; bool consume = false;
    {
      std::lock_guard<std::mutex> lk(camera_queue_mtx);
      if (!camera_queue.empty() &&
          camera_queue.front().timestamp + L < ts_imu_inC) {
        cam = std::move(camera_queue.front());
        camera_queue.pop_front();
        consume = true;
      }
    }
    if (consume) {
      auto t0 = std::chrono::high_resolution_clock::now();
      double update_dt_ms = 1000.0 * (ts_imu_inC - cam.timestamp);
      last_images_timestamp = cam.timestamp;
      last_images = cam.images;

      _app->feed_measurement_camera(cam);
      auto t1 = std::chrono::high_resolution_clock::now();
      publish_cameras();
      visualize();
      auto t2 = std::chrono::high_resolution_clock::now();

      double time_slam  = std::chrono::duration<double>(t1 - t0).count();
      double time_total = std::chrono::duration<double>(t2 - t0).count();
      _dash_board->setNameAndValue(0, "TIME", time_total);
      _dash_board->setNameAndValue(1, "TIME_SLAM", time_slam);
      _dash_board->setNameAndValue(2, "HZ", 1.0 / std::max(1e-6, time_total));
      _dash_board->setNameAndValue(3, "UPDATE_DT_MS", update_dt_ms);
      _dash_board->print();
    }

    // 略微让步，避免满核空转
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // ===== 收尾：停回调 → 停线程 → join =====
  pipe.stop();                 // 1) 先停回调，不再产新帧
  running = false;                 // 2) 让图像线程退出
  if (th_img.joinable()) th_img.join();
  if (SAVE_DEBUG_IMAGES) {
    saving = false; save_cv.notify_all();
    if (th_save.joinable()) th_save.join();
  }
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

  // Publish RMSE if we have it
  if (!gt_states.empty()) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
  }

  // Publish RMSE and NEES if doing simulation
  if (_sim != nullptr) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
    PRINT_INFO(REDPURPLE "NEES: %.3f (deg) orientation\n" RESET, summed_nees_ori / summed_number);
    PRINT_INFO(REDPURPLE "NEES: %.3f (m) position\n\n" RESET, summed_nees_pos / summed_number);
  }

  // Print the total time
  rT2 = boost::posix_time::microsec_clock::local_time();
  PRINT_INFO(REDPURPLE "TIME: %.3f seconds\n\n" RESET, (rT2 - rT1).total_microseconds() * 1e-6);
}

void FGVisualizer::setDevicesAndLatency(const std::string &imu_dev,
                                        const std::string &cam_dev,
                                        const std::string &pose_dev,
                                        double cam_latency){
  imu_device_ = imu_dev;
  cam_device_ = cam_dev;
  pose_serial_device_ = pose_dev;
  cam_fixed_latency_ = cam_latency;
}

void FGVisualizer::startIMUDriver() {
  using namespace ov_sensors;
  if (imu_driver_) return; // guard
  // 固定延迟已参数化 cam_fixed_latency_

  // 调试输出
  if (!imu_log_file_) {
    imu_log_file_ = std::make_shared<std::ofstream>(debug_dir + "/imu_data.txt", std::ios::app);
    if (imu_log_file_ && imu_log_file_->tellp() == 0) {
      (*imu_log_file_) << "# timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,qw,qx,qy,qz,mag_x,mag_y,mag_z\n";
    }
  }
  auto writeIMUFrameCSV = [&](const ov_core::ImuData &msg) {
    if (!imu_log_file_ || !imu_log_file_->good()) return;
    Eigen::Quaterniond q(msg.Rm);
    (*imu_log_file_) << std::fixed << std::setprecision(6) << msg.timestamp << ","
                     << msg.am(0) << "," << msg.am(1) << "," << msg.am(2) << ","
                     << msg.wm(0) << "," << msg.wm(1) << "," << msg.wm(2) << ","
                     << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
                     << msg.hm(0) << "," << msg.hm(1) << "," << msg.hm(2) << "\n";
    imu_log_file_->flush();
  };

  // 创建并启动驱动
  imu_driver_ = std::make_shared<ImuDriver>(imu_device_);
  imu_driver_->setCallback([&](const ImuSample& s){
    // 转换为 ov_core::ImuData
    ov_core::ImuData msg;
    msg.timestamp = s.timestamp;
    if (s.has_gyro)  msg.wm = s.gyro;
    if (s.has_accel) msg.am = s.accel;
    if (s.has_mag)   msg.hm = s.mag;
    if (s.has_R)     msg.Rm = s.R; else msg.Rm.setIdentity();

    if (is_debug) writeIMUFrameCSV(msg);

    _app->feed_measurement_imu(msg);
    _viz->publishIMU("raw_imu", int64_t(msg.timestamp * 1e6), "IMU",
                     msg.am, msg.wm, Eigen::Quaterniond(msg.Rm), msg.hm);
    visualize_odometry(msg.timestamp);

    Eigen::Matrix4f T_w_imu = Eigen::Matrix4f::Identity();
    T_w_imu.block(0, 0, 3, 3) = msg.Rm.cast<float>();
    _viz->showPose("imu_angle", int64_t(msg.timestamp * 1e6), T_w_imu, "LOCAL_WORLD", "IMU_R");

    // 触发消费相机队列（固定延迟落后条件）
    // 更新 IMU 时间 (转换到相机时基) 并唤醒消费者
    double timestamp_imu_inC = msg.timestamp - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
    {
      std::lock_guard<std::mutex> lk(imu_time_mtx_);
      last_imu_timestamp_inC_ = timestamp_imu_inC;
      imu_new_flag_ = true;
    }
    imu_cv_.notify_one();
  });

  if (!imu_driver_->start()) {
    PRINT_ERROR("Failed to start IMU driver (%s).\n", imu_device_.c_str());
  } else {
    PRINT_INFO("IMU driver started on %s\n", imu_device_.c_str());
  }

  // 启动统一消费者线程（只启动一次）
  if (!running_) {
    running_ = true;
    consumer_thread_ = std::thread([&]{
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
            last_images = front.images;
            _app->feed_measurement_camera(front);
            auto t1 = boost::posix_time::microsec_clock::local_time();
            publish_cameras();
            visualize();
            auto t2 = boost::posix_time::microsec_clock::local_time();

            if (_app->initialized()) {
              OdometryData copy;
              copy.timestamp = front.timestamp;
              copy.sensor_ids = front.sensor_ids;
              copy.images = front.images;
              copy.T_local_imu = poses_imu_dq.back().second;
              {
                std::lock_guard<std::mutex> lk1(april_grid_queue_mtx_);
                if (april_grid_image_queue_.size() >= april_grid_queue_max_) {
                  april_grid_image_queue_.pop_front();
                }
                april_grid_image_queue_.push_back(std::move(copy));
              }
              april_grid_cv_.notify_one();
            }
            
            double time_slam = (t1 - t0).total_microseconds() * 1e-6;
            double time_total = (t2 - t0).total_microseconds() * 1e-6;
            _dash_board->setNameAndValue(0, "TIME", time_total);
            _dash_board->setNameAndValue(1, "TIME_SLAM", time_slam);
            _dash_board->setNameAndValue(2, "HZ", 1.0 / std::max(1e-6, time_total));
            _dash_board->setNameAndValue(3, "UPDATE_DT_MS", update_dt_ms);
            _dash_board->print();
        }
      }
    });
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

void FGVisualizer::startCameraDriver() {
  using namespace ov_sensors;
  if (cam_driver_) return; // 已启动
  V4L2CameraDriver::Config cfg; // 默认参数，可后续从 YAML 读取
  cfg.device = cam_device_; // 可配置
  cfg.track_frequency = _app->get_params().track_frequency;
  cam_driver_ = std::make_shared<V4L2CameraDriver>(cfg);
  cam_driver_->setCallback([&](const CameraFrame& f){
    ov_core::CameraData msg;
    msg.timestamp = f.timestamp_raw;
    msg.sensor_ids = f.sensor_ids;
    // clone 防止底层复用缓冲
    msg.images.resize(f.images.size());
    for (size_t i=0;i<f.images.size();++i) msg.images[i] = f.images[i].clone();
    if (_app->get_params().use_mask) {
      msg.masks.push_back(_app->get_params().masks.at(0));
      msg.masks.push_back(_app->get_params().masks.at(1));
    } else {
      msg.masks.emplace_back(cv::Mat::zeros(f.images[0].rows, f.images[0].cols, CV_8UC1));
      msg.masks.emplace_back(cv::Mat::zeros(f.images[1].rows, f.images[1].cols, CV_8UC1));
    }

    // 再入主业务队列（可 move）
    {
      std::lock_guard<std::mutex> lk(camera_queue_mtx);
      constexpr size_t MAX_Q = 240; // ~12s @20Hz
      if (camera_queue.size() >= MAX_Q) camera_queue.pop_front();
      camera_queue.push_back(std::move(msg));
    }
  });
  if (!cam_driver_->start()) {
    PRINT_ERROR("Camera driver start failed.\n");
  } else {
    PRINT_INFO("Camera driver started on %s\n", cam_device_.c_str());
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
  // 停止AprilGrid检测线程
  if (april_grid_running_) {
    april_grid_running_ = false;
    april_grid_cv_.notify_all();
    if (april_grid_thread_.joinable()) april_grid_thread_.join();
  }
  // 停止驱动
  if (imu_driver_) { imu_driver_->stop(); imu_driver_.reset(); }
  if (cam_driver_) { cam_driver_->stop(); cam_driver_.reset(); }
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
  startCameraDriver();
  startIMUDriver();
  startPoseThread();
  startAprilGridLocalization();
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
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

void FGVisualizer::publish_state() {
  int64_t time_us = (last_images_timestamp * 1e6);
  std::shared_ptr<State> state = _app->get_state();

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;

  Eigen::Matrix4d poseIinM = Eigen::Matrix4d::Identity();
  Eigen::Quaterniond q_IinM(state->_imu->quat()(3), state->_imu->quat()(0), state->_imu->quat()(1), state->_imu->quat()(2));
  poseIinM.block(0, 0, 3, 3) = q_IinM.toRotationMatrix();
  poseIinM.block(0, 3, 3, 1) = state->_imu->pos();
  Eigen::Matrix4f poseIinM_f = poseIinM.cast<float>();

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

// AprilGrid检测和PnP定位实现
void FGVisualizer::startAprilGridLocalization() {
  if (april_grid_running_) return;

  // 初始化AprilGrid
  std::string april_grid_config = std::string(PROJ_DIR) + "/thirdparty/kalibrlib/apps/others/aprilgrid.yaml";
  april_grid_ = std::make_shared<CAMERA_CALIB::AprilGrid>(april_grid_config);

  PRINT_INFO("AprilGrid configuration loaded from: %s\n", april_grid_config.c_str());

  april_grid_running_ = true;
  april_grid_thread_ = std::thread(&FGVisualizer::aprilGridDetectionThread, this);
  PRINT_INFO("AprilGrid detection thread started\n");
}

void FGVisualizer::aprilGridDetectionThread() {
  PRINT_INFO("AprilGrid detection thread running...\n");

  const int numTags = april_grid_->getTagCols() * april_grid_->getTagRows();
  PRINT_INFO("AprilGrid detection thread initialized with %d tags.\n", numTags);
  CAMERA_CALIB::ApriltagDetector detector(numTags);

  PRINT_INFO("AprilGrid detection thread ready to process images.\n");

  while (april_grid_running_) {
    OdometryData cam_data;
    {
      std::unique_lock<std::mutex> lk(april_grid_queue_mtx_);
      april_grid_cv_.wait(lk, [&] { return !april_grid_running_ || !april_grid_image_queue_.empty(); });
      if (!april_grid_running_) break;
      if (april_grid_image_queue_.empty()) continue;

      cam_data = std::move(april_grid_image_queue_.front());
      april_grid_image_queue_.pop_front();
    }

    // 基本防御性检查
    if (cam_data.images.size() < 2) {
      PRINT_DEBUG("[APRIL_GRID] invalid images vector size=%zu\n", cam_data.images.size());
      continue;
    }
    if (cam_data.images[0].empty() || cam_data.images[1].empty()) {
      PRINT_DEBUG("[APRIL_GRID] empty image. L=%d R=%d\n", cam_data.images[0].empty(), cam_data.images[1].empty());
      continue;
    }

    PRINT_DEBUG("[APRIL_GRID] processing ts=%.6f size=%dx%d\n",
                cam_data.timestamp, cam_data.images[0].cols, cam_data.images[0].rows);


    // 检测AprilGrid - 左右相机
    CAMERA_CALIB::CalibCornerData corners_left_good, corners_left_bad;
    CAMERA_CALIB::CalibCornerData corners_right_good, corners_right_bad;

    if (cam_data.images[0].empty()) {
      PRINT_ERROR("AprilGrid left image is empty.\n");
    }

    detector.detectTags(cam_data.images[0], corners_left_good.corners, corners_left_good.corner_ids, corners_left_good.radii,
                        corners_left_bad.corners, corners_left_bad.corner_ids, corners_left_bad.radii);

    PRINT_DEBUG("[APRIL_GRID] tags detection for right image ts=%.6f\n", cam_data.timestamp);

    detector.detectTags(cam_data.images[1], corners_right_good.corners, corners_right_good.corner_ids, corners_right_good.radii,
                        corners_right_bad.corners, corners_right_bad.corner_ids, corners_right_bad.radii);

    // 检查是否有足够的角点
    if (corners_left_good.corner_ids.size() < 12 && corners_right_good.corner_ids.size() < 12) {
      continue; // 角点太少，跳过
    }

    double scale = (_app->get_params().downsample_cameras)? 0.5 : 1.0;

    std::vector<Eigen::Vector2d> point2d_left_normalized;
    std::vector<Eigen::Vector3d> point3d_left;
    auto intrinsic_left = _app->get_state()->_cam_intrinsics_cameras.at(0);
    if (!intrinsic_left) {
      PRINT_DEBUG("[APRIL_GRID] left intrinsics not ready, skip\n");
      continue;
    }
    cv::Matx33d K_left_cv = intrinsic_left->get_K();
    std::vector<double> cam_left;
    cam_left.push_back(K_left_cv(0, 0)); // fx
    cam_left.push_back(K_left_cv(1, 1)); // fy
    cam_left.push_back(K_left_cv(0, 2)); // cx
    cam_left.push_back(K_left_cv(1, 2)); // cy
    for (size_t i=0;i<corners_left_good.corner_ids.size();++i) {
      int tag_id = corners_left_good.corner_ids[i];
      point3d_left.push_back(april_grid_->aprilgrid_corner_pos_3d[tag_id].head<3>());
      Eigen::Vector2d pt_norm = intrinsic_left->undistort_d(corners_left_good.corners[i] * scale);

      Eigen::Vector2d pt_homog;
      pt_homog << pt_norm(0) * cam_left[0] + cam_left[2], pt_norm(1) * cam_left[1] + cam_left[3];
      point2d_left_normalized.push_back(pt_homog);
    }

    std::vector<Eigen::Vector2d> point2d_right_normalized;
    std::vector<Eigen::Vector3d> point3d_right;
    auto intrinsic_right = _app->get_state()->_cam_intrinsics_cameras.at(1);
    if (!intrinsic_right) {
      PRINT_DEBUG("[APRIL_GRID] right intrinsics not ready, skip\n");
      continue;
    }
    cv::Matx33d K_right_cv = intrinsic_right->get_K();
    std::vector<double> cam_right;
    cam_right.push_back(K_right_cv(0, 0)); // fx
    cam_right.push_back(K_right_cv(1, 1)); // fy
    cam_right.push_back(K_right_cv(0, 2)); // cx
    cam_right.push_back(K_right_cv(1, 2)); // cy
    for (size_t i=0;i<corners_right_good.corner_ids.size();++i) {
      int tag_id = corners_right_good.corner_ids[i];
      point3d_right.push_back(april_grid_->aprilgrid_corner_pos_3d[tag_id].head<3>());

      Eigen::Vector2d pt_norm = intrinsic_right->undistort_d(corners_right_good.corners[i] * scale);
      Eigen::Vector2d pt_homog;
      pt_homog << pt_norm(0) * cam_right[0] + cam_right[2], pt_norm(1) * cam_right[1] + cam_right[3];
      point2d_right_normalized.push_back(pt_homog);
    }

    Eigen::Matrix4d T_imu_cam_left = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_cam_imu_left = Eigen::Matrix4d::Identity();
    {
      auto calib_left = _app->get_state()->_calib_IMUtoCAM.at(0);
      Eigen::Matrix3d R_imu_to_cam = calib_left->Rot();
      Eigen::Vector3d p_imu_in_cam = calib_left->pos();
      T_cam_imu_left.block<3,3>(0,0) = R_imu_to_cam;
      T_cam_imu_left.block<3,1>(0,3) = p_imu_in_cam;
      Eigen::Matrix3d R_cam_to_imu = R_imu_to_cam.transpose();
      Eigen::Vector3d p_cam_in_imu = -R_cam_to_imu * p_imu_in_cam;
      T_imu_cam_left.block<3,3>(0,0) = R_cam_to_imu;
      T_imu_cam_left.block<3,1>(0,3) = p_cam_in_imu;
    }

    Eigen::Matrix4d T_imu_cam_right = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_cam_imu_right = Eigen::Matrix4d::Identity();
    {
      auto calib_right = _app->get_state()->_calib_IMUtoCAM.at(1);
      Eigen::Matrix3d R_imu_to_cam = calib_right->Rot();
      Eigen::Vector3d p_imu_in_cam = calib_right->pos();
      T_cam_imu_right.block<3,3>(0,0) = R_imu_to_cam;
      T_cam_imu_right.block<3,1>(0,3) = p_imu_in_cam;
      Eigen::Matrix3d R_cam_to_imu = R_imu_to_cam.transpose();
      Eigen::Vector3d p_cam_in_imu = -R_cam_to_imu * p_imu_in_cam;
      T_imu_cam_right.block<3,3>(0,0) = R_cam_to_imu;
      T_imu_cam_right.block<3,1>(0,3) = p_cam_in_imu;
      
    }

    PRINT_DEBUG("[APRIL_GRID] detected tags: L=%zu R=%zu\n",
                corners_left_good.corner_ids.size(), corners_right_good.corner_ids.size());

    Eigen::Matrix4d T_grid_to_cam, T_grid_to_imu;
    if (solvePnP(point2d_left_normalized, point3d_left, cam_left, T_grid_to_cam)) {
      T_grid_to_imu = T_imu_cam_left * T_grid_to_cam;
      PRINT_DEBUG("[APRIL_GRID] Left camera PnP solved, tags=%zu\n", corners_left_good.corner_ids.size());
    } else if (solvePnP(point2d_right_normalized, point3d_right, cam_right, T_grid_to_cam)) {
      T_grid_to_imu = T_imu_cam_right * T_grid_to_cam;
      PRINT_DEBUG("[APRIL_GRID] Right camera PnP solved, tags=%zu\n", corners_right_good.corner_ids.size());
    } else {
      continue; 
    }

    PRINT_DEBUG("[APRIL_GRID] PnP pose estimated\n");

    // 准备用ceres解算新的pose
    bool is_success = optimizeIMUPoseWithCeres(point2d_left_normalized, point2d_right_normalized, 
                                               point3d_left, point3d_right, T_cam_imu_left, T_cam_imu_right,
                                               cam_left, cam_right, T_grid_to_imu);


    Eigen::Matrix4d T_imu_to_grid = Eigen::Matrix4d::Identity();
    T_imu_to_grid.block(0,0,3,3) = T_grid_to_imu.block(0,0,3,3).transpose();
    T_imu_to_grid.block(0,3,3,1) = -T_imu_to_grid.block(0,0,3,3) * T_grid_to_imu.block(0,3,3,1);
    Eigen::Matrix4f T_imu_to_grid_f = T_imu_to_grid.cast<float>();

    // 注意：T_local_grid = T_local_imu * T_imu_to_grid（不需要再取逆）
    Eigen::Matrix4f T_local_grid = cam_data.T_local_imu * T_imu_to_grid_f;

    _viz->showPose("april_grid_pose", int64_t(cam_data.timestamp*1e6), T_imu_to_grid_f, "APRIL_GRID", "IMU2");
    _viz->showPose("april_local_grid_pose", int64_t(cam_data.timestamp*1e6), T_local_grid, "LOCAL", "APRIL_GRID");
    PRINT_DEBUG("[APRIL_GRID] pose updated L=%zu R=%zu right_err=%.2f px%s\n",
                corners_left_good.corner_ids.size(), corners_right_good.corner_ids.size(),
                avg_err_right, is_success? ", ceres_ok":"");
  }

  PRINT_INFO("AprilGrid detection thread stopped\n");
}

bool FGVisualizer::solvePnP(const std::vector<Eigen::Vector2d>& points2d,
                            const std::vector<Eigen::Vector3d>& points3d,
                            const std::vector<double>& cam_params,
                            Eigen::Matrix4d& T_grid_to_cam) {

  if (points2d.size() != points3d.size() || points2d.size() < 4) {
    PRINT_ERROR("solvePnP: need N>=4 and 2D/3D sizes equal, got %zu / %zu\n",
                points2d.size(), points3d.size());
    return false;
  }
  if (cam_params.size() < 4) {
    PRINT_ERROR("solvePnP: cam_params must have at least fx, fy, cx, cy\n");
    return false;
  }

  // 使用RANSAC进行PnP求解
  cv::Mat rvec, tvec;
  cv::Mat inliers;
  cv::Mat K = (cv::Mat_<double>(3, 3) << cam_params[0], 0, cam_params[2],
                                        0, cam_params[1], cam_params[3],
                                        0, 0, 1);

  PRINT_DEBUG("solvePnP: points2d=%zu, points3d=%zu\n", points2d.size(), points3d.size());

  // 将输入点转换为cv::Mat格式
  std::vector<cv::Point2f> cv_points2d;
  for (const auto& pt : points2d) {
    cv_points2d.emplace_back(static_cast<float>(pt.x()), static_cast<float>(pt.y()));
  } 
  std::vector<cv::Point3f> cv_points3d;
  for (const auto& pt : points3d) {
    cv_points3d.emplace_back(static_cast<float>(pt.x()), static_cast<float>(pt.y()), static_cast<float>(pt.z()));
  }

  PRINT_DEBUG("solvePnP: cv_points2d=%zu, cv_points3d=%zu\n", cv_points2d.size(), cv_points3d.size());

  // points2d是归一化平面坐标
  bool success = cv::solvePnPRansac(cv_points3d, cv_points2d, K, cv::Mat::zeros(4, 1, CV_64F),
                                    rvec, tvec, false, 100, 8.0, 0.99, inliers);

  if (!success || inliers.rows < 6) {
    return false; // PnP求解失败或内点太少
  }

  // 将旋转向量转换为旋转矩阵
  cv::Mat R_left;
  cv::Rodrigues(rvec, R_left);

  // 计算左相机到AprilGrid的变换
  T_grid_to_cam = Eigen::Matrix4d::Identity();

  cv::Mat R_eigen, tvec_eigen;
  R_left.convertTo(R_eigen, CV_64F);
  tvec.convertTo(tvec_eigen, CV_64F);

  T_grid_to_cam.block(0, 0, 3, 3) = Eigen::Map<Eigen::Matrix3d>(R_eigen.ptr<double>());
  T_grid_to_cam.block(0, 3, 3, 1) = Eigen::Map<Eigen::Vector3d>(tvec_eigen.ptr<double>());

  return true;
}

// 使用Ceres优化IMU位姿 - 根据新的ReprojectionError设计重写
bool FGVisualizer::optimizeIMUPoseWithCeres(const std::vector<Eigen::Vector2d>& left_points2d,
                                            const std::vector<Eigen::Vector2d>& right_points2d,
                                            const std::vector<Eigen::Vector3d>& left_points3d,
                                            const std::vector<Eigen::Vector3d>& right_points3d,
                                            const Eigen::Matrix4d& T_ItoC_left,
                                            const Eigen::Matrix4d& T_ItoC_right,
                                            const std::vector<double>& cam_left,
                                            const std::vector<double>& cam_right,
                                            Eigen::Matrix4d& T_grid_to_imu) {

  // 创建Ceres优化问题
  ceres::Problem problem;
  ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); // 鲁棒损失函数是不是有问题

  // 初始IMU位姿参数
  // 使用 EigenQuaternionParameterization: 四元数内存布局 [x,y,z,w]
  // pose_q[0-3]: [qx, qy, qz, qw]
  // pose_t[0-2]: 平移 [tx, ty, tz]
  double pose_q[4];
  double pose_t[3] = {0, 0, 0}; // 前4个参数预留，使用后3个存储平移

  // 从当前T_grid_to_imu矩阵初始化参数
  Eigen::Matrix3d R_init = T_grid_to_imu.block<3,3>(0, 0);
  Eigen::Quaterniond q_init(R_init);
  // 存储为 [x,y,z,w]
  pose_q[0] = q_init.x();
  pose_q[1] = q_init.y();
  pose_q[2] = q_init.z();
  pose_q[3] = q_init.w();
  pose_t[0] = T_grid_to_imu(0, 3);
  pose_t[1] = T_grid_to_imu(1, 3);
  pose_t[2] = T_grid_to_imu(2, 3);

  // 添加左相机重投影误差约束
  for (size_t i = 0; i < left_points2d.size() && i < left_points3d.size(); i++) {
    // 转换为Eigen::Vector2d和Eigen::Vector3d
    Eigen::Vector2d point2d = left_points2d[i];
    Eigen::Vector3d point3d = left_points3d[i];

    ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3>(
        new ReprojectionError(point2d, point3d, cam_left, T_ItoC_left));

    problem.AddResidualBlock(cost_function, loss_function, pose_q, pose_t);
  }

  // 添加右相机重投影误差约束
  for (size_t i = 0; i < right_points2d.size() && i < right_points3d.size(); i++) {
    // 转换为Eigen::Vector2d和Eigen::Vector3d
    Eigen::Vector2d point2d = right_points2d[i];
    Eigen::Vector3d point3d = right_points3d[i];

    ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3>(
        new ReprojectionError(point2d, point3d, cam_right, T_ItoC_right));

    problem.AddResidualBlock(cost_function, loss_function, pose_q, pose_t);
  }

  // 添加四元数参数化约束（确保单位四元数）
  // 使用 EigenQuaternionParameterization（期望 [x,y,z,w] 布局）
  ceres::LocalParameterization* quaternion_parameterization = new ceres::EigenQuaternionParameterization;
  problem.SetParameterization(pose_q, quaternion_parameterization);

  // 配置求解器选项
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 10;
  options.num_threads = 2;

  // 求解
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // 从优化后的参数更新变换矩阵（从 [x,y,z,w] 还原 Eigen::Quaterniond(w,x,y,z)）
  Eigen::Quaterniond q_optimized(pose_q[3], pose_q[0], pose_q[1], pose_q[2]);
  T_grid_to_imu.block<3,3>(0, 0) = q_optimized.toRotationMatrix();
  T_grid_to_imu.block<3,1>(0, 3) = Eigen::Vector3d(pose_t[0], pose_t[1], pose_t[2]);

  PRINT_DEBUG("[CERES_OPTIM] Pose optimization: %s, final cost: %.6f, iterations: %d\n",
             summary.BriefReport().c_str(), summary.final_cost, (int)summary.iterations.size());

  return summary.IsSolutionUsable() && summary.final_cost < 1e-3;
}


