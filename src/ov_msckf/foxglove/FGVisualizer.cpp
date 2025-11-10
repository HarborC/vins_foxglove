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

namespace IMUSerial {
Eigen::Matrix3f rpy2R(const Eigen::Vector3f& rpy) {
    float roll = rpy(0);
    float pitch = rpy(1);
    float yaw = rpy(2);

    // 转换为弧度
    float roll_rad = roll * M_PI / 180.0;
    float pitch_rad = pitch * M_PI / 180.0;
    float yaw_rad = yaw * M_PI / 180.0;

    // 构建 Eigen 旋转矩阵
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(yaw_rad,   Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(pitch_rad, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(roll_rad,  Eigen::Vector3f::UnitX());
    
    return R;
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

inline double now_mono_raw() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void FGVisualizer::retrieveIMU() {
  // 配置
  const std::string serial_device = "/dev/ttyS3";
  const double G = 9.81;
  // 串口帧格式与波特率（用于估算分包传输时间）
  constexpr double BAUD = 230400.0;
  constexpr double BITS_PER_BYTE = 10.0;    // 1起始 + 8数据 + 1停止
  constexpr double BYTES_PER_SUBPKT = 11.0; // 你的协议每小包11字节
  constexpr double T_packet = (BITS_PER_BYTE * BYTES_PER_SUBPKT) / BAUD; // ≈0.000478s
  constexpr double T_tx = 4.0 * T_packet;   // 四个分包合计 ≈1.9ms
  // 固定延迟缓冲（把端到端延迟“变长”→“常数”）
  constexpr double CAM_FIXED_LATENCY = 0.030; // 30ms，可按抖动调到 0.03~0.05

  // 调试输出
  std::ofstream imu_file(debug_dir + "/imu_data.txt", std::ios::app);
  imu_file << "# timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,qw,qx,qy,qz,mag_x,mag_y,mag_z\n";
  auto writeIMUFrameCSV = [&](const ov_core::ImuData &msg) {
    Eigen::Quaterniond q(msg.Rm);
    imu_file << std::fixed << std::setprecision(6) << msg.timestamp << ","
             << msg.am(0) << "," << msg.am(1) << "," << msg.am(2) << ","
             << msg.wm(0) << "," << msg.wm(1) << "," << msg.wm(2) << ","
             << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
             << msg.hm(0) << "," << msg.hm(1) << "," << msg.hm(2) << "\n";
    imu_file.flush();
  };

  // 打开串口
  int fd = open(serial_device.c_str(), O_RDWR | O_NOCTTY);
  if (fd == -1) { perror("open serial"); return; }

  PRINT_INFO("open %s success!\n", serial_device.c_str());
  if (isatty(STDIN_FILENO) == 0)
    PRINT_INFO("standard input is not a terminal device\n");
  else
    PRINT_INFO("isatty success!\n");

  // 串口参数
  struct termios tty{}, oldtio;
  if (tcgetattr(fd, &oldtio) != 0) { perror("tcgetattr"); close(fd); return; }
  memset(&tty, 0, sizeof(tty));
  
  tty.c_cflag |= CLOCAL | CREAD;
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;
  tty.c_cflag &= ~PARENB;

  tty.c_cflag &= ~CRTSCTS;

  tty.c_iflag &= ~ICRNL;

  cfsetispeed(&tty, B230400);
  cfsetospeed(&tty, B230400);
  tty.c_cflag &= ~CSTOPB;

  // 关键：一次至少读够一个分包（11字节），降低抖动；超时100ms
  tty.c_cc[VTIME] = 1;
  tty.c_cc[VMIN]  = 11;

  tcflush(fd, TCIFLUSH);

  if (tcsetattr(fd, TCSANOW, &tty) != 0) { perror("tcsetattr"); close(fd); return; }

  // 接收缓冲
  unsigned char chrBuf[100];
  unsigned char chrCnt = 0; // 关键：必须初始化
  char r_buf[1024]; memset(r_buf, 0, sizeof(r_buf));

  IMUSerial::IMUDATA current_imu; // 你已有该结构：含 a/w/angle/h 与 has_* 标志位

  while (true) {
    int ret = read(fd, r_buf, sizeof(r_buf));
    if (ret <= 0) { printf("uart read failed\n"); break; }

    for (int i = 0; i < ret; i++) {
      chrBuf[chrCnt++] = static_cast<unsigned char>(r_buf[i]);
      if (chrCnt < 11) continue;

      // 帧头对齐（保留你原来的判定逻辑）
      if ((chrBuf[0] != 0x55) || ((chrBuf[1] & 0x50) != 0x50)) {
        memmove(&chrBuf[0], &chrBuf[1], 10);
        chrCnt--;
        continue;
      }

      // 解析一个11字节小包
      signed short sData[4];
      memcpy(&sData[0], &chrBuf[2], 8);

      switch (chrBuf[1]) {
        case 0x51: { // 加速度
          for (int j = 0; j < 3; j++)
            current_imu.a(j) = (float)sData[j] / 32768.0f * 16.0f * (float)G; // m/s^2
          current_imu.has_acc = true;
        } break;
        case 0x52: { // 角速度（dps）
          for (int j = 0; j < 3; j++)
            current_imu.w(j) = (float)sData[j] / 32768.0f * 2000.0f; // dps
          current_imu.has_gyro = true;
        } break;
        case 0x53: { // 欧拉角（deg）
          for (int j = 0; j < 3; j++)
            current_imu.angle(j) = (float)sData[j] / 32768.0f * 180.0f;
          current_imu.has_ang = true;
        } break;
        case 0x54: { // 磁力计（原始）
          for (int j = 0; j < 3; j++)
            current_imu.h(j) = (float)sData[j];
          current_imu.has_h = true;
        } break;
        default: break;
      }
      chrCnt = 0;

      // 四个分包到齐 → 组一帧 IMU
      if (current_imu.has_acc && current_imu.has_gyro && current_imu.has_ang && current_imu.has_h) {
        // 用“最后一个分包到达时刻 - T_tx/2”作为采样时刻（居中）
        const double t_last   = now_mono_raw();
        const double t_sample = t_last - 0.5 * T_tx;

        ov_core::ImuData message;
        message.timestamp = t_sample; // 统一 MONOTONIC_RAW 时基
        // gyro: dps -> rad/s
        message.wm << (double)current_imu.w.x(), (double)current_imu.w.y(), (double)current_imu.w.z();
        message.wm *= M_PI / 180.0;
        message.am << (double)current_imu.a.x(), (double)current_imu.a.y(), (double)current_imu.a.z();
        message.hm << (double)current_imu.h.x(), (double)current_imu.h.y(), (double)current_imu.h.z();

        // Rm 仅用于可视化：避免把设备内部姿态参与滤波（易不一致）
        message.Rm = IMUSerial::rpy2R(current_imu.angle).cast<double>();

        if (is_debug) writeIMUFrameCSV(message);

        // 喂给 VIO
        _app->feed_measurement_imu(message);
        _viz->publishIMU("raw_imu", int64_t(message.timestamp * 1e6), "IMU",
                         message.am, message.wm, Eigen::Quaterniond(message.Rm), message.hm);
        visualize_odometry(message.timestamp);

        // 可视化 IMU朝向（此处用单位阵，仅作为占位演示）
        Eigen::Matrix4f T_w_imu = Eigen::Matrix4f::Identity();
        T_w_imu.block(0, 0, 3, 3) = message.Rm.cast<float>();
        _viz->showPose("imu_angle", int64_t(message.timestamp * 1e6), T_w_imu, "LOCAL_WORLD", "IMU_R");

        // 清空等待下一帧
        current_imu = IMUSerial::IMUDATA();

        // 触发消费相机队列（带固定延迟去抖）
        if (!thread_update_running) {
          thread_update_running = true;
          std::thread thread([&, t_sample]() {
            // IMU时刻换到“相机时基下的IMU时刻” （两者都用 MONOTONIC_RAW → 直接用）
            double timestamp_imu_inC = t_sample - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);

            while (true) {
              ov_core::CameraData front;
              {
                std::lock_guard<std::mutex> lck(camera_queue_mtx);
                if (camera_queue.empty()) break;

                // 固定延迟条件：只有“相机戳 + 固定延迟 < imu_inC”才消费
                if ((camera_queue.front().timestamp + CAM_FIXED_LATENCY) >= timestamp_imu_inC) break;

                front = camera_queue.front();
                camera_queue.pop_front();
              }

              auto t0 = boost::posix_time::microsec_clock::local_time();

              double update_dt_ms = 1000.0 * (timestamp_imu_inC - front.timestamp);
              last_images_timestamp = front.timestamp;
              last_images = front.images;

              _app->feed_measurement_camera(front);
              auto t1 = boost::posix_time::microsec_clock::local_time();
              publish_cameras();
              visualize();
              auto t2 = boost::posix_time::microsec_clock::local_time();

              double time_slam = (t1 - t0).total_microseconds() * 1e-6;
              double time_total = (t2 - t0).total_microseconds() * 1e-6;
              _dash_board->setNameAndValue(0, "TIME", time_total);
              _dash_board->setNameAndValue(1, "TIME_SLAM", time_slam);
              _dash_board->setNameAndValue(2, "HZ", 1.0 / time_total);
              _dash_board->setNameAndValue(3, "UPDATE_DT_MS", update_dt_ms);
              _dash_board->print();
              PRINT_INFO(BLUE "[TIME]: %.4f total, %.4f slam (%.1f hz, %.2f ms behind)\n" RESET,
                         time_total, time_slam, 1.0 / time_total, update_dt_ms);
            }
            thread_update_running = false;
          });
          thread.detach();
        }
      }
    }
  }

  imu_file.close();
  close(fd);
  fd = -1;
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


  poses_imu_odom.push_back({timestamp, T_w_i});

  _viz->showPose("pose_imu_odom", time_us, T_w_i, "LOCAL_WORLD", "IMU");

  const double window_duration = 10.0;               // 秒
  const double window_start = timestamp - window_duration;
  while (!poses_imu_odom.empty() && poses_imu_odom.front().first < window_start) {
    poses_imu_odom.pop_front();
  }

  // —— 生成路径（窗口内），可选下采样以限制点数 —— 
  std::vector<Eigen::Matrix4f> path_imu;
  path_imu.reserve(poses_imu_odom.size());

  // 将点数限制到 ~16384（与原逻辑一致）
  const size_t max_pts = 16384;
  const size_t N = poses_imu_odom.size();
  const size_t stride = (N > max_pts) ? (N / max_pts + ((N % max_pts) ? 1 : 0)) : 1;

  size_t idx = 0;
  for (const auto& ps : poses_imu_odom) {
    if ((idx++ % stride) == 0) {
      path_imu.push_back(ps.second);
    }
  }


  _viz->showPath("path_imu_odom_win10", time_us, path_imu, "LOCAL_WORLD");
}

void FGVisualizer::retrieveCamera() {
  // ===== 相机与缓冲配置 =====
  constexpr int WIDTH = 1280;
  constexpr int HEIGHT = 480;
  constexpr int BUFFER_COUNT = 8;
  const std::string device_path = "/dev/video73";

  // 调试保存
  const std::string left_images_dir  = debug_dir + "/images/left/";
  const std::string right_images_dir = debug_dir + "/images/right/";
  fs::create_directories(left_images_dir);
  fs::create_directories(right_images_dir);

  // ===== 打开设备 =====
  int fd = open(device_path.c_str(), O_RDWR);
  if (fd == -1) { perror("open /dev/video"); return; }

  // ===== 查询/打印驱动报告的目标帧率（如果驱动支持） =====
  {
    struct v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_PARM, &parm) == 0) {
      if (parm.parm.capture.timeperframe.numerator != 0) {
        double drv_fps =
            (double)parm.parm.capture.timeperframe.denominator /
            (double)parm.parm.capture.timeperframe.numerator;
        PRINT_INFO("[V4L2] Driver-reported target FPS: %.3f (den=%u, num=%u)\n",
                   drv_fps,
                   parm.parm.capture.timeperframe.denominator,
                   parm.parm.capture.timeperframe.numerator);
      } else {
        PRINT_INFO("[V4L2] VIDIOC_G_PARM returned zero numerator; driver may not report FPS.\n");
      }
    } else {
      PRINT_INFO("[V4L2] VIDIOC_G_PARM not supported on this device.\n");
    }
  }

  // ===== 设置格式（MJPEG：如抖动大可换 RAW/YUYV/MONO8） =====
  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = WIDTH;
  fmt.fmt.pix.height = HEIGHT;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) { perror("VIDIOC_S_FMT"); close(fd); return; }

  // ===== 申请缓冲并 mmap =====
  v4l2_requestbuffers req{};
  req.count = BUFFER_COUNT;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) { perror("VIDIOC_REQBUFS"); close(fd); return; }

  struct Buffer { void* start; size_t length; } buffers[BUFFER_COUNT];
  for (int i = 0; i < BUFFER_COUNT; ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) { perror("VIDIOC_QUERYBUF"); close(fd); return; }

    buffers[i].length = buf.length;
    buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buffers[i].start == MAP_FAILED) { perror("mmap"); close(fd); return; }

    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) { perror("VIDIOC_QBUF"); close(fd); return; }
  }

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) { perror("VIDIOC_STREAMON"); close(fd); return; }

  // ===== 建立 MONOTONIC -> MONOTONIC_RAW 的一次性偏移映射（把相机戳映射到 RAW）=====
  timespec ts_raw{}, ts_mono{};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts_raw);
  clock_gettime(CLOCK_MONOTONIC,     &ts_mono);
  const double offset_raw_minus_mono =
      (ts_raw.tv_sec + ts_raw.tv_nsec * 1e-9) - (ts_mono.tv_sec + ts_mono.tv_nsec * 1e-9);

  // ===== 统计真实采集帧率（基于 V4L2 采集戳）=====
  // 用驱动戳（MONOTONIC）计算瞬时 FPS，再映射到 RAW 做全局统一（不影响差分）
  double last_cam_time_mono = -1.0;
  double ema_fps = -1.0;                 // 指数滑动平均 FPS
  const double ema_alpha = 0.20;         // EMA平滑系数，可调 0.1~0.3
  uint64_t frames_counted = 0;

  // ===== 限频控制（防止前端队列增长造成动态时延）=====
  static double last_timestamp_raw = -1.0;

  // ===== 处理耗时统计 =====
  static double sum_time = 0.0;
  static int count = 0;

  while (true) {
    auto t0 = boost::posix_time::microsec_clock::local_time();

    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) { perror("VIDIOC_DQBUF"); break; }

    // ---- 关键：驱动采集完成时刻（通常 MONOTONIC）----
    const double cam_time_mono = buf.timestamp.tv_sec + buf.timestamp.tv_usec * 1e-6;

    // ---- 采集频率统计（真实 fps）----
    if (last_cam_time_mono > 0.0) {
      double dt = cam_time_mono - last_cam_time_mono;          // 相邻两帧采集时间差
      if (dt > 0.0) {
        double inst_fps = 1.0 / dt;                            // 瞬时 FPS
        if (ema_fps < 0.0) ema_fps = inst_fps;                 // 初始化
        else               ema_fps = ema_alpha * inst_fps + (1.0 - ema_alpha) * ema_fps;

        // 每隔一定帧打印统计（比如每30帧）
        if ((++frames_counted % 30) == 0) {
          PRINT_INFO("[V4L2] Instant FPS: %.3f | EMA FPS: %.3f (dt=%.3f ms)\n",
                     inst_fps, ema_fps, dt * 1000.0);
        }
      }
    }
    last_cam_time_mono = cam_time_mono;

    // ---- 映射到 RAW 时基（与 IMU 统一）----
    const double cam_time_raw = cam_time_mono + offset_raw_minus_mono;

    // ---- 解码 MJPEG ----
    std::vector<uchar> data((uchar*)buffers[buf.index].start,
                            (uchar*)buffers[buf.index].start + buf.bytesused);
    cv::Mat full = cv::imdecode(data, cv::IMREAD_COLOR);

    if (!full.empty() && full.cols == WIDTH && full.rows == HEIGHT) {
      // ---- 前端限频：避免图像过密导致后端排队（动态时延）----
      const double time_delta = 1.0 / _app->get_params().track_frequency;
      if (last_timestamp_raw >= 0.0 && cam_time_raw < last_timestamp_raw + time_delta) {
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) { perror("VIDIOC_QBUF (requeue)"); break; }
        usleep(1);
        continue;
      }
      last_timestamp_raw = cam_time_raw;

      // ---- 构造 CameraData（左右切半，硬同步，同一时间戳 RAW）----
      ov_core::CameraData image_msg;
      image_msg.timestamp = cam_time_raw;   // 关键：与 IMU 完全同一时基（RAW）
      image_msg.sensor_ids = {0, 1};
      image_msg.images.resize(2);

      cv::Mat left_color  = full(cv::Rect(0, 0, WIDTH/2, HEIGHT));
      cv::Mat right_color = full(cv::Rect(WIDTH/2, 0, WIDTH/2, HEIGHT));
      cv::cvtColor(left_color,  image_msg.images[0], cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_color, image_msg.images[1], cv::COLOR_BGR2GRAY);

      if (_app->get_params().use_mask) {
        image_msg.masks.push_back(_app->get_params().masks.at(0));
        image_msg.masks.push_back(_app->get_params().masks.at(1));
      } else {
        image_msg.masks.emplace_back(cv::Mat::zeros(image_msg.images[0].rows, image_msg.images[0].cols, CV_8UC1));
        image_msg.masks.emplace_back(cv::Mat::zeros(image_msg.images[1].rows, image_msg.images[1].cols, CV_8UC1));
      }

      // （可选）保存调试图
      if (is_debug) {
        std::ostringstream oss; oss << std::fixed << std::setprecision(6) << cam_time_raw;
        const std::string ts_str = oss.str();
        cv::imwrite(left_images_dir  + "/" + ts_str + ".png", image_msg.images[0]);
        cv::imwrite(right_images_dir + "/" + ts_str + ".png", image_msg.images[1]);
      }

      // 入队（按时间有序；真正消费在 IMU 线程中结合固定延迟判断）
      {
        std::lock_guard<std::mutex> lock(camera_queue_mtx);
        camera_queue.push_back(image_msg);
      }

      // 统计一帧端到端处理耗时（不含解码外的后端）
      auto t1 = boost::posix_time::microsec_clock::local_time();
      double dt_proc = (t1 - t0).total_microseconds() * 1e-6;
      sum_time += dt_proc; ++count;
      PRINT_INFO("[CAMERA TIME]: %.6f seconds (avg %.6f)\n", dt_proc, sum_time / count);
    } else {
      std::cerr << "图像解码失败或尺寸错误\n";
    }

    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) { perror("VIDIOC_QBUF (requeue)"); break; }
    usleep(1);
  }

  ioctl(fd, VIDIOC_STREAMOFF, &type);
  for (int i = 0; i < BUFFER_COUNT; ++i) {
    munmap(buffers[i].start, buffers[i].length);
  }
  close(fd);
}




void FGVisualizer::run() {
  std::thread cam_thread(&FGVisualizer::retrieveCamera, this);
  cam_thread.detach();

  retrieveIMU();
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

    /***************************************六自由度数据输出start*************************************/
  Eigen::Vector3f pos = state->_imu->pos().cast<float>();
  Eigen::Quaternionf q_IinM_f(q_IinM.cast<float>());
  Eigen::Vector3f euler = q_IinM_f.toRotationMatrix().eulerAngles(0, 1, 2); // RPY
  
  std::array<float, 6> pose6d = {
    pos[0], pos[1], pos[2],   // x,y,z
    euler[0], euler[1], euler[2] // roll,pitch,yaw
  };
  
   // ========== 打开串口5 (/dev/ttyS5) ==========
   int fd = open("/dev/ttyS5", O_RDWR | O_NOCTTY | O_NONBLOCK); // 非阻塞模式
    if (fd < 0) {
        std::cerr << "Error opening /dev/ttyS5" << std::endl;
        //return;
    }

    struct termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error from tcgetattr" << std::endl;
        close(fd);
        //return;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;  // 8位数据
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;   // 原始模式
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0; // 非阻塞读取
    tty.c_cc[VTIME] = 0;

    tty.c_cflag |= (CLOCAL | CREAD); // 打开接收功能
    tty.c_cflag &= ~(PARENB | PARODD); // 无校验
    tty.c_cflag &= ~CSTOPB;            // 1位停止位
    tty.c_cflag &= ~CRTSCTS;           // 无流控

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error from tcsetattr" << std::endl;
        close(fd);
        //return;
    }

    std::vector<uint8_t> frame;
    frame.push_back(0xAA);
    frame.push_back(0x55);

    uint8_t* data_ptr = reinterpret_cast<uint8_t*>(pose6d.data());
    frame.insert(frame.end(), data_ptr, data_ptr + pose6d.size() * sizeof(float));

    frame.push_back(0x0D);

    // ========== 发送数据 ==========
    ssize_t written = write(fd, frame.data(), frame.size());
    if (written < 0) {
        std::cerr << "Error writing to serial" << std::endl;
    } else {
        //std::cout << "x " << pose6d << " bytes" << std::endl;
        //std::cout << "Position: x=" << static_cast<float>(state_plus(4))
         // << ", y=" << static_cast<float>(state_plus(5))
          //<< ", z=" << static_cast<float>(state_plus(6)) << std::endl;
    }

    close(fd);  


  /***************************************六自由度数据输出end*************************************/
  
  

  poses_imu.push_back(std::pair<double, Eigen::Matrix4f>(timestamp_inI, poseIinM_f));

  _viz->showPose("pose_imu", time_us, poseIinM_f, "LOCAL_WORLD", "IMU");

  poses_imu_dq.push_back(std::pair<double, Eigen::Matrix4f>(timestamp_inI, poseIinM_f));

  const double window_duration = 10.0;               // 秒
  const double window_start = timestamp_inI - window_duration;
  while (!poses_imu_dq.empty() && poses_imu_dq.front().first < window_start) {
    poses_imu_dq.pop_front();
  }

  // —— 生成路径（窗口内），可选下采样以限制点数 —— 
  std::vector<Eigen::Matrix4f> path_imu;
  path_imu.reserve(poses_imu_dq.size());

  // 将点数限制到 ~16384（与原逻辑一致）
  const size_t max_pts = 16384;
  const size_t N = poses_imu_dq.size();
  const size_t stride = (N > max_pts) ? (N / max_pts + ((N % max_pts) ? 1 : 0)) : 1;

  size_t idx = 0;
  for (const auto& ps : poses_imu_dq) {
    if ((idx++ % stride) == 0) {
      path_imu.push_back(ps.second);
    }
  }

  _viz->showPath("path_imu_win10", time_us, path_imu, "LOCAL_WORLD");

  std::vector<Eigen::Matrix4f> path_imu2;
  for (size_t i = 0; i < poses_imu.size(); i += std::floor((double)poses_imu.size() / 16384.0) + 1) {
    path_imu2.push_back(poses_imu.at(i).second);
  }
  _viz->showPath("path_imu", time_us, path_imu2, "LOCAL_WORLD");
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

#if ROS_AVAILABLE == 1
void FGVisualizer::setup_subscribers(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<ov_core::YamlParser> parser) {

  // We need a valid parser
  assert(parser != nullptr);

  // Create imu subscriber (handle legacy ros param info)
  std::string topic_imu;
  nh->param<std::string>("topic_imu", topic_imu, "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  sub_imu = nh->subscribe(topic_imu, 1000, &FGVisualizer::callback_inertial, this);
  PRINT_INFO("subscribing to IMU: %s\n", topic_imu.c_str());

  // Logic for sync stereo subscriber
  // https://answers.ros.org/question/96346/subscribe-to-two-image_raws-with-one-function/?answer=96491#post-id-96491
  if (_app->get_params().state_options.num_cameras == 2) {
    // Read in the topics
    std::string cam_topic0, cam_topic1;
    nh->param<std::string>("topic_camera" + std::to_string(0), cam_topic0, "/cam" + std::to_string(0) + "/image_raw");
    nh->param<std::string>("topic_camera" + std::to_string(1), cam_topic1, "/cam" + std::to_string(1) + "/image_raw");
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(0), "rostopic", cam_topic0);
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(1), "rostopic", cam_topic1);
    // Create sync filter (they have unique pointers internally, so we have to use move logic here...)
    auto image_sub0 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*nh, cam_topic0, 1);
    auto image_sub1 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*nh, cam_topic1, 1);
    auto sync = std::make_shared<message_filters::Synchronizer<sync_pol>>(sync_pol(10), *image_sub0, *image_sub1);
    sync->registerCallback(boost::bind(&FGVisualizer::callback_stereo, this, _1, _2, 0, 1));
    // Append to our vector of subscribers
    sync_cam.push_back(sync);
    sync_subs_cam.push_back(image_sub0);
    sync_subs_cam.push_back(image_sub1);
    PRINT_INFO("subscribing to cam (stereo): %s\n", cam_topic0.c_str());
    PRINT_INFO("subscribing to cam (stereo): %s\n", cam_topic1.c_str());
  } else {
    // Now we should add any non-stereo callbacks here
    for (int i = 0; i < _app->get_params().state_options.num_cameras; i++) {
      // read in the topic
      std::string cam_topic;
      nh->param<std::string>("topic_camera" + std::to_string(i), cam_topic, "/cam" + std::to_string(i) + "/image_raw");
      parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
      // create subscriber
      subs_cam.push_back(nh->subscribe<sensor_msgs::Image>(cam_topic, 10, boost::bind(&FGVisualizer::callback_monocular, this, _1, i)));
      PRINT_INFO("subscribing to cam (mono): %s\n", cam_topic.c_str());
    }
  }
}

void FGVisualizer::callback_inertial(const sensor_msgs::Imu::ConstPtr &msg) {
  // convert into correct format
  ov_core::ImuData message;
  message.timestamp = msg->header.stamp.toSec();
  message.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  message.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  Eigen::Quaterniond q_IinM(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
  message.Rm = q_IinM.toRotationMatrix();

  // send it to our VIO system
  _app->feed_measurement_imu(message);

  _viz->publishIMU("raw_imu", int64_t(message.timestamp * 1e6), "IMU", message.am, message.wm, Eigen::Quaterniond(message.Rm),
                         message.hm);

  Eigen::Matrix4f T_w_imu = Eigen::Matrix4f::Identity();
  T_w_imu.block(0, 0, 3, 3) = message.Rm.cast<float>();
  _viz->showPose("imu_angle", int64_t(message.timestamp * 1e6), T_w_imu, "LOCAL_WORLD", "IMU_R");

  if (thread_update_running)
    return;
  thread_update_running = true;
  std::thread thread([&] {
    // Loop through our queue and see if we are able to process any of our camera measurements
    // We are able to process if we have at least one IMU measurement greater than the camera time
    double timestamp_imu_inC = message.timestamp - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
    while (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_imu_inC) {
      auto rT0_1 = boost::posix_time::microsec_clock::local_time();
      double update_dt = 1000.0 * (timestamp_imu_inC - camera_queue.at(0).timestamp);

      last_images_timestamp = camera_queue.at(0).timestamp;
      last_images = camera_queue.at(0).images;

      _app->feed_measurement_camera(camera_queue.at(0));
      auto rT0_2 = boost::posix_time::microsec_clock::local_time();
      publish_cameras();
      visualize();

      {
        std::lock_guard<std::mutex> lck(camera_queue_mtx);
        camera_queue.pop_front();
      }
      
      auto rT0_3 = boost::posix_time::microsec_clock::local_time();
      double time_slam = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
      double time_total = (rT0_3 - rT0_1).total_microseconds() * 1e-6;
      _dash_board->setNameAndValue(0, "TIME", time_total);
      _dash_board->setNameAndValue(1, "TIME_SLAM", time_slam);
      _dash_board->setNameAndValue(2, "HZ", 1.0 / time_total);
      _dash_board->setNameAndValue(3, "UPDATE_DT", update_dt);
      _dash_board->print();
      PRINT_INFO(BLUE "[TIME]: %.4f seconds total, %.4f seconds slam (%.1f hz, %.2f ms behind)\n" RESET, time_total, time_slam, 1.0 / time_total, update_dt);
    }
    
    thread_update_running = false;
  });

  thread.detach();
}

void FGVisualizer::callback_monocular(const sensor_msgs::ImageConstPtr &msg0, int cam_id0) {

  // Check if we should drop this image
  double timestamp = msg0->header.stamp.toSec();
  double time_delta = 1.0 / _app->get_params().track_frequency;
  if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && timestamp < camera_last_timestamp.at(cam_id0) + time_delta) {
    return;
  }
  camera_last_timestamp[cam_id0] = timestamp;

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Create the measurement
  ov_core::CameraData message;
  message.timestamp = cv_ptr->header.stamp.toSec();
  message.sensor_ids.push_back(cam_id0);
  message.images.push_back(cv_ptr->image.clone());

  // Load the mask if we are using it, else it is empty
  // TODO: in the future we should get this from external pixel segmentation
  if (_app->get_params().use_mask) {
    message.masks.push_back(_app->get_params().masks.at(cam_id0));
  } else {
    message.masks.push_back(cv::Mat::zeros(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC1));
  }

  // append it to our queue of images
  std::lock_guard<std::mutex> lck(camera_queue_mtx);
  camera_queue.push_back(message);
  std::sort(camera_queue.begin(), camera_queue.end());
}

void FGVisualizer::callback_stereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1, int cam_id0,
                                     int cam_id1) {

  // // Check if we should drop this image
  // double timestamp = msg0->header.stamp.toSec();
  // double time_delta = 1.0 / _app->get_params().track_frequency;
  // if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && timestamp < camera_last_timestamp.at(cam_id0) + time_delta) {
  //   return;
  // }
  // camera_last_timestamp[cam_id0] = timestamp;

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr0;
  try {
    cv_ptr0 = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s\n", e.what());
    return;
  }

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr1;
  try {
    cv_ptr1 = cv_bridge::toCvShare(msg1, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s\n", e.what());
    return;
  }

  // Create the measurement
  ov_core::CameraData message;
  message.timestamp = cv_ptr0->header.stamp.toSec();
  message.sensor_ids.push_back(cam_id0);
  message.sensor_ids.push_back(cam_id1);
  message.images.push_back(cv_ptr0->image.clone());
  message.images.push_back(cv_ptr1->image.clone());

  // Load the mask if we are using it, else it is empty
  // TODO: in the future we should get this from external pixel segmentation
  if (_app->get_params().use_mask) {
    message.masks.push_back(_app->get_params().masks.at(cam_id0));
    message.masks.push_back(_app->get_params().masks.at(cam_id1));
  } else {
    // message.masks.push_back(cv::Mat(cv_ptr0->image.rows, cv_ptr0->image.cols, CV_8UC1, cv::Scalar(255)));
    message.masks.push_back(cv::Mat::zeros(cv_ptr0->image.rows, cv_ptr0->image.cols, CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv_ptr1->image.rows, cv_ptr1->image.cols, CV_8UC1));
  }

  // append it to our queue of images
  std::lock_guard<std::mutex> lck(camera_queue_mtx);
  camera_queue.push_back(message);
  // std::sort(camera_queue.begin(), camera_queue.end());
}

#endif
