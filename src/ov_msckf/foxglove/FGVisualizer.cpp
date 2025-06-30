#include "FGVisualizer.h"

#include "core/VioManager.h"
#include "serial/imu_serial.h"
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

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;
using namespace std;
namespace fs = std::filesystem;

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

void FGVisualizer::retrieveIMU() {
  std::string serial_device = "/dev/ttyS3";
  std::ofstream imu_file(debug_dir + "/imu_data.txt", std::ios::app);
  imu_file << "# timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,ang_x,ang_y,ang_z\n";

  int fd = open(serial_device.c_str(), O_RDWR | O_NOCTTY);
  if (fd == -1) {
    perror("open serial");
    return;
  }

  struct termios tty;
  tcgetattr(fd, &tty);
  cfsetispeed(&tty, B230400);
  cfsetospeed(&tty, B230400);
  tty.c_cflag |= CLOCAL | CREAD | CS8;
  tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
  tty.c_iflag &= ~ICRNL;
  tty.c_cc[VTIME] = 1;
  tty.c_cc[VMIN] = 1;
  tcsetattr(fd, TCSANOW, &tty);

  if (fd == -1) {
    std::cerr << "串口打开失败：" << serial_device << std::endl;
    return;
  }

  IMUSerial::IMUDATA current_imu;
  char chrBuf[100];
  unsigned char chrCnt;
  float G = 9.81f;

  auto getUTCTimestamp = []() -> double {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
  };

  auto writeIMUFrameCSV = [&](const ov_core::ImuData &message) {
        Eigen::Quaterniond q(message.Rm);
        imu_file << std::fixed << std::setprecision(6) << message.timestamp << ","
                 << message.am(0) << "," << message.am(1) << "," << message.am(2) << ","
                 << message.wm(0) << "," << message.wm(1) << "," << message.wm(2) << ","
                 << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
                 << message.hm(0) << "," << message.hm(1) << "," << message.hm(2) << "\n";
        imu_file.flush();
    };

  char r_buf[1024];
  memset(r_buf, 0, sizeof(r_buf));

  while (true) {
    int ret = read(fd, r_buf, sizeof(r_buf));
    if (ret <= 0) {
      printf("uart read failed\n");
      break;
    }

    for (int i = 0; i < ret; i++) {
      chrBuf[chrCnt++] = r_buf[i];
      if (chrCnt < 11)
        continue;

      if ((chrBuf[0] != 0x55) || (chrBuf[1] & 0x50) != 0x50) {
        memmove(&chrBuf[0], &chrBuf[1], 10);
        chrCnt--;
        continue;
      }

      signed short sData[4];
      memcpy(&sData[0], &chrBuf[2], 8);
      double timestamp = getUTCTimestamp();

      switch (chrBuf[1]) {
      case 0x51:
        for (int j = 0; j < 3; j++)
          current_imu.a(j) = (float)sData[j] / 32768.0 * 16.0 * G;
        current_imu.has_acc = true;
        current_imu.timestamp_acc = timestamp;
        break;
      case 0x52:
        for (int j = 0; j < 3; j++)
          current_imu.w(j) = (float)sData[j] / 32768.0 * 2000.0;
        current_imu.has_gyro = true;
        current_imu.timestamp_gyro = timestamp;
        break;
      case 0x53:
        for (int j = 0; j < 3; j++)
          current_imu.angle(j) = (float)sData[j] / 32768.0 * 180.0;
        current_imu.has_ang = true;
        current_imu.timestamp_ang = timestamp;
        break;
      case 0x54:
        for (int j = 0; j < 3; j++)
          current_imu.h(j) = (float)sData[j];
        current_imu.has_h = true;
        current_imu.timestamp_h = timestamp;
        break;
      }
      chrCnt = 0;

      if (current_imu.has_acc && current_imu.has_gyro && current_imu.has_ang && current_imu.has_h) {
        ov_core::ImuData message;
        message.timestamp = current_imu.timestamp_acc;
        message.wm << double(current_imu.w.x()), double(current_imu.w.y()), double(current_imu.w.z());
        message.am << double(current_imu.a.x()), double(current_imu.a.y()), double(current_imu.a.z());
        message.hm << double(current_imu.h.x()), double(current_imu.h.y()), double(current_imu.h.z());
        message.wm *= 3.14159 / 180.0;                        // convert to rad/s
        message.Rm = IMUSerial::rpy2R(current_imu.angle).cast<double>(); // convert to rotation matrix

        if (is_debug) {
          writeIMUFrameCSV(message);
        }

        // send it to our VIO system
        _app->feed_measurement_imu(message);
        _viz->publishIMU("raw_imu", int64_t(message.timestamp * 1e6), "IMU", message.am, message.wm, Eigen::Quaterniond(message.Rm),
                         message.hm);

        Eigen::Matrix4f T_w_imu = Eigen::Matrix4f::Identity();
        T_w_imu.block(0, 0, 3, 3) = message.Rm.cast<float>();
        _viz->showPose("imu_angle", int64_t(message.timestamp * 1e6), T_w_imu, "LOCAL_WORLD", "IMU_R");

        current_imu = IMUSerial::IMUDATA();

        if (thread_update_running)
          continue;
        thread_update_running = true;
        std::thread thread([&] {
          // Loop through our queue and see if we are able to process any of our camera measurements
          // We are able to process if we have at least one IMU measurement greater than the camera time
          double timestamp_imu_inC = message.timestamp - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
          while (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_imu_inC) {
            auto rT0_1 = boost::posix_time::microsec_clock::local_time();
            double update_dt = 100.0 * (timestamp_imu_inC - camera_queue.at(0).timestamp);

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
    }
  }

  imu_file.close();
  close(fd);
}

void FGVisualizer::retrieveCamera() {
  constexpr int WIDTH = 1280;
  constexpr int HEIGHT = 480;
  constexpr int BUFFER_COUNT = 8;
  std::string device_path = "/dev/video73";

  std::string left_images_dir = debug_dir + "/images/left/";
  std::string right_images_dir = debug_dir + "/images/right/";

  fs::create_directories(left_images_dir);
  fs::create_directories(right_images_dir);

  int fd = open(device_path.c_str(), O_RDWR);
  if (fd == -1) {
    perror("open /dev/video73");
    return;
  }

  v4l2_format fmt = {};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = WIDTH;
  fmt.fmt.pix.height = HEIGHT;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;

  if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
    perror("VIDIOC_S_FMT");
    return;
  }

  v4l2_requestbuffers req = {};
  req.count = BUFFER_COUNT;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
    perror("VIDIOC_REQBUFS");
    return;
  }

  Buffer buffers[BUFFER_COUNT];
  for (int i = 0; i < BUFFER_COUNT; ++i) {
    v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
      perror("VIDIOC_QUERYBUF");
      return;
    }

    buffers[i].length = buf.length;
    buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buffers[i].start == MAP_FAILED) {
      perror("mmap");
      return;
    }

    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
      perror("VIDIOC_QBUF");
      return;
    }
  }

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
    perror("VIDIOC_STREAMON");
    return;
  }

  struct timespec ts_realtime, ts_mono;
  clock_gettime(CLOCK_REALTIME, &ts_realtime);
  clock_gettime(CLOCK_MONOTONIC, &ts_mono);

  double offset = (ts_realtime.tv_sec + ts_realtime.tv_nsec / 1e9) - (ts_mono.tv_sec + ts_mono.tv_nsec / 1e9);

  static double sum_time = 0;
  static int count = 0;
  static double last_timestamp = -1;

  while (true) {
    auto rT0_1 = boost::posix_time::microsec_clock::local_time();
    v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
      perror("VIDIOC_DQBUF");
      break;
    }

    double v4l2_time = buf.timestamp.tv_sec + buf.timestamp.tv_usec / 1e6;
    double utc_time = v4l2_time + offset;

    vector<uchar> data((uchar *)buffers[buf.index].start, (uchar *)buffers[buf.index].start + buf.bytesused);
    cv::Mat full = cv::imdecode(data, cv::IMREAD_COLOR);

    if (!full.empty() && full.cols == WIDTH && full.rows == HEIGHT) {
      // viz->showImage("raw_images", int64_t(utc_time * 1e6), full, "stereo", true);

      ov_core::CameraData image_msg;
      image_msg.timestamp = utc_time;
      image_msg.sensor_ids.push_back(0);
      image_msg.sensor_ids.push_back(1);
      image_msg.images.resize(2);
      // BGR to Gray conversion
      cv::cvtColor(full(cv::Rect(0, 0, WIDTH / 2, HEIGHT)), image_msg.images[0], cv::COLOR_BGR2GRAY);
      cv::cvtColor(full(cv::Rect(WIDTH / 2, 0, WIDTH / 2, HEIGHT)), image_msg.images[1], cv::COLOR_BGR2GRAY);

      if (_app->get_params().use_mask) {
        image_msg.masks.push_back(_app->get_params().masks.at(0));
        image_msg.masks.push_back(_app->get_params().masks.at(1));
      } else {
        // message.masks.push_back(cv::Mat(cv_ptr0->image.rows, cv_ptr0->image.cols, CV_8UC1, cv::Scalar(255)));
        image_msg.masks.push_back(cv::Mat::zeros(image_msg.images[0].rows, image_msg.images[0].cols, CV_8UC1));
        image_msg.masks.push_back(cv::Mat::zeros(image_msg.images[1].rows, image_msg.images[1].cols, CV_8UC1));
      }

      // for debug
      if (is_debug) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << utc_time;
        std::string timestamp_str = oss.str();

        std::string left_path = left_images_dir + "/" + timestamp_str + ".png";
        std::string right_path = right_images_dir + "/" + timestamp_str + ".png";

        cv::imwrite(left_path, image_msg.images[0]);
        cv::imwrite(right_path, image_msg.images[1]); // 保存左右图像
      }

      double time_delta = 1.0 / _app->get_params().track_frequency;
      if (last_timestamp >= 0 && utc_time < last_timestamp + time_delta) {

        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
          perror("VIDIOC_QBUF (requeue)");
          break;
        }

        usleep(1);

        continue;
      }
      last_timestamp = utc_time;

      // _viz->showImage("left_image", int64_t(utc_time * 1e6), image_msg.images[0], "CAM_0", true);
      // _viz->showImage("right_image", int64_t(utc_time * 1e6), image_msg.images[1], "CAM_1", true);

      // for vio
      {
        std::lock_guard<std::mutex> lock(camera_queue_mtx);
        camera_queue.push_back(image_msg);
      }

      auto rT0_2 = boost::posix_time::microsec_clock::local_time();
      double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
      sum_time += time_total;
      count++;
      PRINT_INFO("[CAMERA TIME]:{} seconds total ({})\n", time_total, sum_time / count);
    } else {
      cerr << "图像解码失败或尺寸错误" << endl;
    }

    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
      perror("VIDIOC_QBUF (requeue)");
      break;
    }

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

  // while (1) {
  //   double timestamp_imu_inC = new_imu_timestamp - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
  //   if (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_imu_inC) {
  //     ov_core::CameraData image;
  //     {
  //       std::lock_guard<std::mutex> lock(camera_queue_mtx);
  //       image = camera_queue.front();
  //       camera_queue.pop_front();
  //     }

  //     auto rT0_1 = boost::posix_time::microsec_clock::local_time();
  //     double update_dt = 100.0 * (timestamp_imu_inC - image.timestamp);

  //     last_images_timestamp = image.timestamp;
  //     last_images = image.images;

  //     _app->feed_measurement_camera(image);
      
  //     auto rT0_2 = boost::posix_time::microsec_clock::local_time();

  //     publish_cameras();
  //     visualize();
  //     auto rT0_3 = boost::posix_time::microsec_clock::local_time();
  //     double time_slam = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
  //     double time_total = (rT0_3 - rT0_1).total_microseconds() * 1e-6;
  //     PRINT_INFO(BLUE "[TIME]: %.4f seconds total, %.4f seconds slam (%.1f hz, %.2f ms behind)\n" RESET, time_total, time_slam, 1.0 / time_total, update_dt);
  //     print_memory_usage();
  //   } else {
  //     usleep(1); // Sleep for 10us if queue is empty
  //   }
  // }

}

void FGVisualizer::visualize() {
  // Return if we have already visualized
  if (last_visualization_timestamp == _app->get_state()->_timestamp && _app->initialized())
    return;
  last_visualization_timestamp = _app->get_state()->_timestamp;

  // // publish current image (only if not multi-threaded)
  // if (!_app->get_params().use_multi_threading_pubs)
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

  poses_imu.push_back(poseIinM_f);

  _viz->showPose("pose_imu", time_us, poseIinM_f, "LOCAL_WORLD", "IMU");

  std::vector<Eigen::Matrix4f> path_imu;
  for (size_t i = 0; i < poses_imu.size(); i += std::floor((double)poses_imu.size() / 16384.0) + 1) {
    path_imu.push_back(poses_imu.at(i));
  }
  _viz->showPath("path_imu", time_us, path_imu, "LOCAL_WORLD");
}

void FGVisualizer::publish_features() {
  int64_t time_us = (last_images_timestamp * 1e6);

  // Get our good MSCKF features
  std::vector<Eigen::Vector3d> feats_msckf = _app->get_good_features_MSCKF();
  pcl::PointCloud<pcl::PointXYZRGBA> cloud;
  for (auto &feat : feats_msckf) {
    pcl::PointXYZRGBA point;
    point.x = feat.x();
    point.y = feat.y();
    point.z = feat.z();
    point.r = 0;
    point.g = 255;
    point.b = 0;
    point.a = 255;
    cloud.push_back(point);
  }
  
  _viz->showPointCloudRGBA("points_msckf", time_us, cloud, "LOCAL_WORLD");

  // Get our good SLAM features
  std::vector<Eigen::Vector3d> feats_slam = _app->get_features_SLAM();
  pcl::PointCloud<pcl::PointXYZRGBA> cloud_SLAM;
  for (auto &feat : feats_slam) {
    pcl::PointXYZRGBA point;
    point.x = feat.x();
    point.y = feat.y();
    point.z = feat.z();
    point.r = 255;
    point.g = 0;
    point.b = 0;
    point.a = 255;
    cloud_SLAM.push_back(point);
  }
  _viz->showPointCloudRGBA("points_slam", time_us, cloud_SLAM, "LOCAL_WORLD");


  // Get our good ARUCO features
  std::vector<Eigen::Vector3d> feats_aruco = _app->get_features_ARUCO();
  pcl::PointCloud<pcl::PointXYZRGBA> cloud_ARUCO;
  for (auto &feat : feats_aruco) {
    pcl::PointXYZRGBA point;
    point.x = feat.x();
    point.y = feat.y();
    point.z = feat.z();
    point.r = 0;
    point.g = 0;
    point.b = 255;
    point.a = 255;
    cloud_ARUCO.push_back(point);
  }
  _viz->showPointCloudRGBA("points_aruco", time_us, cloud_ARUCO, "LOCAL_WORLD");

  // Get our good SIMULATION features
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