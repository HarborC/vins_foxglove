#pragma once
#include "ICameraDriver.h"
#include <thread>
#include <atomic>
#include <string>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <chrono>

namespace ov_sensors {

class V4L2CameraDriver final : public ICameraDriver {
public:
  struct Config {
    std::string device = "/dev/video0";
    int width = 1280;
    int height = 480;
    int buffer_count = 8;
    bool mjpeg = true; // if false expect YUYV
    int track_frequency = 20; // target feed frequency
  };

  explicit V4L2CameraDriver(Config cfg): cfg_(std::move(cfg)) {}
  ~V4L2CameraDriver() override { stop(); }

  bool start() override {
    if (running_) return true;
    if (!openDevice()) return false;
    running_ = true;
    worker_ = std::thread(&V4L2CameraDriver::runLoop, this);
    return true;
  }
  void stop() override {
    if (!running_) return;
    running_ = false;
    if (worker_.joinable()) worker_.join();
    closeDevice();
  }
  void setCallback(Callback cb) override {
    std::lock_guard<std::mutex> lk(cb_mtx_); cb_ = std::move(cb);
  }

private:
  struct Buffer { void* start=nullptr; size_t length=0; };

  bool openDevice() {
    fd_ = ::open(cfg_.device.c_str(), O_RDWR);
    if (fd_ < 0) { perror("open video"); return false; }
    // format
    v4l2_format fmt{}; fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; fmt.fmt.pix.width = cfg_.width; fmt.fmt.pix.height = cfg_.height;
    fmt.fmt.pix.pixelformat = cfg_.mjpeg ? V4L2_PIX_FMT_MJPEG : V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) { perror("VIDIOC_S_FMT"); return false; }
    // request buffers
    v4l2_requestbuffers req{}; req.count = cfg_.buffer_count; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) { perror("VIDIOC_REQBUFS"); return false; }
    buffers_.resize(req.count);
    for (int i=0;i<req.count;++i){
      v4l2_buffer buf{}; buf.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; buf.memory=V4L2_MEMORY_MMAP; buf.index=i;
      if (ioctl(fd_, VIDIOC_QUERYBUF, &buf)==-1){ perror("VIDIOC_QUERYBUF"); return false; }
      buffers_[i].length = buf.length;
      buffers_[i].start = mmap(NULL, buf.length, PROT_READ|PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
      if (buffers_[i].start == MAP_FAILED){ perror("mmap"); return false; }
      if (ioctl(fd_, VIDIOC_QBUF, &buf)==-1){ perror("VIDIOC_QBUF"); return false; }
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type)==-1){ perror("STREAMON"); return false; }

    // establish raw-mono offset
    timespec ts_raw{}, ts_mono{}; clock_gettime(CLOCK_MONOTONIC_RAW,&ts_raw); clock_gettime(CLOCK_MONOTONIC,&ts_mono);
    offset_raw_minus_mono_ = (ts_raw.tv_sec + ts_raw.tv_nsec*1e-9) - (ts_mono.tv_sec + ts_mono.tv_nsec*1e-9);
    return true;
  }

  void closeDevice(){
    if (fd_>=0){
      int type=V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(fd_, VIDIOC_STREAMOFF, &type);
      for (auto &b: buffers_) if (b.start) munmap(b.start, b.length);
      ::close(fd_); fd_=-1; buffers_.clear();
    }
  }

  static double nowRaw(){ timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts); return ts.tv_sec + ts.tv_nsec*1e-9; }

  void runLoop(){
    double last_timestamp_raw=-1.0; const double min_dt = 1.0/std::max(1,cfg_.track_frequency);
    while (running_){
      v4l2_buffer buf{}; buf.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; buf.memory=V4L2_MEMORY_MMAP;
      if (ioctl(fd_, VIDIOC_DQBUF, &buf)==-1){ if(errno==EINTR) continue; perror("DQBUF"); break; }
      const double cam_time_mono = buf.timestamp.tv_sec + buf.timestamp.tv_usec*1e-6;
      const double cam_time_raw = cam_time_mono + offset_raw_minus_mono_;
      // throttle
      if (last_timestamp_raw>=0.0 && cam_time_raw < last_timestamp_raw + min_dt){
        if (ioctl(fd_, VIDIOC_QBUF, &buf)==-1) { perror("QBUF"); break; }
        ::usleep(1000); continue;
      }
      last_timestamp_raw = cam_time_raw;

      std::vector<uchar> data((uchar*)buffers_[buf.index].start, (uchar*)buffers_[buf.index].start + buf.bytesused);
      cv::Mat full = cfg_.mjpeg ? cv::imdecode(data, cv::IMREAD_COLOR) : cv::Mat(cfg_.height, cfg_.width, CV_8UC2, buffers_[buf.index].start);
      CameraFrame frame; frame.timestamp_raw = cam_time_raw; frame.sensor_ids = {0,1}; frame.images.resize(2);
      if (!full.empty() && full.cols==cfg_.width && full.rows==cfg_.height){
        if (cfg_.mjpeg){
          cv::Mat left_color = full(cv::Rect(0,0,cfg_.width/2,cfg_.height));
          cv::Mat right_color= full(cv::Rect(cfg_.width/2,0,cfg_.width/2,cfg_.height));
          cv::cvtColor(left_color, frame.images[0], cv::COLOR_BGR2GRAY);
          cv::cvtColor(right_color,frame.images[1], cv::COLOR_BGR2GRAY);
        } else {
          // YUYV split and convert
          cv::Mat left_yuyv (cfg_.height, cfg_.width/2, CV_8UC2, full.data);
          cv::Mat right_yuyv(cfg_.height, cfg_.width/2, CV_8UC2, full.data + (cfg_.width/2)*2);
          cv::cvtColor(left_yuyv,  frame.images[0], cv::COLOR_YUV2GRAY_YUYV);
          cv::cvtColor(right_yuyv, frame.images[1], cv::COLOR_YUV2GRAY_YUYV);
        }
        Callback local; { std::lock_guard<std::mutex> lk(cb_mtx_); local = cb_; }
        if (local) local(frame);
      }

      if (ioctl(fd_, VIDIOC_QBUF, &buf)==-1){ perror("QBUF"); break; }
      ::usleep(1000);
    }
  }

  Config cfg_;
  int fd_ = -1;
  double offset_raw_minus_mono_ = 0.0;
  std::vector<Buffer> buffers_;
  std::thread worker_;
  std::atomic<bool> running_{false};
  Callback cb_{}; std::mutex cb_mtx_;
};

} // namespace ov_sensors
