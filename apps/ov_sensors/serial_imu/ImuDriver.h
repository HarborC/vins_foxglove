#pragma once
#include "IImuDriver.h"
#include "JY901Parser.h"
#include "SerialTransport.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>

namespace ov_sensors {

class ImuDriver final : public IImuDriver {
public:
  ImuDriver(std::string dev);
  ~ImuDriver() override;

  bool start() override;

  void stop() override;

  void setCallback(Callback cb) override;

private:
  // approx transmission time of four sub packets (for timestamp centering)
  static constexpr double TX_TIME_EST = 0.0019; // seconds (~1.9ms)

  void runLoop();

  static double nowRaw() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
  }

  SerialTransport transport_;
  JY901Parser parser_;
  std::thread worker_;
  std::atomic<bool> running_{false};
  Callback cb_{};
  std::mutex cb_mtx_;
};

} // namespace ov_sensors
