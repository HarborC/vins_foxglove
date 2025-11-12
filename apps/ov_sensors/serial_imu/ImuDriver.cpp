#include "ImuDriver.h"
namespace ov_sensors {

ImuDriver::ImuDriver(std::string dev) : transport_(std::move(dev)) {}
ImuDriver::~ImuDriver() { stop(); }

bool ImuDriver::start() {
  if (running_)
    return true;
  if (!transport_.openDevice())
    return false;
  running_ = true;
  worker_ = std::thread(&ImuDriver::runLoop, this);
  return true;
}

void ImuDriver::stop() {
  if (!running_)
    return;
  running_ = false;
  if (worker_.joinable())
    worker_.join();
  transport_.closeDevice();
}

void ImuDriver::setCallback(Callback cb) {
  std::lock_guard<std::mutex> lk(cb_mtx_);
  cb_ = std::move(cb);
}

void ImuDriver::runLoop() {
  unsigned char buf[128];
  while (running_) {
    int ret = transport_.readBytes(buf, sizeof(buf));
    if (ret <= 0) {
      ::usleep(1000);
      continue;
    }
    for (int i = 0; i < ret; ++i) {
      ImuSample samp;
      if (parser_.feed(buf[i], samp)) {
        // timestamp: center around reception time minus half tx window
        const double t_last = nowRaw();
        samp.timestamp = t_last - 0.5 * TX_TIME_EST;
        Callback local;
        {
          std::lock_guard<std::mutex> lk(cb_mtx_);
          local = cb_;
        }
        if (local)
          local(samp);
      }
    }
  }
}
} // namespace ov_sensors