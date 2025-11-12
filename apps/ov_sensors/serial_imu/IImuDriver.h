#pragma once
#include "ImuSample.h"
#include <functional>
#include <memory>

namespace ov_sensors {

class IImuDriver {
public:
  using Callback = std::function<void(const ImuSample &)>;
  virtual ~IImuDriver() = default;
  virtual bool start() = 0;                  // start background reading thread
  virtual void stop() = 0;                   // stop thread and close device
  virtual void setCallback(Callback cb) = 0; // set user callback (thread-safe)
};

using IImuDriverPtr = std::shared_ptr<IImuDriver>;

} // namespace ov_sensors
