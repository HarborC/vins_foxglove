#pragma once
#include <functional>
#include <memory>
#include "CameraFrame.h"

namespace ov_sensors {

class ICameraDriver {
public:
  using Callback = std::function<void(const CameraFrame&)>;
  virtual ~ICameraDriver() = default;
  virtual bool start() = 0;
  virtual void stop() = 0;
  virtual void setCallback(Callback cb) = 0;
};

using ICameraDriverPtr = std::shared_ptr<ICameraDriver>;

} // namespace ov_sensors
