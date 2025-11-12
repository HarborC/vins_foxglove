#pragma once
#include "ImuSample.h"
#include <cstddef>

namespace ov_sensors {

class ImuParser {
public:
  virtual ~ImuParser() = default;
  // feed one byte, return true if a complete sample is ready into `out`
  virtual bool feed(uint8_t byte, ImuSample &out) = 0;
};

} // namespace ov_sensors
