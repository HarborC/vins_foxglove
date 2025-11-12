#pragma once
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

namespace ov_sensors {

class SerialTransport {
public:
  SerialTransport(std::string dev, int baud = 230400);
  ~SerialTransport();

  bool openDevice();

  void closeDevice();

  int readBytes(unsigned char *buf, size_t maxlen);

  bool valid() const;

private:
  bool configure();

  std::string device_;
  int baud_;
  int fd_ = -1;
};

} // namespace ov_sensors
