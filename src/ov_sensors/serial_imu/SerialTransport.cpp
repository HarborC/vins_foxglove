

#include "SerialTransport.h"

namespace ov_sensors {
SerialTransport::SerialTransport(std::string dev, int baud)
    : device_(std::move(dev)), baud_(baud) {}
SerialTransport::~SerialTransport() { closeDevice(); }

bool SerialTransport::openDevice() {
  fd_ = ::open(device_.c_str(), O_RDWR | O_NOCTTY);
  if (fd_ < 0) {
    perror("open serial");
    return false;
  }
  return configure();
}

void SerialTransport::closeDevice() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

int SerialTransport::readBytes(unsigned char *buf, size_t maxlen) {
  if (fd_ < 0)
    return -1;
  int ret = ::read(fd_, buf, maxlen);
  return ret;
}

bool SerialTransport::valid() const { return fd_ >= 0; }

bool SerialTransport::configure() {
  struct termios tty {};
  if (tcgetattr(fd_, &tty) != 0) {
    perror("tcgetattr");
    return false;
  }

  cfmakeraw(&tty);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;
  tty.c_cflag &= ~PARENB;  // no parity
  tty.c_cflag &= ~CSTOPB;  // 1 stop bit
  tty.c_cflag &= ~CRTSCTS; // no hw flow

  // VTIME/VMIN: read returns after >=11 bytes or 0.1s timeout
  tty.c_cc[VTIME] = 1; // 0.1s
  tty.c_cc[VMIN] = 11;

  speed_t sp = B230400; // fixed for minimal impl
  cfsetispeed(&tty, sp);
  cfsetospeed(&tty, sp);

  tcflush(fd_, TCIFLUSH);
  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    perror("tcsetattr");
    return false;
  }
  return true;
}
} // namespace ov_sensors