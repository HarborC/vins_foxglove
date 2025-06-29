#include "imu_serial.h"

#include <ctime>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <termios.h>
#include <string.h>
#include <sys/types.h>
#include <errno.h>

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

// 构造函数
IMUSerial::IMUSerial() : fd(-1), chrCnt(0) {
    memset(chrBuf, 0, sizeof(chrBuf));
}

// 打开串口
int IMUSerial::openSerial(const char* pathname) {
    fd = open(pathname, O_RDWR | O_NOCTTY);
    if (fd == -1) {
        perror("Can't open serial port\n");
        return -1;
    }
    printf("open %s success!\n", pathname);
    if (isatty(STDIN_FILENO) == 0)
        printf("standard input is not a terminal device\n");
    else
        printf("isatty success!\n");
    return fd;
}

// 配置串口参数
int IMUSerial::setSerial() {
    struct termios newtio, oldtio;
    if (tcgetattr(fd, &oldtio) != 0) {
        perror("SetupSerial 1\n");
        return -1;
    }
    memset(&newtio, 0, sizeof(newtio));
    newtio.c_cflag |= CLOCAL | CREAD;
    newtio.c_cflag &= ~CSIZE;
    newtio.c_cflag |= CS8;
    newtio.c_cflag &= ~PARENB;

    newtio.c_cflag &= ~CRTSCTS;

    newtio.c_iflag &= ~ICRNL;

    cfsetispeed(&newtio, B115200);
    cfsetospeed(&newtio, B115200);
    newtio.c_cflag &= ~CSTOPB;
    newtio.c_cc[VTIME] = 10;
    newtio.c_cc[VMIN] = 1;
    tcflush(fd, TCIFLUSH);

    if ((tcsetattr(fd, TCSANOW, &newtio)) != 0) {
        perror("com set error\n");
        return -1;
    }
    printf("set done\n");
    return 0;
}

// 关闭串口
int IMUSerial::closeSerial() {
    assert(fd != -1);
    close(fd);
    fd = -1;
    return 0;
}

// 发送数据
int IMUSerial::sendData(const char* send_buffer, int length) {
    int bytes_written = write(fd, send_buffer, length * sizeof(unsigned char));
    return bytes_written;
}

// 接收数据
int IMUSerial::recvData(char* recv_buffer, int length) {
    int bytes_read = read(fd, recv_buffer, length);
    return bytes_read;
}

// 解析接收到的数据
void IMUSerial::parseData(char chr, IMUDATA& data) {
    chrBuf[chrCnt++] = chr;
    if (chrCnt < 11) return;

    // 检查帧头是否正确
    if ((chrBuf[0] != 0x55) || (chrBuf[1] & 0x50) != 0x50) {
        // printf("Error: %x %x\r\n", chrBuf[0], chrBuf[1]);
        memmove(&chrBuf[0], &chrBuf[1], 10);
        chrCnt--;
        return;
    }

    signed short sData[4];
    memcpy(&sData[0], &chrBuf[2], 8);

    auto now = std::chrono::system_clock::now(); // 获取当前时间点
    double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() * 1e-9;

    switch (chrBuf[1]) {
        case 0x51: // 加速度
            for (int i = 0; i < 3; i++) {
                data.a(i) = (float)sData[i] / 32768.0 * 16.0 * G;
            }
            data.time = timestamp;
            break;

        case 0x52: // 角速度
            for (int i = 0; i < 3; i++) {
                data.w(i) = (float)sData[i] / 32768.0 * 2000.0;
            }
            break;

        case 0x53: // 角度
            for (int i = 0; i < 3; i++) {
                data.angle(i) = (float)sData[i] / 32768.0 * 180.0;
            }
            break;

        case 0x54: // 磁场
            for (int i = 0; i < 3; i++) {
                data.h(i) = (float)sData[i];
            }
            break;
    }
    chrCnt = 0;
}
