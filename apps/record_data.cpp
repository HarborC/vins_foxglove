#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <ctime>
#include <thread>
#include <deque>
#include <filesystem>
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

namespace fs = std::filesystem;

#include <foxglove/visualizer.h>

#include "serial/imu_serial.h"
#include "utils/sensor_data.h"
#include "utils/memory_utils.h"

using namespace std;

foxglove_viz::Visualizer::Ptr viz;
std::deque<ov_core::CameraData> camera_queue;
std::mutex camera_queue_mtx;

std::string root_dir = "/home/cat/projects/data/";

struct Buffer {
    void* start;
    size_t length;
};

void retrieveIMU() {
    std::string serial_device = "/dev/ttyS3"; 
    std::ofstream imu_file(root_dir + "/imu_data.txt", std::ios::app);
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

    auto writeIMUFrameCSV = [&](const IMUSerial::IMUDATA& frame, double timestamp) {
        Eigen::Quaterniond q(rpy2R(frame.angle).cast<double>());
        imu_file << std::fixed << std::setprecision(6) << timestamp << ","
                 << frame.a(0) << "," << frame.a(1) << "," << frame.a(2) << ","
                 << frame.w(0) << "," << frame.w(1) << "," << frame.w(2) << ","
                 << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
                 << frame.h(0) << "," << frame.h(1) << "," << frame.h(2) << "\n";
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
            if (chrCnt < 11) continue;

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
                        current_imu.w(j) = (float)sData[j] / 32768.0 * 2000.0 * 3.14159 / 180.0;
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
                // double avg_ts = (current_imu.timestamp_acc + current_imu.timestamp_gyro + current_imu.timestamp_ang + current_imu.timestamp_h) / 4.0;
                writeIMUFrameCSV(current_imu, current_imu.timestamp_acc);

                ov_core::ImuData message;
                message.timestamp = current_imu.timestamp_acc;
                message.wm << double(current_imu.w.x()), double(current_imu.w.y()), double(current_imu.w.z());
                message.am << double(current_imu.a.x()), double(current_imu.a.y()), double(current_imu.a.z());
                message.hm << double(current_imu.h.x()), double(current_imu.h.y()), double(current_imu.h.z());
                message.Rm = rpy2R(current_imu.angle).cast<double>(); // convert to rotation matrix

                viz->publishIMU("raw_imu", int64_t(message.timestamp * 1e6), "IMU", message.am, message.wm, Eigen::Quaterniond(message.Rm), message.hm);

                Eigen::Matrix4f T_w_imu = Eigen::Matrix4f::Identity();
                T_w_imu.block(0, 0, 3, 3) = message.Rm.cast<float>();
                viz->showPose("imu_angle", int64_t(message.timestamp * 1e6), T_w_imu, "LOCAL_WORLD", "IMU_R");
            
                current_imu = IMUSerial::IMUDATA();
            }
        }
    }

    close(fd);
    imu_file.close();
}

void retrieveCamera() {
    std::string left_images_dir = root_dir + "/images/left/";
    std::string right_images_dir = root_dir + "/images/right/";

    fs::create_directories(left_images_dir);
    fs::create_directories(right_images_dir);

    constexpr int WIDTH = 1280;
    constexpr int HEIGHT = 480;
    constexpr int BUFFER_COUNT = 8;
    std::string device_path = "/dev/video73";

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

    ov_core::CameraData image_msg;
    image_msg.sensor_ids.push_back(0);
    image_msg.sensor_ids.push_back(1);
    cv::Mat mask = cv::Mat::zeros(cv::Size(WIDTH / 2, HEIGHT), CV_8UC1);
    image_msg.masks.push_back(mask);
    image_msg.masks.push_back(mask);
    image_msg.images.resize(2);

    while (true) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            perror("VIDIOC_DQBUF");
            break;
        }

        double v4l2_time = buf.timestamp.tv_sec + buf.timestamp.tv_usec / 1e6;
        double utc_time = v4l2_time + offset;

        vector<uchar> data((uchar*)buffers[buf.index].start,
                           (uchar*)buffers[buf.index].start + buf.bytesused);
        cv::Mat full = cv::imdecode(data, cv::IMREAD_COLOR);

        if (!full.empty() && full.cols == WIDTH && full.rows == HEIGHT) {
            // viz->showImage("raw_images", int64_t(utc_time * 1e6), full, "stereo", true);

            image_msg.timestamp = utc_time;
            image_msg.images[0] = full(cv::Rect(0, 0, WIDTH / 2, HEIGHT)).clone();
            image_msg.images[1] = full(cv::Rect(WIDTH / 2, 0, WIDTH / 2, HEIGHT)).clone();

            cv::Mat gray_left, gray_right;
            cv::cvtColor(image_msg.images[0], gray_left, cv::COLOR_BGR2GRAY);
            cv::cvtColor(image_msg.images[1], gray_right, cv::COLOR_BGR2GRAY);

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << utc_time;
            std::string timestamp_str = oss.str();

            std::string left_path = left_images_dir + "/" + timestamp_str + ".png";
            std::string right_path = right_images_dir + "/" + timestamp_str + ".png";

            cv::imwrite(left_path, gray_left);
            cv::imwrite(right_path, gray_right);

            {
                std::lock_guard<std::mutex> lock(camera_queue_mtx);
                camera_queue.push_back(image_msg);
            }
            print_memory_usage();
        } else {
            cerr << "图像解码失败或尺寸错误" << endl;
        }

        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            perror("VIDIOC_QBUF (requeue)");
            break;
        }

        usleep(10);
    }

    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }
    close(fd);
}

int main(int argc, char **argv) {
    bool is_record_camera = false;
    if (argc > 1) {
        is_record_camera = true;
    }

    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");  // eg. 20250609_162300
    std::string timestamp_str = oss.str();

    root_dir += "/" + timestamp_str + "/";
    fs::create_directories(root_dir);

    viz = std::make_shared<foxglove_viz::Visualizer>(8088, 2);

    if (is_record_camera) {
        std::thread cam_thread(&retrieveCamera);
        cam_thread.detach();
        std::cout << "Camera thread started." << std::endl;
    }
    
    std::thread imu_thread(&retrieveIMU);
    imu_thread.detach();
    std::cout << "IMU thread started." << std::endl;
    
    while (1)
    {
        if (!camera_queue.empty()) {
            ov_core::CameraData image;
            {
                std::lock_guard<std::mutex> lock(camera_queue_mtx);
                image = camera_queue.front();
                camera_queue.pop_front();
            }

            int64_t time_us = (image.timestamp * 1e6);
            viz->showImage("left_image", time_us, image.images[0], "left_camera", true);
            viz->showImage("right_image", time_us, image.images[1], "right_camera", true);

            // if (!image.empty()) {
            //     // viz->showImage("raw_images", int64_t(std::chrono::system_clock::now().time_since_epoch().count() * 1e-6), image, "stereo", true);
                
            // } else {
            //     std::cout << "No image to display." << std::endl;
            // }
        } 
        else {
            usleep(10); // Sleep for 100ms if queue is empty
        }
    }
    
    return 0;
}
