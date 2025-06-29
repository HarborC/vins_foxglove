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

#include <foxglove/visualizer.h>

#include "serial/imu_serial.h"
#include "utils/sensor_data.h"
#include "utils/memory_utils.h"

using namespace std;

foxglove_viz::Visualizer::Ptr viz;
std::deque<ov_core::CameraData> camera_queue;
std::mutex camera_queue_mtx;

struct Buffer {
    void* start;
    size_t length;
};

void retrieveIMU() {
    IMUSerial imu_serial;

    if (imu_serial.openSerial("/dev/ttyS3") == -1) {
        printf("open ttyS3 error\n");
        exit(EXIT_FAILURE);
    }

    if (imu_serial.setSerial() == -1) {
        printf("set ttyS3 error\n");
        exit(EXIT_FAILURE);
    }

    char r_buf[1024];
    memset(r_buf, 0, sizeof(r_buf));

    while (true) {
        int ret = imu_serial.recvData(r_buf, sizeof(r_buf));
        if (ret <= 0) {
            printf("uart read failed\n");
            break;
        }
        IMUSerial::IMUDATA imu_data;
        for (int i = 0; i < ret; i++) {
            imu_serial.parseData(r_buf[i], imu_data);
        }

        // convert into correct format
        ov_core::ImuData message;
        message.timestamp = double(imu_data.time);
        message.wm << double(imu_data.w.x()), double(imu_data.w.y()), double(imu_data.w.z());
        message.am << double(imu_data.a.x()), double(imu_data.a.y()), double(imu_data.a.z());
        message.wm *= 3.14159 / 180.0; // convert to rad/s

        // std::cout << "IMU: " << std::fixed << message.timestamp << " " << message.wm.transpose() << " " << message.am.transpose() << std::endl;
        viz->publishIMU("imu", int64_t(message.timestamp * 1e6), "IMU", message.am, message.wm);

        usleep(1000);
    }

    imu_serial.closeSerial();
    return;
}

void retrieveCamera() {
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

        usleep(1000);
    }

    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }
    close(fd);
}

int main(int argc, char **argv) {
    viz = std::make_shared<foxglove_viz::Visualizer>(8088, 2);

    std::thread cam_thread(&retrieveCamera);
    cam_thread.detach();
    std::cout << "Camera thread started." << std::endl;

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
            usleep(1000); // Sleep for 100ms if queue is empty
        }
    }
    
    return 0;
}
