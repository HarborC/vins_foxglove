// 包含标准头文件和外部库
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

// 使用 C++17 文件系统命名空间
namespace fs = std::filesystem;

// 包含项目中的头文件
#include <foxglove/visualizer.h>
#include "utils/sensor_data.h"
#include "utils/memory_utils.h"
#include "foxglove/FGVisualizer.h"
#include "calibration/apriltags.h"
#include "calibration/aprilgrid.h" 
#include "calibration/frame_quality_evaluator.h"

using namespace std;
using namespace cv;

// 全局变量定义
foxglove_viz::Visualizer::Ptr viz; // 可视化工具指针
std::deque<ov_core::CameraData> camera_queue; // 相机数据队列
std::mutex camera_queue_mtx; // 互斥锁
std::atomic<bool> thread_update_running = false;
int is_record_camera = 0;
std::shared_ptr<CAMERA_CALIB::AprilGrid> april_grid;
std::shared_ptr<FrameQualityEvaluator> dqe_left;
std::shared_ptr<FrameQualityEvaluator> dqe_right;

std::string root_dir = std::string(PROJ_DIR) + "/calib_data/";
std::string left_images_dir, right_images_dir;

// 视频缓冲区结构
struct Buffer {
    void* start;
    size_t length;
};

// 采集 IMU 数据的函数
void retrieveIMU() {
    std::string serial_device = "/dev/ttyS3"; // 串口设备路径
    std::ofstream imu_file(root_dir + "/imu_data.txt", std::ios::app);
    imu_file << "# timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,ang_x,ang_y,ang_z\n";

    // 打开串口设备
    int fd = open(serial_device.c_str(), O_RDWR | O_NOCTTY);
    if (fd == -1) {
        perror("open serial");
        return;
    }

    // 配置串口参数
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

    // 初始化 IMU 数据结构
    IMUSerial::IMUDATA current_imu;
    char chrBuf[100];
    unsigned char chrCnt = 0;
    float G = 9.81f;

    // 获取 UTC 时间戳的 Lambda
    auto getUTCTimestamp = []() -> double {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return ts.tv_sec + ts.tv_nsec / 1e9;
    };

    // 写入 IMU 数据到 CSV 文件
    auto writeIMUFrameCSV = [&](const IMUSerial::IMUDATA& frame, double timestamp) {
        Eigen::Quaterniond q(IMUSerial::rpy2R(frame.angle).cast<double>());
        imu_file << std::fixed << std::setprecision(6) << timestamp << ","
                 << frame.a(0) << "," << frame.a(1) << "," << frame.a(2) << ","
                 << frame.w(0) << "," << frame.w(1) << "," << frame.w(2) << ","
                 << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
                 << frame.h(0) << "," << frame.h(1) << "," << frame.h(2) << "\n";
        imu_file.flush();
    };

    char r_buf[1024];
    memset(r_buf, 0, sizeof(r_buf));

    // 无限循环读取串口数据
    while (true) {
        int ret = read(fd, r_buf, sizeof(r_buf));
        if (ret <= 0) {
            printf("uart read failed\n");
            break;
        }

        // 逐字节分析数据帧
        for (int i = 0; i < ret; i++) {
            chrBuf[chrCnt++] = r_buf[i];
            if (chrCnt < 11) continue;

            // 检查帧头合法性
            if ((chrBuf[0] != 0x55) || (chrBuf[1] & 0x50) != 0x50) {
                memmove(&chrBuf[0], &chrBuf[1], 10);
                chrCnt--;
                continue;
            }

            // 解析数据内容
            signed short sData[4];
            memcpy(&sData[0], &chrBuf[2], 8);
            double timestamp = getUTCTimestamp();

            switch (chrBuf[1]) {
                case 0x51: // 加速度
                    for (int j = 0; j < 3; j++)
                        current_imu.a(j) = (float)sData[j] / 32768.0 * 16.0 * G;
                    current_imu.has_acc = true;
                    current_imu.timestamp_acc = timestamp;
                    break;
                case 0x52: // 陀螺仪
                    for (int j = 0; j < 3; j++)
                        current_imu.w(j) = (float)sData[j] / 32768.0 * 2000.0 * 3.14159 / 180.0;
                    current_imu.has_gyro = true;
                    current_imu.timestamp_gyro = timestamp;
                    break;
                case 0x53: // 姿态角
                    for (int j = 0; j < 3; j++)
                        current_imu.angle(j) = (float)sData[j] / 32768.0 * 180.0;
                    current_imu.has_ang = true;
                    current_imu.timestamp_ang = timestamp;
                    break;
                case 0x54: // 磁力计
                    for (int j = 0; j < 3; j++)
                        current_imu.h(j) = (float)sData[j];
                    current_imu.has_h = true;
                    current_imu.timestamp_h = timestamp;
                    break;
            }
            chrCnt = 0;

            // 当收集完一帧完整数据时
            if (current_imu.has_acc && current_imu.has_gyro && current_imu.has_ang && current_imu.has_h) {
                writeIMUFrameCSV(current_imu, current_imu.timestamp_acc);

                ov_core::ImuData message;
                message.timestamp = current_imu.timestamp_acc;
                message.wm << double(current_imu.w.x()), double(current_imu.w.y()), double(current_imu.w.z());
                message.am << double(current_imu.a.x()), double(current_imu.a.y()), double(current_imu.a.z());
                message.hm << double(current_imu.h.x()), double(current_imu.h.y()), double(current_imu.h.z());
                message.Rm = IMUSerial::rpy2R(current_imu.angle).cast<double>();

                // 发布 IMU 数据和姿态
                viz->publishIMU("raw_imu", int64_t(message.timestamp * 1e6), "IMU", message.am, message.wm, Eigen::Quaterniond(message.Rm), message.hm);

                Eigen::Matrix4f T_w_imu = Eigen::Matrix4f::Identity();
                T_w_imu.block(0, 0, 3, 3) = message.Rm.cast<float>();
                viz->showPose("imu_angle", int64_t(message.timestamp * 1e6), T_w_imu, "LOCAL_WORLD", "IMU_R");

                // 重置 IMU 数据结构
                current_imu = IMUSerial::IMUDATA();
            }
        }
    }

    close(fd);
    imu_file.close();
}

void stereoCalibDataGet(const ov_core::CameraData& image_msg) {
    static int image_num = 0;
    const int numTags = april_grid->getTagCols() * april_grid->getTagRows();
    CAMERA_CALIB::ApriltagDetector ad(numTags);

    CAMERA_CALIB::CalibCornerData ccd_good_left;
    CAMERA_CALIB::CalibCornerData ccd_bad_left;
    ad.detectTags(image_msg.images[0], ccd_good_left.corners, ccd_good_left.corner_ids, ccd_good_left.radii,
                    ccd_bad_left.corners, ccd_bad_left.corner_ids, ccd_bad_left.radii);

    vector<Point2f> current_corners_left;
    for (int i_p = 0; i_p < ccd_good_left.corner_ids.size(); i_p++) {
        const Eigen::Vector2d& pt = ccd_good_left.corners[i_p];
        current_corners_left.emplace_back(pt.x(), pt.y());
    }

    bool isFrameAcceptable_left = dqe_left->isFrameAcceptable(image_msg.images[0], current_corners_left);

    CAMERA_CALIB::CalibCornerData ccd_good_right;
    CAMERA_CALIB::CalibCornerData ccd_bad_right;
    ad.detectTags(image_msg.images[1], ccd_good_right.corners, ccd_good_right.corner_ids, ccd_good_right.radii,
                    ccd_bad_right.corners, ccd_bad_right.corner_ids, ccd_bad_right.radii);

    vector<Point2f> current_corners_right;
    for (int i_p = 0; i_p < ccd_good_right.corner_ids.size(); i_p++) {
        const Eigen::Vector2d& pt = ccd_good_right.corners[i_p];
        current_corners_right.emplace_back(pt.x(), pt.y());
    }

    bool isFrameAcceptable_right = dqe_right->isFrameAcceptable(image_msg.images[1], current_corners_right);

    if (isFrameAcceptable_left || isFrameAcceptable_right) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << image_msg.timestamp;
        std::string timestamp_str = oss.str();

        std::string left_path = left_images_dir + "/" + timestamp_str + ".png";
        std::string right_path = right_images_dir + "/" + timestamp_str + ".png";

        cv::imwrite(left_path, image_msg.images[0]);
        cv::imwrite(right_path, image_msg.images[1]);
        image_num++;
    }

    cv::Mat image_combined;
    cv::hconcat(dqe_left->global_coverage_.viz_mat, dqe_right->global_coverage_.viz_mat, image_combined);
    // 图像中间用一条线分割
    cv::line(image_combined, cv::Point(image_combined.cols / 2, 0), cv::Point(image_combined.cols / 2, image_combined.rows), cv::Scalar(255, 255, 255), 2);
    // 图片下方黑框显示各种状态
    cv::line(image_combined, cv::Point(0, image_combined.rows - 50), cv::Point(image_combined.cols, image_combined.rows - 50), cv::Scalar(0, 0, 0), 2);
    cv::Mat status_image = cv::Mat::zeros(cv::Size(image_combined.cols, 50), CV_8UC3);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Image Num: " << image_num << " " << " | Left Coverage Ratio: " << dqe_left->calcCoverageRatio() << " " << " | Right Coverage Ratio: " << dqe_right->calcCoverageRatio();
    cv::putText(status_image, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::vconcat(image_combined, status_image, image_combined);
    
    int64_t time_us = (image_msg.timestamp * 1e6);
    viz->showImage("dete_images", time_us, image_combined, "dete_images", true);

    thread_update_running = false;
}

// 采集相机图像数据
void retrieveCamera() {
    left_images_dir = root_dir + "/images/left/";
    right_images_dir = root_dir + "/images/right/";

    fs::create_directories(left_images_dir);
    fs::create_directories(right_images_dir);

    constexpr int WIDTH = 1280;
    constexpr int HEIGHT = 480;
    constexpr int BUFFER_COUNT = 8;
    std::string device_path = "/dev/video73";

    // 打开视频设备
    int fd = open(device_path.c_str(), O_RDWR);
    if (fd == -1) {
        perror("open /dev/video73");
        return;
    }

    // 设置视频格式
    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    ioctl(fd, VIDIOC_S_FMT, &fmt);

    // 请求缓冲区
    v4l2_requestbuffers req = {};
    req.count = BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ioctl(fd, VIDIOC_REQBUFS, &req);

    Buffer buffers[BUFFER_COUNT];
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        ioctl(fd, VIDIOC_QUERYBUF, &buf);

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        ioctl(fd, VIDIOC_QBUF, &buf);
    }

    // 开启视频流
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd, VIDIOC_STREAMON, &type);

    // 时间同步偏移计算（MONO->UTC）
    struct timespec ts_realtime, ts_mono;
    clock_gettime(CLOCK_REALTIME, &ts_realtime);
    clock_gettime(CLOCK_MONOTONIC, &ts_mono);
    double offset = (ts_realtime.tv_sec + ts_realtime.tv_nsec / 1e9) - (ts_mono.tv_sec + ts_mono.tv_nsec / 1e9);

    // 图像采集循环
    while (true) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        ioctl(fd, VIDIOC_DQBUF, &buf);

        // 计算图像 UTC 时间
        double v4l2_time = buf.timestamp.tv_sec + buf.timestamp.tv_usec / 1e6;
        double utc_time = v4l2_time + offset;

        vector<uchar> data((uchar*)buffers[buf.index].start,
                           (uchar*)buffers[buf.index].start + buf.bytesused);
        cv::Mat full = cv::imdecode(data, cv::IMREAD_COLOR);

        if (!full.empty() && full.cols == WIDTH && full.rows == HEIGHT) {
            ov_core::CameraData image_msg;
            image_msg.images.resize(2);
            image_msg.timestamp = utc_time;
            image_msg.images[0] = full(cv::Rect(0, 0, WIDTH / 2, HEIGHT)).clone();
            image_msg.images[1] = full(cv::Rect(WIDTH / 2, 0, WIDTH / 2, HEIGHT)).clone();

            cv::cvtColor(image_msg.images[0], image_msg.images[0], cv::COLOR_BGR2GRAY);
            cv::cvtColor(image_msg.images[1], image_msg.images[1], cv::COLOR_BGR2GRAY);

            if (is_record_camera == 2) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(6) << utc_time;
                std::string timestamp_str = oss.str();

                std::string left_path = left_images_dir + "/" + timestamp_str + ".png";
                std::string right_path = right_images_dir + "/" + timestamp_str + ".png";

                cv::imwrite(left_path, image_msg.images[0]);
                cv::imwrite(right_path, image_msg.images[1]);
            }
            
            {
                std::lock_guard<std::mutex> lock(camera_queue_mtx);
                camera_queue.push_back(image_msg);
            }

            // print_memory_usage();
            ioctl(fd, VIDIOC_QBUF, &buf); // 重新排队该缓冲区
            usleep(10); // 微小延迟

            if (is_record_camera == 1) {
                if (thread_update_running)
                    continue;
                thread_update_running = true;
                std::thread thread_update(&stereoCalibDataGet, image_msg);
                thread_update.detach();
            }

        } else {
            cerr << "图像解码失败或尺寸错误" << endl;
            ioctl(fd, VIDIOC_QBUF, &buf); // 重新排队该缓冲区
            usleep(10); // 微小延迟
        }
    }

    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }
    close(fd);
}



// 主函数
int main(int argc, char **argv) {
    if (argc > 1) {
        is_record_camera = atoi(argv[1]);
    }

    // 创建时间戳目录
    if (is_record_camera == 0) {
        root_dir += "/imu_calib/";
    } else if (is_record_camera == 1) {
        root_dir += "/camera_calib/";
    } else if (is_record_camera == 2) {
        root_dir += "/camera_imu_calib/";
    } else {
        std::cout << "Invalid argument" << std::endl;
        return -1;
    }

    if (fs::exists(root_dir)) 
        fs::remove_all(root_dir);
    fs::create_directories(root_dir);

    // 初始化可视化器
    viz = std::make_shared<foxglove_viz::Visualizer>(8088, 2);

    // 启动 IMU 线程
    std::thread imu_thread(&retrieveIMU);
    imu_thread.detach();
    std::cout << "IMU thread started." << std::endl;

    // 启动相机线程（可选）
    if (is_record_camera) {
        if (is_record_camera == 1) {
            april_grid = std::make_shared<CAMERA_CALIB::AprilGrid>(std::string(PROJ_DIR) + "/thirdparty/kalibrlib/apps/others/aprilgrid.yaml");
            dqe_left = std::make_shared<FrameQualityEvaluator>();
            dqe_right = std::make_shared<FrameQualityEvaluator>();
        }

        std::thread cam_thread(&retrieveCamera);
        cam_thread.detach();
        std::cout << "Camera thread started." << std::endl;

        // if (is_record_camera == 2) {
            // 主循环：从相机队列中取图并显示
            while (1)
            {
                if (!camera_queue.empty()) {
                    ov_core::CameraData image;
                    {
                        std::lock_guard<std::mutex> lock(camera_queue_mtx);
                        image = camera_queue.front();
                        camera_queue.pop_front();
                    }

                    cv::Mat image_combined;
                    cv::hconcat(image.images[0], image.images[1], image_combined);
                    int64_t time_us = (image.timestamp * 1e6);
                    viz->showImage("track_images", time_us, image_combined, "track_images", true);
                } else {
                    usleep(10); // 队列为空时休眠
                }
            }
        // } else {
        //     while(1) {
        //         usleep(10);
        //     }
        // }
    }

    return 0;
}
