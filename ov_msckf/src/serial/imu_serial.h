#ifndef OV_MSCKF_IMUSERIAL_H
#define OV_MSCKF_IMUSERIAL_H

#include <Eigen/Core>
#include <Eigen/Geometry>

Eigen::Matrix3f rpy2R(const Eigen::Vector3f& rpy);

class IMUSerial {
    
private:
    int fd; // 文件描述符
    char chrBuf[100]; // 缓存接收到的数据
    unsigned char chrCnt; // 缓存计数器
    float G = 9.81; // 重力加速度

public:
    struct IMUDATA {
        double time; // 时间戳
        Eigen::Vector3f a; // 加速度
        Eigen::Vector3f w; // 角速度
        Eigen::Vector3f angle; // 角度(Roll,pitch,yaw)
        Eigen::Vector3f h; // 磁场
        double timestamp_acc = 0;
        double timestamp_gyro = 0;
        double timestamp_ang = 0;
        double timestamp_h = 0;
        bool has_acc = false;
        bool has_gyro = false;
        bool has_ang = false;
        bool has_h = false;
    };

    // 构造函数
    IMUSerial();

    // 打开串口
    int openSerial(const char* pathname);

    // 配置串口参数
    int setSerial();

    // 关闭串口
    int closeSerial();

    // 发送数据
    int sendData(const char* send_buffer, int length);

    // 接收数据
    int recvData(char* recv_buffer, int length);

    // 解析接收到的数据
    void parseData(char chr, IMUDATA& data);

    // 主循环
    void run();
};

#endif // OV_MSCKF_IMUSERIAL_H