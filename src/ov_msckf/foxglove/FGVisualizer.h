#ifndef OV_MSCKF_FGVISUALIZER_H
#define OV_MSCKF_FGVISUALIZER_H

// 标准库
#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>

// 第三方库
#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <foxglove/visualizer.h>

#include "foxglove/dash_board.h"  // 可视化仪表板模块

// ========== 命名空间定义 ==========

namespace ov_core {
class YamlParser;         // 配置文件解析器
struct CameraData;        // 相机数据结构
}

namespace IMUSerial {

// IMU 数据结构定义
struct IMUDATA {
  double time;                  // 时间戳（暂未使用）
  Eigen::Vector3f a;            // 加速度
  Eigen::Vector3f w;            // 角速度
  Eigen::Vector3f angle;        // 欧拉角（roll, pitch, yaw）
  Eigen::Vector3f h;            // 地磁传感器

  // 各类数据的时间戳
  double timestamp_acc = 0;
  double timestamp_gyro = 0;
  double timestamp_ang = 0;
  double timestamp_h = 0;

  // 标记是否接收到该类数据
  bool has_acc = false;
  bool has_gyro = false;
  bool has_ang = false;
  bool has_h = false;
};

// 将欧拉角转换为旋转矩阵的函数声明
Eigen::Matrix3f rpy2R(const Eigen::Vector3f& rpy);
} // namespace IMUSerial

namespace ov_msckf {

// 前置声明
class VioManager;
class Simulator;

// mmap 视频帧缓冲区结构
struct Buffer {
  void* start;
  size_t length;
};

// 可视化主类定义
class FGVisualizer {
public:
  // 构造函数，传入 VIO 管理器与可选模拟器
  FGVisualizer(std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr);

  // 结束后输出最终可视化（如参数、RMSE等）
  void visualize_final();

  // 启动 IMU 数据采集线程
  void retrieveIMU();

  // 启动相机图像采集线程
  void retrieveCamera();

  // 运行主函数，启动各线程
  void run();

  // 可视化主接口，调用状态、轨迹、特征等发布
  void visualize();

  // 发布历史图像
  void publish_images();

  // 发布状态估计结果
  void publish_state();

  // 发布 3D 特征点云（MSCKF/SLAM/Aruco）
  void publish_features();

  // 发布相机的位姿、内参矩阵等
  void publish_cameras();

  // 创建调试输出文件夹
  void create_debug_dir();

#if ROS_AVAILABLE == 1
  // ROS 下的订阅器初始化函数
  void setup_subscribers(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<ov_core::YamlParser> parser);

  // IMU 消息回调
  void callback_inertial(const sensor_msgs::Imu::ConstPtr &msg);

  // 单目图像回调
  void callback_monocular(const sensor_msgs::ImageConstPtr &msg0, int cam_id0);

  // 双目图像同步回调
  void callback_stereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1, int cam_id0, int cam_id1);

  // ROS 订阅器与同步器
  ros::Subscriber sub_imu;
  std::vector<ros::Subscriber> subs_cam;

  // 同步策略定义（近似时间同步）
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;

  // 同步器列表（双目）
  std::vector<std::shared_ptr<message_filters::Synchronizer<sync_pol>>> sync_cam;

  // 原始图像订阅器列表（双目）
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> sync_subs_cam;
#endif

  // VIO 管理器核心对象
  std::shared_ptr<VioManager> _app;

  // 模拟器（为空则非仿真模式）
  std::shared_ptr<Simulator> _sim;

  // 是否已有后端线程在运行
  std::atomic<bool> thread_update_running;

  // 相机数据队列，用于同步 IMU 与图像
  std::deque<ov_core::CameraData> camera_queue;
  std::mutex camera_queue_mtx;

  // 每路相机最近一次图像时间戳
  std::map<int, double> camera_last_timestamp;

  // 最近一次状态/图像可视化的时间戳
  double last_visualization_timestamp = 0;
  double last_visualization_timestamp_image = 0;

  // 真值轨迹（用于评估）
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;

  // 累积误差统计
  double summed_mse_ori = 0.0;
  double summed_mse_pos = 0.0;
  double summed_nees_ori = 0.0;
  double summed_nees_pos = 0.0;
  size_t summed_number = 0;

  // 系统运行起止时间
  bool start_time_set = false;
  boost::posix_time::ptime rT1, rT2;

  // IMU 轨迹
  unsigned int poses_seq_imu = 0;
  std::vector<Eigen::Matrix4f> poses_imu;

  // 可视化器指针
  foxglove_viz::Visualizer::Ptr _viz;

  // 实时仪表板
  DashBoard::Ptr _dash_board;

  // 最近一帧图像的时间戳与数据（用于可视化）
  double last_images_timestamp = 0;
  std::vector<cv::Mat> last_images;

  // 新一帧 IMU 的时间戳（备用）
  double new_imu_timestamp = -1;

  // 日志输出根目录
  std::string debug_dir = "/home/cat/projects/debug/";

  // 是否为调试模式（控制是否保存图像/IMU）
  bool is_debug = false;
};

} // namespace ov_msckf

#endif // OV_MSCKF_FGVISUALIZER_H
