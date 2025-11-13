#ifndef OV_MSCKF_FGVISUALIZER_H
#define OV_MSCKF_FGVISUALIZER_H

// 标准库
#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <array>

// 第三方库
#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <foxglove/visualizer.h>

#include "foxglove/dash_board.h"  // 可视化仪表板模块
#include "utils/sensor_data.h"


// ========== 命名空间定义 ==========

namespace ov_core {
class YamlParser;         // 配置文件解析器
struct CameraData;        // 相机数据结构
}

namespace ov_msckf {

// 前置声明
class VioManager;
class Simulator;

// 可视化主类定义
class FGVisualizer {
public:
  // 构造函数，传入 VIO 管理器与可选模拟器
  FGVisualizer(std::shared_ptr<VioManager> app, bool is_viz = true);
  ~FGVisualizer();

  // 结束后输出最终可视化（如参数、RMSE等）
  void visualize_final();

  // 使用独立驱动后的启动接口（向后兼容）
  // void startIMUDriver();
  // void startCameraDriver();
  void startCore();
  void stopDrivers();
  void setDevicesAndLatency(const std::string &imu_dev,
                            const std::string &cam_dev,
                            const std::string &pose_dev,
                            double cam_latency);

  // 运行主函数，启动各线程
  void run();

  void feedIMU(const ov_core::ImuData& msg);

  void feedStereo(ov_core::CameraData& msg);

  // 可视化主接口，调用状态、轨迹、特征等发布
  void visualize();

  void visualize_odometry(double timestamp) ;

  // 发布历史图像
  void publish_images();
    
  // 发布状态估计结果
  void publish_state();

  // 发布 3D 特征点云（MSCKF/SLAM/Aruco）
  void publish_features();

  // 发布相机的位姿、内参矩阵等
  void publish_cameras();

  // 发布IMU等
  void publish_raw_imu(const ov_core::ImuData& s);

  // 创建调试输出文件夹
  void create_debug_dir();

  // VIO 管理器核心对象
  std::shared_ptr<VioManager> _app;

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
  std::vector<std::pair<double, Eigen::Matrix4f>> poses_imu;
  std::deque<std::pair<double, Eigen::Matrix4f>> poses_imu_dq;
  std::deque<std::pair<double, Eigen::Matrix4f>> poses_imu_odom;

  // 可视化器指针
  foxglove_viz::Visualizer::Ptr _viz = nullptr;
  std::string pose_serial_device_ = "/dev/ttyS5";
  double cam_fixed_latency_ = 0.030; // 默认 30ms

  // 实时仪表板
  DashBoard::Ptr _dash_board;

  // 最近一帧图像的时间戳与数据（用于可视化）
  double last_images_timestamp = 0;

  // 新一帧 IMU 的时间戳（备用）
  double new_imu_timestamp = -1;

  // 日志输出根目录
  std::string debug_dir = std::string(PROJ_DIR) + "/debug_data/";
  std::shared_ptr<std::ofstream> odom_out_ptr_;
  std::shared_ptr<std::ofstream> odom_cam_ptr_;

  // IMU 时间（camera 时基下）
  std::mutex imu_time_mtx_;
  double last_imu_timestamp_inC_ = -1.0;
  bool imu_new_flag_ = false;
  std::condition_variable imu_cv_;
  std::thread consumer_thread_;
  std::atomic<bool> running_{false};

  // 姿态串口
  int pose_serial_fd_ = -1;
  bool pose_serial_open_ = false;

  // ========= 六自由度姿态异步发送线程（队列版，确保每帧发送） =========
  struct Pose6DFrame {
    std::array<float,6> data{}; // x,y,z,roll,pitch,yaw
    double timestamp{0.0};
  };
  std::deque<Pose6DFrame> pose_queue_;
  std::mutex pose_mtx_;
  std::condition_variable pose_cv_;
  std::atomic<bool> pose_thread_running_{false};
  std::thread pose_thread_;
  size_t pose_queue_max_ = 2048; // 极端情况下的上限，避免无限增长

  // 启动 / 停止 姿态发送线程
  void startPoseThread();
};

} // namespace ov_msckf

#endif // OV_MSCKF_FGVISUALIZER_H
