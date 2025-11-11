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
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "foxglove/dash_board.h"  // 可视化仪表板模块
#include "serial_imu/IImuDriver.h"   // IMU 驱动接口
#include "camera_v4l2/ICameraDriver.h" // 相机驱动接口
#include "calibration/aprilgrid.h"

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
  struct OdometryData {
    /// Timestamp of the reading
    double timestamp;

    /// Camera ids for each of the images collected
    std::vector<int> sensor_ids;

    /// Raw image we have collected for each camera
    std::vector<cv::Mat> images;

    Eigen::Matrix4f T_local_imu; // 相机时基下的 IMU 位姿
  };
public:
  // 构造函数，传入 VIO 管理器与可选模拟器
  FGVisualizer(std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr);

  // 结束后输出最终可视化（如参数、RMSE等）
  void visualize_final();

  // 使用独立驱动后的启动接口（向后兼容）
  void startIMUDriver();
  void startCameraDriver();
  void stopDrivers();
  void setDevicesAndLatency(const std::string &imu_dev,
                            const std::string &cam_dev,
                            const std::string &pose_dev,
                            double cam_latency);

  void runRealsenseIO();

  // 运行主函数，启动各线程
  void run();

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

  // 创建调试输出文件夹
  void create_debug_dir();

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
  std::vector<std::pair<double, Eigen::Matrix4f>> poses_imu;
  std::deque<std::pair<double, Eigen::Matrix4f>> poses_imu_dq;
  std::deque<std::pair<double, Eigen::Matrix4f>> poses_imu_odom;

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
  std::string debug_dir = "./debug_data/";

  // 是否为调试模式（控制是否保存图像/IMU）
  bool is_debug = false;

  // 新增：相机驱动指针（最小实现）
  std::shared_ptr<ov_sensors::ICameraDriver> cam_driver_;

  // 新增：独立IMU驱动指针（最小实现）
  std::shared_ptr<class ov_sensors::IImuDriver> imu_driver_;
  std::shared_ptr<std::ofstream> imu_log_file_;

  // ========= 新增配置与状态 =========
  std::string imu_device_ = "/dev/ttyS3";
  std::string cam_device_ = "/dev/video73";
  std::string pose_serial_device_ = "/dev/ttyS5";
  double cam_fixed_latency_ = 0.030; // 默认 30ms

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

  // AprilGrid检测和PnP定位相关函数
  void startAprilGridLocalization();
  void aprilGridDetectionThread();
  bool solvePnP(const std::vector<Eigen::Vector2d>& points2d,
                const std::vector<Eigen::Vector3d>& points3d,
                const std::vector<double>& cam_params,
                Eigen::Matrix4d& T_grid_to_cam);

  // AprilGrid检测相关成员变量
  std::atomic<bool> april_grid_running_{false};
  std::thread april_grid_thread_;
  std::shared_ptr<CAMERA_CALIB::AprilGrid> april_grid_;
  std::deque<OdometryData> april_grid_image_queue_;
  std::mutex april_grid_queue_mtx_;
  std::condition_variable april_grid_cv_;
  size_t april_grid_queue_max_ = 10; // 限制队列长度

  // ========== Ceres AprilGrid位姿优化器 ==========

  // 简化的重投影误差代价函数
  class ReprojectionError {
  public:
    ReprojectionError(const Eigen::Vector2d& point2d,
                      const Eigen::Vector3d& point3d,
                      const std::vector<double>& cam_params,
                      const Eigen::Matrix4d& T_ci)
      : point3d_(point3d), cam_params_(cam_params), T_ci_(T_ci) {
        point2d_.x() = point2d.x() - cam_params[2];
        point2d_.y() = point2d.y() - cam_params[3];
      }

    // 注意：此处使用 EigenQuaternionParameterization，对应存储顺序为 [x,y,z,w]
    template <typename T>
    bool operator()(const T* const pose_q, const T* const pose_t, T* residuals) const {
      // pose_q: [x,y,z,w]  -> 构造 Eigen::Quaternion(w,x,y,z)
      Eigen::Quaternion<T> q_iw(pose_q[3], pose_q[0], pose_q[1], pose_q[2]);
      // pose_t: [tx,ty,tz]
      Eigen::Matrix<T,3,1> t_iw(pose_t[0], pose_t[1], pose_t[2]);

      Eigen::Quaternion<T> q_cw = Eigen::Quaternion<T>(T_ci_.block<3,3>(0,0).cast<T>()) * q_iw;
      Eigen::Matrix<T,3,1> t_cw = Eigen::Matrix<T,3,1>(T_ci_.block<3,1>(0,3).cast<T>()) + q_cw * t_iw;

      Eigen::Matrix<T,3,1> point_3d_T = point3d_.cast<T>();
      Eigen::Matrix<T,3,1> point_in_c = q_cw * point_3d_T + t_cw;

      if (point_in_c(2) <= T(0)) {
        return false;
      }

      residuals[0] = point_in_c(0) / point_in_c(2) * T(cam_params_[0]) - T(point2d_(0));
      residuals[1] = point_in_c(1) / point_in_c(2) * T(cam_params_[1]) - T(point2d_(1));

      return true;
    }

  private:
    Eigen::Vector2d point2d_;
    Eigen::Vector3d point3d_;
    Eigen::Matrix4d T_ci_;
    std::vector<double> cam_params_;
  };

  // Ceres优化器函数
  bool optimizeIMUPoseWithCeres(const std::vector<Eigen::Vector2d>& left_points2d,
                                const std::vector<Eigen::Vector2d>& right_points2d,
                                const std::vector<Eigen::Vector3d>& left_points3d,
                                const std::vector<Eigen::Vector3d>& right_points3d,
                                const Eigen::Matrix4d& T_ItoC_left,
                                const Eigen::Matrix4d& T_ItoC_right,
                                const std::vector<double>& cam_left,
                                const std::vector<double>& cam_right,
                                Eigen::Matrix4d& T_grid_to_imu);
};

} // namespace ov_msckf

#endif // OV_MSCKF_FGVISUALIZER_H
