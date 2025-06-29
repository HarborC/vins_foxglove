#ifndef OV_MSCKF_FGVISUALIZER_H
#define OV_MSCKF_FGVISUALIZER_H

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>

#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

#if ROS_AVAILABLE == 1
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#endif

#include <foxglove/visualizer.h>

#include "foxglove/dash_board.h"

namespace ov_core {
class YamlParser;
struct CameraData;
} // namespace ov_core

namespace ov_msckf {

class VioManager;
class Simulator;

struct Buffer {
  void* start;
  size_t length;
};

class FGVisualizer {

public:
  /**
   * @brief Default constructor
   * @param nh ROS node handler
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */
  FGVisualizer(std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr);

    /**
   * @brief After the run has ended, print results
   */
  void visualize_final();

  void retrieveIMU();

  void retrieveCamera();

  void run();

  void visualize();

  void publish_images();

  void publish_state();

  void publish_features();

  void publish_cameras();

  void create_debug_dir();

  #if ROS_AVAILABLE == 1
  void setup_subscribers(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<ov_core::YamlParser> parser);

  /// Callback for inertial information
  void callback_inertial(const sensor_msgs::Imu::ConstPtr &msg);

  /// Callback for monocular cameras information
  void callback_monocular(const sensor_msgs::ImageConstPtr &msg0, int cam_id0);

  /// Callback for synchronized stereo camera information
  void callback_stereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1, int cam_id0, int cam_id1);
  
  // Our subscribers and camera synchronizers
  ros::Subscriber sub_imu;
  std::vector<ros::Subscriber> subs_cam;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  std::vector<std::shared_ptr<message_filters::Synchronizer<sync_pol>>> sync_cam;
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> sync_subs_cam;
  #endif

  /// Core application of the filter system
  std::shared_ptr<VioManager> _app;

  /// Simulator (is nullptr if we are not sim'ing)
  std::shared_ptr<Simulator> _sim;

  // Thread atomics
  std::atomic<bool> thread_update_running;

  /// Queue up camera measurements sorted by time and trigger once we have
  /// exactly one IMU measurement with timestamp newer than the camera measurement
  /// This also handles out-of-order camera measurements, which is rare, but
  /// a nice feature to have for general robustness to bad camera drivers.
  std::deque<ov_core::CameraData> camera_queue;
  std::mutex camera_queue_mtx;

  // Last camera message timestamps we have received (mapped by cam id)
  std::map<int, double> camera_last_timestamp;

  // Last timestamp we visualized at
  double last_visualization_timestamp = 0;
  double last_visualization_timestamp_image = 0;

  // Our groundtruth states
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;

  // Groundtruth infomation
  double summed_mse_ori = 0.0;
  double summed_mse_pos = 0.0;
  double summed_nees_ori = 0.0;
  double summed_nees_pos = 0.0;
  size_t summed_number = 0;

  // Start and end timestamps
  bool start_time_set = false;
  boost::posix_time::ptime rT1, rT2;

  // For path viz
  unsigned int poses_seq_imu = 0;
  std::vector<Eigen::Matrix4f> poses_imu;

  // Our visualizer
  foxglove_viz::Visualizer::Ptr _viz;
  DashBoard::Ptr _dash_board;

  double last_images_timestamp = 0;
  std::vector<cv::Mat> last_images;

  double new_imu_timestamp = -1;

  std::string debug_dir = "/home/cat/projects/debug/";

  bool is_debug = false;
};
}

#endif // OV_MSCKF_FGVISUALIZER_H