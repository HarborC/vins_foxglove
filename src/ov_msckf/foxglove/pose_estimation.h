#pragma once
#include "calibration/aprilgrid.h"
#include "calibration/apriltags.h"

#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

namespace ov_msckf {

class VioManager;

class PoseEstimation {
public:
    PoseEstimation(std::shared_ptr<VioManager> app);
    ~PoseEstimation() {}

    bool feed(const double& timestamp, const std::vector<cv::Mat> images);
private:
    std::shared_ptr<VioManager> _app;
    std::shared_ptr<CAMERA_CALIB::AprilGrid> april_grid_;
    std::shared_ptr<CAMERA_CALIB::ApriltagDetector> detector_;

    bool optimizeIMUPoseWithCeres(const std::vector<Eigen::Vector2d>& left_points2d,
                                  const std::vector<Eigen::Vector2d>& right_points2d,
                                  const std::vector<Eigen::Vector3d>& left_points3d,
                                  const std::vector<Eigen::Vector3d>& right_points3d,
                                  const Eigen::Matrix4d& T_ItoC_left,
                                  const Eigen::Matrix4d& T_ItoC_right,
                                  const std::vector<double>& cam_left,
                                  const std::vector<double>& cam_right,
                                  Eigen::Matrix4d& T_grid_to_imu);

    bool solvePnP(const std::vector<Eigen::Vector2d>& points2d,
                  const std::vector<Eigen::Vector3d>& points3d,
                  const std::vector<double>& cam_params,
                  Eigen::Matrix4d& T_grid_to_cam);
};

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

} // namespace ov_msckf