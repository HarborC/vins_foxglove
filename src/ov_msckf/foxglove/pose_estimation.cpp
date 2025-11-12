#include "pose_estimation.h"
#include "core/VioManager.h"
#include "state/State.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "calibration/apriltags.h"
#include "calibration/aprilgrid.h"

namespace ov_msckf {

PoseEstimation::PoseEstimation(std::shared_ptr<VioManager> app) : _app(app) {
    std::string april_grid_config = std::string(PROJ_DIR) + "/thirdparty/kalibrlib/apps/others/aprilgrid.yaml";
    april_grid_ = std::make_shared<CAMERA_CALIB::AprilGrid>(april_grid_config);

    const int numTags = april_grid_->getTagCols() * april_grid_->getTagRows();
    PRINT_INFO("AprilGrid detection thread initialized with %d tags.\n", numTags);
    detector_ = std::make_shared<CAMERA_CALIB::ApriltagDetector>(numTags);
}

bool PoseEstimation::feed(const double& timestamp, const std::vector<cv::Mat> images) {
    // 基本防御性检查
    if (images.size() < 2) {
      PRINT_DEBUG("[APRIL_GRID] invalid images vector size=%zu\n", images.size());
      return false;
    }
    if (images[0].empty() || images[1].empty()) {
      PRINT_DEBUG("[APRIL_GRID] empty image. L=%d R=%d\n", images[0].empty(), images[1].empty());
      return false;
    }

    PRINT_DEBUG("[APRIL_GRID] processing ts=%.6f size=%dx%d\n", timestamp, images[0].cols, images[0].rows);


    // 检测AprilGrid - 左右相机
    CAMERA_CALIB::CalibCornerData corners_left_good, corners_left_bad;
    CAMERA_CALIB::CalibCornerData corners_right_good, corners_right_bad;

    if (images[0].empty()) {
      PRINT_ERROR("AprilGrid left image is empty.\n");
    }

    detector_->detectTags(images[0], corners_left_good.corners, corners_left_good.corner_ids, corners_left_good.radii,
                          corners_left_bad.corners, corners_left_bad.corner_ids, corners_left_bad.radii);

    PRINT_DEBUG("[APRIL_GRID] tags detection for right image ts=%.6f\n", timestamp);

    detector_->detectTags(images[1], corners_right_good.corners, corners_right_good.corner_ids, corners_right_good.radii,
                          corners_right_bad.corners, corners_right_bad.corner_ids, corners_right_bad.radii);

    // 检查是否有足够的角点
    if (corners_left_good.corner_ids.size() < 12 && corners_right_good.corner_ids.size() < 12) {
      return false; // 角点太少，跳过
    }

    double scale = (_app->get_params().downsample_cameras)? 0.5 : 1.0;

    std::vector<Eigen::Vector2d> point2d_left_normalized;
    std::vector<Eigen::Vector3d> point3d_left;
    auto intrinsic_left = _app->get_state()->_cam_intrinsics_cameras.at(0);
    if (!intrinsic_left) {
      PRINT_DEBUG("[APRIL_GRID] left intrinsics not ready, skip\n");
      return false;
    }
    cv::Matx33d K_left_cv = intrinsic_left->get_K();
    std::vector<double> cam_left;
    cam_left.push_back(K_left_cv(0, 0)); // fx
    cam_left.push_back(K_left_cv(1, 1)); // fy
    cam_left.push_back(K_left_cv(0, 2)); // cx
    cam_left.push_back(K_left_cv(1, 2)); // cy
    for (size_t i=0;i<corners_left_good.corner_ids.size();++i) {
      int tag_id = corners_left_good.corner_ids[i];
      point3d_left.push_back(april_grid_->aprilgrid_corner_pos_3d[tag_id].head<3>());
      Eigen::Vector2d pt_norm = intrinsic_left->undistort_d(corners_left_good.corners[i] * scale);

      Eigen::Vector2d pt_homog;
      pt_homog << pt_norm(0) * cam_left[0] + cam_left[2], pt_norm(1) * cam_left[1] + cam_left[3];
      point2d_left_normalized.push_back(pt_homog);
    }

    std::vector<Eigen::Vector2d> point2d_right_normalized;
    std::vector<Eigen::Vector3d> point3d_right;
    auto intrinsic_right = _app->get_state()->_cam_intrinsics_cameras.at(1);
    if (!intrinsic_right) {
      PRINT_DEBUG("[APRIL_GRID] right intrinsics not ready, skip\n");
      return false;
    }
    cv::Matx33d K_right_cv = intrinsic_right->get_K();
    std::vector<double> cam_right;
    cam_right.push_back(K_right_cv(0, 0)); // fx
    cam_right.push_back(K_right_cv(1, 1)); // fy
    cam_right.push_back(K_right_cv(0, 2)); // cx
    cam_right.push_back(K_right_cv(1, 2)); // cy
    for (size_t i=0;i<corners_right_good.corner_ids.size();++i) {
      int tag_id = corners_right_good.corner_ids[i];
      point3d_right.push_back(april_grid_->aprilgrid_corner_pos_3d[tag_id].head<3>());

      Eigen::Vector2d pt_norm = intrinsic_right->undistort_d(corners_right_good.corners[i] * scale);
      Eigen::Vector2d pt_homog;
      pt_homog << pt_norm(0) * cam_right[0] + cam_right[2], pt_norm(1) * cam_right[1] + cam_right[3];
      point2d_right_normalized.push_back(pt_homog);
    }

    Eigen::Matrix4d T_imu_cam_left = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_cam_imu_left = Eigen::Matrix4d::Identity();
    {
      auto calib_left = _app->get_state()->_calib_IMUtoCAM.at(0);
      Eigen::Matrix3d R_imu_to_cam = calib_left->Rot();
      Eigen::Vector3d p_imu_in_cam = calib_left->pos();
      T_cam_imu_left.block<3,3>(0,0) = R_imu_to_cam;
      T_cam_imu_left.block<3,1>(0,3) = p_imu_in_cam;
      Eigen::Matrix3d R_cam_to_imu = R_imu_to_cam.transpose();
      Eigen::Vector3d p_cam_in_imu = -R_cam_to_imu * p_imu_in_cam;
      T_imu_cam_left.block<3,3>(0,0) = R_cam_to_imu;
      T_imu_cam_left.block<3,1>(0,3) = p_cam_in_imu;
    }

    Eigen::Matrix4d T_imu_cam_right = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_cam_imu_right = Eigen::Matrix4d::Identity();
    {
      auto calib_right = _app->get_state()->_calib_IMUtoCAM.at(1);
      Eigen::Matrix3d R_imu_to_cam = calib_right->Rot();
      Eigen::Vector3d p_imu_in_cam = calib_right->pos();
      T_cam_imu_right.block<3,3>(0,0) = R_imu_to_cam;
      T_cam_imu_right.block<3,1>(0,3) = p_imu_in_cam;
      Eigen::Matrix3d R_cam_to_imu = R_imu_to_cam.transpose();
      Eigen::Vector3d p_cam_in_imu = -R_cam_to_imu * p_imu_in_cam;
      T_imu_cam_right.block<3,3>(0,0) = R_cam_to_imu;
      T_imu_cam_right.block<3,1>(0,3) = p_cam_in_imu;
      
    }

    PRINT_DEBUG("[APRIL_GRID] detected tags: L=%zu R=%zu\n",
                corners_left_good.corner_ids.size(), corners_right_good.corner_ids.size());

    Eigen::Matrix4d T_grid_to_cam, T_grid_to_imu;
    if (solvePnP(point2d_left_normalized, point3d_left, cam_left, T_grid_to_cam)) {
      T_grid_to_imu = T_imu_cam_left * T_grid_to_cam;
      PRINT_DEBUG("[APRIL_GRID] Left camera PnP solved, tags=%zu\n", corners_left_good.corner_ids.size());
    } else if (solvePnP(point2d_right_normalized, point3d_right, cam_right, T_grid_to_cam)) {
      T_grid_to_imu = T_imu_cam_right * T_grid_to_cam;
      PRINT_DEBUG("[APRIL_GRID] Right camera PnP solved, tags=%zu\n", corners_right_good.corner_ids.size());
    } else {
      return false;
    }

    PRINT_DEBUG("[APRIL_GRID] PnP pose estimated\n");

    // 准备用ceres解算新的pose
    bool is_success = optimizeIMUPoseWithCeres(point2d_left_normalized, point2d_right_normalized, 
                                               point3d_left, point3d_right, T_cam_imu_left, T_cam_imu_right,
                                               cam_left, cam_right, T_grid_to_imu);


    Eigen::Matrix4d T_imu_to_grid = Eigen::Matrix4d::Identity();
    T_imu_to_grid.block(0,0,3,3) = T_grid_to_imu.block(0,0,3,3).transpose();
    T_imu_to_grid.block(0,3,3,1) = -T_imu_to_grid.block(0,0,3,3) * T_grid_to_imu.block(0,3,3,1);
    Eigen::Matrix4f T_imu_to_grid_f = T_imu_to_grid.cast<float>();
    return true;
}

bool PoseEstimation::solvePnP(const std::vector<Eigen::Vector2d>& points2d,
                              const std::vector<Eigen::Vector3d>& points3d,
                              const std::vector<double>& cam_params,
                              Eigen::Matrix4d& T_grid_to_cam) {

  if (points2d.size() != points3d.size() || points2d.size() < 4) {
    PRINT_ERROR("solvePnP: need N>=4 and 2D/3D sizes equal, got %zu / %zu\n",
                points2d.size(), points3d.size());
    return false;
  }
  if (cam_params.size() < 4) {
    PRINT_ERROR("solvePnP: cam_params must have at least fx, fy, cx, cy\n");
    return false;
  }

  // 使用RANSAC进行PnP求解
  cv::Mat rvec, tvec;
  cv::Mat inliers;
  cv::Mat K = (cv::Mat_<double>(3, 3) << cam_params[0], 0, cam_params[2],
                                        0, cam_params[1], cam_params[3],
                                        0, 0, 1);

  PRINT_DEBUG("solvePnP: points2d=%zu, points3d=%zu\n", points2d.size(), points3d.size());

  // 将输入点转换为cv::Mat格式
  std::vector<cv::Point2f> cv_points2d;
  for (const auto& pt : points2d) {
    cv_points2d.emplace_back(static_cast<float>(pt.x()), static_cast<float>(pt.y()));
  } 
  std::vector<cv::Point3f> cv_points3d;
  for (const auto& pt : points3d) {
    cv_points3d.emplace_back(static_cast<float>(pt.x()), static_cast<float>(pt.y()), static_cast<float>(pt.z()));
  }

  PRINT_DEBUG("solvePnP: cv_points2d=%zu, cv_points3d=%zu\n", cv_points2d.size(), cv_points3d.size());

  // points2d是归一化平面坐标
  bool success = cv::solvePnPRansac(cv_points3d, cv_points2d, K, cv::Mat::zeros(4, 1, CV_64F),
                                    rvec, tvec, false, 100, 8.0, 0.99, inliers);

  if (!success || inliers.rows < 6) {
    return false; // PnP求解失败或内点太少
  }

  // 将旋转向量转换为旋转矩阵
  cv::Mat R_left;
  cv::Rodrigues(rvec, R_left);

  // 计算左相机到AprilGrid的变换
  T_grid_to_cam = Eigen::Matrix4d::Identity();

  cv::Mat R_eigen, tvec_eigen;
  R_left.convertTo(R_eigen, CV_64F);
  tvec.convertTo(tvec_eigen, CV_64F);

  T_grid_to_cam.block(0, 0, 3, 3) = Eigen::Map<Eigen::Matrix3d>(R_eigen.ptr<double>());
  T_grid_to_cam.block(0, 3, 3, 1) = Eigen::Map<Eigen::Vector3d>(tvec_eigen.ptr<double>());

  return true;
}

// 使用Ceres优化IMU位姿 - 根据新的ReprojectionError设计重写
bool PoseEstimation::optimizeIMUPoseWithCeres(const std::vector<Eigen::Vector2d>& left_points2d,
                                              const std::vector<Eigen::Vector2d>& right_points2d,
                                              const std::vector<Eigen::Vector3d>& left_points3d,
                                              const std::vector<Eigen::Vector3d>& right_points3d,
                                              const Eigen::Matrix4d& T_ItoC_left,
                                              const Eigen::Matrix4d& T_ItoC_right,
                                              const std::vector<double>& cam_left,
                                              const std::vector<double>& cam_right,
                                              Eigen::Matrix4d& T_grid_to_imu) {

  // 创建Ceres优化问题
  ceres::Problem problem;
  ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); // 鲁棒损失函数是不是有问题

  // 初始IMU位姿参数
  // 使用 EigenQuaternionParameterization: 四元数内存布局 [x,y,z,w]
  // pose_q[0-3]: [qx, qy, qz, qw]
  // pose_t[0-2]: 平移 [tx, ty, tz]
  double pose_q[4];
  double pose_t[3] = {0, 0, 0}; // 前4个参数预留，使用后3个存储平移

  // 从当前T_grid_to_imu矩阵初始化参数
  Eigen::Matrix3d R_init = T_grid_to_imu.block<3,3>(0, 0);
  Eigen::Quaterniond q_init(R_init);
  // 存储为 [x,y,z,w]
  pose_q[0] = q_init.x();
  pose_q[1] = q_init.y();
  pose_q[2] = q_init.z();
  pose_q[3] = q_init.w();
  pose_t[0] = T_grid_to_imu(0, 3);
  pose_t[1] = T_grid_to_imu(1, 3);
  pose_t[2] = T_grid_to_imu(2, 3);

  // 添加左相机重投影误差约束
  for (size_t i = 0; i < left_points2d.size() && i < left_points3d.size(); i++) {
    // 转换为Eigen::Vector2d和Eigen::Vector3d
    Eigen::Vector2d point2d = left_points2d[i];
    Eigen::Vector3d point3d = left_points3d[i];

    ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3>(
        new ReprojectionError(point2d, point3d, cam_left, T_ItoC_left));

    problem.AddResidualBlock(cost_function, loss_function, pose_q, pose_t);
  }

  // 添加右相机重投影误差约束
  for (size_t i = 0; i < right_points2d.size() && i < right_points3d.size(); i++) {
    // 转换为Eigen::Vector2d和Eigen::Vector3d
    Eigen::Vector2d point2d = right_points2d[i];
    Eigen::Vector3d point3d = right_points3d[i];

    ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3>(
        new ReprojectionError(point2d, point3d, cam_right, T_ItoC_right));

    problem.AddResidualBlock(cost_function, loss_function, pose_q, pose_t);
  }

  // 添加四元数参数化约束（确保单位四元数）
  // 使用 EigenQuaternionParameterization（期望 [x,y,z,w] 布局）
  ceres::LocalParameterization* quaternion_parameterization = new ceres::EigenQuaternionParameterization;
  problem.SetParameterization(pose_q, quaternion_parameterization);

  // 配置求解器选项
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 10;
  options.num_threads = 2;

  // 求解
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // 从优化后的参数更新变换矩阵（从 [x,y,z,w] 还原 Eigen::Quaterniond(w,x,y,z)）
  Eigen::Quaterniond q_optimized(pose_q[3], pose_q[0], pose_q[1], pose_q[2]);
  T_grid_to_imu.block<3,3>(0, 0) = q_optimized.toRotationMatrix();
  T_grid_to_imu.block<3,1>(0, 3) = Eigen::Vector3d(pose_t[0], pose_t[1], pose_t[2]);

  PRINT_DEBUG("[CERES_OPTIM] Pose optimization: %s, final cost: %.6f, iterations: %d\n",
             summary.BriefReport().c_str(), summary.final_cost, (int)summary.iterations.size());

  return summary.IsSolutionUsable() && summary.final_cost < 1e-3;
}

}