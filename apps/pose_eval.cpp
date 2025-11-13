#include "vi_dataset.h"
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"
#include "foxglove/FGVisualizer.h"

#include "utils/sensor_data.h"
#include "utils/memory_utils.h"
#include "foxglove/pose_estimation.h"
#include "camera_v4l2/CameraFrame.h"

using namespace ov_msckf;

// 主函数
int main(int argc, char **argv) {
    std::string data_dir = "/home/cat/projects/open_vins/debug_data/20251112_214818";
    if (argc > 1) {
        data_dir = argv[1];
    }

    std::string config_path = std::string(PROJ_DIR) + "/config/ours/estimator_config.yaml";
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);

    // 初始化 VIO 管理器参数
    VioManagerOptions params;
    params.print_and_load(parser);
    std::shared_ptr<VioManager> sys = std::make_shared<VioManager>(params);
    PoseEstimation pose_estimator(sys);
    std::shared_ptr<FGVisualizer> viz_ = std::make_shared<FGVisualizer>(sys);

    VIDataset::Ptr g_vi_dataset(new VIDataset(data_dir, 1));
    std::shared_ptr<std::ofstream> gt_out_ptr_(new std::ofstream(data_dir + "/gt_poses.txt"));

    std::vector<std::vector<float>> april_grid_pcd = pose_estimator.getAprilGridPCD();

    ov_sensors::CameraFrame cam_frame;
    std::vector<Eigen::Matrix4f> gt_poses;
    while(g_vi_dataset->loadNextStereoData(cam_frame)) {
        Eigen::Matrix4f T_imu_to_grid_f;
        if (pose_estimator.feed(cam_frame.timestamp_raw, cam_frame.images, T_imu_to_grid_f)) {
            Eigen::Quaternionf q(T_imu_to_grid_f.block<3,3>(0,0));
            Eigen::Vector3f t = T_imu_to_grid_f.block<3,1>(0,3);
            (*gt_out_ptr_) << std::fixed << std::setprecision(6) << cam_frame.timestamp_raw << " "
                           << t.x() << " " << t.y() << " " << t.z() << " "
                           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            gt_poses.push_back(T_imu_to_grid_f);
        }
        if (viz_->_viz) {
            int64_t time_us = int64_t(cam_frame.timestamp_raw * 1e6);
            viz_->_viz->showPath("gt_path", time_us, gt_poses, "LOCAL_WORLD");
            viz_->_viz->showImage("dete_images", time_us, cam_frame.images[0], "dete_images", true);
            viz_->_viz->showPointCloud("points_aprilgrid", time_us, april_grid_pcd, {}, "LOCAL_WORLD", 1);
        }
    }

    return 0;
}