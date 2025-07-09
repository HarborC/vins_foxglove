#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>

namespace CAMERA_CALIB {

struct CalibCornerData {
  std::vector<Eigen::Vector2d> corners;
  std::vector<int> corner_ids;
  std::vector<double> radii;  //!< threshold used for maximum displacement
                              //! during sub-pix refinement; Search region is
  //! slightly larger.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void showInImage(cv::Mat &img, const std::vector<uint8_t>& color = {0, 255, 0}) {
    if (corners.size() != corner_ids.size()) {
      std::cerr << "[ERROR] Corners and IDs size mismatch!" << std::endl;
      return;
    }

    if (color.size() != 3) {
      std::cerr << "[ERROR] Color must be a 3-element vector [R, G, B]." << std::endl;
      return;
    }

    // 确保图像是三通道彩色图
    if (img.empty()) {
      std::cerr << "[ERROR] Input image is empty." << std::endl;
      return;
    }
    if (img.channels() == 1) {
      cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    } else if (img.channels() != 3) {
      std::cerr << "[ERROR] Unsupported image format. Only 1 or 3 channels supported." << std::endl;
      return;
    }

    for (size_t i = 0; i < corners.size(); ++i) {
      const auto& pt = corners[i];
      int id = corner_ids[i];

      // 转换为 OpenCV 坐标格式
      cv::Point2f cv_pt(static_cast<float>(pt.x()), static_cast<float>(pt.y()));

      // 绘制圆心
      cv::circle(img, cv_pt, 2, cv::Scalar(color[0], color[1], color[2]), -1);

      // 如果有 radius，画出搜索区域
      if (!radii.empty() && i < radii.size()) {
        cv::circle(img, cv_pt, static_cast<int>(radii[i]), cv::Scalar(0, 255, 0), 1);
      }

      // 显示角点 ID
      cv::putText(img, std::to_string(id), cv_pt,
                  cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(0, 0, 0), 1);  // 黑色文字
      cv::putText(img, std::to_string(id), cv_pt,
                  cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(color[0], color[1], color[2]), 1);
    }
  }
};

typedef std::map<std::string, CalibCornerData> CalibCornerDataMap;

struct ApriltagDetectorData;

class ApriltagDetector {
 public:
  ApriltagDetector(int numTags);

  ~ApriltagDetector();

  void detectTags(const cv::Mat& img_raw,
                  std::vector<Eigen::Vector2d>& corners,
                  std::vector<int>& ids, std::vector<double>& radii,
                  std::vector<Eigen::Vector2d>& corners_rejected,
                  std::vector<int>& ids_rejected, std::vector<double>& radii_rejected);

 private:
  ApriltagDetectorData* data;
};

}  // namespace CAMERA_CALIB
