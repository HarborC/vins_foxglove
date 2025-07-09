#include "aprilgrid.h"
#include <opencv2/opencv.hpp>

namespace CAMERA_CALIB {

AprilGrid::AprilGrid(const std::string &config_path) {
  cv::FileStorage config_node(config_path, cv::FileStorage::READ);

  if (!config_node["tagCols"].empty())
    config_node["tagCols"] >> tagCols;

  if (!config_node["tagRows"].empty())
    config_node["tagRows"] >> tagRows;

  if (!config_node["tagSize"].empty())
    config_node["tagSize"] >> tagSize;

  if (!config_node["tagSpacing"].empty())
    config_node["tagSpacing"] >> tagSpacing;  

  double x_corner_offsets[4] = {0, tagSize, tagSize, 0};
  double y_corner_offsets[4] = {0, 0, tagSize, tagSize};

  aprilgrid_corner_pos_3d.resize(tagCols * tagRows * 4);

  for (int y = 0; y < tagRows; y++) {
    for (int x = 0; x < tagCols; x++) {
      int tag_id = tagCols * y + x;
      double x_offset = x * tagSize * (1 + tagSpacing);
      double y_offset = y * tagSize * (1 + tagSpacing);

      for (int i = 0; i < 4; i++) {
        int corner_id = (tag_id << 2) + i;

        Eigen::Vector4d &pos_3d = aprilgrid_corner_pos_3d[corner_id];

        pos_3d[0] = x_offset + x_corner_offsets[i];
        pos_3d[1] = y_offset + y_corner_offsets[i];
        pos_3d[2] = 0;
        pos_3d[3] = 1;
      }
    }
  }
}

}  // namespace CAMERA_CALIB
