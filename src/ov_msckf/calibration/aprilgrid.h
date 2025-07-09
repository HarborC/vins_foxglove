#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>

namespace CAMERA_CALIB {

struct AprilGrid {
  AprilGrid(const std::string &config_path);

  std::vector<Eigen::Vector4d> aprilgrid_corner_pos_3d;

  inline int getTagCols() const { return tagCols; }
  inline int getTagRows() const { return tagRows; }

 private:
  int tagCols;        // number of apriltags
  int tagRows;        // number of apriltags
  double tagSize;     // size of apriltag, edge to edge [m]
  double tagSpacing;  // ratio of space between tags to tagSize
};

}  // namespace CAMERA_CALIB