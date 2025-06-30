#pragma once

#include "google/protobuf/util/time_util.h"

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "foxglove/FrameTransform.pb.h"
#include "foxglove/PosesInFrame.pb.h"
#include "foxglove/Color.pb.h"
#include "foxglove/Point3.pb.h"
#include "foxglove/RawImage.pb.h"
#include "foxglove/CompressedImage.pb.h"
#include "foxglove/PointCloud.pb.h"
#include "foxglove/LinePrimitive.pb.h"

#include <chrono>

namespace foxglove_viz {
namespace foxglove {
namespace utility {
inline void Transformation3ToPosOri(const Eigen::Matrix4f& pose,
                                    ::foxglove::Vector3* pos, ::foxglove::Quaternion* rot) {
  const Eigen::Quaternionf quat_in = Eigen::Quaternionf(pose.block<3, 3>(0, 0)).normalized();
  const Eigen::Vector3f& pos_in = pose.block<3, 1>(0, 3);
  pos->set_x(pos_in.x());
  pos->set_y(pos_in.y());
  pos->set_z(pos_in.z());
  rot->set_x(quat_in.x());
  rot->set_y(quat_in.y());
  rot->set_z(quat_in.z());
  rot->set_w(quat_in.w());
}

inline int64_t UsecSinceEpoch() {
  return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count());
}

template <typename MsgT>
void SetMsgTimeStamp(const int64_t usec, MsgT* msg) {
  *(msg->mutable_timestamp()) =
      google::protobuf::util::TimeUtil::NanosecondsToTimestamp(usec * 1000l);
}

inline void SetColor(const std::array<float, 4>& rgba, ::foxglove::Color* out) {
  out->set_r(rgba[0]);
  out->set_g(rgba[1]);
  out->set_b(rgba[2]);
  out->set_a(rgba[3]);
}

template <typename ValT>
void SetPoint(const std::vector<ValT>& pt, ::foxglove::Point3* out) {
  out->set_x(pt[0]);
  out->set_y(pt[1]);
  out->set_z(pt[2]);
}

std::string CompressAsStr(const cv::Mat& img, const std::string& fmt);

void SetImgMsg(const cv::Mat& img, const std::string& fmt, ::foxglove::CompressedImage* msg);

bool SetImgMsg(const cv::Mat& img, ::foxglove::RawImage* msg);

void SetPointCloudMsgProperties(::foxglove::PointCloud* pc_msg, bool no_color = false);

bool AddPointsToMsg(const std::vector<std::vector<float>>& raw_pc, const size_t skip_n,
                    const std::vector<std::vector<uint8_t>>& colors,
                    ::foxglove::PointCloud* pc_msg);

void addPointsToLine(const std::vector<std::vector<float>>& points, ::foxglove::LinePrimitive* line);

template <typename MarkerT>
void setMarkerProps(const ::foxglove::Vector3& size, const ::foxglove::Color& color,
                    const Eigen::Matrix4f& marker_pose, MarkerT* marker) {
  (*marker->mutable_size()) = size;
  (*marker->mutable_color()) = color;
  auto* mpose = marker->mutable_pose();
  Transformation3ToPosOri(marker_pose, mpose->mutable_position(),
                          mpose->mutable_orientation());
}

inline ::foxglove::Vector3 fgVec3(const double s) {
  ::foxglove::Vector3 result;
  result.set_x(s);
  result.set_y(s);
  result.set_z(s);
  return result;
}

inline ::foxglove::Vector3 fgVec3(const double x, const double y, const double z) {
  ::foxglove::Vector3 result;
  result.set_x(x);
  result.set_y(y);
  result.set_z(z);
  return result;
}

inline ::foxglove::Quaternion fgQuat(const double w, const double x, const double y,
                                     const double z) {
  ::foxglove::Quaternion result;
  result.set_w(w);
  result.set_x(x);
  result.set_y(y);
  result.set_z(z);
  return result;
}

void setLineProps(const ::foxglove::LinePrimitive::Type& line_type,
                  const ::foxglove::Color& color, const double thickness,
                  ::foxglove::LinePrimitive* line);

}  // namespace utility

}  // namespace foxglove
}  // namespace foxglove_viz