// Unit tests for the target checks (no ROS, no robot).
// Run: colcon test --packages-select cartesian_impedance_control --ctest-args -R test_target_validation

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "cartesian_impedance_control/target_validation.hpp"

using cartesian_impedance_control::check_orientation_target;
using cartesian_impedance_control::check_position_target;
using cartesian_impedance_control::check_target_limits;
using cartesian_impedance_control::TargetLimits;

namespace {
const Eigen::Vector3d kCurrent(0.5, 0.0, 0.5);  // start pose of the yaml
const double kNan = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();
}  // namespace

TEST(TargetLimits, DefaultsAreValid) {
  EXPECT_EQ(check_target_limits(TargetLimits{}), "");
}

TEST(TargetLimits, RejectsBadLimits) {
  TargetLimits l;
  l.max_distance = 0.0;
  EXPECT_NE(check_target_limits(l), "");
  l = TargetLimits{};
  l.max_distance = kNan;
  EXPECT_NE(check_target_limits(l), "");
  l = TargetLimits{};
  l.workspace_min.z() = 0.9;  // min > max
  EXPECT_NE(check_target_limits(l), "");
  l.workspace_enabled = false;  // box not used -> not checked
  EXPECT_EQ(check_target_limits(l), "");
}

TEST(PositionTarget, AcceptsNearbyTargetInsideBox) {
  EXPECT_EQ(check_position_target(Eigen::Vector3d(0.5, 0.4, 0.5), kCurrent, TargetLimits{}), "");
  EXPECT_EQ(check_position_target(kCurrent, kCurrent, TargetLimits{}), "");
}

TEST(PositionTarget, RejectsNanAndInf) {
  EXPECT_NE(check_position_target(Eigen::Vector3d(kNan, 0.0, 0.5), kCurrent, TargetLimits{}), "");
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.5, kInf, 0.5), kCurrent, TargetLimits{}), "");
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.5, 0.0, -kInf), kCurrent, TargetLimits{}), "");
}

TEST(PositionTarget, RejectsTooFar) {
  TargetLimits l;
  l.workspace_enabled = false;
  EXPECT_EQ(check_position_target(Eigen::Vector3d(0.5, 0.0, 0.0001), kCurrent, l), "");   // 0.4999 m
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.5, 0.0, -0.0001), kCurrent, l), "");  // 0.5001 m
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.5, -0.4, 0.5), Eigen::Vector3d(0.5, 0.4, 0.5), l), "");  // 0.8 m step
}

TEST(PositionTarget, RejectsOutsideWorkspace) {
  TargetLimits l;
  l.max_distance = 10.0;
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.5, 0.0, 0.09), kCurrent, l), "");   // below z_min 0.10
  EXPECT_EQ(check_position_target(Eigen::Vector3d(0.5, 0.0, 0.10), kCurrent, l), "");   // on the boundary
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.10, 0.0, 0.5), kCurrent, l), "");   // x < 0.15
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.85, 0.0, 0.5), kCurrent, l), "");   // x > 0.80
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.5, 0.65, 0.5), kCurrent, l), "");   // y > 0.60
  EXPECT_NE(check_position_target(Eigen::Vector3d(0.5, 0.0, 0.85), kCurrent, l), "");   // z > 0.80
  l.workspace_enabled = false;
  EXPECT_EQ(check_position_target(Eigen::Vector3d(0.5, 0.0, 0.09), kCurrent, l), "");
}

TEST(OrientationTarget, NormalisesSlightlyOffQuaternion) {
  Eigen::Quaterniond q(0.0, 1.05, 0.0, 0.0);  // w, x, y, z; norm 1.05
  EXPECT_EQ(check_orientation_target(q), "");
  EXPECT_NEAR(q.norm(), 1.0, 1e-12);
  EXPECT_NEAR(q.x(), 1.0, 1e-12);
}

TEST(OrientationTarget, RejectsBadNorm) {
  Eigen::Quaterniond zero(0.0, 0.0, 0.0, 0.0);
  EXPECT_NE(check_orientation_target(zero), "");
  Eigen::Quaterniond small(0.0, 0.85, 0.0, 0.0);
  EXPECT_NE(check_orientation_target(small), "");
  EXPECT_DOUBLE_EQ(small.x(), 0.85);  // unchanged on rejection
  Eigen::Quaterniond large(0.0, 1.2, 0.0, 0.0);
  EXPECT_NE(check_orientation_target(large), "");
  Eigen::Quaterniond boundary(0.0, 0.9, 0.0, 0.0);
  EXPECT_EQ(check_orientation_target(boundary), "");
}

TEST(OrientationTarget, RejectsNanAndInf) {
  Eigen::Quaterniond q_nan(kNan, 1.0, 0.0, 0.0);
  EXPECT_NE(check_orientation_target(q_nan), "");
  Eigen::Quaterniond q_inf(0.0, kInf, 0.0, 0.0);
  EXPECT_NE(check_orientation_target(q_inf), "");
}
