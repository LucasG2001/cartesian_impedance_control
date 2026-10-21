// Checks for external pose targets (reference_pose topic, set_pose service) before they reach the 1 kHz loop.
// Pure functions (Eigen only, no ROS), unit-tested in test/test_target_validation.cpp. Added 2026-10-19
// (fix-list a13, docs/controller-setup.md; parameters target.* in launch/controllers.yaml).

#pragma once

#include <string>

#include <Eigen/Dense>

namespace cartesian_impedance_control {

struct TargetLimits {
  double max_distance{0.5};                                      // [m] from the current EE position
  bool workspace_enabled{true};                                  // box check on/off
  Eigen::Vector3d workspace_min = Eigen::Vector3d(0.15, -0.60, 0.10);  // [m], base frame (fr3_link0)
  Eigen::Vector3d workspace_max = Eigen::Vector3d(0.80, 0.60, 0.80);   // [m], base frame
};

// Quaternion norm accepted before normalisation
constexpr double kQuaternionNormMin = 0.9;
constexpr double kQuaternionNormMax = 1.1;

// Empty string if the limits themselves are usable, otherwise the reason (checked once at configure).
std::string check_target_limits(const TargetLimits& limits);

// Empty string if the position target is accepted, otherwise the reason (finite, distance, workspace box).
std::string check_position_target(const Eigen::Vector3d& target, const Eigen::Vector3d& current,
                                  const TargetLimits& limits);

// Empty string if the orientation is accepted (finite, norm in [0.9, 1.1]); then q is normalised in place.
// On rejection q is left unchanged.
std::string check_orientation_target(Eigen::Quaterniond& q);

}  // namespace cartesian_impedance_control
