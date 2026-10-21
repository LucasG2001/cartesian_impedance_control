#include "cartesian_impedance_control/target_validation.hpp"

#include <cmath>
#include <sstream>

namespace cartesian_impedance_control {

namespace {
std::string format_vector(const Eigen::Vector3d& v) {
  std::ostringstream s;
  s << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
  return s.str();
}
}  // namespace

std::string check_target_limits(const TargetLimits& limits) {
  if (!std::isfinite(limits.max_distance) || !(limits.max_distance > 0.0)) {
    return "target.max_distance must be a finite value > 0";
  }
  if (limits.workspace_enabled) {
    if (!limits.workspace_min.allFinite() || !limits.workspace_max.allFinite()) {
      return "target.workspace_min / workspace_max must be finite";
    }
    if (!(limits.workspace_min.array() < limits.workspace_max.array()).all()) {
      return "target.workspace_min must be < target.workspace_max on every axis";
    }
  }
  return "";
}

std::string check_position_target(const Eigen::Vector3d& target, const Eigen::Vector3d& current,
                                  const TargetLimits& limits) {
  if (!target.allFinite()) {
    return "position contains NaN/Inf";
  }
  const double distance = (target - current).norm();
  if (!(distance <= limits.max_distance)) {
    std::ostringstream s;
    s << "position " << format_vector(target) << " is " << distance << " m from the current position "
      << format_vector(current) << " (target.max_distance " << limits.max_distance << " m)";
    return s.str();
  }
  if (limits.workspace_enabled &&
      !((target.array() >= limits.workspace_min.array()).all() &&
        (target.array() <= limits.workspace_max.array()).all())) {
    return "position " + format_vector(target) + " is outside the workspace box " +
           format_vector(limits.workspace_min) + " .. " + format_vector(limits.workspace_max);
  }
  return "";
}

std::string check_orientation_target(Eigen::Quaterniond& q) {
  if (!q.coeffs().allFinite()) {
    return "quaternion contains NaN/Inf";
  }
  const double norm = q.norm();
  if (!(norm >= kQuaternionNormMin && norm <= kQuaternionNormMax)) {
    std::ostringstream s;
    s << "quaternion norm " << norm << " outside [" << kQuaternionNormMin << ", " << kQuaternionNormMax << "]";
    return s.str();
  }
  q.normalize();
  return "";
}

}  // namespace cartesian_impedance_control
