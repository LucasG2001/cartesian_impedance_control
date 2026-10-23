#pragma once

// Robot model computed with Pinocchio from a URDF.
//
// Drop-in replacement for the values the controller otherwise reads from
// franka_semantic_components::FrankaRobotModel (libfranka), so the same
// controller can run on hardware that only exposes joint position / velocity
// state and joint effort commands (e.g. Isaac Sim's in-process ros2_control).
//
// Conventions match libfranka:
//   mass()      M(q), 7x7
//   coriolis()  C(q, dq) * dq, 7x1, gravity NOT included
//   coriolis_transpose_dq()  C(q, dq)^T * dq (momentum observer)
//   gravity()   g(q) (only for an observer on measured link torques, which include gravity on the FR3)
//   jacobian()  6x7, rows [linear; angular], expressed in the base frame,
//               reference point = end-effector frame origin  (= getZeroJacobian)
//   pose()      end-effector pose in the base frame          (= getPoseMatrix)
//
// Pinocchio headers are kept out of this header on purpose (pimpl), so the
// controller does not pull them in.

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace cartesian_impedance_control {

// Exported explicitly: src/pinocchio_model.cpp is compiled with -fvisibility=hidden.
class __attribute__((visibility("default"))) PinocchioModel {
 public:
  // arm_joints: the 7 joints the controller commands, in controller order.
  // All other joints in the URDF (e.g. gripper fingers) are locked at their
  // neutral position. ee_frame: URDF frame used as end-effector.
  // armature: optional rotor inertia per arm joint [kg m^2] (reflected through the gearbox), added to the
  // mass matrix; empty = none (URDF link inertias only).
  PinocchioModel(const std::string& urdf_path,
                 const std::vector<std::string>& arm_joints,
                 const std::string& ee_frame,
                 const std::vector<double>& armature = {});
  ~PinocchioModel();

  PinocchioModel(const PinocchioModel&) = delete;
  PinocchioModel& operator=(const PinocchioModel&) = delete;

  void compute(const Eigen::Matrix<double, 7, 1>& q, const Eigen::Matrix<double, 7, 1>& dq);

  const Eigen::Matrix<double, 7, 7>& mass() const { return M_; }
  const Eigen::Matrix<double, 7, 1>& coriolis() const { return coriolis_; }
  const Eigen::Matrix<double, 7, 1>& coriolis_transpose_dq() const { return coriolis_transpose_dq_; }
  const Eigen::Matrix<double, 7, 1>& gravity() const { return gravity_; }
  const Eigen::Matrix<double, 6, 7>& jacobian() const { return J_; }
  const Eigen::Affine3d& pose() const { return pose_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  Eigen::Matrix<double, 7, 7> M_ = Eigen::Matrix<double, 7, 7>::Identity();
  Eigen::Matrix<double, 7, 1> coriolis_ = Eigen::Matrix<double, 7, 1>::Zero();
  Eigen::Matrix<double, 7, 1> coriolis_transpose_dq_ = Eigen::Matrix<double, 7, 1>::Zero();
  Eigen::Matrix<double, 7, 1> gravity_ = Eigen::Matrix<double, 7, 1>::Zero();
  Eigen::Matrix<double, 6, 7> J_ = Eigen::Matrix<double, 6, 7>::Zero();
  Eigen::Affine3d pose_ = Eigen::Affine3d::Identity();
};

}  // namespace cartesian_impedance_control
