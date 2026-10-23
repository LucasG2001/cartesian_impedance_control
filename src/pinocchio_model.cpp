// Pinocchio headers first (Boost.MPL limits must be set before other Boost users).
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "cartesian_impedance_control/pinocchio_model.hpp"

#include <algorithm>
#include <stdexcept>

namespace cartesian_impedance_control {

struct PinocchioModel::Impl {
  pinocchio::Model model;
  pinocchio::Data data;
  pinocchio::FrameIndex ee_frame_id{0};
  std::vector<int> q_idx;  // controller joint i -> index in model q
  std::vector<int> v_idx;  // controller joint i -> index in model v
  Eigen::VectorXd q, v;
  Eigen::Matrix<double, 6, Eigen::Dynamic> J_full;
};

PinocchioModel::PinocchioModel(const std::string& urdf_path,
                               const std::vector<std::string>& arm_joints,
                               const std::string& ee_frame,
                               const std::vector<double>& armature)
    : impl_(std::make_unique<Impl>()) {
  if (arm_joints.size() != 7) {
    throw std::invalid_argument("PinocchioModel: expected 7 arm joints");
  }

  pinocchio::Model full;
  pinocchio::urdf::buildModel(urdf_path, full);  // fixed base

  for (const auto& name : arm_joints) {
    if (!full.existJointName(name)) {
      throw std::invalid_argument("PinocchioModel: joint '" + name + "' not in " + urdf_path);
    }
  }

  // Lock every joint that is not commanded by the controller (fingers, ...).
  std::vector<pinocchio::JointIndex> to_lock;
  for (pinocchio::JointIndex j = 1; j < static_cast<pinocchio::JointIndex>(full.njoints); ++j) {
    if (std::find(arm_joints.begin(), arm_joints.end(), full.names[j]) == arm_joints.end()) {
      to_lock.push_back(j);
    }
  }
  const Eigen::VectorXd q_ref = pinocchio::neutral(full);
  impl_->model = pinocchio::buildReducedModel(full, to_lock, q_ref);
  impl_->data = pinocchio::Data(impl_->model);

  auto& m = impl_->model;
  if (m.nv != 7 || m.nq != 7) {
    throw std::runtime_error("PinocchioModel: reduced model does not have 7 DOF");
  }
  if (!m.existFrame(ee_frame)) {
    throw std::invalid_argument("PinocchioModel: frame '" + ee_frame + "' not in " + urdf_path);
  }
  impl_->ee_frame_id = m.getFrameId(ee_frame);

  for (const auto& name : arm_joints) {
    const auto jid = m.getJointId(name);
    impl_->q_idx.push_back(m.joints[jid].idx_q());
    impl_->v_idx.push_back(m.joints[jid].idx_v());
  }
  if (!armature.empty()) {
    if (armature.size() != 7) {
      throw std::invalid_argument("PinocchioModel: armature needs 7 values");
    }
    for (size_t i = 0; i < 7; ++i) {
      m.armature[impl_->v_idx[i]] = armature[i];  // crba adds it to the mass matrix diagonal
    }
  }
  impl_->q = Eigen::VectorXd::Zero(m.nq);
  impl_->v = Eigen::VectorXd::Zero(m.nv);
  impl_->J_full = Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, m.nv);
}

PinocchioModel::~PinocchioModel() = default;

void PinocchioModel::compute(const Eigen::Matrix<double, 7, 1>& q,
                             const Eigen::Matrix<double, 7, 1>& dq) {
  auto& m = impl_->model;
  auto& d = impl_->data;
  for (int i = 0; i < 7; ++i) {
    impl_->q[impl_->q_idx[i]] = q[i];
    impl_->v[impl_->v_idx[i]] = dq[i];
  }

  // Mass matrix (crba fills the upper triangle only).
  pinocchio::crba(m, d, impl_->q);
  d.M.triangularView<Eigen::StrictlyLower>() = d.M.transpose().triangularView<Eigen::StrictlyLower>();

  // Coriolis/centrifugal vector C(q, dq) dq, without gravity.
  pinocchio::computeCoriolisMatrix(m, d, impl_->q, impl_->v);
  const Eigen::VectorXd c = d.C * impl_->v;
  const Eigen::VectorXd ct = d.C.transpose() * impl_->v;  // for the momentum observer
  const Eigen::VectorXd g = pinocchio::computeGeneralizedGravity(m, d, impl_->q);

  // Kinematics, end-effector pose and Jacobian.
  pinocchio::computeJointJacobians(m, d, impl_->q);  // also runs forward kinematics
  pinocchio::updateFramePlacements(m, d);
  impl_->J_full.setZero();
  pinocchio::getFrameJacobian(m, d, impl_->ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED,
                              impl_->J_full);

  for (int i = 0; i < 7; ++i) {
    const int vi = impl_->v_idx[i];
    coriolis_[i] = c[vi];
    coriolis_transpose_dq_[i] = ct[vi];
    gravity_[i] = g[vi];
    J_.col(i) = impl_->J_full.col(vi);
    for (int k = 0; k < 7; ++k) {
      M_(i, k) = d.M(vi, impl_->v_idx[k]);
    }
  }

  const auto& oMf = d.oMf[impl_->ee_frame_id];
  pose_.setIdentity();
  pose_.linear() = oMf.rotation();
  pose_.translation() = oMf.translation();
}

}  // namespace cartesian_impedance_control
