// Sanity check of PinocchioModel against libfranka reference values.
// Usage: pinocchio_model_check [urdf_path] [ee_frame]
// Exits with 0 if all checks pass.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "cartesian_impedance_control/pinocchio_model.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what, double value) {
  std::printf("[%s] %-40s %.3e\n", ok ? " OK " : "FAIL", what.c_str(), value);
  if (!ok) {
    ++failures;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string urdf = argc > 1 ? argv[1] : "/home/njohnson/ci_mpc_assets/fr3.urdf";
  const std::string ee_frame = argc > 2 ? argv[2] : "fr3_hand_tcp";
  std::vector<std::string> joints;
  for (int i = 1; i <= 7; ++i) {
    joints.push_back("fr3_joint" + std::to_string(i));
  }
  cartesian_impedance_control::PinocchioModel model(urdf, joints, ee_frame);

  // libfranka home pose
  Eigen::Matrix<double, 7, 1> q;
  q << 0.0, -M_PI_4, 0.0, -3.0 * M_PI_4, 0.0, M_PI_2, M_PI_4;
  Eigen::Matrix<double, 7, 1> dq = Eigen::Matrix<double, 7, 1>::Zero();
  model.compute(q, dq);

  const Eigen::Vector3d p_ref(0.3069, 0.0, 0.4869);
  const Eigen::Matrix3d R_ref = Eigen::Vector3d(1.0, -1.0, -1.0).asDiagonal();
  const Eigen::Vector3d p = model.pose().translation();
  std::printf("home TCP position: %.4f %.4f %.4f\n", p.x(), p.y(), p.z());
  check((p - p_ref).norm() < 1e-3, "home TCP position vs libfranka", (p - p_ref).norm());
  check((model.pose().linear() - R_ref).norm() < 1e-6, "home TCP rotation = diag(1,-1,-1)",
        (model.pose().linear() - R_ref).norm());
  check(model.coriolis().norm() < 1e-12, "coriolis(dq = 0) = 0", model.coriolis().norm());

  // Random state
  q << 0.3, -0.5, 0.2, -2.0, 0.4, 1.8, -0.3;
  dq << 0.5, -0.4, 0.3, 0.6, -0.7, 0.2, 0.9;
  model.compute(q, dq);
  const Eigen::Matrix<double, 7, 7> M = model.mass();
  check((M - M.transpose()).norm() < 1e-12, "M symmetric", (M - M.transpose()).norm());
  const double min_eig = Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 7, 7>>(M)
                             .eigenvalues().minCoeff();
  check(min_eig > 0.0, "M positive definite (min eigenvalue)", min_eig);
  // C(q, dq) dq is quadratic in dq
  const Eigen::Matrix<double, 7, 1> c1 = model.coriolis();
  model.compute(q, 2.0 * dq);
  check((model.coriolis() - 4.0 * c1).norm() < 1e-9, "coriolis quadratic in dq",
        (model.coriolis() - 4.0 * c1).norm());

  // Jacobian vs finite differences of the pose
  model.compute(q, dq);
  const Eigen::Matrix<double, 6, 7> J = model.jacobian();
  const Eigen::Affine3d T0 = model.pose();
  Eigen::Matrix<double, 6, 7> J_fd;
  const double h = 1e-7;
  for (int i = 0; i < 7; ++i) {
    Eigen::Matrix<double, 7, 1> qp = q;
    qp[i] += h;
    model.compute(qp, dq);
    const Eigen::Affine3d T1 = model.pose();
    J_fd.block<3, 1>(0, i) = (T1.translation() - T0.translation()) / h;
    const Eigen::AngleAxisd aa(T1.linear() * T0.linear().transpose());
    J_fd.block<3, 1>(3, i) = aa.axis() * aa.angle() / h;
  }
  check((J - J_fd).cwiseAbs().maxCoeff() < 1e-5, "jacobian vs finite differences",
        (J - J_fd).cwiseAbs().maxCoeff());

  std::printf("%s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
  return failures == 0 ? 0 : 1;
}
