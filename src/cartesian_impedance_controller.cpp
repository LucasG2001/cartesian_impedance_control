// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cartesian_impedance_control/cartesian_impedance_controller.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace {

template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}

namespace cartesian_impedance_control {


inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
  double lambda_ = damped ? 0.2 : 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);   
  Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
  Eigen::MatrixXd S_ = M_;  // copying the dimensions of M_, its content is not needed.
  S_.setZero();

  for (int i = 0; i < sing_vals_.size(); i++)
     S_(i, i) = (sing_vals_(i)) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);

  M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
}

void CartesianImpedanceController::calculate_tau_friction(){
	  //Nullspace-calculation for friction compensation
    pseudoInverse(J, J_pinv);
    N = (Eigen::MatrixXd::Identity(7, 7) - J_pinv * J);
    Eigen::Matrix<double, 6, 1> pose_error;
    pose_error.setZero();
    pose_error = error;
    // Component-wise thresholding
    pose_error = (pose_error.array() < 0.0008).select(0.0, pose_error); //set errors smaller than 0.0008 to 0
    double alpha = 0.1;//constant for exponential filter in relation to static friction moment        
    dq_filtered = alpha* dq_ + (1 - alpha) * dq_filtered; //Filtering dq of every joint
    tau_impedance_filtered = alpha*tau_impedance + (1 - alpha) * tau_impedance_filtered; //Filtering tau_impedance
    //Creating and filtering a "fake" tau_impedance with own weights, optimized for friction compensation (else friction compensation would get stronger with higher stiffnesses)
    tau_friction_impedance = J.transpose() * (-alpha * (D_friction*(J*dq_) + K_friction * (pose_error))) + (1 - alpha) * tau_friction_impedance;
    //Creating "fake" dq, that acts only in the impedance-space, else dq in the nullspace also gets compensated, which we do not want due to null-space movement
    dq_imp = dq_filtered - N * dq_filtered;
    //Calculation of friction force according to Bachelor Thesis: https://polybox.ethz.ch/index.php/s/iYj8ALPijKTAC2z?path=%2FFriction%20compensation
    f = beta.cwiseProduct(dq_imp) + offset_friction;
    g = coulomb_friction;
    g(4) = (coulomb_friction(4) + (static_friction(4) - coulomb_friction(4)) * exp(-1 * std::abs(dq_imp(4)/(dq_s(4) + 0.0001))));
    g(6) = (coulomb_friction(6) + (static_friction(6) - coulomb_friction(6)) * exp(-1 * std::abs(dq_imp(6)/dq_s(6))));
           
    dz = dq_imp.array() - dq_imp.array().abs() / g.array() * sigma_0.array() * z.array() + 0.025 * tau_friction_impedance.array()/*(J.transpose() * K * error).array()*/;
    dz(6) -= 0.02*tau_friction_impedance(6);
    z = 0.001 * dz + z;
    tau_friction = sigma_0.array() * z.array() + 100 * sigma_1.array() * dz.array() + f.array();  
    // clamp elementwise between -4 and 4s
    tau_friction = tau_friction.cwiseMin(3.5).cwiseMax(-3.5);
   
}

// Validates a pose target and hands it to update() (fix-list a13). Called from the reference_pose callback and
// the set_pose service (non-RT threads). Rejected targets are logged and ignored; the current target stays.
bool CartesianImpedanceController::submit_pose_target(const Eigen::Vector3d& position,
                                                      const Eigen::Quaterniond* orientation,
                                                      bool reset_integrator, const char* source) {
  auto logger = get_node()->get_logger();
  Eigen::Vector3d current;
  {
    std::lock_guard<std::mutex> lock(current_position_mutex_);
    if (!current_position_valid_) {
      RCLCPP_WARN(logger, "%s rejected: controller not active", source);
      return false;
    }
    current = current_position_;
  }
  Eigen::Quaterniond q = orientation ? *orientation : Eigen::Quaterniond::Identity();
  std::string reason = orientation ? check_orientation_target(q) : "";
  if (reason.empty()) {
    reason = check_position_target(position, current, target_limits_);
  }
  if (!reason.empty()) {
    RCLCPP_WARN(logger, "%s rejected: %s", source, reason.c_str());
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    command_nonrt_.pose_seq++;
    command_nonrt_.position = position;
    command_nonrt_.has_orientation = orientation != nullptr;
    command_nonrt_.orientation = q;
    command_nonrt_.reset_integrator = reset_integrator;
    command_buffer_.writeFromNonRT(command_nonrt_);
  }
  RCLCPP_INFO(logger, "%s accepted: position [%.4f %.4f %.4f]%s", source, position.x(), position.y(), position.z(),
              orientation ? " + orientation" : " (orientation unchanged)");
  return true;
}

void CartesianImpedanceController::reference_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg)
{
    // Full pose target (position + orientation); resets the integrator as before
    const Eigen::Vector3d position(msg->position.x, msg->position.y, msg->position.z);
    const Eigen::Quaterniond orientation(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
    submit_pose_target(position, &orientation, true, "reference_pose");
}

void CartesianImpedanceController::setPose(const std::shared_ptr<messages_fr3::srv::SetPose::Request> request,
    std::shared_ptr<messages_fr3::srv::SetPose::Response> response)
{
    // POSITION ONLY: roll / pitch / yaw are ignored and the orientation target stays unchanged. The old
    // user_input_client always sends a hardcoded roll 180 deg / yaw 90 deg, which must not rotate the hand.
    // Integrator: reset only if integrator.reset_on_new_target (as before).
    response->success = submit_pose_target(Eigen::Vector3d(request->x, request->y, request->z), nullptr, false,
                                            "set_pose");
}

void CartesianImpedanceController::setParam(const std::shared_ptr<messages_fr3::srv::SetParam::Request> /*request*/,
    std::shared_ptr<messages_fr3::srv::SetParam::Response> response)
{
    // The old server wrote T and D from this thread; D is recomputed every cycle and T is unused, so it never had
    // an effect. Gains are parameters in controllers.yaml (read at configure).
    RCLCPP_WARN(get_node()->get_logger(),
                "set_param is not supported (no effect): set the gains in controllers.yaml and restart the launch");
    response->success = false;
}

void CartesianImpedanceController::control_mode_callback(const std_msgs::msg::Int16::SharedPtr msg)
{
  RCLCPP_INFO(get_node()->get_logger(), "Received Control Mode Callback.");
  // applied in update() (apply_external_commands): 3 lift, 2 float, 1 compliant, anything else normal;
  // logged here (non-RT) with the same texts as before
  if (msg->data == 2) {
    RCLCPP_INFO(get_node()->get_logger(), "Control mode set to gripper closed.");
  } else if (msg->data == 1) {
    RCLCPP_INFO(get_node()->get_logger(), "Control mode set to compliant");
  } else if (msg->data != 3) {
    RCLCPP_INFO(get_node()->get_logger(), "Control mode set to normal (gripper open).");
  }
  std::lock_guard<std::mutex> lock(command_mutex_);
  command_nonrt_.mode_seq++;
  command_nonrt_.mode = msg->data;
  command_buffer_.writeFromNonRT(command_nonrt_);
}

// Runs in update(): applies a new pose / control mode from the handoff buffer (one consistent read per cycle).
void CartesianImpedanceController::apply_external_commands() {
  const TargetCommand* command = command_buffer_.readFromRT();
  if (command->pose_seq != applied_pose_seq_) {
    applied_pose_seq_ = command->pose_seq;
    position_d_target_ = command->position;
    if (command->has_orientation) {
      orientation_d_target_ = command->orientation;
    }
    if (command->reset_integrator) {
      I_error.setZero(); //reset integral error when a new target is received (reference_pose)
    }
    start_move_active_ = false;  // a new target cancels the start move
  }
  if (command->mode_seq != applied_mode_seq_) {
    applied_mode_seq_ = command->mode_seq;
    // same assignments as the original control_mode callback (the log lines are in control_mode_callback)
    if (command->mode == 3){
      //lift -> high extraction stiffness
        K.diagonal() = stiffness_mode_lift_; // parameter modes.lift_stiffness
        control_mode = 3;
    }
    else if (command->mode == 2) {
        control_mode = 2; // low stiffness float mode
        I_error.setZero(); //reset integral error
        K.diagonal() = stiffness_mode_float_; // parameter modes.float_stiffness
    }
    else if (command->mode == 1) {
        control_mode = 1; // Calibration mode
    }
    else {
        control_mode = 0; // Normal mode
        K.diagonal() = stiffness_mode_normal_; // parameter modes.normal_stiffness
    }
  }
}


void CartesianImpedanceController::update_stiffness_and_references(){
  //target by filtering
  /** at the moment we do not use dynamic reconfigure and control the robot via D, K and T **/
  //K = filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * K;
  //D = filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * D;
  nullspace_stiffness_ = filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  //std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  // low-pass step towards the target, limited to max_reference_velocity (0 = no limit)
  Eigen::Vector3d position_step = filter_params_ * (position_d_target_ - position_d_);
  const double max_position_step = max_reference_velocity_translational_ * dt;
  if (max_position_step > 0.0 && position_step.norm() > max_position_step) {
    position_step *= max_position_step / position_step.norm();
  }
  position_d_ += position_step;
  double slerp_fraction = filter_params_;
  const double angle_to_target = orientation_d_.angularDistance(orientation_d_target_);
  const double max_angle_step = max_reference_velocity_rotational_ * dt;
  if (max_angle_step > 0.0 && filter_params_ * angle_to_target > max_angle_step) {
    slerp_fraction = max_angle_step / angle_to_target;
  }
  orientation_d_ = orientation_d_.slerp(slerp_fraction, orientation_d_target_);
}


void CartesianImpedanceController::arrayToMatrix(const std::array<double,7>& inputArray, Eigen::Matrix<double,7,1>& resultMatrix)
{
 for(long unsigned int i = 0; i < 7; ++i){
     resultMatrix(i,0) = inputArray[i];
   }
}

void CartesianImpedanceController::arrayToMatrix(const std::array<double,6>& inputArray, Eigen::Matrix<double,6,1>& resultMatrix)
{
 for(long unsigned int i = 0; i < 6; ++i){
     resultMatrix(i,0) = inputArray[i];
   }
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceController::saturateTorqueRate(
  const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
  const Eigen::Matrix<double, 7, 1>& tau_J_d_M) {  
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
  double difference = tau_d_calculated[i] - tau_J_d_M[i];
  tau_d_saturated[i] =
         tau_J_d_M[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}


controller_interface::InterfaceConfiguration
CartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}


controller_interface::InterfaceConfiguration CartesianImpedanceController::state_interface_configuration()
  const {
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
    //state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }

  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    state_interfaces_config.names.push_back(franka_robot_model_name);
    std::cout << franka_robot_model_name << std::endl;
  }

  const std::string full_interface_name = arm_id_ + "/" + state_interface_name_;

  return state_interfaces_config;
}


void CartesianImpedanceController::declare_tuning_parameters() {
  // Defaults = the values hardcoded before 2026-10-13 (docs/controller-tuning.md)
  auto_declare<std::vector<double>>("stiffness", {1000.0, 1000.0, 1000.0, 130.0, 130.0, 45.0});
  auto_declare<double>("damping.ratio_translational", 1.0);
  auto_declare<double>("damping.ratio_rotational", 1.3);
  auto_declare<double>("damping.rot_z_min_factor", 2.0);
  auto_declare<std::vector<double>>("integrator.weights", {350.0, 350.0, 0.0, 50.0, 50.0, 50.0});
  auto_declare<double>("integrator.limit_force", 2.5);
  auto_declare<double>("integrator.limit_torque", 1.5);
  auto_declare<double>("integrator.limit_torque_z", 0.5);
  auto_declare<bool>("integrator.reset_on_new_target", false);
  auto_declare<double>("saturation.max_force", 18.0);
  auto_declare<double>("saturation.max_torque", 9.0);
  auto_declare<double>("saturation.max_torque_z", 2.0);
  auto_declare<bool>("saturation.clamp_spring_only", false);
  auto_declare<double>("saturation.max_damping_force", 36.0);
  auto_declare<double>("saturation.max_damping_torque", 18.0);
  auto_declare<double>("filter_params", 0.002);
  auto_declare<double>("delta_tau_max", 1.0);
  auto_declare<double>("max_reference_velocity.translational", 0.0);
  auto_declare<double>("max_reference_velocity.rotational", 0.0);
  auto_declare<double>("lambda_regularization", 0.0);
  auto_declare<double>("nullspace.stiffness", 0.0);
  auto_declare<double>("nullspace.damping", 0.0);
  auto_declare<double>("nullspace.max_torque", 5.0);
  auto_declare<bool>("nullspace.target_from_activation", true);
  auto_declare<std::vector<double>>("nullspace.q_target", {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785});
  auto_declare<bool>("start_pose.enabled", false);
  auto_declare<std::vector<double>>("start_pose.position", {0.5, 0.0, 0.5});
  auto_declare<bool>("start_pose.use_orientation", false);
  auto_declare<std::vector<double>>("start_pose.orientation_xyzw", {1.0, 0.0, 0.0, 0.0});
  auto_declare<double>("start_pose.max_velocity", 0.1);
  auto_declare<double>("start_pose.max_angular_velocity", 0.3);
  auto_declare<std::vector<double>>("modes.normal_stiffness", {2500.0, 2500.0, 1500.0, 130.0, 130.0, 45.0});
  auto_declare<std::vector<double>>("modes.float_stiffness", {150.0, 150.0, 150.0, 40.0, 40.0, 20.0});
  auto_declare<std::vector<double>>("modes.lift_stiffness", {0.0, 0.0, 8500.0, 0.0, 0.0, 0.0});
  auto_declare<std::vector<double>>("modes.contact_stiffness", {2500.0, 2500.0, 100.0, 40.0, 40.0, 20.0});
  auto_declare<double>("debug.publish_rate", 100.0);
  // checks on external pose targets (target_validation.hpp); defaults = TargetLimits
  const TargetLimits limits;
  auto_declare<double>("target.max_distance", limits.max_distance);
  auto_declare<bool>("target.workspace_enabled", limits.workspace_enabled);
  auto_declare<std::vector<double>>("target.workspace_min",
                                    {limits.workspace_min.x(), limits.workspace_min.y(), limits.workspace_min.z()});
  auto_declare<std::vector<double>>("target.workspace_max",
                                    {limits.workspace_max.x(), limits.workspace_max.y(), limits.workspace_max.z()});
}

bool CartesianImpedanceController::read_tuning_parameters() {
  auto node = get_node();
  auto logger = node->get_logger();
  // 6-vector parameter, all entries >= 0
  auto read_vec6 = [&](const std::string& name, Eigen::Matrix<double, 6, 1>& out) {
    const auto v = node->get_parameter(name).as_double_array();
    if (v.size() != 6) {
      RCLCPP_ERROR(logger, "Parameter %s needs 6 values, got %zu", name.c_str(), v.size());
      return false;
    }
    for (size_t i = 0; i < 6; ++i) {
      if (!(v[i] >= 0.0)) {
        RCLCPP_ERROR(logger, "Parameter %s[%zu] must be >= 0", name.c_str(), i);
        return false;
      }
      out(i) = v[i];
    }
    return true;
  };
  // scalar parameter, value >= min_value
  auto read_scalar = [&](const std::string& name, double& out, double min_value) {
    const double v = node->get_parameter(name).as_double();
    if (!(v >= min_value)) {
      RCLCPP_ERROR(logger, "Parameter %s = %f must be >= %f", name.c_str(), v, min_value);
      return false;
    }
    out = v;
    return true;
  };

  Eigen::Matrix<double, 6, 1> stiffness;
  bool ok = read_vec6("stiffness", stiffness) &&
            read_scalar("damping.ratio_translational", damping_ratio_translational_, 0.0) &&
            read_scalar("damping.ratio_rotational", damping_ratio_rotational_, 0.0) &&
            read_scalar("damping.rot_z_min_factor", damping_rot_z_min_factor_, 0.0) &&
            read_vec6("integrator.weights", integrator_weights) &&
            read_scalar("integrator.limit_force", integrator_limit_force_, 0.0) &&
            read_scalar("integrator.limit_torque", integrator_limit_torque_, 0.0) &&
            read_scalar("integrator.limit_torque_z", integrator_limit_torque_z_, 0.0) &&
            read_scalar("saturation.max_force", max_force_, 0.0) &&
            read_scalar("saturation.max_torque", max_torque_, 0.0) &&
            read_scalar("saturation.max_torque_z", max_torque_z_, 0.0) &&
            read_scalar("saturation.max_damping_force", max_damping_force_, 0.0) &&
            read_scalar("saturation.max_damping_torque", max_damping_torque_, 0.0) &&
            read_scalar("filter_params", filter_params_, 0.0) &&
            read_scalar("delta_tau_max", delta_tau_max_, 0.0) &&
            read_scalar("max_reference_velocity.translational", max_reference_velocity_translational_, 0.0) &&
            read_scalar("max_reference_velocity.rotational", max_reference_velocity_rotational_, 0.0) &&
            read_scalar("lambda_regularization", lambda_regularization_, 0.0) &&
            read_scalar("nullspace.stiffness", nullspace_stiffness_target_, 0.0) &&
            read_scalar("nullspace.damping", nullspace_damping_, 0.0) &&
            read_scalar("nullspace.max_torque", nullspace_max_torque_, 0.0) &&
            read_vec6("modes.normal_stiffness", stiffness_mode_normal_) &&
            read_vec6("modes.float_stiffness", stiffness_mode_float_) &&
            read_vec6("modes.lift_stiffness", stiffness_mode_lift_) &&
            read_vec6("modes.contact_stiffness", stiffness_mode_contact_) &&
            read_scalar("debug.publish_rate", debug_publish_rate_, 0.0);
  if (!ok) {
    return false;
  }
  if (debug_publish_rate_ > 1000.0) {
    RCLCPP_ERROR(logger, "Parameter debug.publish_rate = %f must be <= 1000 (controller rate)", debug_publish_rate_);
    return false;
  }
  if (filter_params_ > 1.0) {
    RCLCPP_ERROR(logger, "Parameter filter_params = %f must be <= 1", filter_params_);
    return false;
  }
  integrator_reset_on_new_target_ = node->get_parameter("integrator.reset_on_new_target").as_bool();

  nullspace_target_from_activation_ = node->get_parameter("nullspace.target_from_activation").as_bool();
  const auto q_target = node->get_parameter("nullspace.q_target").as_double_array();
  if (q_target.size() != 7) {
    RCLCPP_ERROR(logger, "Parameter nullspace.q_target needs 7 values, got %zu", q_target.size());
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    nullspace_q_target_param_(i) = q_target[i];
  }

  start_pose_enabled_ = node->get_parameter("start_pose.enabled").as_bool();
  start_pose_use_orientation_ = node->get_parameter("start_pose.use_orientation").as_bool();
  const auto start_position = node->get_parameter("start_pose.position").as_double_array();
  const auto start_orientation = node->get_parameter("start_pose.orientation_xyzw").as_double_array();
  if (start_position.size() != 3 || start_orientation.size() != 4) {
    RCLCPP_ERROR(logger, "start_pose.position needs 3 values and start_pose.orientation_xyzw 4 values");
    return false;
  }
  start_position_ << start_position[0], start_position[1], start_position[2];
  start_orientation_.coeffs() << start_orientation[0], start_orientation[1], start_orientation[2], start_orientation[3];
  if (start_orientation_.norm() < 1e-6) {
    RCLCPP_ERROR(logger, "start_pose.orientation_xyzw must not be zero");
    return false;
  }
  start_orientation_.normalize();
  if (!read_scalar("start_pose.max_velocity", start_pose_max_velocity_, 1e-3) ||
      !read_scalar("start_pose.max_angular_velocity", start_pose_max_angular_velocity_, 1e-3)) {
    return false;
  }
  if (start_pose_max_velocity_ > 0.5) {
    RCLCPP_ERROR(logger, "start_pose.max_velocity = %f must be <= 0.5 m/s", start_pose_max_velocity_);
    return false;
  }
  clamp_spring_only_ = node->get_parameter("saturation.clamp_spring_only").as_bool();

  target_limits_.max_distance = node->get_parameter("target.max_distance").as_double();
  target_limits_.workspace_enabled = node->get_parameter("target.workspace_enabled").as_bool();
  const auto workspace_min = node->get_parameter("target.workspace_min").as_double_array();
  const auto workspace_max = node->get_parameter("target.workspace_max").as_double_array();
  if (workspace_min.size() != 3 || workspace_max.size() != 3) {
    RCLCPP_ERROR(logger, "target.workspace_min and target.workspace_max need 3 values each");
    return false;
  }
  target_limits_.workspace_min << workspace_min[0], workspace_min[1], workspace_min[2];
  target_limits_.workspace_max << workspace_max[0], workspace_max[1], workspace_max[2];
  const std::string limits_error = check_target_limits(target_limits_);
  if (!limits_error.empty()) {
    RCLCPP_ERROR(logger, "%s", limits_error.c_str());
    return false;
  }
  K.setZero();
  K.diagonal() = stiffness;

  RCLCPP_INFO_STREAM(logger, "Tuning: K = [" << K.diagonal().transpose() << "], damping ratio trans/rot = "
      << damping_ratio_translational_ << "/" << damping_ratio_rotational_ << ", integrator weights = ["
      << integrator_weights.transpose() << "], clamp F/T/Tz = " << max_force_ << "/" << max_torque_ << "/"
      << max_torque_z_ << (clamp_spring_only_ ? " (spring only)" : " (total)") << ", filter_params = "
      << filter_params_ << ", max reference velocity trans/rot = " << max_reference_velocity_translational_
      << "/" << max_reference_velocity_rotational_ << ", delta_tau_max = " << delta_tau_max_ << ", lambda_regularization = "
      << lambda_regularization_ << ", nullspace k/d = " << nullspace_stiffness_target_ << "/"
      << nullspace_damping_ << " (target " << (nullspace_target_from_activation_ ? "activation q" : "nullspace.q_target")
      << "), integrator reset on new target = " << integrator_reset_on_new_target_ << ", start pose "
      << (start_pose_enabled_ ? "ON" : "off"));
  RCLCPP_INFO_STREAM(logger, "Target checks: max distance " << target_limits_.max_distance << " m, workspace box "
      << (target_limits_.workspace_enabled ? "ON [" : "off [") << target_limits_.workspace_min.transpose() << "] .. ["
      << target_limits_.workspace_max.transpose() << "] m (base frame)");
  return true;
}

void CartesianImpedanceController::setup_debug_publisher() {
  debug_publisher_.reset();
  debug_publisher_raw_.reset();
  if (debug_publish_rate_ <= 0.0) {
    return;
  }
  debug_decimation_ = std::max(1, static_cast<int>(std::lround(1000.0 / debug_publish_rate_)));

  // field order of the debug message; vectors expand to <name>_<axis>
  debug_field_names_.clear();
  auto add = [this](const std::string& name, const std::vector<std::string>& axes) {
    for (const auto& axis : axes) {
      debug_field_names_.push_back(axes.size() == 1 ? name : name + "_" + axis);
    }
  };
  const std::vector<std::string> one{""};
  const std::vector<std::string> xyz{"x", "y", "z"};
  const std::vector<std::string> quat{"qx", "qy", "qz", "qw"};
  const std::vector<std::string> twist{"x", "y", "z", "rx", "ry", "rz"};
  const std::vector<std::string> joints{"1", "2", "3", "4", "5", "6", "7"};
  add("t", one);                  // controller time [s]
  add("error", twist);            // pose error (position [m], rotation [rad, ~sin(angle/2)*2]), base frame
  add("position", xyz);           // measured end-effector position [m]
  add("position_d", xyz);         // filtered reference used in this cycle [m]
  add("position_d_target", xyz);  // target from set_pose / reference_pose [m]
  add("orientation", quat);       // measured
  add("orientation_d", quat);     // filtered reference
  add("v", twist);                // end-effector twist J*dq [m/s, rad/s]
  add("F_cmd", twist);            // commanded wrench after clamping [N, Nm]
  add("F_unclamped", twist);      // -(K*e + D*v + I) before clamping
  add("F_spring", twist);         // -K*e
  add("F_damping", twist);        // -D*v
  add("I_error", twist);          // integrator state
  add("K", twist);                // diag(K)
  add("D", twist);                // diag(D)
  add("Lambda", twist);           // diag(Lambda)
  add("q", joints);               // [rad]
  add("dq", joints);              // [rad/s]
  add("tau_raw", joints);         // desired torque before the torque rate limit [Nm]
  add("tau_cmd", joints);         // commanded torque (after rate limit) [Nm]
  add("tau_nullspace", joints);   // [Nm]
  add("coriolis", joints);        // [Nm]
  add("sigma_min_J", one);        // smallest singular value of J (-> 0 near a singularity)
  add("control_mode", one);

  debug_publisher_raw_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
      "~/debug", rclcpp::SystemDefaultsQoS());
  debug_publisher_ =
      std::make_unique<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(debug_publisher_raw_);
  debug_publisher_->lock();
  auto& msg = debug_publisher_->msg_;
  msg.layout.dim.resize(1);
  std::string label;
  for (const auto& name : debug_field_names_) {
    label += (label.empty() ? "" : ",") + name;
  }
  msg.layout.dim[0].label = label;
  msg.layout.dim[0].size = debug_field_names_.size();
  msg.layout.dim[0].stride = debug_field_names_.size();
  msg.data.assign(debug_field_names_.size(), 0.0);  // preallocated, the 1 kHz loop only overwrites values
  debug_publisher_->unlock();
  RCLCPP_INFO(get_node()->get_logger(), "Debug topic %s: %zu fields at %.0f Hz",
              debug_publisher_raw_->get_topic_name(), debug_field_names_.size(), 1000.0 / debug_decimation_);
}

CallbackReturn CartesianImpedanceController::on_init() {
   declare_tuning_parameters();
   // No input-server thread any more (fix-list a13): /set_pose and /set_param are services of this node
   // (on_configure), executed by the controller manager's executor.
   return CallbackReturn::SUCCESS;
}


CallbackReturn CartesianImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    if (!read_tuning_parameters()) {
      return CallbackReturn::ERROR;
    }
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
    franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name,
                                                 arm_id_ + "/" + k_robot_state_interface_name));
                                               
  // handoff buffers start defined (Eigen vectors are not zero-initialised); nothing else runs yet at configure
  F_ext.setZero();
  f_ext_buffer_.initRT(F_ext);
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    command_buffer_.initRT(command_nonrt_);
  }

  try {
    rclcpp::QoS qos_profile(1); // Depth of the message queue
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    franka_state_subscriber = get_node()->create_subscription<franka_msgs::msg::FrankaRobotState>(
    "franka_robot_state_broadcaster/robot_state", qos_profile, 
    std::bind(&CartesianImpedanceController::topic_callback, this, std::placeholders::_1));
    std::cout << "Succesfully subscribed to robot_state_broadcaster" << std::endl;
    setup_debug_publisher();
    // absolute names: the old user_input_client calls "set_pose" / "set_param" from the root namespace
    pose_srv_ = get_node()->create_service<messages_fr3::srv::SetPose>(
        "/set_pose", std::bind(&CartesianImpedanceController::setPose, this, std::placeholders::_1, std::placeholders::_2));
    param_srv_ = get_node()->create_service<messages_fr3::srv::SetParam>(
        "/set_param", std::bind(&CartesianImpedanceController::setParam, this, std::placeholders::_1, std::placeholders::_2));
  }

  catch (const std::exception& e) {
    fprintf(stderr,  "Exception thrown during publisher creation at configure stage with message : %s \n",e.what());
    return CallbackReturn::ERROR;
    }


  RCLCPP_DEBUG(get_node()->get_logger(), "configured successfully");
  return CallbackReturn::SUCCESS;
}


CallbackReturn CartesianImpedanceController::on_activate(
  const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  // joint state interfaces by name (fix-list a12): the order of state_interfaces_ is not guaranteed by the API
  for (int i = 0; i < num_joints; ++i) {
    const std::string joint = arm_id_ + "_joint" + std::to_string(i + 1);
    for (const auto& [type, index] : {std::pair<std::string, size_t*>{"position", &joint_position_index_[i]},
                                      std::pair<std::string, size_t*>{"velocity", &joint_velocity_index_[i]}}) {
      const std::string name = joint + "/" + type;
      const auto it = std::find_if(state_interfaces_.begin(), state_interfaces_.end(),
                                   [&name](const auto& interface) { return interface.get_name() == name; });
      if (it == state_interfaces_.end()) {
        RCLCPP_ERROR(get_node()->get_logger(), "State interface %s not found, activation aborted", name.c_str());
        franka_robot_model_->release_interfaces();
        return CallbackReturn::ERROR;
      }
      *index = static_cast<size_t>(std::distance(state_interfaces_.begin(), it));
    }
  }

  std::array<double, 16> initial_pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose.data()));
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target_ = position_d_;
  orientation_d_target_ = orientation_d_;
  last_position_d_target_ = position_d_target_;
  I_error.setZero();
  // ignore commands left in the handoff buffer from a previous activation
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    applied_pose_seq_ = command_nonrt_.pose_seq;
    applied_mode_seq_ = command_nonrt_.mode_seq;
  }
  {
    std::lock_guard<std::mutex> lock(current_position_mutex_);
    current_position_ = position_d_;
    current_position_valid_ = true;
  }
  updateJointStates();
  // nullspace posture target: configuration at activation or the fixed nullspace.q_target
  q_d_nullspace_ = nullspace_target_from_activation_ ? Vector7d(q_) : nullspace_q_target_param_;
  nullspace_stiffness_ = 0.0;  // ramps to nullspace.stiffness via filter_params

  // smooth move to the start pose (minimum-jerk, duration from the speed limits)
  start_move_active_ = false;
  if (start_pose_enabled_) {
    start_move_p0_ = position_d_;
    start_move_o0_ = orientation_d_;
    start_move_p1_ = start_position_;
    start_move_o1_ = start_pose_use_orientation_ ? start_orientation_ : orientation_d_;
    if (start_move_o0_.coeffs().dot(start_move_o1_.coeffs()) < 0.0) {
      start_move_o1_.coeffs() << -start_move_o1_.coeffs();  // shortest rotation
    }
    const double distance = (start_move_p1_ - start_move_p0_).norm();
    const double angle = start_move_o0_.angularDistance(start_move_o1_);
    // minimum-jerk peak speed = 1.875 * distance / duration
    start_move_duration_ = std::max({1.875 * distance / start_pose_max_velocity_,
                                     1.875 * angle / start_pose_max_angular_velocity_, 0.5});
    start_move_time_ = 0.0;
    start_move_last_target_ = position_d_target_;
    start_move_active_ = true;
    RCLCPP_INFO(get_node()->get_logger(),
                "Start pose: moving %.3f m / %.2f rad to [%.3f %.3f %.3f] in %.1f s (new targets cancel the move)",
                distance, angle, start_move_p1_(0), start_move_p1_(1), start_move_p1_(2), start_move_duration_);
  }

  K_friction.topLeftCorner(3, 3) = 400 * Eigen::Matrix3d::Identity();
  K_friction.bottomRightCorner(3, 3) << 49, 0, 0, 0, 49, 0, 0, 0, 16;
  D_friction.topLeftCorner(3, 3) = 40 * Eigen::Matrix3d::Identity();
  D_friction.bottomRightCorner(3, 3) << 14, 0, 0, 0, 20, 0, 0, 0, 8;

  // Create the subscriber in the on_activate method
  desired_pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::Pose>(
    "/cartesian_impedance_controller/reference_pose", 1, std::bind(&CartesianImpedanceController::reference_pose_callback, this, std::placeholders::_1));

  // Create the subscriber in the on_activate method
  control_mode_sub_ = get_node()->create_subscription<std_msgs::msg::Int16>(
    "/cartesian_impedance_controller/control_mode", 1, std::bind(&CartesianImpedanceController::control_mode_callback, this, std::placeholders::_1));

  std::cout << "Completed Activation process" << std::endl;
  return CallbackReturn::SUCCESS;
}


controller_interface::CallbackReturn CartesianImpedanceController::on_deactivate(
  const rclcpp_lifecycle::State& /*previous_state*/) {
  {
    std::lock_guard<std::mutex> lock(current_position_mutex_);
    current_position_valid_ = false;  // reject targets while inactive
  }
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

std::array<double, 6> CartesianImpedanceController::convertToStdArray(const geometry_msgs::msg::WrenchStamped& wrench) {
    std::array<double, 6> result;
    result[0] = wrench.wrench.force.x;
    result[1] = wrench.wrench.force.y;
    result[2] = wrench.wrench.force.z;
    result[3] = wrench.wrench.torque.x;
    result[4] = wrench.wrench.torque.y;
    result[5] = wrench.wrench.torque.z;
    return result;
}

void CartesianImpedanceController::topic_callback(const std::shared_ptr<franka_msgs::msg::FrankaRobotState> msg) {
  O_F_ext_hat_K = convertToStdArray(msg->o_f_ext_hat_k);
  Eigen::Matrix<double, 6, 1> F_ext_measured;
  arrayToMatrix(O_F_ext_hat_K, F_ext_measured);
  F_ext = 0.1 * F_ext_measured + 0.9 * F_ext; //filtering external force with low pass filter
  f_ext_buffer_.writeFromNonRT(F_ext);  // the z-contact switch (control_mode 1) runs in update()
}

void CartesianImpedanceController::updateJointStates() {
  // indices resolved by name in on_activate (fix-list a12)
  for (auto i = 0; i < num_joints; ++i) {
    q_(i) = state_interfaces_[joint_position_index_[i]].get_value();
    dq_(i) = state_interfaces_[joint_velocity_index_[i]].get_value();
  }
}

void CartesianImpedanceController::step_start_move() {
  // a target from set_pose / reference_pose cancels the move; the reference then continues from where it is
  if ((position_d_target_ - start_move_last_target_).norm() > 1e-9) {
    start_move_active_ = false;
    return;
  }
  start_move_time_ += dt;
  const double s = std::min(start_move_time_ / start_move_duration_, 1.0);
  const double sigma = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);  // minimum-jerk profile 0 -> 1
  position_d_ = start_move_p0_ + sigma * (start_move_p1_ - start_move_p0_);
  orientation_d_ = start_move_o0_.slerp(sigma, start_move_o1_);
  position_d_target_ = position_d_;
  orientation_d_target_ = orientation_d_;
  start_move_last_target_ = position_d_target_;
  if (s >= 1.0) {
    start_move_active_ = false;
    I_error.setZero();
  }
}

controller_interface::return_type CartesianImpedanceController::update(const rclcpp::Time& time, const rclcpp::Duration& /*period*/) {  
  std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  std::array<double, 42> jacobian_array =  franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 16> pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  J = Eigen::Map<Eigen::Matrix<double, 6, 7>>(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 7>> M(mass.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(pose.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());
  //orientation_d_target_ = Eigen::AngleAxisd(rotation_d_target_[0], Eigen::Vector3d::UnitX())
  //                      * Eigen::AngleAxisd(rotation_d_target_[1], Eigen::Vector3d::UnitY())
  //                      * Eigen::AngleAxisd(rotation_d_target_[2], Eigen::Vector3d::UnitZ());
  updateJointStates(); 
  {
    // current EE position for the target distance check in the callbacks; skipped if a callback holds the lock
    std::unique_lock<std::mutex> lock(current_position_mutex_, std::try_to_lock);
    if (lock.owns_lock()) {
      current_position_ = position;
    }
  }
  apply_external_commands();  // new pose / control mode from the callbacks (fix-list a13)
  if (start_move_active_) {
    step_start_move();
  }

  
  error.head(3) << position - position_d_;

  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);

  // z-contact switch in control_mode 1 (moved here from the robot_state callback 2026-10-19, same condition; it
  // uses the error of this cycle and the latest filtered F_ext)
  const Eigen::Matrix<double, 6, 1>& F_ext_filtered = *f_ext_buffer_.readFromRT();
  if (F_ext_filtered(2) < -4.0 && K(2,2) >= 700 && error(2) > 0 && control_mode == 1 && std::abs(error(2)) < 0.03 && position_d_(2) <= 0.05) //only if grasped
  {
    RCLCPP_INFO(get_node()->get_logger(), "Force in z-direction detected: %.2f N, switching to compliant mode",
                F_ext_filtered(2));  // once per switch: K(2,2) drops below 700 afterwards
    I_error.setZero(); //reset integral error
    K.diagonal() = stiffness_mode_contact_; // parameter modes.contact_stiffness (low z, see controllers.yaml)
    //K.bottomRightCorner(3,3) = 0.25 * K.bottomRightCorner(3,3); //decrease rotational stiffness as well
  }

  Lambda = (J * M.inverse() * J.transpose() + lambda_regularization_ * IDENTITY).inverse();
  //Theta = Lambda;

  // correcting D to be critically damped
  D =  2.0 * K.cwiseMax(0.0).cwiseSqrt() * Lambda.cwiseMax(0.0).diagonal().cwiseSqrt().asDiagonal();
  D.topLeftCorner(3,3) *= damping_ratio_translational_;
  D.bottomRightCorner(3,3) *= damping_ratio_rotational_; //increase rotational damping a bit
  D(5,5) = std::max(D(5,5), damping_rot_z_min_factor_ * std::sqrt(K(5,5))); //minimal rotational damping in z-direction to avoid instabilities
  //D(4,4) = std::max(D(4,4), 2*std::sqrt(K(4,4))); //minimal rotational damping in z-direction to avoid instabilities
  //D(3,3) = std::max(D(3,3), 2*std::sqrt(K(3,3))); //minimal rotational damping in z-direction to avoid instabilities
  D.topRightCorner(3,3).setZero();
  D.bottomLeftCorner(3,3).setZero();
  
  if (integrator_reset_on_new_target_ && (position_d_target_ - last_position_d_target_).norm() > 1e-9) {
    I_error.setZero(); //new target (e.g. set_pose service): reset integral error
  }
  last_position_d_target_ = position_d_target_;
  I_error = I_error + integrator_weights.cwiseProduct(error) * 0.001; //weighted integral action
  //I_error.tail(4).setZero(); //only integrate position error in x and y, not rotation error
  I_error.head(3) = I_error.head(3).cwiseMin(integrator_limit_force_).cwiseMax(-integrator_limit_force_); //anti-windup
  I_error.tail(3) = I_error.tail(3).cwiseMin(integrator_limit_torque_).cwiseMax(-integrator_limit_torque_); //anti-windup
  I_error.tail(1) = I_error.tail(1).cwiseMin(integrator_limit_torque_z_).cwiseMax(-integrator_limit_torque_z_); //anti-windup

  // clamp to +-(max_force, max_force, max_force, max_torque, max_torque, max_torque_z)
  auto clamp_wrench = [](Eigen::Matrix<double, 6, 1> w, double f_max, double t_max, double tz_max) {
    w.head(3) = w.head(3).cwiseMin(f_max).cwiseMax(-f_max);
    w.tail(3) = w.tail(3).cwiseMin(t_max).cwiseMax(-t_max);
    w(5) = std::max(-tz_max, std::min(tz_max, w(5)));
    return w;
  };
  if (clamp_spring_only_) {
    // spring + integrator clamped, damping clamped separately, so damping stays active while the spring saturates
    Eigen::Matrix<double, 6, 1> F_spring = clamp_wrench(-1 * (K * error + I_error), max_force_, max_torque_, max_torque_z_);
    Eigen::Matrix<double, 6, 1> F_damping = clamp_wrench(-1 * (D * (J * dq_)), max_damping_force_, max_damping_torque_, max_damping_torque_);
    F_impedance = F_spring + F_damping;
  } else {
    F_impedance = -1 * (D * (J * dq_) + K * error + I_error);
    F_impedance = clamp_wrench(F_impedance, max_force_, max_torque_, max_torque_z_); //clamping to avoid instabilities
  }

  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7), tau_impedance(7);
  tau_nullspace.setZero();
  if (nullspace_stiffness_ > 0.0 || nullspace_damping_ > 0.0) {
    // posture spring + joint damping, projected into the nullspace of J with the exact projector
    // N = I - J^+ J = I - sum(v_i v_i^T) over the right singular vectors with sigma_i > threshold
    // (the damped pinv used before leaked part of the posture spring into the end-effector force)
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 7>> svd_J(J, Eigen::ComputeFullV);
    Eigen::Matrix<double, 7, 7> N_exact = Eigen::Matrix<double, 7, 7>::Identity();
    for (int k = 0; k < 6; ++k) {
      if (svd_J.singularValues()(k) > 1e-3) {
        N_exact -= svd_J.matrixV().col(k) * svd_J.matrixV().col(k).transpose();
      }
    }
    tau_nullspace << N_exact * (nullspace_stiffness_ * (q_d_nullspace_ - q_) - nullspace_damping_ * dq_);
    tau_nullspace = tau_nullspace.cwiseMin(nullspace_max_torque_).cwiseMax(-nullspace_max_torque_);
  }

  //tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
  //                  J.transpose() * jacobian_transpose_pinv) *
  //                  (nullspace_stiffness_ * config_control * (q_d_nullspace_ - q_) - //if config_control = true we control the whole robot configuration
  //                  (2.0 * sqrt(nullspace_stiffness_)) * dq_);  // if config control ) false we don't care about the joint position

  // tau_impedance = J.transpose() * Sm * (F_impedance /*+ F_repulsion + F_potential*/) + J.transpose() * Sf * F_cmd;
  tau_impedance = J.transpose() * F_impedance;
  //auto tau_d_placeholder = tau_impedance 
  tau_d << tau_impedance + tau_nullspace + coriolis; //tau_nullspace = 0 unless nullspace.stiffness/damping > 0 //add nullspace and coriolis components to desired torque
  const Eigen::Matrix<double, 7, 1> tau_raw = tau_d;  // for the debug topic
  tau_d << saturateTorqueRate(tau_d, tau_J_d_M);  // Saturate torque rate to avoid discontinuities
  tau_J_d_M = tau_d;
  //calculate_tau_friction();

  for (size_t i = 0; i < 7; ++i) {
    command_interfaces_[i].set_value(tau_d(i) /*+ tau_friction(i)*/); //adding friction compensation
  }
  
  if (outcounter % 2500/update_frequency == 0){
    //std::cout << "F_ext_robot [N]" << std::endl;
    //std::cout << O_F_ext_hat_K << std::endl;
    //std::cout << O_F_ext_hat_K_M << std::endl;
    //std::cout << "Lambda  Thetha.inv(): " << std::endl;
    //std::cout << Lambda*Theta.inverse() << std::endl;
    //std::cout << "dq: " << dq_.transpose() << std::endl;
    //std::cout << "K: " << K.diagonal().transpose() << std::endl;
    //std::cout << "velocity contribution is: " << (D * (J * dq_)).transpose() << std::endl;
    //std::cout << "I_error: " <<  I_error.transpose() << std::endl;
    //std::cout << "error: " << error.transpose() << std::endl;
    //std::cout << "F: " << F_impedance.transpose() << std::endl;
  }
  outcounter++;

  // debug topic: decimated, non-blocking (sample skipped if the publisher thread is still busy)
  if (debug_publisher_ && ++debug_counter_ >= debug_decimation_) {
    debug_counter_ = 0;
    if (debug_publisher_->trylock()) {
      auto& data = debug_publisher_->msg_.data;
      size_t i = 0;
      auto put = [&data, &i](const auto& vec) {  // bounds-checked; a count mismatch is caught below
        for (Eigen::Index k = 0; k < vec.size(); ++k, ++i) {
          if (i < data.size()) {
            data[i] = vec(k);
          }
        }
      };
      auto put_scalar = [&data, &i](double value) {
        if (i < data.size()) {
          data[i] = value;
        }
        ++i;
      };
      const Eigen::Matrix<double, 6, 1> v = J * dq_;
      const Eigen::Matrix<double, 6, 1> F_spring = -K * error;
      const Eigen::Matrix<double, 6, 1> F_damping = -D * v;
      const Eigen::Matrix<double, 6, 1> F_unclamped = F_spring + F_damping - I_error;
      Eigen::JacobiSVD<Eigen::Matrix<double, 6, 7>> svd(J);
      put_scalar(time.seconds());
      put(error);
      put(position);
      put(position_d_);
      put(position_d_target_);
      put(orientation.coeffs());
      put(orientation_d_.coeffs());
      put(v);
      put(F_impedance);
      put(F_unclamped);
      put(F_spring);
      put(F_damping);
      put(I_error);
      put(K.diagonal());
      put(D.diagonal());
      put(Lambda.diagonal());
      put(q_);
      put(dq_);
      put(tau_raw);
      put(tau_d);
      put(tau_nullspace);
      put(coriolis);
      put_scalar(svd.singularValues()(5));
      put_scalar(control_mode);
      if (i == data.size()) {
        debug_publisher_->unlockAndPublish();
      } else {
        debug_publisher_->unlock();  // field list and data out of sync: never publish a wrong layout
        RCLCPP_ERROR_ONCE(get_node()->get_logger(), "Debug topic: %zu values for %zu fields", i, data.size());
      }
    }
  }

  update_stiffness_and_references();
  return controller_interface::return_type::OK;
}
}

// namespace cartesian_impedance_control
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cartesian_impedance_control::CartesianImpedanceController,
                       controller_interface::ControllerInterface)