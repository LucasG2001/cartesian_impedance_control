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
    if (friction_abs_threshold_) {
      pose_error = (pose_error.array().abs() < 0.0008).select(0.0, pose_error); //set errors smaller than 0.0008 to 0
    } else {
      pose_error = (pose_error.array() < 0.0008).select(0.0, pose_error); // upstream: zeroes all negative errors too
    }
    // filter constant 0.1 is per 1 ms step; rescale so the time constant does not depend on the rate
    const double alpha = 1.0 - std::pow(1.0 - 0.1, dt_ / 0.001);
    dq_filtered = alpha* dq_ + (1 - alpha) * dq_filtered; //Filtering dq of every joint
    tau_impedance_filtered = alpha*tau_impedance + (1 - alpha) * tau_impedance_filtered; //Filtering tau_impedance
    //Creating and filtering a "fake" tau_impedance with own weights, optimized for friction compensation (else friction compensation would get stronger with higher stiffnesses)
    tau_friction_impedance = J.transpose() * (-alpha * (D_friction*(J*dq_) + K_friction * (pose_error))) + (1 - alpha) * tau_friction_impedance;
    //Creating "fake" dq, that acts only in the impedance-space, else dq in the nullspace also gets compensated, which we do not want due to null-space movement
    dq_imp = dq_filtered - N * dq_filtered;
    //Calculation of friction force according to Bachelor Thesis: https://polybox.ethz.ch/index.php/s/iYj8ALPijKTAC2z?path=%2FFriction%20compensation
    f = beta.cwiseProduct(dq_imp) + (friction_offsets_ ? offset_friction : Eigen::Matrix<double, 7, 1>::Zero());
    g = coulomb_friction;
    g(4) = (coulomb_friction(4) + (static_friction(4) - coulomb_friction(4)) * exp(-1 * std::abs(dq_imp(4)/(dq_s(4) + 0.0001))));
    g(6) = (coulomb_friction(6) + (static_friction(6) - coulomb_friction(6)) * exp(-1 * std::abs(dq_imp(6)/dq_s(6))));
           
    dz = dq_imp.array() - dq_imp.array().abs() / g.array() * sigma_0.array() * z.array() + 0.025 * tau_friction_impedance.array()/*(J.transpose() * K * error).array()*/;
    dz(6) -= 0.02*tau_friction_impedance(6);
    z = dt_ * dz + z;  // upstream: 0.001 * dz (1 kHz)
    tau_friction = sigma_0.array() * z.array() + 100 * sigma_1.array() * dz.array() + f.array();  
    // clamp elementwise between -4 and 4s
    tau_friction = tau_friction.cwiseMin(3.5).cwiseMax(-3.5);
   
}

void CartesianImpedanceController::reference_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg)
{
    // Handle the incoming pose message
    std::cout << "received reference posistion as " <<  msg->position.x << ", " << msg->position.y << ", " << msg->position.z << std::endl;
    position_d_target_ << msg->position.x, msg->position.y, msg->position.z;
    orientation_d_target_.coeffs() << msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w;
    if (integrator_reset_on_target_) {
      I_error.setZero(); //reset integral error when a new target is received
    }
    //control_mode = 0; // Set to normal mode when a new pose is received
    
  
}

void CartesianImpedanceController::mpc_command_callback(const ci_mpc_msgs::msg::MpcCommand::SharedPtr msg)
{
  mpc_buffer_.writeFromNonRT(msg);  // applied in update(); a new pointer marks a new command
}

void CartesianImpedanceController::control_mode_callback(const std_msgs::msg::Int16::SharedPtr msg)
{
  RCLCPP_INFO(get_node()->get_logger(), "Received Control Mode Callback.");
    // Handle the incoming control mode message
    if (msg->data == 3){
      //lift -> high extraction stiffness
        K(2,2) = 8500.0; 
        K(0,0) = 0.0; 
        K(1,1) = 0.0;
        K.bottomRightCorner(3,3) = Eigen::MatrixXd::Identity(3,3) * 0; //reset rotational stiffness as well
        K(5,5) = 0;
        control_mode = 3;
    }
    else if (msg->data ==2 ) {
        control_mode = 2; // low stiffness float mode
        I_error.setZero(); //reset integral error
        K(2,2) = 150.0; //if a force above 5N is detected in z-direction set stiffness in z to very low
        K(0,0) = 150.0; //decrease stiffness in x and y direction as well
        K(1,1) = 150.0;
        K.bottomRightCorner(3,3) = Eigen::MatrixXd::Identity(3,3) * 40; //reset rotational stiffness as well
        K(5,5) = 20;
        RCLCPP_INFO(get_node()->get_logger(), "Control mode set to gripper closed.");
    }
    else if (msg->data == 1) {
        control_mode = 1; // Calibration mode
        RCLCPP_INFO(get_node()->get_logger(), "Control mode set to compliant");
    } 
    else {
        control_mode = 0; // Normal mode
        K(2,2) = 1500.0; 
        K(0,0) = 2500.0; 
        K(1,1) = 2500.0;
        K.bottomRightCorner(3,3) = Eigen::MatrixXd::Identity(3,3) * 130; //reset rotational stiffness as well
        K(5,5) = 45;
        RCLCPP_INFO(get_node()->get_logger(), "Control mode set to normal (gripper open).");
    }
}


void CartesianImpedanceController::update_stiffness_and_references(){
  //target by filtering
  /** at the moment we do not use dynamic reconfigure and control the robot via D, K and T **/
  //K = filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * K;
  //D = filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * D;
  // filter_params_ is defined per 1 ms step; rescale so the time constant does not depend on the rate
  const double a = 1.0 - std::pow(1.0 - filter_params_, dt_ / 0.001);
  nullspace_stiffness_ = a * nullspace_stiffness_target_ + (1.0 - a) * nullspace_stiffness_;
  //std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  position_d_ = a * position_d_target_ + (1.0 - a) * position_d_;
  orientation_d_ = orientation_d_.slerp(a, orientation_d_target_);
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
  const double max_step = delta_tau_max_ * dt_ / 0.001;  // delta_tau_max_ is per 1 ms step
  for (size_t i = 0; i < 7; i++) {
  double difference = tau_d_calculated[i] - tau_J_d_M[i];
  tau_d_saturated[i] =
         tau_J_d_M[i] + std::max(std::min(difference, max_step), -max_step);
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
  if (usePinocchio() && observer_torque_source_ == "measured") {  // after all position / velocity interfaces
    for (int i = 1; i <= num_joints; ++i) {
      state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
    }
  }

  if (!usePinocchio()) {
    for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
      state_interfaces_config.names.push_back(franka_robot_model_name);
      std::cout << franka_robot_model_name << std::endl;
    }
  }

  const std::string full_interface_name = arm_id_ + "/" + state_interface_name_;

  return state_interfaces_config;
}


CallbackReturn CartesianImpedanceController::on_init() {
   try {
     auto_declare<std::string>("arm_id", "fr3");
     auto_declare<std::string>("model_source", "franka");
     auto_declare<std::string>("urdf_path", "");
     auto_declare<std::string>("ee_frame", "fr3_hand_tcp");
     auto_declare<bool>("enable_input_server", true);
     auto_declare<double>("nullspace_damping", 0.0);
     auto_declare<std::vector<double>>("armature", std::vector<double>{});
     auto_declare<std::vector<double>>("integrator_weights", {350, 350, 0, 50, 50, 50});
     auto_declare<std::vector<double>>("integrator_limits", {2.5, 2.5, 2.5, 1.5, 1.5, 0.5});
     auto_declare<std::vector<double>>("integrator_windows", {0, 0, 0, 0, 0, 0});
     auto_declare<double>("integrator_velocity_window", 0.0);
     auto_declare<double>("integrator_leak", 0.0);
     auto_declare<bool>("integrator_freeze_on_saturation", false);
     auto_declare<bool>("integrator_reset_on_target", true);
     auto_declare<double>("mpc_timeout", 0.05);
     auto_declare<double>("observer_gain", 30.0);
     auto_declare<std::string>("observer_torque_source", "measured");
     auto_declare<bool>("observer_gravity", false);
     auto_declare<double>("observer_torque_noise", 0.0);
     auto_declare<std::vector<double>>("observer_torque_bias", std::vector<double>{});
     auto_declare<std::vector<double>>("observer_armature", std::vector<double>{});
     auto_declare<int>("external_wrench_every", 2);
     auto_declare<double>("mpc_integral_off_leak", 5.0);
     auto_declare<std::string>("friction_compensation", "none");
     auto_declare<bool>("friction_offsets", true);
     auto_declare<bool>("friction_abs_threshold", false);
     auto_declare<double>("friction_scale", 1.0);
     auto_declare<double>("friction_tau_eps", 0.3);
     auto_declare<double>("friction_vel_eps", 0.02);
   } catch (const std::exception& e) {
     RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
     return CallbackReturn::ERROR;
   }
   // The input server spins its own node and calls rclcpp::shutdown() when done, so it must be
   // disabled when the controller runs inside another process (e.g. Isaac Sim).
   if (get_node()->get_parameter("enable_input_server").as_bool()) {
     UserInputServer input_server_obj(&position_d_target_, &rotation_d_target_, &K, &D, &T);
     std::thread input_thread(&UserInputServer::main, input_server_obj, 0, nullptr);
     input_thread.detach();
   }
   return CallbackReturn::SUCCESS;
}


CallbackReturn CartesianImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    model_source_ = get_node()->get_parameter("model_source").as_string();
    urdf_path_ = get_node()->get_parameter("urdf_path").as_string();
    ee_frame_ = get_node()->get_parameter("ee_frame").as_string();
    enable_input_server_ = get_node()->get_parameter("enable_input_server").as_bool();
    nullspace_damping_ = get_node()->get_parameter("nullspace_damping").as_double();
    const auto weights = get_node()->get_parameter("integrator_weights").as_double_array();
    const auto limits = get_node()->get_parameter("integrator_limits").as_double_array();
    if (weights.size() != 6 || limits.size() != 6) {
      RCLCPP_ERROR(get_node()->get_logger(), "integrator_weights and integrator_limits need 6 entries");
      return CallbackReturn::ERROR;
    }
    integrator_weights = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(weights.data());
    integrator_limits = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(limits.data());
    const auto windows = get_node()->get_parameter("integrator_windows").as_double_array();
    if (windows.size() != 6) {
      RCLCPP_ERROR(get_node()->get_logger(), "integrator_windows needs 6 entries");
      return CallbackReturn::ERROR;
    }
    integrator_windows = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(windows.data());
    integrator_velocity_window_ = get_node()->get_parameter("integrator_velocity_window").as_double();
    integrator_leak_ = get_node()->get_parameter("integrator_leak").as_double();
    integrator_freeze_on_saturation_ = get_node()->get_parameter("integrator_freeze_on_saturation").as_bool();
    integrator_reset_on_target_ = get_node()->get_parameter("integrator_reset_on_target").as_bool();
    mpc_timeout_ = get_node()->get_parameter("mpc_timeout").as_double();
    observer_gain_ = get_node()->get_parameter("observer_gain").as_double();
    observer_torque_source_ = get_node()->get_parameter("observer_torque_source").as_string();
    observer_gravity_ = get_node()->get_parameter("observer_gravity").as_bool();
    observer_torque_noise_ = get_node()->get_parameter("observer_torque_noise").as_double();
    {
      const auto bias = get_node()->get_parameter("observer_torque_bias").as_double_array();
      if (bias.size() == 7) {
        observer_torque_bias_ = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(bias.data());
      } else if (!bias.empty()) {
        RCLCPP_ERROR(get_node()->get_logger(), "observer_torque_bias needs 7 entries");
        return CallbackReturn::ERROR;
      }
    }
    if (observer_torque_source_ != "measured" && observer_torque_source_ != "command") {
      RCLCPP_ERROR(get_node()->get_logger(), "observer_torque_source must be measured or command");
      return CallbackReturn::ERROR;
    }
    {
      const auto arm = get_node()->get_parameter("observer_armature").as_double_array();
      if (arm.size() == 7) {
        observer_armature_ = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(arm.data());
      } else if (!arm.empty()) {
        RCLCPP_ERROR(get_node()->get_logger(), "observer_armature needs 7 entries");
        return CallbackReturn::ERROR;
      }
    }
    wrench_pub_every_ = std::max<int>(1, get_node()->get_parameter("external_wrench_every").as_int());
    wrench_pub_ = std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>(
        get_node()->create_publisher<geometry_msgs::msg::WrenchStamped>("~/external_wrench", 10));
    mpc_integral_off_leak_ = get_node()->get_parameter("mpc_integral_off_leak").as_double();
    friction_compensation_ = get_node()->get_parameter("friction_compensation").as_string();
    friction_offsets_ = get_node()->get_parameter("friction_offsets").as_bool();
    friction_abs_threshold_ = get_node()->get_parameter("friction_abs_threshold").as_bool();
    friction_scale_ = get_node()->get_parameter("friction_scale").as_double();
    friction_tau_eps_ = get_node()->get_parameter("friction_tau_eps").as_double();
    friction_vel_eps_ = get_node()->get_parameter("friction_vel_eps").as_double();
    if (friction_compensation_ != "none" && friction_compensation_ != "lugre" && friction_compensation_ != "coulomb" &&
        friction_compensation_ != "velocity") {
      RCLCPP_ERROR(get_node()->get_logger(), "friction_compensation must be none, lugre, coulomb or velocity");
      return CallbackReturn::ERROR;
    }

    if (model_source_ == "pinocchio") {
      std::vector<std::string> arm_joints;
      for (int i = 1; i <= num_joints; ++i) {
        arm_joints.push_back(arm_id_ + "_joint" + std::to_string(i));
      }
      try {
        pinocchio_model_ = std::make_unique<PinocchioModel>(
            urdf_path_, arm_joints, ee_frame_, get_node()->get_parameter("armature").as_double_array());
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_node()->get_logger(), "Failed to build Pinocchio model from '%s': %s",
                     urdf_path_.c_str(), e.what());
        return CallbackReturn::ERROR;
      }
      RCLCPP_INFO(get_node()->get_logger(), "Model source: pinocchio (%s, ee frame %s, %s)",
                  urdf_path_.c_str(), ee_frame_.c_str(),
                  get_node()->get_parameter("armature").as_double_array().empty() ? "no armature" : "with armature");
    } else if (model_source_ == "franka") {
      franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name,
                                                   arm_id_ + "/" + k_robot_state_interface_name));
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Unknown model_source '%s' (expected 'franka' or 'pinocchio')",
                   model_source_.c_str());
      return CallbackReturn::ERROR;
    }
                                               
  try {
    rclcpp::QoS qos_profile(1); // Depth of the message queue
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    franka_state_subscriber = get_node()->create_subscription<franka_msgs::msg::FrankaRobotState>(
    "franka_robot_state_broadcaster/robot_state", qos_profile, 
    std::bind(&CartesianImpedanceController::topic_callback, this, std::placeholders::_1));
    std::cout << "Succesfully subscribed to robot_state_broadcaster" << std::endl;
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
  if (!usePinocchio()) {
    franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  }

  // Hold the current pose and start from a clean torque / integrator state.
  updateJointStates();
  Eigen::Matrix<double, 7, 7> M;
  Eigen::Matrix<double, 7, 1> coriolis;
  Eigen::Affine3d initial_transform;
  computeModel(M, coriolis, initial_transform);
  tau_J_d_M.setZero();
  I_error.setZero();
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target_ = position_d_;
  orientation_d_target_ = orientation_d_;

  K_friction.topLeftCorner(3, 3) = 400 * Eigen::Matrix3d::Identity();
  K_friction.bottomRightCorner(3, 3) << 49, 0, 0, 0, 49, 0, 0, 0, 16;
  D_friction.topLeftCorner(3, 3) = 40 * Eigen::Matrix3d::Identity();
  D_friction.bottomRightCorner(3, 3) << 14, 0, 0, 0, 20, 0, 0, 0, 8;

  // Create the subscriber in the on_activate method
  desired_pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::Pose>(
    "/cartesian_impedance_controller/reference_pose", 1, std::bind(&CartesianImpedanceController::reference_pose_callback, this, std::placeholders::_1));

  // Create the subscriber in the on_activate method
  mpc_command_sub_ = get_node()->create_subscription<ci_mpc_msgs::msg::MpcCommand>(
    "/cartesian_impedance_controller/mpc_command", rclcpp::SensorDataQoS(),
    std::bind(&CartesianImpedanceController::mpc_command_callback, this, std::placeholders::_1));
  mpc_buffer_.writeFromNonRT(nullptr);
  mpc_last_.reset();
  mpc_active_ = false;
  twist_d_.setZero();
  F_ff_.setZero();
  integral_enabled_ = true;

  // Create the subscriber in the on_activate method
  control_mode_sub_ = get_node()->create_subscription<std_msgs::msg::Int16>(
    "/cartesian_impedance_controller/control_mode", 1, std::bind(&CartesianImpedanceController::control_mode_callback, this, std::placeholders::_1));

  std::cout << "Completed Activation process" << std::endl;
  return CallbackReturn::SUCCESS;
}


controller_interface::CallbackReturn CartesianImpedanceController::on_deactivate(
  const rclcpp_lifecycle::State& /*previous_state*/) {
  if (!usePinocchio()) {
    franka_robot_model_->release_interfaces();
  }
  return CallbackReturn::SUCCESS;
}

void CartesianImpedanceController::updateExternalWrench(const Eigen::Matrix<double, 7, 7>& M_model,
                                                       const Eigen::Matrix<double, 7, 1>& coriolis,
                                                       const Eigen::Matrix<double, 7, 1>& tau_applied) {
  const bool measured = observer_torque_source_ == "measured";
  Eigen::Matrix<double, 7, 7> M = M_model;
  if (!measured) {
    M.diagonal() += observer_armature_;  // the commanded torque also accelerates the rotors
  }
  const Eigen::Matrix<double, 7, 1> p = M * dq_;
  if (!observer_initialized_) {
    observer_initialized_ = true;
    observer_M_prev_ = M;
    observer_p0_ = p;
    observer_integral_.setZero();
    tau_ext_hat_.setZero();
    return;
  }
  observer_M_prev_ = M;
  (void)coriolis;
  const Eigen::Matrix<double, 7, 1> ct_dq = pinocchio_model_->coriolis_transpose_dq();
  const Eigen::Matrix<double, 7, 1> g = observer_gravity_ ? pinocchio_model_->gravity()
                                                          : Eigen::Matrix<double, 7, 1>::Zero();
  observer_integral_ += (tau_applied + ct_dq - g + tau_ext_hat_) * dt_;
  tau_ext_hat_ = observer_gain_ * (p - observer_integral_ - observer_p0_);
  // wrench at the EE frame: tau_ext = J^T F_env  ->  F_env = (J J^T)^-1 J tau_ext; Franka sign: robot on environment
  const Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
  F_ext_hat_ = -JJt.ldlt().solve(J * tau_ext_hat_);
  F_ext = F_ext_hat_;
  if (wrench_pub_ && outcounter % wrench_pub_every_ == 0 && wrench_pub_->trylock()) {
    auto& msg = wrench_pub_->msg_;
    msg.header.stamp = get_node()->now();
    msg.header.frame_id = arm_id_ + "_link0";
    msg.wrench.force.x = F_ext_hat_(0);
    msg.wrench.force.y = F_ext_hat_(1);
    msg.wrench.force.z = F_ext_hat_(2);
    msg.wrench.torque.x = F_ext_hat_(3);
    msg.wrench.torque.y = F_ext_hat_(4);
    msg.wrench.torque.z = F_ext_hat_(5);
    wrench_pub_->unlockAndPublish();
  }
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
  if (F_ext(2) < -4.0 && K(2,2) >= 700 && error(2) > 0 && control_mode == 1 && std::abs(error(2)) < 0.03 && position_d_(2) <= 0.05) //only if grasped
  {
    std::cout << "Force in z-direction detected: " << O_F_ext_hat_K[2] << "N, switching to compliant mode" << std::endl;
    I_error.setZero(); //reset integral error
    K(2,2) = 100.0; //if a force above 5N is detected in z-direction set stiffness in z to very low
    K(0,0) = 2500.0; //decrease stiffness in x and y direction as well
    K(1,1) = 2500.0;
    K.bottomRightCorner(3,3) = Eigen::MatrixXd::Identity(3,3) * 40; //reset rotational stiffness as well
    K(5,5) = 20;
    //K.bottomRightCorner(3,3) = 0.25 * K.bottomRightCorner(3,3); //decrease rotational stiffness as well
  }
  
}

void CartesianImpedanceController::updateJointStates() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

void CartesianImpedanceController::computeModel(Eigen::Matrix<double, 7, 7>& M,
                                               Eigen::Matrix<double, 7, 1>& coriolis,
                                               Eigen::Affine3d& transform) {
  if (usePinocchio()) {
    // needs q_, dq_ from updateJointStates()
    pinocchio_model_->compute(q_, dq_);
    M = pinocchio_model_->mass();
    coriolis = pinocchio_model_->coriolis();
    J = pinocchio_model_->jacobian();
    transform = pinocchio_model_->pose();
    return;
  }
  std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  std::array<double, 42> jacobian_array =  franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 16> pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  coriolis = Eigen::Map<Eigen::Matrix<double, 7, 1>>(coriolis_array.data());
  J = Eigen::Map<Eigen::Matrix<double, 6, 7>>(jacobian_array.data());
  M = Eigen::Map<Eigen::Matrix<double, 7, 7>>(mass.data());
  transform = Eigen::Affine3d(Eigen::Matrix4d::Map(pose.data()));
}

controller_interface::return_type CartesianImpedanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {  
  dt_ = period.seconds();
  if (dt_ <= 0.0 || dt_ > 0.1) {
    dt_ = 0.001;
  }
  updateJointStates();  // first: the Pinocchio model needs q_, dq_
  Eigen::Matrix<double, 7, 7> M;
  Eigen::Matrix<double, 7, 1> coriolis;
  Eigen::Affine3d transform;
  computeModel(M, coriolis, transform);
  if (usePinocchio()) {
    Eigen::Matrix<double, 7, 1> tau = tau_J_d_M;  // command applied during the last period
    if (observer_torque_source_ == "measured") {
      for (int i = 0; i < num_joints; ++i) {
        tau(i) = state_interfaces_.at(2 * num_joints + i).get_value() + observer_torque_bias_(i) +
                 observer_torque_noise_ * observer_normal_(observer_rng_);
      }
    }
    updateExternalWrench(M, coriolis, tau);
  }
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());
  //orientation_d_target_ = Eigen::AngleAxisd(rotation_d_target_[0], Eigen::Vector3d::UnitX())
  //                      * Eigen::AngleAxisd(rotation_d_target_[1], Eigen::Vector3d::UnitY())
  //                      * Eigen::AngleAxisd(rotation_d_target_[2], Eigen::Vector3d::UnitZ());

  
  // MPC command: newest setpoint, or hold the current pose after mpc_timeout_ without one
  const auto mpc = *mpc_buffer_.readFromRT();
  if (mpc && mpc != mpc_last_) {
    mpc_last_ = mpc;
    mpc_age_ = 0.0;
    if (!mpc_active_) {
      RCLCPP_INFO(get_node()->get_logger(), "MPC tracking active");
    }
    mpc_active_ = true;
    position_d_ << mpc->pose.position.x, mpc->pose.position.y, mpc->pose.position.z;
    Eigen::Quaterniond q_cmd(mpc->pose.orientation.w, mpc->pose.orientation.x, mpc->pose.orientation.y,
                             mpc->pose.orientation.z);
    if (q_cmd.norm() > 1e-6) {
      orientation_d_ = q_cmd.normalized();
    }
    position_d_target_ = position_d_;  // the reference filter then leaves the setpoint unchanged
    orientation_d_target_ = orientation_d_;
    twist_d_ << mpc->twist.linear.x, mpc->twist.linear.y, mpc->twist.linear.z, mpc->twist.angular.x,
        mpc->twist.angular.y, mpc->twist.angular.z;
    F_ff_ << mpc->wrench.force.x, mpc->wrench.force.y, mpc->wrench.force.z, mpc->wrench.torque.x,
        mpc->wrench.torque.y, mpc->wrench.torque.z;
    const Eigen::Map<const Eigen::Matrix<double, 6, 1>> k_cmd(mpc->stiffness.data());
    if ((k_cmd.array() != 0.0).any()) {
      K = k_cmd.asDiagonal();
    }
    mpc_damping_ = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(mpc->damping.data());
    mpc_damping_set_ = (mpc_damping_.array() != 0.0).any();
    integral_enabled_ = mpc->integral_enabled;
  } else if (mpc_active_) {
    mpc_age_ += dt_;
    if (mpc_age_ > mpc_timeout_) {
      RCLCPP_WARN(get_node()->get_logger(), "no MPC command for %.0f ms: holding the current pose", mpc_age_ * 1000);
      mpc_active_ = false;
      position_d_ = position;
      position_d_target_ = position;
      orientation_d_ = orientation;
      orientation_d_target_ = orientation;
      twist_d_.setZero();
      F_ff_.setZero();
      mpc_damping_set_ = false;
      integral_enabled_ = true;
      I_error.setZero();  // built for the old setpoint; at the new one (current pose) no force is needed
    }
  }

  error.head(3) << position - position_d_;

  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);

  Lambda = (J * M.inverse() * J.transpose()).inverse();
  //Theta = Lambda;

  // correcting D to be critically damped
  D =  2.0 * K.cwiseMax(0.0).cwiseSqrt() * Lambda.cwiseMax(0.0).diagonal().cwiseSqrt().asDiagonal();
  D.bottomRightCorner(3,3) *= 1.3; //increase rotational damping a bit
  D(5,5) = std::max(D(5,5), 2*std::sqrt(K(5,5))); //minimal rotational damping in z-direction to avoid instabilities
  //D(4,4) = std::max(D(4,4), 2*std::sqrt(K(4,4))); //minimal rotational damping in z-direction to avoid instabilities
  //D(3,3) = std::max(D(3,3), 2*std::sqrt(K(3,3))); //minimal rotational damping in z-direction to avoid instabilities
  D.topRightCorner(3,3).setZero();
  D.bottomLeftCorner(3,3).setZero();
  if (mpc_active_ && mpc_damping_set_) {
    D = mpc_damping_.asDiagonal();
  }
  
  {
    const double ee_speed = (J * dq_).head(3).norm();
    const bool slow = integrator_velocity_window_ <= 0.0 || ee_speed < integrator_velocity_window_;
    const double decay = std::exp(-(integral_enabled_ ? integrator_leak_ : mpc_integral_off_leak_) * dt_);
    for (int i = 0; i < 6; ++i) {
      const bool in_window = integrator_windows(i) <= 0.0 || std::abs(error(i)) < integrator_windows(i);
      const bool frozen = integrator_freeze_on_saturation_ && F_saturated_(i) > 0.5;
      if (integral_enabled_ && slow && in_window && !frozen) {
        I_error(i) += integrator_weights(i) * error(i) * dt_; //weighted integral action
      } else {
        I_error(i) *= decay;
      }
    }
  }
  //I_error.tail(4).setZero(); //only integrate position error in x and y, not rotation error
  I_error = I_error.cwiseMin(integrator_limits).cwiseMax(-integrator_limits); //anti-windup
 
  // twist_d_ and F_ff_ are zero unless an MPC command sets them
  F_impedance = -1 * (D * (J * dq_ - twist_d_) + K * error + I_error) + F_ff_;
  const Eigen::Matrix<double, 6, 1> F_unclamped = F_impedance;
  F_impedance.head(3) = F_impedance.head(3).cwiseMin(18.0).cwiseMax(-18.0); //clamping forces to avoid instabilities
  F_impedance.tail(3) = F_impedance.tail(3).cwiseMin(9.0).cwiseMax(-9.0); //clamping torques to avoid
  F_impedance(5) = std::max(-2.0, std::min(2.0, F_impedance(5))); //clamping torques to avoid
  F_saturated_ = ((F_unclamped - F_impedance).array().abs() > 1e-9).cast<double>();

  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);
  //pseudoInverse(J.transpose(), jacobian_transpose_pinv);

  //tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
  //                  J.transpose() * jacobian_transpose_pinv) *
  //                  (nullspace_stiffness_ * config_control * (q_d_nullspace_ - q_) - //if config_control = true we control the whole robot configuration
  //                  (2.0 * sqrt(nullspace_stiffness_)) * dq_);  // if config control ) false we don't care about the joint position

  // tau_impedance = J.transpose() * Sm * (F_impedance /*+ F_repulsion + F_potential*/) + J.transpose() * Sf * F_cmd;
  tau_impedance = J.transpose() * F_impedance;
  tau_nullspace.setZero();
  if (nullspace_damping_ > 0.0) {
    pseudoInverse(J.transpose(), jacobian_transpose_pinv);
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) - J.transpose() * jacobian_transpose_pinv) *
                     (-nullspace_damping_ * dq_);
  }
  //auto tau_d_placeholder = tau_impedance
  tau_friction.setZero();
  if (friction_compensation_ == "lugre") {
    calculate_tau_friction();
  } else if (friction_compensation_ == "coulomb") {
    for (int i = 0; i < 7; ++i) {
      tau_friction(i) = friction_scale_ * (coulomb_friction(i) * std::tanh(tau_impedance(i) / friction_tau_eps_) +
                                           beta(i) * dq_(i));
    }
  } else if (friction_compensation_ == "velocity") {
    for (int i = 0; i < 7; ++i) {
      tau_friction(i) = friction_scale_ * (coulomb_friction(i) * std::tanh(dq_(i) / friction_vel_eps_) +
                                           beta(i) * dq_(i));
    }
  }
  // friction compensation goes through the torque-rate limit too (upstream added it after the limit)
  tau_d << tau_impedance + tau_nullspace + coriolis + tau_friction; //add nullspace and coriolis components to desired torque
  tau_d << saturateTorqueRate(tau_d, tau_J_d_M);  // Saturate torque rate to avoid discontinuities
  tau_J_d_M = tau_d;

  for (size_t i = 0; i < 7; ++i) {
    command_interfaces_[i].set_value(tau_d(i));
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
  update_stiffness_and_references();
  return controller_interface::return_type::OK;
}
}

// namespace cartesian_impedance_control
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cartesian_impedance_control::CartesianImpedanceController,
                       controller_interface::ControllerInterface)