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

void CartesianImpedanceController::reference_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg)
{
    // Handle the incoming pose message
    std::cout << "received reference posistion as " <<  msg->position.x << ", " << msg->position.y << ", " << msg->position.z << std::endl;
    position_d_target_ << msg->position.x, msg->position.y, msg->position.z;
    orientation_d_target_.coeffs() << msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w;
    I_error.setZero(); //reset integral error when a new target is received 
    //control_mode = 0; // Set to normal mode when a new pose is received
    
  
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
  nullspace_stiffness_ = filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  //std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
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


CallbackReturn CartesianImpedanceController::on_init() {
   UserInputServer input_server_obj(&position_d_target_, &rotation_d_target_, &K, &D, &T);
   std::thread input_thread(&UserInputServer::main, input_server_obj, 0, nullptr);
   input_thread.detach();
   return CallbackReturn::SUCCESS;
}


CallbackReturn CartesianImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
    franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name,
                                                 arm_id_ + "/" + k_robot_state_interface_name));
                                               
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
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  std::array<double, 16> initial_pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose.data()));
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
  control_mode_sub_ = get_node()->create_subscription<std_msgs::msg::Int16>(
    "/cartesian_impedance_controller/control_mode", 1, std::bind(&CartesianImpedanceController::control_mode_callback, this, std::placeholders::_1));

  std::cout << "Completed Activation process" << std::endl;
  return CallbackReturn::SUCCESS;
}


controller_interface::CallbackReturn CartesianImpedanceController::on_deactivate(
  const rclcpp_lifecycle::State& /*previous_state*/) {
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

controller_interface::return_type CartesianImpedanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {  
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
  
  I_error = I_error + integrator_weights.cwiseProduct(error) * 0.001; //weighted integral action
  //I_error.tail(4).setZero(); //only integrate position error in x and y, not rotation error
  I_error.head(3) = I_error.head(3).cwiseMin(2.5).cwiseMax(-2.5); //anti-windup
  I_error.tail(3) = I_error.tail(3).cwiseMin(1.5).cwiseMax(-1.5); //anti-windup
  I_error.tail(1) = I_error.tail(1).cwiseMin(0.5).cwiseMax(-0.5); //anti-windup
 
  F_impedance = -1 * (D * (J * dq_) + K * error + I_error);
  F_impedance.head(3) = F_impedance.head(3).cwiseMin(18.0).cwiseMax(-18.0); //clamping forces to avoid instabilities
  F_impedance.tail(3) = F_impedance.tail(3).cwiseMin(9.0).cwiseMax(-9.0); //clamping torques to avoid
  double value = F_impedance(5);
  F_impedance(5) = std::max(-2.0, std::min(2.0, F_impedance(5))); //clamping torques to avoid

  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7), tau_impedance(7);
  //pseudoInverse(J.transpose(), jacobian_transpose_pinv);

  //tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
  //                  J.transpose() * jacobian_transpose_pinv) *
  //                  (nullspace_stiffness_ * config_control * (q_d_nullspace_ - q_) - //if config_control = true we control the whole robot configuration
  //                  (2.0 * sqrt(nullspace_stiffness_)) * dq_);  // if config control ) false we don't care about the joint position

  // tau_impedance = J.transpose() * Sm * (F_impedance /*+ F_repulsion + F_potential*/) + J.transpose() * Sf * F_cmd;
  tau_impedance = J.transpose() * F_impedance;
  //auto tau_d_placeholder = tau_impedance 
  tau_d << tau_impedance + coriolis; /*+ tau_nullspace + coriolis */; //add nullspace and coriolis components to desired torque
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
  update_stiffness_and_references();
  return controller_interface::return_type::OK;
}
}

// namespace cartesian_impedance_control
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cartesian_impedance_control::CartesianImpedanceController,
                       controller_interface::ControllerInterface)