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
#include <numeric>

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

void CartesianImpedanceController::update_stiffness_and_references(){
  //target by filtering
  /** at the moment we do not use dynamic reconfigure and control the robot via D, K and T **/
  //K = filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * K;
  //D = filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * D;
  nullspace_stiffness_ = filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  //std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  
  
  if (control_act){
    
    // orientation_d_target_ = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitX())
    //                     * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY())
    //                     * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ());
    
    // save the current orientation as reference when drilling controller is activated
    if (orientation_set == false)
    {
      orientation_d_target_ = orientation;
      position_d_target_ = position;
      orientation_set = true;
    }

    mode_ = false;

  }

  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
  F_contact_des = 0.05 * F_contact_target + 0.95 * F_contact_des;
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


controller_interface::InterfaceConfiguration
CartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}


controller_interface::InterfaceConfiguration CartesianImpedanceController::state_interface_configuration()
  const {
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    state_interfaces_config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/position");
    state_interfaces_config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/velocity");
  }

  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    state_interfaces_config.names.push_back(franka_robot_model_name);
    std::cout << franka_robot_model_name << std::endl;
  }

  const std::string full_interface_name = robot_name_ + "/" + state_interface_name_;

  return state_interfaces_config;
}


CallbackReturn CartesianImpedanceController::on_init() {
   UserInputServer input_server_obj(&position_d_target_, &rotation_d_target_, &K, &D, &T, &mode_,&control_act, &drill_act);
   std::thread input_thread(&UserInputServer::main, input_server_obj, 0, nullptr);
   input_thread.detach();
   return CallbackReturn::SUCCESS;
}


CallbackReturn CartesianImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
  franka_semantic_components::FrankaRobotModel(robot_name_ + "/" + k_robot_model_interface_name,
                                               robot_name_ + "/" + k_robot_state_interface_name));
                                               
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

  // Initialize publisher here
  jacobian_ee_publisher_ = get_node()->create_publisher<messages_fr3::msg::JacobianEE>("/jacobian_ee", 10);
  dt_Fext_desired_publisher_ = get_node()->create_publisher<std_msgs::msg::Float64>("/dt_fext_z", 10);
  D_z_publisher_ = get_node()->create_publisher<std_msgs::msg::Float64>("/D_z", 10);
  VelocityErrorPublisher_ = get_node()->create_publisher<std_msgs::msg::Float64>("/velocity_error", 10);
  joint_config_srv_ = get_node()->create_service<messages_fr3::srv::JointConfig>("/joint_config", std::bind(&CartesianImpedanceController::setJointConfig, this, std::placeholders::_1, std::placeholders::_2));

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
  arrayToMatrix(O_F_ext_hat_K, O_F_ext_hat_K_M);
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

void CartesianImpedanceController::publishJacobianEE(const std::array<double, 42>& jacobian_EE, const std::array<double, 42>& jacobian_EE_derivative) {
    messages_fr3::msg::JacobianEE jacobian_ee_msg;
  
    // Assign the arrays directly to the message fields
    jacobian_ee_msg.jacobianee = jacobian_EE;
    jacobian_ee_msg.dtjacobianee = jacobian_EE_derivative;
  
    // Publish the combined message
    jacobian_ee_publisher_->publish(jacobian_ee_msg);
}

void CartesianImpedanceController::calculate_accel_pose(double delta_time, double z_position) {
    // Calculate the velocity
    z_velocity = (z_position - previous_z_position_) / delta_time;

    // Low-pass filter the velocity
    z_velocity = 0.1 * z_velocity + 0.9 * previous_z_velocity_;

    // Calculate acceleration before updating `previous_z_velocity_`
    z_acceleration = (z_velocity - previous_z_velocity_) / delta_time;

    // Low-pass filter the acceleration
    z_acceleration = 0.1 * z_acceleration + 0.9 * previous_z_acceleration_;

    // Update previous velocity and acceleration for the next iteration
    previous_z_velocity_ = z_velocity;
    previous_z_acceleration_ = z_acceleration;
}

void CartesianImpedanceController::calculate_dt_f_ext(double delta_time, double F_ext_desired) {
    // Calculate the dt_F_ext_desired
    dt_F_ext_desired = (F_ext_desired - previous_F_ext_desired) / delta_time;

    // Low-pass filter dt_F_ext_desired
    dt_F_ext_desired = 0.1 * dt_F_ext_desired + 0.9 * previous_dt_F_ext_desired;

    // Update previous dt_F_ext_desired for the next iteration
    previous_dt_F_ext_desired = dt_F_ext_desired;

    // Publish the jointEEState message
    std_msgs::msg::Float64 dt_F_ext_desired_msg;
    dt_F_ext_desired_msg.data = dt_F_ext_desired;
    dt_Fext_desired_publisher_->publish(dt_F_ext_desired_msg);
}

void CartesionImpedanceController::update_joint_config(Eigen::Quaterniond orientation_d, Eigen::Vector3d position_d, Eigen::Vector3d direction_d){

    // convert orientation to euler angles
    Eigen::Vector3d euler_angles = orientation_d.toRotationMatrix().eulerAngles(0,1,2);

    // send the desired orientation, position and direction to the joint_config service
    messages_fr3::srv::JointConfig::Request::SharedPtr request = std::make_shared<messages_fr3::srv::JointConfig::Request>();
    request->x = position_d_[0];
    request->y = position_d_[1];
    request->z = position_d_[2];
    request->roll = euler_angles[0]; 
    request->pitch = euler_angles[1]; 
    request->yaw = euler_angles[2]; 
    request->directionx = direction_current[0];
    request->directiony = direction_current[1];
    request->directionz = direction_current[2];

    // Send the service request
    if (joint_config_srv_->wait_for_service(std::chrono::seconds(5))) {
        auto future = joint_config_srv_->async_send_request(request);

        if (rclcpp::spin_until_future_complete(get_node(), future) ==
            rclcpp::FutureReturnCode::SUCCESS) {
            auto response = future.get();
            
            // Access the joint configuration from the response
            joint_config << response->joint1, response->joint2, response->joint3, 
                            response->joint4, response->joint5, response->joint6, 
                            response->joint7;

            // Use the joint configuration as needed
            RCLCPP_INFO(get_node()->get_logger(), "Received Joint Config: [%f, %f, %f, %f, %f, %f, %f]",
                        joint_config[0], joint_config[1], joint_config[2], joint_config[3], 
                        joint_config[4], joint_config[5], joint_config[6]);

        } else {
            RCLCPP_ERROR(get_node()->get_logger(), "Failed to call joint_config service.");
        }
    } else {
        RCLCPP_ERROR(get_node()->get_logger(), "Service joint_config not available.");
    }

}

controller_interface::return_type CartesianImpedanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {  

  std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  std::array<double, 42> jacobian_array =  franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 42> jacobian_array_EE =  franka_robot_model_->getBodyJacobian(franka::Frame::kEndEffector);
  std::array<double, 16> pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 7>> M(mass.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(pose.data()));
  position = transform.translation();
  orientation = transform.rotation();
  
  double z_position = position.z();
  double previous_z_position = z_position;

  // Calculate the Jacobian derivative using finite differences
  Eigen::Matrix<double, 6, 7> jacobian_EE_derivative;
  
  updateJointStates();

  calculate_accel_pose(dt, z_position);
  
  // Update previous z position for the next iteration
  previous_z_position_ = z_position;

  // in free float mode we do not control the robot but to not have a jump in orientation when reactivated we set the desired orientation to the current one
  if (mode_){
    orientation_d_target_ = orientation;
  } 
  else if (!control_act){
    orientation_d_target_ = Eigen::AngleAxisd(rotation_d_target_[0], Eigen::Vector3d::UnitX())
                        * Eigen::AngleAxisd(rotation_d_target_[1], Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(rotation_d_target_[2], Eigen::Vector3d::UnitZ());
  }

  if (control_act){

    if(drill_position_set == false){

      // save the current orientation and pose as reference when drilling controller is activated
      orientation_d_ = orientation;
      position_d_ = position;
      drill_position_set = true;
    }
    
    if (projection_matrix_decrease_set == false){
      
      K_original = K;

      // determine relative rotation
      relative_rotation = orientation*rotation_ref.inverse();

      // change to rotation matrix
      Eigen::Matrix3d relative_rotation_matrix = relative_rotation.toRotationMatrix();

      // determine the direction of the relative rotation
      direction_current = relative_rotation_matrix * direction_ref;

      direction_current.normalize();

      // update the joint configuration to be optimal for the drilling task and before stiffness is adapted
      if (joint_optimization_set == false){
        update_joint_config(orientation_d_, position_d_, direction_current);
        joint_optimization_set = true;
      }

      projection_matrix_decrease.topLeftCorner(3,3) = Eigen::Matrix3d::Identity() - direction_current * direction_current.transpose();
      projection_matrix_increase.topLeftCorner(3,3) = Eigen::Matrix3d::Identity() + K_increase_gain*(direction_current * direction_current.transpose());
      projection_matrix_decrease.bottomRightCorner(3,3) = Eigen::Matrix3d::Identity();
      projection_matrix_increase.bottomRightCorner(3,3) = Eigen::Matrix3d::Identity();

      K.topLeftCorner(3,3) = projection_matrix_decrease.topLeftCorner(3,3) * K_original.topLeftCorner(3,3);
      
      projection_matrix_decrease_set = true;

    }

  }

  double F_ext_desired = O_F_ext_hat_K_M.head(3).dot(direction_current);
  double previous_F_ext_desired = F_ext_desired;

  calculate_dt_f_ext(dt,F_ext_desired);

  // save the position when the drill is activated
  if (drill_act && drill_start_posistion_set == false){
    drill_start_position = position;
    drill_start_posistion_set = true;
  }

  // save all the velocities from start point of drilling into an array
  if (drill_start_posistion_set && target_drill_velocity_set == false){
    drill_velocities_.push_back(z_velocity);
    drill_forces_.push_back(F_ext_desired);

    // once we have drilled 1cm take the average of the drill velocities and set this to the target velocity
    if (drill_start_position.z() - position.z() > 0.005){
      sum_drill_velocity_ = std::accumulate(drill_velocities_.begin(), drill_velocities_.end(), 0.0);
      sum_drill_force_ = std::accumulate(drill_forces_.begin(), drill_forces_.end(), 0.0);
      target_drill_force_ = sum_drill_force_ / drill_forces_.size();
      target_dampening = target_drill_force_ / target_drill_velocity_;
      target_drill_velocity_set = true;
    }
  }
  
  velocity_error = target_drill_velocity_ - z_velocity;

  // publish velicity error
  std_msgs::msg::Float64 velocity_error_msg;
  velocity_error_msg.data = velocity_error;
  VelocityErrorPublisher_->publish(velocity_error_msg);



  if (dt_F_ext_desired < -5500 && control_act && accel_trigger == false) {
    // Start the ramping process if the condition is met
    ramping_active_ = true;       
    position_set_ = true;
    position_accel_lim = position - 0.01 * direction_current;   // ADAJUST TO THE POSITION IN DRILLING DIRECTION
    accel_trigger = true;
  }

  if (ramping_active_) {
        time_constant = 0.01; // Adjust this to control the response speed
        alpha = 1.0 - exp(-period.seconds() / time_constant);

        // Calculate the target stiffness value
        target_K = projection_matrix_increase.topLeftCorner(3,3) * K_original.topLeftCorner(3,3);
        
        // Gradually increase K.diagonal()[2] towards the target value
        K.topLeftCorner(3,3) = alpha * target_K + (1.0 - alpha) * K.topLeftCorner(3,3);

        // Stop ramping once we reach the target value
        if (K.topLeftCorner(3, 3).isApprox(target_K, 1e-6)) {
            K.topLeftCorner(3,3) = target_K;
            elapsed_time += period.seconds();
        }

        if (elapsed_time > 3.0) {
          // Reset the ramping process after 5 seconds
          ramping_active_ = false;
          accel_trigger = false;
          elapsed_time = 0.0;
          K.topLeftCorner(3,3) = projection_matrix_decrease.topLeftCorner(3,3) * K_original.topLeftCorner(3,3);
          position_set_ = false;
          position_d_ = position;
        }
  } 

  if (position_set_){
      position_d_ = position_accel_lim; // setting breakthrough position
      D_gain = 2.05;
  }

  error.head(3) << position - position_d_;

  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);

  Lambda = (jacobian * M.inverse() * jacobian.transpose()).inverse();    
    // correcting D to be critically damped
  D =  D_gain* K.cwiseMax(0.0).cwiseSqrt() * Lambda.diagonal().cwiseSqrt().asDiagonal();
  
  D.topRightCorner(3,3).setZero();
  D.bottomLeftCorner(3,3).setZero();

  // when not in drilling mode we use the damoing term to be critically damped and dependant on K
  // TODO: Potentially add accel_trigger bool so if triggered we go back to the dynaic damping term which is dependant on K
  if (drill_act){ 
    D.diagonal()[2] = D_drilling_target; // this is in drilling mode
    
    // increase D to maintain target velocity
    if (target_drill_velocity_set && abs(dt_F_ext_desired) > 5000){
      brake_through = true;
    }

    if (brake_through){
      // Calculate the integral of the velocity error
      velocity_error_sum += velocity_error * dt;

      target_D_z = D_drilling_target + Kp_drilling * velocity_error + Ki_drilling * velocity_error_sum - Kd_drilling * z_acceleration;

      target_D_z = std::max(target_D_z, 0.0);

      // Define the time constant for exponential response
      time_constant_D = 0.01; // Adjust this for ramping speed
      alpha_D = 1.0 - exp(-dt / time_constant_D); // Calculate smoothing factor

      // Smoothly adjust D.diagonal()[2] toward the changing target_D_z
      D.diagonal()[2] = (alpha_D * abs(target_dampening) + (1.0 - alpha_D) * D.diagonal()[2]) * sqrt(Lambda.diagonal()[2]);
      D.diagonal()[2] = 600;
      // Clamp the value of D.diagonal()[2] to min_D and max_D
      //D.diagonal()[2] = std::clamp(D.diagonal()[2], min_D, max_D);
    }
  }

  // publish D_z
  std_msgs::msg::Float64 D_z_msg;
  D_z_msg.data = D.diagonal()[2];
  D_z_publisher_->publish(D_z_msg);

  F_impedance = -1 * (D * (jacobian * dq_) + K * error );     

  F_ext = 0.9 * F_ext + 0.1 * O_F_ext_hat_K_M; //Filtering 
  I_F_error += dt * Sf* (F_contact_des - F_ext);
  F_cmd = Sf*(0.4 * (F_contact_des - F_ext) + 0.9 * I_F_error + 0.9 * F_contact_des);

  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7), tau_impedance(7);
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                    (nullspace_stiffness_ * config_control * (q_d_nullspace_ - q_) - //if config_control = true we control the whole robot configuration
                    (2.0 * sqrt(nullspace_stiffness_)) * dq_);  // if config control ) false we don't care about the joint position

  tau_impedance = jacobian.transpose() * Sm * (F_impedance /*+ F_repulsion + F_potential*/) + jacobian.transpose() * Sf * F_cmd;
  Eigen::VectorXd tau_d_placeholder = tau_impedance + tau_nullspace + coriolis; //add nullspace and coriolis components to desired torque
  
  // free floating mode
  if (mode_) {
    tau_d_placeholder.setZero();
  }

  tau_d << tau_d_placeholder;
  tau_d << saturateTorqueRate(tau_d, tau_J_d_M);  // Saturate torque rate to avoid discontinuities
  tau_J_d_M = tau_d;

  for (size_t i = 0; i < 7; ++i) {
    command_interfaces_[i].set_value(tau_d(i));
  }
  
  if (outcounter % 1000/update_frequency == 0){
    /* std::cout << "F_ext_robot [N]" << std::endl;*/
    std::cout << O_F_ext_hat_K << std::endl;
    /* std::cout << O_F_ext_hat_K_M << std::endl; */
    /*std::cout << "Lambda  Thetha.inv(): " << std::endl;
    std::cout << Lambda*Theta.inverse() << std::endl;
    std::cout << "tau_d" << std::endl;
    std::cout << tau_d << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << tau_nullspace << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << tau_impedance << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << coriolis << std::endl;*/
    /* std::cout << "Inertia scaling:\n " << std::endl;
    std::cout << T << std::endl;
    std::cout << "Lambda:\n" << Lambda << std::endl; */
    /* std::cout << "Control mode: " << mode_ << std::endl; */
    /* std::cout << "Position target :" << position_d_target_ << std::endl; */
    // print the phrobenius norm of the K
    //std::cout << "K_norm: " << K.norm() << std::endl;
    // std::cout << "Drilling actived: " << drill_act << std::endl;
    std::cout << "Stiffness:\n " << K << std::endl;
    //std::cout << "Damping:\n " << D << std::endl;
    //std::cout << " F_impedance: " << F_impedance << std::endl;
    // std::cout << "Damping_z:\n " << D.diagonal()[2] << std::endl;
    // std::cout << "z_acceleration:\n " << z_acceleration << std::endl;
    // std::cout << "target_drill_velocity_:\n " << target_drill_velocity_ << std::endl;
    // std::cout << "targer_drill_force_:\n " << target_drill_force_ << std::endl;
    // std::cout << "target_dampening:\n " << target_dampening << std::endl;
    // std::cout << "projection_matrix_decrease:\n " << projection_matrix_decrease << std::endl;
    //std::cout << "k_top_left:\n " << K.topLeftCorner<3, 3>() << std::endl;
    //std::cout << "projection_top_left:\n " << projection_top_left << std::endl;
    std::cout << "direction_current:\n " << direction_current << std::endl;
    /* std::cout << "elapsed_time:\n " << elapsed_time << std::endl; */
    std::cout << "dt_fext_desired:\n " << dt_F_ext_desired << std::endl;
    std::cout << "magnitude of direction_current:\n " << direction_current.norm() << std::endl;
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