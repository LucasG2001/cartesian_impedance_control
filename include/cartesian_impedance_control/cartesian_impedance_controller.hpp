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

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <thread>
#include <chrono>         
#include <vector>

#include "cartesian_impedance_control/user_input_server.hpp"

#include <rclcpp/rclcpp.hpp>
#include "rclcpp/subscription.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigen>

#include <controller_interface/controller_interface.hpp>

#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include "franka_hardware/franka_hardware_interface.hpp"
#include <franka_hardware/model.hpp>

#include "franka_msgs/msg/franka_robot_state.hpp"
#include "franka_msgs/msg/errors.hpp"
#include "messages_fr3/srv/set_pose.hpp"
#include "messages_fr3/msg/jacobian_ee.hpp"
#include "messages_fr3/msg/pose_direction.hpp"
#include "messages_fr3/msg/joint_config.hpp"
#include "std_msgs/msg/float64.hpp"

#include "franka_semantic_components/franka_robot_model.hpp"
#include "franka_semantic_components/franka_robot_state.hpp"

#define IDENTITY Eigen::MatrixXd::Identity(6, 6)

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using Vector7d = Eigen::Matrix<double, 7, 1>;

namespace cartesian_impedance_control {

class CartesianImpedanceController : public controller_interface::ControllerInterface {
public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;

  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;

    void setPose(const std::shared_ptr<messages_fr3::srv::SetPose::Request> request, 
    std::shared_ptr<messages_fr3::srv::SetPose::Response> response);
      

 private:
    //Nodes
    rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr franka_state_subscriber = nullptr;
    rclcpp::Service<messages_fr3::srv::SetPose>::SharedPtr pose_srv_;
    rclcpp::Publisher<messages_fr3::msg::JacobianEE>::SharedPtr jacobian_ee_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr dt_Fext_desired_publisher_; 
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr D_z_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr VelocityErrorPublisher_;
    rclcpp::Publisher<messages_fr3::msg::PoseDirection>::SharedPtr pose_direction_publisher_;
    rclcpp::Subscription<messages_fr3::msg::JointConfig>::SharedPtr joint_config_subscriber_;


    //Functions
    void topic_callback(const std::shared_ptr<franka_msgs::msg::FrankaRobotState> msg);
    void updateJointStates();
    void update_stiffness_and_references();
    void publishJacobianEE(const std::array<double, 42>& jacobian_EE, const std::array<double, 42>& jacobian_EE_derivative);
    void arrayToMatrix(const std::array<double, 6>& inputArray, Eigen::Matrix<double, 6, 1>& resultMatrix);
    void arrayToMatrix(const std::array<double, 7>& inputArray, Eigen::Matrix<double, 7, 1>& resultMatrix);
    void calculate_accel_pose(double delta_time, double z_position);
    void calculate_dt_f_ext(double delta_time, double F_ext_desired);
    void update_joint_config(Eigen::Quaterniond orientation_d, Eigen::Vector3d position_d, Eigen::Vector3d direction_d);
    void jointConfigCallback(const messages_fr3::msg::JointConfig::SharedPtr msg);
    Eigen::Matrix<double, 7, 1> saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d);  
    std::array<double, 6> convertToStdArray(const geometry_msgs::msg::WrenchStamped& wrench);
    
    //State vectors and matrices
    std::array<double, 7> q_subscribed;
    std::array<double, 7> tau_J_d = {0,0,0,0,0,0,0};
    std::array<double, 6> O_F_ext_hat_K = {0,0,0,0,0,0};
    Eigen::Matrix<double, 7, 1> q_subscribed_M;
    Eigen::Matrix<double, 7, 1> tau_J_d_M = Eigen::MatrixXd::Zero(7, 1);
    Eigen::Matrix<double, 6, 1> O_F_ext_hat_K_M = Eigen::MatrixXd::Zero(6,1);
    Eigen::Matrix<double, 7, 1> q_;
    Eigen::Matrix<double, 7, 1> dq_;
    Eigen::Matrix<double, 7, 1> dq_prev_;
    Eigen::MatrixXd jacobian_transpose_pinv;  

    //Robot parameters
    const int num_joints = 7;
    const std::string state_interface_name_{"robot_state"};
    //Name here needs to be fr3 and not panda
    const std::string robot_name_{"fr3"};
    const std::string k_robot_state_interface_name{"robot_state"};
    const std::string k_robot_model_interface_name{"robot_model"};
    franka_hardware::FrankaHardwareInterface interfaceClass;
    std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
    const double delta_tau_max_{1.0};
    const double dt = 0.001;
                
    //Impedance control variables              
    Eigen::Matrix<double, 6, 6> Lambda = IDENTITY;                                           // operational space mass matrix
    Eigen::Matrix<double, 6, 6> Sm = IDENTITY;                                               // task space selection matrix for positions and rotation
    Eigen::Matrix<double, 6, 6> Sf = Eigen::MatrixXd::Zero(6, 6);                            // task space selection matrix for forces
    Eigen::Matrix<double, 6, 6> K =  (Eigen::MatrixXd(6,6) << 1000,   0,   0,   0,   0,   0,
                                                                0, 1000,   0,   0,   0,   0,
                                                                0,   0, 1000,   0,   0,   0,  // impedance stiffness term
                                                                0,   0,   0, 100,   0,   0,
                                                                0,   0,   0,   0, 100,   0,
                                                                0,   0,   0,   0,   0,  20).finished();

    Eigen::Matrix<double, 6, 6> D =  (Eigen::MatrixXd(6,6) <<  30,   0,   0,   0,   0,   0,
                                                                0,  30,   0,   0,   0,   0,
                                                                0,   0,  30,   0,   0,   0,  // impedance damping term
                                                                0,   0,   0,   18,   0,   0,
                                                                0,   0,   0,   0,   18,   0,
                                                                0,   0,   0,   0,   0,   9).finished();

    double D_gain = 2.05;
    Eigen::Matrix<double, 6, 6> Theta = IDENTITY;
    Eigen::Matrix<double, 6, 6> T = (Eigen::MatrixXd(6,6) <<       10,   0,   0,   0,   0,   0,
                                                                   0,   10,   0,   0,   0,   0,
                                                                   0,   0,   1,   0,   0,   0,  // Inertia term
                                                                   0,   0,   0,   10,   0,   0,
                                                                   0,   0,   0,   0,   10,   0,
                                                                   0,   0,   0,   0,   0,   1).finished();                                               // impedance inertia term

    Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;                                 // impedance damping term
    Eigen::Matrix<double, 6, 6> cartesian_damping_target_;                                   // impedance damping term
    Eigen::Matrix<double, 6, 6> cartesian_inertia_target_;                                   // impedance damping term
    Eigen::Matrix<double, 6, 6> K_original;
    Eigen::Matrix<double, 3, 3> projection_top_left;
    Eigen::Vector3d position_d_target_ = {0.5, 0.0, 0.5};
    Eigen::Vector3d rotation_d_target_ = {M_PI, 0.0, 0.0};
    Eigen::Vector3d direction_ref = {0.0, -1.0, 0.0};
    Eigen::Quaterniond rotation_ref = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())
                                    * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())
                                    * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond relative_rotation;
    Eigen::Quaterniond orientation_d_target_;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d position;
    Eigen::Matrix<double, 6, 6> projection_matrix_decrease;
    Eigen::Matrix<double, 6, 6> projection_matrix_increase;
    Eigen::Matrix3d target_K;
    Eigen::Vector3d direction_current;
    Eigen::Vector3d position_d_;
    Eigen::Quaterniond orientation_d_; 
    Eigen::Matrix<double, 6, 1> F_impedance;  
    Eigen::Matrix<double, 6, 1> F_contact_des = Eigen::MatrixXd::Zero(6, 1);                 // desired contact force
    Eigen::Matrix<double, 6, 1> F_contact_target = Eigen::MatrixXd::Zero(6, 1);              // desired contact force used for filtering
    Eigen::Matrix<double, 6, 1> F_ext = Eigen::MatrixXd::Zero(6, 1);                         // external forces
    Eigen::Matrix<double, 6, 1> F_cmd = Eigen::MatrixXd::Zero(6, 1);                         // commanded contact force
    Eigen::Matrix<double, 7, 1> q_d_nullspace_;
    Eigen::Matrix<double, 6, 1> error;
    Eigen::Vector3d drill_start_position; 
    Eigen::Matrix<double, 7, 1> joint_config;
    std::vector<double> drill_velocities_;
    std::vector<double> drill_forces_;
    double nullspace_stiffness_{0.001};
    double nullspace_stiffness_target_{0.001};
    // Previous values for z position, velocity, and acceleration
    double previous_z_position_ = 0.0;
    double previous_z_velocity_ = 0.0;
    double previous_z_acceleration_ = 0.0;
    double z_acceleration = 0.0;
    double z_velocity = 0.0;
    double dt_F_ext_desired = 0.0;
    double previous_dt_F_ext_desired = 0.0;
    double previous_F_ext_desired = 0.0;
    double target_drill_velocity_ = -0.008;
    double sum_drill_velocity_ = 0.0;
    double velocity_error = 0.0;
    double D_drilling_target = 0;
    double Kp_drilling = 10000;
    double Ki_drilling = 0.0;
    double Kd_drilling = 2*sqrt(Kp_drilling);
    double min_D = 10.0;
    double max_D = 750.0;
    double target_D_z = 0.0;
    double alpha_D = 0.0;
    double time_constant_D = 0.0;
    double target_K_z = 0.0;
    double target_dampening = 0.0;
    double target_drill_force_ = 0.0;
    double sum_drill_force_ = 0.0;
    double K_increase_gain = 1.0;

    double alpha = 0.0;
    double time_constant = 0.0;
    bool ramping_active_ = false;
    double target_stiffness_z_ = 4000;
    bool position_set_ = false;
    Eigen::Vector3d position_accel_lim;
    bool accel_trigger = false;
    double elapsed_time = 0.0;
    double velocity_error_sum = 0.0;

    //Logging
    int outcounter = 0;
    const int update_frequency = 2; //frequency for update outputs
    Eigen::Matrix<double, 6, 1> I_F_error = Eigen::MatrixXd::Zero(6, 1);                    // force error integral

   
  
    std::mutex position_and_orientation_d_target_mutex_;

    //Flags
    bool config_control = false;           // sets if we want to control the configuration of the robot in nullspace
    bool do_logging = false;               // set if we do log values

    //Filter-parameters
    double filter_params_{0.001};

    // mode selection between impedance control and free floating
    bool mode_ = false; // false = impedance control, true = free floating

    bool control_act = false; // controller activation flag
    bool drill_act = false; // drill activation flag
    bool drill_start_posistion_set = false; // drill start position saved flag
    bool target_drill_velocity_set = false; // target drill velocity set flag
    bool brake_through = false; // brake through flag
    bool orientation_set = false; // orientation set flag
    bool projection_matrix_decrease_set = false; // projection matrix set flag
    bool projection_matrix_increase_set = false; // projection matrix set flag
    bool drill_position_set = false; // drill position set flag
    bool joint_optimization_set = false; // joint optimization set flag


    int accel_mode_ = 0; // acceleration calculation mode flag
};
}  // namespace cartesian_impedance_control