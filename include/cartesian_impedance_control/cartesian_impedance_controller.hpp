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
#include <cstdint>
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

#include "cartesian_impedance_control/target_validation.hpp"

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
#include "messages_fr3/srv/set_param.hpp"
#include "std_msgs/msg/int16.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <realtime_tools/realtime_publisher.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <geometry_msgs/msg/pose.hpp>

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

    // /set_pose service (position only, see the .cpp) and /set_param (not supported, logs and returns false);
    // on the controller node since 2026-10-19 (replaces the detached UserInputServer thread, fix-list a13)
    void setPose(const std::shared_ptr<messages_fr3::srv::SetPose::Request> request, 
    std::shared_ptr<messages_fr3::srv::SetPose::Response> response);
    void setParam(const std::shared_ptr<messages_fr3::srv::SetParam::Request> request,
    std::shared_ptr<messages_fr3::srv::SetParam::Response> response);

//subscriptions
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr desired_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr control_mode_sub_;
      

 private:
    
    int control_mode = 0;
    // friction compensation
    Eigen::Matrix<double, 6,6> K_friction = Eigen::MatrixXd::Identity(6, 6); //impedance stiffness term for friction compensation
    Eigen::Matrix<double, 6,6> D_friction = Eigen::MatrixXd::Identity(6, 6); //impedance damping term for friction compensation
    Eigen::Matrix<double, 7, 1> tau_impedance = Eigen::MatrixXd::Zero(7,1); //torque for every joint from Jacobi * F_cmd
    Eigen::Matrix<double, 7, 1> tau_impedance_filtered = Eigen::MatrixXd::Zero(7,1); //filtered impedance torque for friction compensation
    Eigen::Matrix<double, 7, 1> tau_friction = Eigen::MatrixXd::Zero(7,1); //torque compensating friction
    Eigen::Matrix<double, 7, 1> tau_friction_impedance = Eigen::MatrixXd::Zero(7,1); //impedance torque needed for tau_frictio  
    Eigen::Matrix<double, 7, 1> dq_s = (Eigen::VectorXd(7) << 0, 0, 0, 0.0001, 0, 0, 0.05).finished();
    Eigen::Matrix<double, 7, 1> static_friction = (Eigen::VectorXd(7) << 1.025412896, 1.259913793, 0.8380147058, 1.005214968, 1.2928, 0.41525, 0.5341655).finished();
    Eigen::Matrix<double, 7, 1> offset_friction = (Eigen::VectorXd(7) << -0.05, -0.70, -0.07, -0.13, -0.1025, 0.103, -0.02).finished();
    Eigen::Matrix<double, 7, 1> coulomb_friction = (Eigen::VectorXd(7) << 1.025412896, 1.259913793, 0.8380147058, 0.96, 1.2928, 0.41525, 0.5341655).finished();
    Eigen::Matrix<double, 7, 1> beta = (Eigen::VectorXd(7) << 1.18, 0, 0.55, 0.87, 0.935, 0.54, 0.45).finished();//component b of linear friction model (a + b*dq)
    Eigen::Matrix<double, 7, 1> dq_filtered = Eigen::MatrixXd::Zero(7,1); //rotational speed filtered for friction compensation
    Eigen::Matrix<double, 7, 1> dq_d = Eigen::MatrixXd::Zero(7,1); //desired rotational speed
    Eigen::Matrix<double, 7, 1> dq_imp = Eigen::MatrixXd::Zero(7,1); //"impedance dq", dq without the nullspace-part of it

    //state tuner stuf  
    Eigen::Matrix<double, 7, 1> dz = Eigen::MatrixXd::Zero(7,1);
    Eigen::Matrix<double, 7, 1> z = Eigen::MatrixXd::Zero(7,1);
    Eigen::Matrix<double, 7, 1> f = Eigen::MatrixXd::Zero(7,1);
    Eigen::Matrix<double, 7, 1> g = Eigen::MatrixXd::Zero(7,1);
    const Eigen::Matrix<double, 7, 1> sigma_0 = (Eigen::VectorXd(7) << 76.95, 37.94, 71.07, 44.02, 21.32, 21.83, 53).finished();
    const Eigen::Matrix<double, 7, 1> sigma_1 = (Eigen::VectorXd(7) << 0.056, 0.06, 0.064, 0.073, 0.1, 0.0755, 0.000678).finished();

    //Nodes
    rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr franka_state_subscriber = nullptr;
    rclcpp::Service<messages_fr3::srv::SetPose>::SharedPtr pose_srv_;
    rclcpp::Service<messages_fr3::srv::SetParam>::SharedPtr param_srv_;


    //Functions
    void topic_callback(const std::shared_ptr<franka_msgs::msg::FrankaRobotState> msg);
    void updateJointStates();
    void calculate_tau_friction();
    void update_stiffness_and_references();
    void reference_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg);
    void control_mode_callback(const std_msgs::msg::Int16::SharedPtr msg);
    void arrayToMatrix(const std::array<double, 6>& inputArray, Eigen::Matrix<double, 6, 1>& resultMatrix);
    void arrayToMatrix(const std::array<double, 7>& inputArray, Eigen::Matrix<double, 7, 1>& resultMatrix);
    Eigen::Matrix<double, 7, 1> saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d);  
    std::array<double, 6> convertToStdArray(const geometry_msgs::msg::WrenchStamped& wrench);
    //State vectors and matrices
    std::array<double, 7> q_subscribed;
    std::array<double, 7> tau_J_d = {0,0,0,0,0,0,0};
    std::array<double, 6> O_F_ext_hat_K = {0,0,0,0,0,0};
    Eigen::Matrix<double, 7, 1> q_subscribed_M;
    Eigen::Matrix<double, 7, 1> tau_J_d_M = Eigen::MatrixXd::Zero(7, 1);
    Eigen::Matrix<double, 6, 1> F_ext = Eigen::MatrixXd::Zero(6,1);
    Eigen::Matrix<double, 7, 1> q_;
    Eigen::Matrix<double, 7, 1> dq_;
    Eigen::MatrixXd jacobian_transpose_pinv;  
    Eigen::Matrix<double, 7, 7> N = Eigen::MatrixXd::Zero(7, 7); //Nullspace matrix of the end-effector
    Eigen::Matrix<double, 6, 1> I_error = Eigen::VectorXd::Zero(6, 1); //Integrator
    Eigen::Matrix<double, 6, 7> J; //Jacobian of the end-effector
    Eigen::MatrixXd J_pinv; //pseudoinverse jacobian matrix
    //Robot parameters
    const int num_joints = 7;
    const std::string state_interface_name_{"robot_state"};
    std::string arm_id_;
    const std::string k_robot_state_interface_name{"robot_state"};
    const std::string k_robot_model_interface_name{"robot_model"};
    std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
    double delta_tau_max_{1.0};  // torque rate limit per cycle [Nm], parameter delta_tau_max
    const double dt = 0.001;
                
    //Impedance control variables              
    Eigen::Matrix<double, 6, 6> Lambda = IDENTITY;                                           // operational space mass matrix
    Eigen::Matrix<double, 6, 1> integrator_weights = (Eigen::VectorXd(6) << 350, 350, 0, 50, 50, 50).finished(); 
    Eigen::Matrix<double, 6, 6> Sm = IDENTITY;                                               // task space selection matrix for positions and rotation
    Eigen::Matrix<double, 6, 6> Sf = Eigen::MatrixXd::Zero(6, 6);                            // task space selection matrix for forces
   
    Eigen::Matrix<double, 6, 6> K =  (Eigen::MatrixXd(6,6) << 1000,   0,   0,   0,   0,   0,
                                                                0, 1000,   0,   0,   0,   0,
                                                                0,   0, 1000,   0,   0,   0,  // impedance stiffness term
                                                                0,   0,   0, 130,   0,   0,
                                                                0,   0,   0,   0, 130,   0,
                                                                0,   0,   0,   0,   0,  45).finished();

    Eigen::Matrix<double, 6, 6> D =  (Eigen::MatrixXd(6,6) <<  35,   0,   0,   0,   0,   0,
                                                                0,  35,   0,   0,   0,   0,
                                                                0,   0,  35,   0,   0,   0,  // impedance damping term
                                                                0,   0,   0,   25,   0,  0,
                                                                0,   0,   0,   0,   25,  0,
                                                                0,   0,   0,   0,   0,   6).finished();

                                                                

    Eigen::Matrix<double, 6, 6> Theta = IDENTITY;
    Eigen::Matrix<double, 6, 6> T = (Eigen::MatrixXd(6,6) <<       1,   0,   0,   0,   0,   0,
                                                                   0,   1,   0,   0,   0,   0,
                                                                   0,   0,   2.5,   0,   0,   0,  // Inertia term
                                                                   0,   0,   0,   1,   0,   0,
                                                                   0,   0,   0,   0,   1,   0,
                                                                   0,   0,   0,   0,   0,   2.5).finished();                                               // impedance inertia term
     /*   

    Eigen::Matrix<double, 6, 6> K =  (Eigen::MatrixXd(6,6) << 400,   0,   0,   0,   0,   0,
                                                                0, 400,   0,   0,   0,   0,
                                                                0,   0, 400,   0,   0,   0,  // impedance stiffness term
                                                                0,   0,   0, 75,   0,   0,
                                                                0,   0,   0,   0, 75,   0,
                                                                0,   0,   0,   0,   0,  20).finished();

    Eigen::Matrix<double, 6, 6> D =  (Eigen::MatrixXd(6,6) <<  28,   0,   0,   0,   0,   0,
                                                                0,  28,   0,   0,   0,   0,
                                                                0,   0,  28,   0,   0,   0,  // impedance damping term
                                                                0,   0,   0,   8,   0,   0,
                                                                0,   0,   0,   0,   8,   0,
                                                                0,   0,   0,   0,   0,   7).finished();

     */

    Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;                                 // impedance damping term
    Eigen::Matrix<double, 6, 6> cartesian_damping_target_;                                   // impedance damping term
    Eigen::Matrix<double, 6, 6> cartesian_inertia_target_;                                   // impedance damping term
    Eigen::Vector3d position_d_target_ = {0.5, 0.0, 0.5};
    Eigen::Vector3d rotation_d_target_ = {M_PI, 0.0, 0.0};
    Eigen::Quaterniond orientation_d_target_;
    Eigen::Vector3d position_d_;
    Eigen::Quaterniond orientation_d_; 
    Eigen::Matrix<double, 6, 1> F_impedance;  
    Eigen::Matrix<double, 6, 1> F_contact_des = Eigen::MatrixXd::Zero(6, 1);                 // desired contact force
    Eigen::Matrix<double, 6, 1> F_contact_target = Eigen::MatrixXd::Zero(6, 1);              // desired contact force used for filtering
    Eigen::Matrix<double, 6, 1> F_cmd = Eigen::MatrixXd::Zero(6, 1);                         // commanded contact force
    Eigen::Matrix<double, 7, 1> q_d_nullspace_;
    Eigen::Matrix<double, 6, 1> error;                                                       // pose error (6d)
    double nullspace_stiffness_{0.0};
    double nullspace_stiffness_target_{0.0};
    double D_gain = 3.0;

    //Logging
    int outcounter = 0;
    const int update_frequency = 2; //frequency for update outputs
  
    // Handoff of external inputs to update() (fix-list a13, 2026-10-19). Callbacks run on the controller manager's
    // executor, not in update(): they validate, merge into command_nonrt_ under command_mutex_ (non-RT only) and
    // publish the complete struct through a RealtimeBuffer; update() reads it once per cycle (try_lock, never
    // blocks) and applies a pose / mode only when its sequence number is new. Only update() writes the targets,
    // K and I_error.
    struct TargetCommand {
      uint64_t pose_seq{0};                       // incremented for every accepted pose
      Eigen::Vector3d position = Eigen::Vector3d::Zero();
      bool has_orientation{false};                // false: keep the current orientation target (set_pose)
      Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
      bool reset_integrator{false};               // reference_pose resets I_error (as before), set_pose does not
      uint64_t mode_seq{0};                       // incremented for every control_mode message
      int mode{0};
    };
    bool submit_pose_target(const Eigen::Vector3d& position, const Eigen::Quaterniond* orientation,
                            bool reset_integrator, const char* source);
    void apply_external_commands();
    std::mutex command_mutex_;                    // non-RT writers only
    TargetCommand command_nonrt_;
    realtime_tools::RealtimeBuffer<TargetCommand> command_buffer_;
    uint64_t applied_pose_seq_{0};                // RT side
    uint64_t applied_mode_seq_{0};
    // filtered external wrench from the robot_state callback; the mode-1 z-contact switch runs in update()
    realtime_tools::RealtimeBuffer<Eigen::Matrix<double, 6, 1>> f_ext_buffer_;
    // current EE position for the distance check: written by update() with try_lock, read by the callbacks
    std::mutex current_position_mutex_;
    Eigen::Vector3d current_position_ = Eigen::Vector3d::Zero();
    bool current_position_valid_{false};          // false while inactive -> all targets rejected
    TargetLimits target_limits_;                  // parameters target.*, read in on_configure

    // State interface indices resolved by name in on_activate (fix-list a12)
    std::array<size_t, 7> joint_position_index_{};
    std::array<size_t, 7> joint_velocity_index_{};

    // Tuning parameters, declared in on_init and read in on_configure (launch/controllers.yaml,
    // docs/controller-tuning.md). Defaults = the values hardcoded before 2026-10-13.
    void declare_tuning_parameters();
    bool read_tuning_parameters();
    double damping_ratio_translational_{1.0};       // D_ii = 2 * ratio * sqrt(K_ii * Lambda_ii)
    double damping_ratio_rotational_{1.3};
    double damping_rot_z_min_factor_{2.0};          // D(5,5) >= factor * sqrt(K(5,5))
    double integrator_limit_force_{2.5};            // anti-windup [N]
    double integrator_limit_torque_{1.5};           // anti-windup [Nm]
    double integrator_limit_torque_z_{0.5};         // anti-windup [Nm]
    bool integrator_reset_on_new_target_{false};    // reset I_error when position_d_target_ changes (set_pose)
    Eigen::Vector3d last_position_d_target_ = Eigen::Vector3d::Zero();
    double max_force_{18.0};                        // clamp on F_impedance [N]
    double max_torque_{9.0};                        // clamp on F_impedance [Nm]
    double max_torque_z_{2.0};                      // clamp on F_impedance(5) [Nm]
    bool clamp_spring_only_{false};                 // clamp only K*e + I, damping term clamped separately
    double max_damping_force_{36.0};                // clamp on D*v if clamp_spring_only [N]
    double max_damping_torque_{18.0};               // clamp on D*v if clamp_spring_only [Nm]
    double max_reference_velocity_translational_{0.0};  // [m/s], 0 = no limit
    double max_reference_velocity_rotational_{0.0};     // [rad/s], 0 = no limit
    double lambda_regularization_{0.0};             // Lambda = (J M^-1 J^T + reg * I)^-1
    double nullspace_damping_{0.0};                 // [Nms/rad], 0 = off
    double nullspace_max_torque_{5.0};              // clamp on tau_nullspace per joint [Nm]
    bool nullspace_target_from_activation_{true};   // true: posture target = q at activation, else nullspace.q_target
    Vector7d nullspace_q_target_param_ = Vector7d::Zero();

    // Smooth move to a start pose at activation (minimum-jerk), parameters start_pose.*
    void step_start_move();
    bool start_pose_enabled_{false};
    Eigen::Vector3d start_position_ = Eigen::Vector3d::Zero();
    bool start_pose_use_orientation_{false};
    Eigen::Quaterniond start_orientation_ = Eigen::Quaterniond::Identity();
    double start_pose_max_velocity_{0.1};           // peak speed of the move [m/s]
    double start_pose_max_angular_velocity_{0.3};   // [rad/s]
    bool start_move_active_{false};
    double start_move_time_{0.0};
    double start_move_duration_{0.0};
    Eigen::Vector3d start_move_p0_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d start_move_p1_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond start_move_o0_ = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond start_move_o1_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d start_move_last_target_ = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 6, 1> stiffness_mode_normal_;   // control_mode 0
    Eigen::Matrix<double, 6, 1> stiffness_mode_float_;    // control_mode 2
    Eigen::Matrix<double, 6, 1> stiffness_mode_lift_;     // control_mode 3
    Eigen::Matrix<double, 6, 1> stiffness_mode_contact_;  // z-contact switch in control_mode 1

    // Debug topic ~/debug (std_msgs/Float64MultiArray, field names in layout.dim[0].label; docs/controller-tuning.md)
    void setup_debug_publisher();
    double debug_publish_rate_{100.0};              // [Hz], 0 = off
    int debug_decimation_{10};                      // publish every n-th 1 kHz cycle
    int debug_counter_{0};
    std::vector<std::string> debug_field_names_;
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>> debug_publisher_raw_;
    std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> debug_publisher_;

    //Flags
    bool config_control = false;           // sets if we want to control the configuration of the robot in nullspace
    bool do_logging = false;               // set if we do log values

    //Filter-parameters
    double filter_params_{0.002};
    int mode_ = 1;
};
}  // namespace cartesian_impedance_control