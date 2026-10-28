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
#include <random>
#include <mutex>
#include <string>
#include <unistd.h>
#include <thread>
#include <chrono>         

#include "cartesian_impedance_control/pinocchio_model.hpp"
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
#include "std_msgs/msg/int16.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include "ci_mpc_msgs/msg/mpc_command.hpp"

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

//subscriptions
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr desired_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr control_mode_sub_;
  rclcpp::Subscription<ci_mpc_msgs::msg::MpcCommand>::SharedPtr mpc_command_sub_;
  void mpc_command_callback(const ci_mpc_msgs::msg::MpcCommand::SharedPtr msg);
      

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


    //Functions
    void topic_callback(const std::shared_ptr<franka_msgs::msg::FrankaRobotState> msg);
    void updateJointStates();
    // Fills M, coriolis (no gravity), the EE pose and member J from the selected model source.
    void computeModel(Eigen::Matrix<double, 7, 7>& M, Eigen::Matrix<double, 7, 1>& coriolis,
                      Eigen::Affine3d& transform);
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
    // Model source: "franka" (libfranka via franka_hardware) or "pinocchio" (URDF, e.g. Isaac Sim)
    std::string model_source_{"franka"};
    std::string urdf_path_;
    std::string ee_frame_{"fr3_hand_tcp"};
    bool enable_input_server_{true};
    // Joint-space damping projected into the nullspace [Nms/rad]. 0 = off (real robot: joint
    // friction damps the self-motion). The sim has no joint friction, so the elbow/wrist drift.
    double nullspace_damping_{0.0};
    // Friction compensation: "none" (default), "lugre" (the upstream model in calculate_tau_friction(),
    // now scaled with dt), "coulomb" (Coulomb friction in the direction of the impedance torque plus
    // viscous friction on the measured velocity), "velocity" (Coulomb + viscous friction from the
    // measured joint velocity only: nothing at rest, so the stiffness stays as set). Identified values below.
    std::string friction_compensation_{"none"};
    bool friction_offsets_{true};          // lugre: include offset_friction (asymmetric bias)
    bool friction_abs_threshold_{false};   // lugre: zero |error| < 0.8 mm (upstream zeroes all negative errors)
    double friction_scale_{1.0};           // coulomb: scale on the identified friction
    double friction_tau_eps_{0.3};         // coulomb: tanh width on the impedance torque [Nm]
    double friction_vel_eps_{0.02};        // velocity: tanh width on the joint velocity [rad/s]
    double friction_scale_rest_{0.5};      // hybrid: torque-direction scale at rest (friction_scale while moving)
    Eigen::Matrix<double, 7, 1> joint_torque_offset_ = Eigen::Matrix<double, 7, 1>::Zero();  // added to tau_d [Nm]
    double rotational_damping_scale_{1.3};  // D_rot = scale * 2 sqrt(K Lambda)
    double rz_damping_floor_{1.0};          // D_rz >= floor * 2 sqrt(K_rz)
    std::unique_ptr<PinocchioModel> pinocchio_model_;
    bool usePinocchio() const { return model_source_ == "pinocchio"; }
    bool observer_side_model_{false};  // franka model + Pinocchio side model for the observer only (real robot)
    const double delta_tau_max_{1.0};  // per 1 ms step, scaled with dt_
    double dt_ = 0.001;                // control period [s], set from update()'s period
                
    //Impedance control variables              
    Eigen::Matrix<double, 6, 6> Lambda = IDENTITY;                                           // operational space mass matrix
    Eigen::Matrix<double, 6, 1> integrator_weights = (Eigen::VectorXd(6) << 350, 350, 0, 50, 50, 50).finished(); 
    Eigen::Matrix<double, 6, 1> integrator_limits = (Eigen::VectorXd(6) << 2.5, 2.5, 2.5, 1.5, 1.5, 0.5).finished();  // anti-windup [N, Nm]
    // Conditional integration (defaults = always integrate, as upstream). A component integrates only while
    // |error_i| < integrator_windows(i) (0 = no window), the EE speed is below integrator_velocity_window
    // [m/s] (0 = off) and, with integrator_freeze_on_saturation, its output was not clamped last cycle.
    // Otherwise it decays with rate integrator_leak [1/s] (0 = holds its value).
    Eigen::Matrix<double, 6, 1> integrator_windows = Eigen::Matrix<double, 6, 1>::Zero();
    double integrator_velocity_window_{0.0};
    double integrator_leak_{0.0};
    bool integrator_freeze_on_saturation_{false};
    bool integrator_reset_on_target_{true};  // upstream: new reference_pose zeroes the integral
    // MPC tracking (ci_mpc_msgs/MpcCommand on /cartesian_impedance_controller/mpc_command): the newest command
    // sets pose (no reference filter), desired twist (damping on the velocity error), feedforward wrench,
    // optional K / D and integral gating. Without a new command for mpc_timeout [s] (counted in control
    // cycles) the controller holds the current pose and drops twist and feedforward wrench.
    realtime_tools::RealtimeBuffer<ci_mpc_msgs::msg::MpcCommand::SharedPtr> mpc_buffer_;
    // External wrench estimate (model_source pinocchio; the real robot also has Franka's O_F_ext_hat_K): generalized
    // momentum observer r = K_O (M dq - int(tau + C^T dq - g + r) dt - p0); wrench at the EE frame = -pinv(J^T) r
    // in Franka's sign convention (force of the robot on the environment positive). tau (observer_torque_source):
    //   measured: the joints' effort state = link-side joint torque (FR3 torque sensors; Isaac: the torque PhysX
    //             transmits through the joint), friction stays out; M without rotor inertia; g if
    //             observer_gravity (FR3: tau_J includes gravity; sim: robot gravity is off);
    //   command:  the commanded torque; joint friction not cancelled by the compensation (stiction at rest) shows
    //             up in it (sim, known wrench at rest: 2-4.7 N off on the other axes); M + observer_armature.
    // Published on ~/external_wrench (geometry_msgs/WrenchStamped, base frame) and fed into F_ext (mode 1).
    std::string observer_torque_source_{"measured"};
    // sim only: torque sensor model added to the measured torques (Isaac's are exact): white noise [Nm] and a
    // per-joint offset [Nm] (FR3 torque sensors drift / have offsets; values assumed, to be measured on the robot)
    double observer_torque_noise_{0.0};
    Eigen::Matrix<double, 7, 1> observer_torque_bias_ = Eigen::Matrix<double, 7, 1>::Zero();
    std::mt19937 observer_rng_{12345};
    std::normal_distribution<double> observer_normal_{0.0, 1.0};
    bool observer_gravity_{false};
    double observer_gain_{30.0};
    // rotor inertia added to M for the observer only (the impedance law keeps its tuned M / Lambda)
    Eigen::Matrix<double, 7, 1> observer_armature_ = Eigen::Matrix<double, 7, 1>::Zero();
    bool observer_initialized_{false};
    Eigen::Matrix<double, 7, 7> observer_M_prev_ = Eigen::Matrix<double, 7, 7>::Zero();
    Eigen::Matrix<double, 7, 1> observer_integral_ = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> observer_p0_ = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> tau_ext_hat_ = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 6, 1> F_ext_hat_ = Eigen::Matrix<double, 6, 1>::Zero();
    std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>> wrench_pub_;
    int wrench_pub_every_{2};
    void updateExternalWrench(const Eigen::Matrix<double, 7, 7>& M, const Eigen::Matrix<double, 7, 1>& coriolis,
                              const Eigen::Matrix<double, 7, 1>& tau_applied);
    ci_mpc_msgs::msg::MpcCommand::SharedPtr mpc_last_;
    bool mpc_active_{false};
    double mpc_age_{0.0};
    double mpc_timeout_{0.05};
    double mpc_integral_off_leak_{5.0};    // [1/s] decay of the integral while the command disables it
    bool integral_enabled_{true};
    bool mpc_damping_set_{false};
    Eigen::Matrix<double, 6, 1> mpc_damping_ = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> twist_d_ = Eigen::Matrix<double, 6, 1>::Zero();  // desired EE twist
    Eigen::Matrix<double, 6, 1> F_ff_ = Eigen::Matrix<double, 6, 1>::Zero();     // feedforward wrench
    Eigen::Matrix<double, 6, 1> F_saturated_ = Eigen::Matrix<double, 6, 1>::Zero();  // 1 where clamped last cycle
    Eigen::Matrix<double, 6, 6> Sm = IDENTITY;                                               // task space selection matrix for positions and rotation
    Eigen::Matrix<double, 6, 6> Sf = Eigen::MatrixXd::Zero(6, 6);                            // task space selection matrix for forces
   
    Eigen::Matrix<double, 6, 6> K =  (Eigen::MatrixXd(6,6) << 2500,   0,   0,   0,   0,   0,
                                                                0, 2500,   0,   0,   0,   0,
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
  
    std::mutex position_and_orientation_d_target_mutex_;

    //Flags
    bool config_control = false;           // sets if we want to control the configuration of the robot in nullspace
    bool do_logging = false;               // set if we do log values

    //Filter-parameters
    double filter_params_{0.002};  // per 1 ms step, rescaled to dt_ in update_stiffness_and_references()
    int mode_ = 1;
};
}  // namespace cartesian_impedance_control