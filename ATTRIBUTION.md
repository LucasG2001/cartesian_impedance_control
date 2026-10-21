# Attribution

- **Copied from:** `external/cartesian_impedance_control/` in this repo (supervisor repo, branch `main`, commit
  `f8d5fe258afe9f45fc1fe1f3821a4b85a7be8f98`, imported 2026-10-09; see docs/external-repos.md).
- **Original author (per README.md):** Curdin Deplazes (Pd|Z, ETH Zurich),
  https://github.com/CurdinDeplazes/cartesian_impedance_control
- **License:** Apache 2.0 (`LICENSE`, `package.xml`); source headers keep the Franka Emika GmbH copyright.
- **Version:** package.xml 0.1.8. The README targets franka_ros2 v0.13.1; this workspace uses franka_ros2 v2.2.0 +
  libfranka 0.19.0 (container thesis_franka, docs/docker.md). Needed changes: docs/controller-setup.md.
- **Changes vs. the original:** copied identical to `external/cartesian_impedance_control/` (checked with
  `diff -r` on 2026-10-13), plus this file. Changed since:
  - 2026-10-13 (Matteo, first robot test): translational stiffness x/y 2500 → 1000 in `K`
    (`include/cartesian_impedance_control/cartesian_impedance_controller.hpp`); z (1000) and rotation unchanged.
  - 2026-10-13: launch file adapted to franka_ros2 v2.2.0 (`launch/cartesian_impedance_controller.launch.py`):
    `robot_type` (default fr3) instead of `arm_id`, `robot_ip` default 192.168.1.200, `load_gripper` default false,
    `controllers_yaml:=` this package's `launch/controllers.yaml`; removed the `/home/lucas/.../set_force_torque_limits.sh`
    ExecuteProcess + OnProcessExit chain (controller spawned directly, active) and the `use_rviz` argument.
    `launch/controllers.yaml` replaced with the content of `ros2_ws/src/thesis_bringup/config/controllers.yaml`
    (v2.2.0 base: update_rate, thread_priority, both broadcasters, controller type + `arm_id`/`robot_type` fr3).
  - 2026-10-13: tuning parameters in `launch/controllers.yaml` (docs/controller-tuning.md): all gains (K, damping
    ratios, integrator, clamps, filter, torque rate, mode presets) read in `on_configure`, defaults = the old
    hardcoded values; new optional (off by default): nullspace posture spring + damping, Λ regularisation,
    integrator reset on a new target, spring-only clamp. Note: the start-up K default is the .hpp value (x/y 1000).
  - 2026-10-13: `max_reference_velocity` (rate limit of the reference, default off); yaml switched to the first
    tuning set (docs/decisions.md).
  - 2026-10-13: debug topic `~/debug` (realtime_tools RealtimePublisher, `debug.publish_rate`); new dependency
    realtime_tools (package.xml, CMakeLists.txt). yaml back to the baseline values.
  - 2026-10-13: exact nullspace projector (SVD) instead of the damped pinv, fixed posture target
    (`nullspace.q_target`), smooth start move at activation (`start_pose.*`).
  - 2026-10-13: launch file records the debug topic with ros2 bag (`record`, `bag_dir`, `bag_name` arguments).
  - 2026-10-19: target interface made thread-safe (docs/controller-setup.md fix-list a12 + a13):
    - **Deleted** `src/user_input_server.cpp` and `include/cartesian_impedance_control/user_input_server.hpp`.
      Reason: the server ran in a detached thread with its own node, wrote K / D / T and the target through raw
      pointers without synchronisation while `update()` ran at 1 kHz, and called `rclcpp::shutdown()` when its spin
      returned (would shut down the controller manager's context). `/set_pose` and `/set_param` are now services
      of the controller node (same absolute names; `user_input_client` unchanged). `set_pose` is position only;
      `set_param` logs "not supported" and changes nothing.
    - External inputs (reference_pose, set_pose, control_mode) reach `update()` through a
      `realtime_tools::RealtimeBuffer`; the robot_state callback only filters F_ext (also via a buffer), the mode-1
      z-contact switch moved into `update()` with the same condition.
    - New target checks (`include/.../target_validation.hpp`, `src/target_validation.cpp`, parameters `target.*`)
      and a gtest (`test/test_target_validation.cpp`, test_depend ament_cmake_gtest).
    - Joint state interfaces looked up by name in `on_activate` instead of index order (assert removed).
    - yaml: `max_reference_velocity` 0.05 m/s / 0.2 rad/s, new `target.*` block.
    - `realtime_publisher.h` → `realtime_publisher.hpp` (deprecated header).
- **Depends on:** `messages_fr3` (ros2_ws/src/messages_fr3, see its ATTRIBUTION.md).
