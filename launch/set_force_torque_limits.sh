#!/bin/bash

ros2 service call /service_server/set_force_torque_collision_behavior franka_msgs/srv/SetForceTorqueCollisionBehavior "{
  lower_torque_thresholds_nominal: [100.0, 100.0, 100.0, 100.0, 60.0, 60.0, 18.0],
  upper_torque_thresholds_nominal: [100.0, 100.0, 100.0, 100.0, 60.0, 60.0, 18.0],
  lower_force_thresholds_nominal: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
  upper_force_thresholds_nominal: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
}"