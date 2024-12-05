#!/bin/bash

ros2 service call /service_server/set_load franka_msgs/srv/SetLoad "{
mass: 0,
center_of_mass: [0.0, 0.0, 0.0],
load_inertia: [0.000001, 0, 0, 0 , 0.000001, 0, 0, 0 , 0.000001]
}"
