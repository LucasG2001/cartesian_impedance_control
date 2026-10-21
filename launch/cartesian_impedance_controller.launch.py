# Adapted to franka_ros2 v2.2.0 (2026-10-13, see ATTRIBUTION.md): includes franka_bringup/launch/franka.launch.py
# with robot_type (instead of arm_id) and this package's launch/controllers.yaml, then spawns
# cartesian_impedance_controller directly (active). Removed: the set_force_torque_limits.sh ExecuteProcess +
# OnProcessExit chain (absolute path of the original author's machine) and the use_rviz argument (not declared
# by franka.launch.py in v2.2.0).
# Added 2026-10-13: record:=true (default) records /cartesian_impedance_controller/debug and
# /franka_robot_state_broadcaster/robot_state with ros2 bag into
# bag_dir/<YYYY-MM-DD_HHMM>_<bag_name> (docs/controller-tuning.md); Ctrl+C on the launch also closes the bag.

import datetime
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def bag_recorder(context):
    # Bag folder name is fixed at launch time: <bag_dir>/<YYYY-MM-DD_HHMM>_<bag_name>
    if LaunchConfiguration('record').perform(context).lower() not in ('true', '1'):
        return []
    bag_dir = os.path.expanduser(LaunchConfiguration('bag_dir').perform(context))
    bag_name = LaunchConfiguration('bag_name').perform(context)
    os.makedirs(bag_dir, exist_ok=True)
    output = os.path.join(bag_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S') + '_' + bag_name)
    return [ExecuteProcess(
        # robot_state: external force/torque estimate, contact/collision flags, current/last motion errors (reflexes)
        cmd=['ros2', 'bag', 'record', '-o', output, '/cartesian_impedance_controller/debug',
             '/franka_robot_state_broadcaster/robot_state'],
        output='screen',
    )]


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    robot_type_parameter_name = 'robot_type'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    robot_type = LaunchConfiguration(robot_type_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)

    controllers_yaml = PathJoinSubstitution(
        [FindPackageShare('cartesian_impedance_control'), 'launch', 'controllers.yaml'])

    # Spawn impedance controller node (loaded, configured and activated)
    impedance_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['cartesian_impedance_controller'],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            default_value='192.168.1.200',
            description='Hostname or IP address of the robot.'),
        DeclareLaunchArgument(
            robot_type_parameter_name,
            default_value='fr3',
            description='ID of the type of arm used. Supported values: fer, fr3, fp3'),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            default_value='false',
            description='Use fake hardware'),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value='false',
            description="Fake sensor commands. Only valid when '{}' is true".format(
                use_fake_hardware_parameter_name)),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='false',
            description='Use Franka Gripper as an end-effector, otherwise, the robot is loaded '
                        'without an end-effector.'),
        DeclareLaunchArgument(
            'record',
            default_value='true',
            description='Record /cartesian_impedance_controller/debug with ros2 bag.'),
        DeclareLaunchArgument(
            'bag_dir',
            default_value='~/matteo/thesis/bags',
            description='Folder for the bags (git-ignored).'),
        DeclareLaunchArgument(
            'bag_name',
            default_value='run',
            description='Suffix of the bag folder name, e.g. test4_posture.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])]),
            launch_arguments={
                robot_ip_parameter_name: robot_ip,
                robot_type_parameter_name: robot_type,
                load_gripper_parameter_name: load_gripper,
                use_fake_hardware_parameter_name: use_fake_hardware,
                fake_sensor_commands_parameter_name: fake_sensor_commands,
                'controllers_yaml': controllers_yaml,
            }.items(),
        ),

        OpaqueFunction(function=bag_recorder),
        impedance_spawner,
    ])
