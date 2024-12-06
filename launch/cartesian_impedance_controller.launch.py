from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    arm_id_parameter_name = 'arm_id'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'
    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    arm_id = LaunchConfiguration(arm_id_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    
    # Execute the set_load.sh script
    set_load = ExecuteProcess(
        cmd=['/home/nilsjohnson/franka_ros2_ws/src/cartesian_impedance_control/launch/set_load.sh'],
        output='screen',
    )
    # Start the cartesian_impedance_controller after set_load.sh finishes
    start_controller = Node(
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
            arm_id_parameter_name,
            default_value='fr3',
            description='ID of the type of arm used. Supported values: fer, fr3, fp3'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='false',
            description='Visualize the robot in Rviz'),
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
        # Include the main Franka bringup launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])]),
            launch_arguments={robot_ip_parameter_name: robot_ip,
                              arm_id_parameter_name: arm_id,
                              load_gripper_parameter_name: load_gripper,
                              use_fake_hardware_parameter_name: use_fake_hardware,
                              fake_sensor_commands_parameter_name: fake_sensor_commands,
                              use_rviz_parameter_name: use_rviz
                              }.items(),
        ),

        ExecuteProcess(
            cmd=['/home/nilsjohnson/franka_ros2_ws/src/cartesian_impedance_control/launch/set_force_torque_limits.sh'],  # Reference to the shell script in the same folder
            output='screen',
        ),

        # First, run set_load.sh
        set_load,
        # Then, start the controller only after set_load.sh completes
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=set_load,
                on_exit=[start_controller],
            )
        ),
    ])










