# cartesian_impedance_control

### Usage for Orthopedic Drilling

#### Endeffector-config

Before launching the controller make sure the surgical drill is attached securley to the flansch of the Franka Emika Robot. Additionally the endeffector configuration needs to be updated on the FCI website. This can be done by selecting under the Settings tab in the End-Effector subpage. Select "other" in the drop down menu and upload the [endeffector-config_med_drill.json](endeffector-config_med_drill.json) file. Save the settings before you launch the controller.

#### Controller launching

To use the orthopedic drilling controller launch the cartesian impedance controller, the user input client and if logging is wished the robot trajectory logger as well in individual terminals. 

    ros2 launch cartesian_impedance_control cartesian_impedance_controller.launch.py
    ros2 run cartesian_impedance_control user_input_client
    ros2 run robot_trajectory_logger robot_trajectory_logger


#### Input server node

The input server is needed to send commands to the cartesion impedance controller. Connect the foot pedal to the usb port of your computer to be able to switch between the free floating and drilling modes without the use of your keyboard. Once the the user input node and the cartesian impedance controller have been launched the robot will move to it's reference position. By tapping on the foot pedal once, the robot will enter free floating mode. You can now place the end-effector in your desired drilling position. Next press on the foot pedal and the drilling controller will be started. If you wish to log the data from the drilling procedure make sure that you have activated the log file. 

    1) Press on foot pedal to activate free floating mode and move end-effector to deired position
    2) Press on foot pedal to activate drilling controller. Robot will allow movement along drilling axis and restrain movement deviating from it.
    #5 --> #1: activate logging
    START DRILLING
    #5 --> #2: stop logging once outlier has been triggerd
    3) Press on foot pedal to deactivate drilling controller and reenter free floating mode

It is still possible to use the keyboard to activate individual tasks over the input server node.

    #3 --> #1: Free floating mode actived. Place the end-effector in desires position
    #4 --> #1: Stiffness adapted in drilling direction. Robot will allow movement along drilling axis and restrain movement deviating from it.
    #5 --> #1: activate logging
    START DRILLING
    #5 --> #2: stop logging once outlier has been triggerd
    #4 --> #2: Drilling controller deactivated. Robot has returned to free floating mode and drill can be removed from bone and placed to next drilling position
    #4 --> #1: Activate drilling mode again if wished.

The drilling controller will restrict 
### ROS2 cartesian_impedance_controller from Pd|Z

Prerequisites:
* ROS2 humble <br />
* Libfranka 0.13.0 or newer <br />
* franka_ros2 v0.13.1 <br />

For further information, please refer to the [Franka ROS2 FCI documentation](https://support.franka.de/docs/franka_ros2.html)

Once you have everything set up, follow the steps below to get the controller running.

Clone this repository in the src directory of your franka_ros2_ws: <br />
```bash
cd franka_ros2_ws/src 
git clone https://github.com/CurdinDeplazes/cartesian_impedance_control.git
```
For the moment, you need to add the following lines of code, to your controllers.yaml file inside franka_ros2/franka_bringup/config/:
```bash
cartesian_impedance_controller:
      type: cartesian_impedance_control/CartesianImpedanceController
```

Clone the messages package in the src directory: <br />
```bash
git clone https://github.com/CurdinDeplazes/messages_fr3.git
```

Build the package or whole workspace: <br />
```bash
colcon build --packages-select cartesian_impedance_control --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release #Builds all the packages in your src folder
```

If not yet done, ensure your setup is always source by adding the following line to the end of your .bashrc file (to get access to it, you need to execute `nano .bashrc` in your home directory). : <br />
```bash
source /home/<user>/franka_ros2_ws/install/setup.sh 
```

Launch the controller: <br />
```bash
ros2 launch cartesian_impedance_control cartesian_impedance_controller.launch.py
```

Launch the client if you want to adjust parameters: <br />
``` bash
ros2 run cartesian_impedance_control user_input_client 
```

You have multiple options: 
1. Change target position of the end-effector (only works in mode 1 and not in free-float)
2. Change impedance parameter
3. Change mode (mode 1 = cartesian impedance control --> with a position / mode 2 = free-float with variable inertia)

This section should give the reader a better understanding of how the controller
works and what the functions are used for.
### Controller
After launching the controller the three startup functions on_init(), on_configure(), and on_activate() are being called before the controller starts. All these functions were taken over from the franka_example_controller package from Franka Emika and have been equipped with the required steps.
#### Startup functions
In on_init(), the constructor of the input server is called, and together with the references to the variables that need to be changed during runtime, an object of this class is created. Then, this object is multithreaded in order to make sure that the controller and server can run in parallel. During on_configure(), a FrankaRobotModel object is assigned robot parameters, over which some of the states from the robot can be read. Then, the controller subscribes to the franka_forbot_state_broadcaster, using a reliable quality of service setting. If this setting would not be specified, it might happen, that the robot does not receive the states in the beginning and has an undefined behaviour. After that, the collision behavior is assigned to the robot. For that, a client is created and a request is sent to the robot to change the collision behavior to what is specified. When on_activate() is called, the franka_robot_model_ variable is assigned its states, which have been before declared in the state_interface_configuration(). Then the initial states are assigned and mapped to the respective variables. Next to the StateInterface, also a so-called CommandInterface is created before the controller starts. This interface is later used to send the desired torques to the robot.
#### Update()
The update() function is where the controller itself is running. At the start, the robot states are again read and mapped to the respective variables. Then the rotation quaternion is created given the desired Euler angles. When changing these angles, attention has to be paid as the standard position (gripper pointing downwards and the flat part of the end-effector pointing in x-direction), has the Euler angles (PI, 0, 0). The next step is to get the position and velocity of all the joints, using the StateInterface. Then the pseudo-inverse of the Jacobian is created using the pseudoinverse() function, the error terms are alculated and the inertia of the robot is calculated. To improve the robot's behavior, the external force is smoothened using an exponential filter. The smoothing factor of choice is equal to 0.1. For increased precision, the controller uses friction compensation as described in Luba 2024. The integrator in there makes the robot push the interactor away when not at the desired position and for that, a Boolean is introduced which empties the integrator as soon as the robot has an external force of more than 10N. In the switch statement, the controller checks whether free-float or impedance control is used. If mode_ = 1, the controller uses impedance control. If mode_ = 2, the free-float is chosen. Then the nullspace torque is calculated, the impedance force is converted into joint torques, and the desired torque, which is the sum of impedance, nullspace, Coriolis, and friction torques is calculated. This result is saturated, to limit the torque ramp. For this project, a torque ramp of 100 Nm/s was used. This saturated torque is then commanded to the robot using the command interface. After that, the position and orientation intermediate steps are updated, which also use an exponential filter. At the very end of the update() function, a condition with variable terminal output frequency for debugging purposes is defined. This update loop is running at 1000 Hz and is only interrupted through robot errors and manual shutting down of the controller. At the very end, the on_deactivate() is called where the franka_robot_model_ states are released. 

### Input service
The service consists of a server and a client. Each of those subsystems has its own .cpp and .hpp file, called user_input_client.cpp, user_input_client.hpp, user_input_server.cpp and user_input_client.hpp respectively. The client has to be called in a second terminal, while the server is automatically launched when launching the controller.

#### Input server
The input server consists of a main function and one callback function per client. At first, the node is created, and then one server per changeable parameter. As of now, the pose, impedance parameter and mode have an own server. As soon as a client with the same service name is called, a callback function is executed, where the sent request is assigned to the parameter running on the controller. 

#### Input client
The input client consists of one main function only. At first rclcpp::init(argc,argv) is called to initialize communication. Then, the node on which all different clients are created on. After that, a request for all the different parameters one wants to change. Next, using a std::cin statement, the desired parameters to change are read in by the program, assigned to the respective parts of the request, the request is sent and the node spined until a successful response is received. When the server is stopped, the node is shut down. If it is desired to change a new parameter, this first needs to be given into the constructor of the UserInputServer so that the correct parameter can be changed during runtime. Then a client and a server have to be implemented in the same way as already inside the respective files. In addition to that, if
required new messages have to be defined.

### Custom messages
In the messages_fr3 package, some messages are already defined. The message definition for the service can be found in meessages_fr3/srv. When creating a new message, first, the definition has to be created in .srv file in CamelCase. Then, type the request, together with the type of the variable, followed by three hyphens and the desired response. The last step, before the message can be included as a lowercase_lowerscase.hpp file is to add it to the CMakeLists.txt file below the already existing messages in the rosidl_generate_interface block.
