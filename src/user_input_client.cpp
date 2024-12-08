#include <rclcpp/rclcpp.hpp>
#include <messages_fr3/srv/set_pose.hpp>
#include <messages_fr3/srv/set_param.hpp>
#include <messages_fr3/srv/set_stiffness.hpp>
#include <messages_fr3/srv/set_mode.hpp>
#include <messages_fr3/srv/controller_activation.hpp>
#include <messages_fr3/srv/planner_service.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <array>
#include <cmath>
#include <termios.h> // For terminal control definitions
#include <unistd.h>  // For STDIN_FILENO
#include <cstdio>    // For getchar()

// Function to get a single key press (Linux-compatible)
char get_key() {
    struct termios oldt, newt;
    char ch;
    tcgetattr(STDIN_FILENO, &oldt); // Get current terminal attributes
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &newt); // Apply new attributes
    ch = getchar(); // Get a single character
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt); // Restore original attributes
    return ch;
}

int main(int argc, char **argv) {
    
    rclcpp::init(argc, argv);
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("user_input_client");

    rclcpp::Client<messages_fr3::srv::SetPose>::SharedPtr pose_client =
        node->create_client<messages_fr3::srv::SetPose>("set_pose");
    auto pose_request = std::make_shared<messages_fr3::srv::SetPose::Request>();

    rclcpp::Client<messages_fr3::srv::SetParam>::SharedPtr param_client =
        node->create_client<messages_fr3::srv::SetParam>("set_param");
    auto param_request = std::make_shared<messages_fr3::srv::SetParam::Request>();

    rclcpp::Client<messages_fr3::srv::SetMode>::SharedPtr mode_client =
        node->create_client<messages_fr3::srv::SetMode>("set_mode");
    auto mode_request = std::make_shared<messages_fr3::srv::SetMode::Request>();

    rclcpp::Client<messages_fr3::srv::ControllerActivation>::SharedPtr activation_client =
        node->create_client<messages_fr3::srv::ControllerActivation>("controller_activation");
    auto activation_request = std::make_shared<messages_fr3::srv::ControllerActivation::Request>();

    rclcpp::Client<messages_fr3::srv::SetStiffness>::SharedPtr stiffness_client =
        node->create_client<messages_fr3::srv::SetStiffness>("set_stiffness");
    auto stiffness_request = std::make_shared<messages_fr3::srv::SetStiffness::Request>();

    rclcpp::Client<messages_fr3::srv::PlannerService>::SharedPtr planner_client =
        node->create_client<messages_fr3::srv::PlannerService>("planner_service");
    auto planner_request = std::make_shared<messages_fr3::srv::PlannerService::Request>();

    int task_selection, pose_selection, param_selection, mode_selection, activation_selection, planner_selection;

    int step_counter = 0;

    while (rclcpp::ok()){

        std::cout << "Enter the next task: \n [1] --> Change position \n [2] --> Change impedance parameters \n"
                << " [3] --> Choose control mode \n [4] --> Drill Position activation \n [5] --> Logging" << std::endl;
                
        // Read input as a string to handle both numeric and character inputs
        char key = get_key();

       if (key == 'b') {

            step_counter += 1.0;

            switch (step_counter)
            {
            case 1:{

                mode_request->mode = true;

                auto mode_result = mode_client->async_send_request(mode_request);
                if(rclcpp::spin_until_future_complete(node, mode_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", mode_result.get()->success);
                } else {
                    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service setMode");
                }

                break;
            }

            case 2:{

                activation_request->controller_activation = true;
                
                auto activation_result = activation_client->async_send_request(activation_request);
                if(rclcpp::spin_until_future_complete(node, activation_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", activation_result.get()->success);
                } else {
                    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service controllerActivation");
                }

                planner_request->command = "a";
                auto planner_result = planner_client->async_send_request(planner_request);
                if(rclcpp::spin_until_future_complete(node, planner_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", planner_result.get()->success);
                } else {
                    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service plannerService");
                }

                break;
            }

            case 3:{
                
                activation_request->controller_activation = false;
                
                auto activation_result = activation_client->async_send_request(activation_request);
                if(rclcpp::spin_until_future_complete(node, activation_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", activation_result.get()->success);
                } else {
                    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service controllerActivation");
                }

                planner_request->command = "d";
                auto planner_result = planner_client->async_send_request(planner_request);
                if(rclcpp::spin_until_future_complete(node, planner_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", planner_result.get()->success);
                } else {
                    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service plannerService");
                }

                step_counter = 1;
            
                break;
            }

            default:{
                    std::cout << "Invalid input, please try again\n";
                    break;   
                }
            }

        }
        
        if (isdigit(key)) {
            task_selection = key - '0'; // Convert char to int
            switch (task_selection){
                case 1:{ 
                    std::cout << "Enter new goal position: \n [1] --> 0.5, -0.4, 0.5 \n [2] --> DO NOT USE \n [3] --> 0.5, 0.4, 0.5\n";
                    std::cin >> pose_selection;
                    switch (pose_selection){
                        case 1:{
                            pose_request->x = 0.5;
                            pose_request->y = -0.4;
                            pose_request->z = 0.5;
                            pose_request->roll = M_PI;
                            pose_request->pitch = 0.0;
                            pose_request->yaw = M_PI_2;
                            break;
                        }
                        case 2:{
                            std::cout << "Enter your desired position and orientation" << std::endl;
                            std::array<double, 6> pose;
                            for (long unsigned int i = 0; i<pose.size(); ++i){
                                std::cin >> pose[i];
                            }
                            pose_request->x = pose[0];
                            pose_request->y = pose[1];
                            pose_request->z = pose[2];
                            pose_request->roll = pose[3];
                            pose_request->pitch = pose[4];
                            pose_request->yaw = pose[5];
                            break;
                        }
                        case 3:{
                            pose_request->x = 0.5;
                            pose_request->y = 0.4;
                            pose_request->z = 0.5;
                            pose_request->roll = M_PI;
                            pose_request->pitch = 0.0;
                            pose_request->yaw = M_PI_2;
                            break;
                        }
                        default:{
                            pose_request->x = 0.5;
                            pose_request->y = 0.0;
                            pose_request->z = 0.4;
                            pose_request->roll = M_PI;
                            pose_request->pitch = 0.0;
                            pose_request->yaw = M_PI_2;
                            break;
                        }
                    }
                    auto pose_result = pose_client->async_send_request(pose_request);
                    if(rclcpp::spin_until_future_complete(node, pose_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                        std::cout << "Hot geklappt";
                        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Worked: %d", pose_result.get()->success);
                    } else {
                        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service setPose");
                    }
                    break;
                }
                case 2:{
                    std::cout << "Enter new inertia: \n [1] --> N/A \n [2] --> N/A \n [3] --> N/A\n";
                    std::cin >> param_selection;
                    switch(param_selection){
                        case 1:{
                            param_request->a = 2;
                            param_request->b = 0.5;
                            param_request->c = 0.5;
                            param_request->d = 2;
                            param_request->e = 0.5;
                            param_request->f = 0.5;
                            break;
                        }
                        default:{
                            param_request->a = 1;
                            param_request->b = 1;
                            param_request->c = 1;
                            param_request->d = 1;
                            param_request->e = 1;
                            param_request->f = 1;
                            break;
                        }
                    }
                    auto param_result = param_client->async_send_request(param_request);
                    if(rclcpp::spin_until_future_complete(node, param_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", param_result.get()->success);
                    } else {
                        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service setParam");
                    }
                    break;
                }
                case 3:{
                    std::cout << "Enter new mode: \n [1] --> Free float \n [2] --> Impedance Control\n";
                    std::cin >> mode_selection;
                    switch(mode_selection){
                        case 1:{
                            mode_request->mode = true;
                            break;
                        }
                        case 2:{
                            mode_request->mode = false;
                            break;
                        }
                        default:{
                            mode_request->mode = false;
                            break;
                        }
                    }
                    auto mode_result = mode_client->async_send_request(mode_request);
                    if(rclcpp::spin_until_future_complete(node, mode_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", mode_result.get()->success);
                    } else {
                        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service setMode");
                    }
                    break;
                }
                case 4:{
                    std::cout << "Activate drill control: \n [1] --> Activate \n [2] --> Deactivate\n";
                    std::cin >> activation_selection;
                    switch(activation_selection){
                        case 1:{
                            activation_request->controller_activation = true;
                            break;
                        }
                        case 2:{
                            activation_request->controller_activation = false;
                            break;
                        }
                        default:{
                            activation_request->controller_activation = false;
                            break;
                        }
                    }
                    auto activation_result = activation_client->async_send_request(activation_request);
                    if(rclcpp::spin_until_future_complete(node, activation_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", activation_result.get()->success);
                    } else {
                        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service controllerActivation");
                    }
                    break;
                }
                /* case 5: {
                    std::cout << "Set stiffness: \n [1] --> adjust z component \n [2] --> default values\n";
                    std::cin >> param_selection;
                    switch(param_selection){
                        case 1:{
                            stiffness_request->a = 2500.0;
                            stiffness_request->b = 2500.0;
                            stiffness_request->c = 0.0;
                            stiffness_request->d = 100.0;
                            stiffness_request->e = 100.0;
                            stiffness_request->f = 20.0;
                            stiffness_request-> drillactivation = true;
                            break;
                        }
                        case 2:{
                            stiffness_request->a = 2000.0;
                            stiffness_request->b = 2000.0;
                            stiffness_request->c = 2000.0;
                            stiffness_request->d = 100.0;
                            stiffness_request->e = 100.0;
                            stiffness_request->f = 20.0;
                            stiffness_request-> drillactivation = false;
                            break;
                        }
                        default:{
                            stiffness_request->a = 2000.0;
                            stiffness_request->b = 2000.0;
                            stiffness_request->c = 2000.0;
                            stiffness_request->d = 100.0;
                            stiffness_request->e = 100.0;
                            stiffness_request->f = 20.0;
                            stiffness_request-> drillactivation = true;
                            break;
                        }
                    }
                    auto stiffness_result = stiffness_client->async_send_request(stiffness_request);
                    if(rclcpp::spin_until_future_complete(node, stiffness_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", stiffness_result.get()->success);
                    } else {
                        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service setStiffness");
                    } break;
                } */
                case 5:{
                    std::cout << "Activate logging: \n [1] --> Activate \n [2] --> Deactivate\n";
                    std::cin >> planner_selection;
                    switch(planner_selection){
                        case 1:{
                            planner_request->command = "a";
                            break;
                        }
                        case 2:{
                            planner_request->command = "d";
                            break;
                        }
                        default:{
                            planner_request->command = "d";
                            break;
                        }           
                    }
                    auto planner_result = planner_client->async_send_request(planner_request);
                    if(rclcpp::spin_until_future_complete(node, planner_result) ==  rclcpp::FutureReturnCode::SUCCESS){
                        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sum: %d", planner_result.get()->success);
                    } else {
                        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service plannerService");
                    } break;
                }
                default:{
                    std::cout << "Invalid selection, please try again\n";
                    break;   
                }
            }
        }
    }    
    rclcpp::shutdown();
    return 0;
}
