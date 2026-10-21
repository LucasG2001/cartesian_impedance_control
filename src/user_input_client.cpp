#include <rclcpp/rclcpp.hpp>
#include "messages_fr3/srv/set_pose.hpp"
#include "messages_fr3/srv/set_param.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <array>
#include <cmath>

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("user_input_client");

    rclcpp::Client<messages_fr3::srv::SetPose>::SharedPtr pose_client =
        node->create_client<messages_fr3::srv::SetPose>("set_pose");
    auto pose_request = std::make_shared<messages_fr3::srv::SetPose::Request>();

    rclcpp::Client<messages_fr3::srv::SetParam>::SharedPtr param_client =
        node->create_client<messages_fr3::srv::SetParam>("set_param");
    auto param_request = std::make_shared<messages_fr3::srv::SetParam::Request>();

    int task_selection, pose_selection, param_selection;

    while (rclcpp::ok()){
        std::cout << "Enter the next task: \n [1] --> Change position \n [2] --> Change impedance parameters" << std::endl;
        std:: cin >> task_selection;
        switch (task_selection){
            case 1:{ 
                std::cout << "Enter new goal position: \n [1] --> 0.5, -0.4, 0.5 \n [2] --> custom position \n [3] --> 0.5, 0.4, 0.5\n";
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
                    case 2: {
                        std::cout << "Enter your desired position and orientation." << std::endl;

                        double x, y, z;
                        std::cout << "Enter X position: ";
                        std::cin >> x;
                        std::cout << "Enter Y position: ";
                        std::cin >> y;
                        std::cout << "Enter Z position: ";
                        std::cin >> z;

                        // Confirm input
                        std::cout << "\nYou entered:"
                                << "\n  X = " << x
                                << "\n  Y = " << y
                                << "\n  Z = " << z;

                        pose_request->x = x;
                        pose_request->y = y;
                        pose_request->z = z;

                        // Fixed orientation
                        pose_request->roll = M_PI;      // 180 degrees
                        pose_request->pitch = 0.0;
                        pose_request->yaw = M_PI_2;     // 90 degrees

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
            default:{
                std::cout << "Invalid selection, please try again\n";
                break;   
            }
        }
    }    
    rclcpp::shutdown();
    return 0;
}