// Copyright (c) 2024 Franka Robotics GmbH
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

#include <tinyxml2.h>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <string>

namespace robot_utils {
using namespace std::chrono_literals;

inline std::string getRobotNameFromDescription(const std::string& robot_description,
                                               const rclcpp::Logger& logger) {
  std::string robot_name;
  tinyxml2::XMLDocument doc;

  if (doc.Parse(robot_description.c_str()) != tinyxml2::XML_SUCCESS) {
    RCLCPP_ERROR(logger, "Failed to parse robot_description.");
    return "";
  }

  tinyxml2::XMLElement* robot_xml = doc.FirstChildElement("robot");
  if (robot_xml) {
    robot_name = robot_xml->Attribute("name");
    if (robot_name.empty()) {
      RCLCPP_ERROR(logger, "Failed to get robot name from XML.");
      return "";
    }
    RCLCPP_INFO(logger, "Extracted Robot Name: %s", robot_name.c_str());
  } else {
    RCLCPP_ERROR(logger, "Robot element not found in XML.");
  }
  return robot_name;
}

const auto time_out = 1000ms;

}  // namespace robot_utils
