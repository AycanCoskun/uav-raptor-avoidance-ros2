cmake_minimum_required(VERSION 3.8)
project(raptor_lidar)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(raptor_interfaces REQUIRED)

# ROS-independent detection logic, unit tested on its own.
add_library(hazard_detector STATIC src/hazard_detector.cpp)
target_include_directories(hazard_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(hazard_detector PUBLIC cxx_std_17)

add_executable(lidar_hazard_node src/lidar_hazard_node.cpp)
target_link_libraries(lidar_hazard_node
  hazard_detector
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${raptor_interfaces_TARGETS})

install(TARGETS lidar_hazard_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # ament_copyright does not know AGPL-3.0; cpplint still checks that every file has a
  # copyright line.
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_hazard_detector test/test_hazard_detector.cpp)
  target_link_libraries(test_hazard_detector hazard_detector)
endif()

ament_package()
