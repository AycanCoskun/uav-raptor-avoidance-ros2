cmake_minimum_required(VERSION 3.8)
project(raptor_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch models rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
