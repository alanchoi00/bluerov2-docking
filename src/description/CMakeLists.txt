cmake_minimum_required(VERSION 3.8)
project(description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY models worlds rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/hooks.dsv.in")

ament_package()
