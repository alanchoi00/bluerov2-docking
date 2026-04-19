cmake_minimum_required(VERSION 3.8)
project(sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY models worlds launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES docking_sim.rviz
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/hooks.dsv.in")

ament_package()
