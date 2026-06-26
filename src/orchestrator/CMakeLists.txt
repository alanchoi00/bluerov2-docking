cmake_minimum_required(VERSION 3.8)
project(orchestrator)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_transitions orchestrator/tests/test_transitions.py)
  ament_add_pytest_test(test_vehicle_io orchestrator/tests/test_vehicle_io.py TIMEOUT 30)
endif()

ament_package()
