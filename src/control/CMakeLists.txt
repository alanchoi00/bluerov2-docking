cmake_minimum_required(VERSION 3.8)
project(control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Node executables land here in #30; for now this package ships the pure-Python
# control law (control/pbvs.py) consumed by the #29 prototype and the #30 node.

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_pbvs control/tests/test_pbvs.py)
endif()

ament_package()
