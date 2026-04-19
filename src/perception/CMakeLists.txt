cmake_minimum_required(VERSION 3.8)
project(perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  perception/led_mock_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME led_mock_publisher
)

install(PROGRAMS
  perception/aruco_relay.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME aruco_relay
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_transforms perception/utils/tests/test_transforms.py)
endif()

ament_package()
