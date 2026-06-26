cmake_minimum_required(VERSION 3.8)
project(orchestrator)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  orchestrator/docking_fsm_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME docking_fsm_node
)

install(PROGRAMS
  orchestrator/autonomy_deadman_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME autonomy_deadman_node
)

install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_transitions orchestrator/tests/test_transitions.py)
  ament_add_pytest_test(test_vehicle_io orchestrator/tests/test_vehicle_io.py TIMEOUT 30)
  ament_add_pytest_test(test_docking_fsm_node
    orchestrator/integration/test_docking_fsm_node.py
    TIMEOUT 90
  )
  ament_add_pytest_test(test_autonomy_deadman_node
    orchestrator/integration/test_autonomy_deadman_node.py
    TIMEOUT 60
  )
endif()

ament_package()
