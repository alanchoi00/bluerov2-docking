from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="orchestrator",
                executable="docking_fsm_node",
                name="docking_fsm",
                parameters=[
                    PathJoinSubstitution(
                        [FindPackageShare("orchestrator"), "config", "docking_fsm.yaml"]
                    )
                ],
                output="screen",
            ),
        ]
    )
