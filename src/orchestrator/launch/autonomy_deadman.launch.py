from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("deadman_button", default_value="5"),
            Node(
                package="orchestrator",
                executable="autonomy_deadman_node",
                name="autonomy_deadman",
                parameters=[
                    {
                        "deadman_button": LaunchConfiguration("deadman_button"),
                        "input_topic": "/cmd_vel_auto",
                        "output_topic": "/cmd_vel",
                    }
                ],
                output="screen",
            ),
        ]
    )
