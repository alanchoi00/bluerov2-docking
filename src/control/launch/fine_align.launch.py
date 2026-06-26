from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("target_frame", default_value="map"),
            Node(
                package="control",
                executable="fine_align_node",
                name="fine_align",
                parameters=[
                    PathJoinSubstitution(
                        [FindPackageShare("control"), "config", "fine_pbvs.yaml"]
                    ),
                    {"target_frame": LaunchConfiguration("target_frame")},
                ],
                output="screen",
            ),
        ]
    )
