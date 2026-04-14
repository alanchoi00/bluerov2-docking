from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("use_teleop_joy", default_value="false"),
            DeclareLaunchArgument("use_key", default_value="false"),
            DeclareLaunchArgument("model", default_value="bluerov2_heavy"),
            DeclareLaunchArgument(
                "world",
                default_value="ocean",
                choices=["ocean"],  # TODO: add more worlds
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("blue_sim"), "launch/sim.launch.py"]
                    )
                ),
                launch_arguments={
                    "use_sim": "true",
                    "use_rviz": LaunchConfiguration("use_rviz"),
                    "use_teleop_joy": LaunchConfiguration("use_teleop_joy"),
                    "use_key": LaunchConfiguration("use_key"),
                    "model": LaunchConfiguration("model"),
                    "gazebo_world_file": [
                        PathJoinSubstitution([FindPackageShare("sim"), "worlds/"]),
                        LaunchConfiguration("world"),
                        ".world",
                    ],
                }.items(),
            ),
        ]
    )
