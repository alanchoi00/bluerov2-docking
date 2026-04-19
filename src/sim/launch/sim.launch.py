from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("use_joy", default_value="false"),
            DeclareLaunchArgument("use_key", default_value="false"),
            DeclareLaunchArgument("use_ardusub", default_value="true"),
            DeclareLaunchArgument("flight_mode", default_value="POSHOLD"),
            DeclareLaunchArgument("use_mock_led", default_value="true"),
            DeclareLaunchArgument("use_aruco", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("blue_sim"), "launch/sim.launch.py"]
                    )
                ),
                launch_arguments={
                    "use_sim": "true",
                    "use_rviz": LaunchConfiguration("use_rviz"),
                    "use_joy": LaunchConfiguration("use_joy"),
                    "use_key": LaunchConfiguration("use_key"),
                    "model": "bluerov2_heavy",
                    "use_ardusub": LaunchConfiguration("use_ardusub"),
                    "flight_mode": LaunchConfiguration("flight_mode"),
                    "gazebo_world_file": [
                        PathJoinSubstitution([FindPackageShare("sim"), "worlds"]),
                        "/ocean.world",
                    ],
                }.items(),
            ),
            # Needs to add this camera info bridge since 3rd party ardusub_driver only bridges /camera/image_raw
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=[
                    "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
                ],
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("perception"), "launch/led_mock.launch.py"]
                    )
                ),
                condition=IfCondition(LaunchConfiguration("use_mock_led")),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("perception"), "launch/aruco.launch.py"]
                    )
                ),
                condition=IfCondition(LaunchConfiguration("use_aruco")),
            ),
        ]
    )
