from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="aruco_ros",
                executable="marker_publisher",
                name="arucos",
                remappings=[
                    ("image", "/camera/image_raw"),
                    ("camera_info", "/camera/camera_info"),
                    ("markers", "/aruco/detections"),
                ],
                parameters=[
                    {
                        "image_is_rectified": True,
                        "reference_frame": "camera_link",
                        "camera_frame": "camera_link",
                    }
                ],
                output="screen",
            ),
            Node(
                package="perception",
                executable="aruco_relay",
                name="aruco_relay",
                output="screen",
            ),
        ]
    )
