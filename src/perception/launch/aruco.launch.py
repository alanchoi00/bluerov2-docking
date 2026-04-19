from launch import LaunchDescription
from launch_ros.actions import Node

REFERENCE_MARKER_SIZE = 0.1  # m

# Physical sizes are total print dimensions. aruco_ros measures the black square
# only, so each size is multiplied by the black/total pixel ratio (1470/1889 =
# 0.7782).
MARKER_SIZES = [
    "201:0.15564",  # 200mm total × 0.7782
    "202:0.15564",  # 200mm total × 0.7782
    "301:0.07782",  # 100mm total × 0.7782
    "302:0.04669",  # 60mm total × 0.7782
    "303:0.04669",  # 60mm total × 0.7782
    "304:0.04669",  # 60mm total × 0.7782
    "305:0.04669",  # 60mm total × 0.7782
    "401:0.03696",  # 47.5mm total × 0.7782
    "402:0.03696",  # 47.5mm total × 0.7782
]


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
                ],
                parameters=[
                    {
                        "image_is_rectified": True,
                        "reference_frame": "camera_link",
                        "camera_frame": "camera_link",
                        "marker_size": REFERENCE_MARKER_SIZE,
                    }
                ],
                output="screen",
            ),
            Node(
                package="perception",
                executable="aruco_relay",
                name="aruco_relay",
                parameters=[
                    {
                        "reference_marker_size": REFERENCE_MARKER_SIZE,
                        "marker_sizes": MARKER_SIZES,
                    }
                ],
                output="screen",
            ),
        ]
    )
