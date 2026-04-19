from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="aruco_ros",
            executable="aruco_ros",
            name="aruco_slope",
            remappings=[
                ("image", "/camera/image_raw"),
                ("camera_info", "/camera/camera_info"),
                ("detections", "/aruco/slope/detections"),
            ],
            parameters=[{
                "image_is_rectified": True,
                "aruco_dictionary_id": "DICT_5X5_1000",
                "marker_size": 0.2,
                "reference_frame": "camera_link",
                "camera_frame": "camera_link",
            }],
            output="screen",
        ),
        Node(
            package="aruco_ros",
            executable="aruco_ros",
            name="aruco_backplate",
            remappings=[
                ("image", "/camera/image_raw"),
                ("camera_info", "/camera/camera_info"),
                ("detections", "/aruco/backplate/detections"),
            ],
            parameters=[{
                "image_is_rectified": True,
                "aruco_dictionary_id": "DICT_5X5_1000",
                "marker_size": 0.06,
                "reference_frame": "camera_link",
                "camera_frame": "camera_link",
            }],
            output="screen",
        ),
        Node(
            package="perception",
            executable="aruco_relay",
            name="aruco_relay",
            output="screen",
        ),
    ])
